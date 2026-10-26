"""Daily yield triage: is a yield drop a calculation error, a measurement
error, or a true quality shift in upstream material, process or equipment?

Pipeline for each failure mode:
1. Detect: exact binomial test of each day's fail count against a reference
   window (one-sided, p < 0.00135, the 3-sigma tail). Consecutive flagged days
   become one episode.
2. Calculation check: recompute the metric from raw readings in SQL and
   compare with the stored value.
3. Localize: G-test and Cramer's V of fail vs. each factor (tester, racks,
   lots, shift, protocol), plus a gradient-boosting model with permutation
   importance as an independent ML ranking of the same factors.
4. Retest evidence: share of episode fails that pass when retested on other
   equipment.
5. Classify with explicit rules and write a preliminary root-cause note for
   engineering review.

The code never reads the injected ground truth; scoring happens in score().
"""
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

MODES = {"K_HIGH": "fail_k", "CAP_LOW": "fail_cap", "RLS_HIGH": "fail_rls"}
RETEST_COL = {"K_HIGH": ("retest_k_mv_day", ">", "k_usl_mv_day"),
              "CAP_LOW": ("retest_cap_ah", "<", "cap_lsl_ah"),
              "RLS_HIGH": ("retest_rls_mohm", ">", "rls_usl_mohm")}
EQUIPMENT = ["grading_tester", "formation_rack", "aging_rack"]
MATERIAL = ["electrode_lot", "electrolyte_lot"]
FACTORS = EQUIPMENT + MATERIAL + ["shift", "protocol"]
ALPHA = 0.00135


@dataclass
class Episode:
    mode: str
    day_first: int
    day_last: int
    n_cells: int = 0
    n_fail: int = 0
    fail_rate: float = 0.0
    ref_rate: float = 0.0
    calc_mismatch_rate: float = 0.0
    retest_pass_rate: float = float("nan")
    factor_stats: list = field(default_factory=list)
    ml_importance: dict = field(default_factory=dict)
    top_factor: str = ""
    hot_levels: list = field(default_factory=list)
    hot_share_of_fails: float = 0.0
    classification: str = ""
    action: str = ""


def detect(con: duckdb.DuckDBPyConnection, ref_days=(10, 25), max_gap: int = 2) -> list[Episode]:
    daily = con.execute(
        "SELECT grade_day, COUNT(*) n, SUM(fail_k::INT) k, SUM(fail_cap::INT) cap, "
        "SUM(fail_rls::INT) rls, AVG((protocol='fast')::INT) fast "
        "FROM cell_results GROUP BY grade_day ORDER BY grade_day").df()
    ref = daily[(daily.grade_day >= ref_days[0]) & (daily.grade_day <= ref_days[1])]
    col = {"K_HIGH": "k", "CAP_LOW": "cap", "RLS_HIGH": "rls"}
    episodes = []
    for mode, c in col.items():
        p0 = ref[c].sum() / ref.n.sum()
        p = stats.binom.sf(daily[c] - 1, daily.n, p0)
        flagged = daily.grade_day[(p < ALPHA) & (daily.grade_day > ref_days[1])].tolist()
        run = []
        for d in flagged:
            if run and d - run[-1] > max_gap + 1:
                episodes.append(Episode(mode, run[0], run[-1], ref_rate=float(p0)))
                run = []
            run.append(d)
        if run:
            episodes.append(Episode(mode, run[0], run[-1], ref_rate=float(p0)))
    return episodes


def _calc_mismatch(con, ep: Episode) -> float:
    if ep.mode != "K_HIGH":
        return 0.0   # capacity and R_LS are stored as measured, nothing derived
    return con.execute(
        "SELECT AVG((ABS(k_stored_mv_day - k_raw) > 1e-6 * GREATEST(ABS(k_raw),1e-3))::INT) "
        "FROM (SELECT k_stored_mv_day, (ocv1_v-ocv2_v)*1000.0/((epoch(ocv2_ts)-epoch(ocv1_ts))"
        "/86400.0) k_raw FROM cells WHERE grade_day BETWEEN ? AND ?)",
        [ep.day_first, ep.day_last]).fetchone()[0]


def _retest_pass(con, ep: Episode) -> float:
    col, op, lim = RETEST_COL[ep.mode]
    fail = MODES[ep.mode]
    return con.execute(
        f"SELECT AVG((NOT (t.{col} {op} l.{lim}))::INT) FROM cell_results r "
        f"JOIN retests t USING (cell_id) CROSS JOIN limits l "
        f"WHERE r.{fail} AND r.grade_day BETWEEN ? AND ?",
        [ep.day_first, ep.day_last]).fetchone()[0]


def _localize(df: pd.DataFrame, fail: str, ref_rate: float, seed: int = 0):
    y = df[fail].to_numpy().astype(int)
    rows = []
    for f in FACTORS:
        tab = pd.crosstab(df[f], y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            rows.append(dict(factor=f, g_stat=0.0, p_value=1.0, cramers_v=0.0))
            continue
        g, p, dof, _ = stats.chi2_contingency(tab, lambda_="log-likelihood")
        v = np.sqrt(g / (tab.values.sum() * (min(tab.shape) - 1)))
        rows.append(dict(factor=f, g_stat=float(g), p_value=float(p), cramers_v=float(v)))
    rows.sort(key=lambda r: r["cramers_v"], reverse=True)

    # Independent ML ranking: gradient boosting on the categorical factors,
    # permutation importance scored on held-out cells so it is not inflated
    # by memorized noise on high-cardinality factors.
    X = np.column_stack([pd.Categorical(df[f]).codes for f in FACTORS])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))[:min(len(df), 80_000)]
    tr, te = idx[: idx.size // 2], idx[idx.size // 2:]
    clf = HistGradientBoostingClassifier(categorical_features=list(range(len(FACTORS))),
                                         max_iter=100, learning_rate=0.05,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    imp = permutation_importance(clf, X[te], y[te], scoring="roc_auc",
                                 n_repeats=5, random_state=seed)
    ml = {f: float(m) for f, m in zip(FACTORS, imp.importances_mean)}

    top = rows[0]["factor"]
    lv = df.groupby(top)[fail].agg(["sum", "count"])
    n_levels = len(lv)
    pv = stats.binom.sf(lv["sum"] - 1, lv["count"], ref_rate)
    hot = lv[(pv < 0.01 / n_levels) & (lv["sum"] / lv["count"] > 2 * ref_rate)]
    share = float(hot["sum"].sum() / max(lv["sum"].sum(), 1))
    return rows, ml, top, sorted(hot.index.tolist()), share


def classify(ep: Episode) -> None:
    localized = ep.hot_share_of_fails >= 0.6 and ep.factor_stats[0]["cramers_v"] >= 0.05
    if ep.calc_mismatch_rate > 0.05:
        ep.classification = "calculation_error"
        ep.action = ("Stored metric disagrees with the value recomputed from raw readings. "
                     "Fix the ETL, recompute the metric for the window, re-disposition the "
                     "cells, and do not quarantine material.")
    elif localized and ep.top_factor in EQUIPMENT and ep.retest_pass_rate >= 0.7:
        ep.classification = "measurement_error"
        ep.action = (f"Fails concentrate on {ep.top_factor} {', '.join(ep.hot_levels)} and "
                     f"{ep.retest_pass_rate:.0%} pass on other equipment. Take the unit "
                     "down for calibration, regrade its cells, and release the ones that pass.")
    elif localized and ep.top_factor in MATERIAL:
        ep.classification = "true_quality_shift"
        ep.action = (f"Fails track upstream {ep.top_factor} {', '.join(ep.hot_levels)} across "
                     "equipment and persist on retest. Quarantine the lots, notify the upstream "
                     "process owner, and hold related WIP pending root cause.")
    elif localized and ep.top_factor in EQUIPMENT:
        ep.classification = "true_quality_shift"
        ep.action = (f"Fails persist on retest and concentrate on {ep.top_factor} "
                     f"{', '.join(ep.hot_levels)}. Treat as an equipment-induced process "
                     "shift: hold output from that unit and inspect.")
    elif ep.retest_pass_rate >= 0.7:
        ep.classification = "measurement_error"
        ep.action = "Not localized but fails clear on retest. Review test-system health."
    else:
        ep.classification = "unresolved"
        ep.action = "No single factor or calculation explains the excursion. Escalate."


def run(con: duckdb.DuckDBPyConnection) -> list[Episode]:
    episodes = detect(con)
    for ep in episodes:
        cols = ", ".join(FACTORS)
        df = con.execute(
            f"SELECT {cols}, {MODES[ep.mode]} FROM cell_results "
            "WHERE grade_day BETWEEN ? AND ?", [ep.day_first, ep.day_last]).df()
        fail = MODES[ep.mode]
        ep.n_cells, ep.n_fail = len(df), int(df[fail].sum())
        ep.fail_rate = ep.n_fail / ep.n_cells
        ep.calc_mismatch_rate = float(_calc_mismatch(con, ep))
        ep.retest_pass_rate = float(_retest_pass(con, ep))
        (ep.factor_stats, ep.ml_importance, ep.top_factor,
         ep.hot_levels, ep.hot_share_of_fails) = _localize(df, fail, ep.ref_rate)
        classify(ep)
    return episodes


def score(episodes: list[Episode], truth: dict) -> dict:
    """Compare triage output with the hidden ground truth."""
    matched, results = set(), []
    for ev in truth["events"]:
        hits = [i for i, ep in enumerate(episodes)
                if ep.mode == ev["mode"] and ep.day_first <= ev["grade_day_last"]
                and ep.day_last >= ev["grade_day_first"]]
        matched.update(hits)
        ep = episodes[hits[0]] if hits else None
        results.append(dict(
            event_id=ev["event_id"], kind=ev["kind"], detected=bool(hits),
            classified_as=ep.classification if ep else None,
            correct=bool(ep and ep.classification == ev["kind"]),
            delay_days=(ep.day_first - ev["grade_day_first"]) if ep else None,
        ))
    false_alarms = [dict(mode=episodes[i].mode, days=[episodes[i].day_first, episodes[i].day_last],
                         classification=episodes[i].classification)
                    for i in range(len(episodes)) if i not in matched]
    return dict(events=results,
                n_detected=sum(r["detected"] for r in results),
                n_correct=sum(r["correct"] for r in results),
                n_events=len(results), false_alarm_episodes=false_alarms)


def rca_markdown(ep: Episode) -> str:
    lines = [
        f"### {ep.mode} excursion, grading days {ep.day_first}-{ep.day_last}",
        f"*Preliminary root-cause analysis generated automatically for engineering review.*",
        "",
        f"- Fail rate {ep.fail_rate:.2%} vs reference {ep.ref_rate:.2%} "
        f"({ep.n_fail:,} of {ep.n_cells:,} cells)",
        f"- Stored vs recomputed metric mismatch: {ep.calc_mismatch_rate:.0%}",
        f"- Retest pass rate on other equipment: {ep.retest_pass_rate:.0%}",
        f"- Most associated factor: **{ep.top_factor}** "
        f"(Cramer's V {ep.factor_stats[0]['cramers_v']:.3f}); hot levels "
        f"{', '.join(ep.hot_levels) or 'none'} hold {ep.hot_share_of_fails:.0%} of fails",
        f"- ML permutation importance (ROC-AUC drop): "
        + ", ".join(f"{k} {v:.3f}" for k, v in
                    sorted(ep.ml_importance.items(), key=lambda kv: -kv[1])[:3]),
        "",
        f"**Classification: {ep.classification.replace('_', ' ')}.** {ep.action}",
        "",
        "| factor | G | Cramer's V |", "|---|---:|---:|",
    ]
    lines += [f"| {r['factor']} | {r['g_stat']:.0f} | {r['cramers_v']:.3f} |"
              for r in ep.factor_stats]
    return "\n".join(lines) + "\n"
