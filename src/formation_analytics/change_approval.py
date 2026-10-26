"""Data-driven change approval for a Formation process change.

Compares fast-formation pilot cells with baseline cells built on the same
days (pilot racks vs. the rest) using equivalence (TOST) and non-inferiority
tests, then states a decision and its conditions.
"""
import numpy as np
from scipy import stats

from .config import LineConfig


def tost(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided Welch tests: is |mean(a) - mean(b)| < margin?"""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    df = se ** 4 / ((a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
                    + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1))
    p_low = stats.t.sf((diff + margin) / se, df)
    p_up = stats.t.cdf((diff - margin) / se, df)
    tcrit = stats.t.ppf(1 - alpha, df)
    return dict(diff=float(diff), margin=float(margin),
                ci90=[float(diff - tcrit * se), float(diff + tcrit * se)],
                p=float(max(p_low, p_up)), equivalent=bool(max(p_low, p_up) < alpha))


def noninferior_proportion(x_new, n_new, x_ref, n_ref, margin: float, alpha: float = 0.05) -> dict:
    p1, p0 = x_new / n_new, x_ref / n_ref
    se = np.sqrt(p1 * (1 - p1) / n_new + p0 * (1 - p0) / n_ref)
    lower = (p1 - p0) - stats.norm.ppf(1 - alpha) * se
    return dict(new=float(p1), ref=float(p0), diff=float(p1 - p0),
                lower_bound=float(lower), margin=-margin, noninferior=bool(lower > -margin))


def evaluate(con, cfg: LineConfig, life: dict) -> dict:
    df = con.execute(
        "SELECT protocol, cap_meas_ah, rls_meas_mohm, k_stored_mv_day, first_pass "
        "FROM cell_results WHERE build_day BETWEEN ? AND ?",
        [cfg.pilot_start_day, cfg.rollout_day - 1]).df()
    new, ref = df[df.protocol == "fast"], df[df.protocol == "baseline"]
    cap = tost(new.cap_meas_ah.to_numpy(), ref.cap_meas_ah.to_numpy(),
               margin=0.005 * ref.cap_meas_ah.mean())
    k = tost(np.log(new.k_stored_mv_day.clip(lower=1e-3)).to_numpy(),
             np.log(ref.k_stored_mv_day.clip(lower=1e-3)).to_numpy(), margin=np.log(1.10))
    rls = stats.ttest_ind(new.rls_meas_mohm, ref.rls_meas_mohm, equal_var=False)
    fpy = noninferior_proportion(new.first_pass.sum(), len(new),
                                 ref.first_pass.sum(), len(ref), margin=0.003)
    rt = life["protocol_lifetime"]["RT"]
    checks = {"capacity_equivalent_0p5pct": cap["equivalent"],
              "k_value_equivalent_10pct": k["equivalent"],
              "first_pass_yield_noninferior_0p3pp": fpy["noninferior"],
              "lifetime_not_worse_public_data": rt["gain_ci95"][0] > 0}
    decision = "APPROVE" if all(checks.values()) else "REJECT"
    return dict(
        n_pilot=int(len(new)), n_concurrent_baseline=int(len(ref)),
        capacity_tost=cap, k_value_log_tost=k,
        rls_shift_mohm=float(new.rls_meas_mohm.mean() - ref.rls_meas_mohm.mean()),
        rls_welch_p=float(rls.pvalue), first_pass_yield=fpy,
        lifetime_gain_cycles_RT=rt["gain_cycles"], lifetime_gain_ci95=rt["gain_ci95"],
        checks=checks, decision=decision,
        conditions=[
            "R_LS limit was derived on baseline cells; fast formation moves R_LS "
            f"by {new.rls_meas_mohm.mean() - ref.rls_meas_mohm.mean():.1f} mOhm and the public "
            "data show no within-protocol R_LS-life link for fast cells, so re-derive or "
            "retire the R_LS screen for fast-formed cells instead of reading the yield jump "
            "as a quality gain.",
            "Capacity is equivalent within 0.5% but lower by about "
            f"{-cap['diff'] * 1000:.1f} mAh (more lithium consumed in formation): confirm "
            "against the capacity spec at the next grading-limit review.",
            "Keep daily triage on K-value through rollout: self-discharge equivalence was "
            "shown on pilot volume only.",
        ],
    )
