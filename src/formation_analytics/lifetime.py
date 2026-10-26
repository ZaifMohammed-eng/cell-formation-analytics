"""Real-data side: what formation protocol and low-SOC resistance do to life.

Uses the public per-cell summary from Weng et al. (2021). Everything in this
module is MEASURED data, the only modelling choice is a linear fit of cycle
life on low-SOC resistance (R_LS, 10 s pulse at 5% SOC), the screening signal
that paper proposed.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

from .config import WENG_URL, CostConfig

DATA = Path(__file__).resolve().parents[2] / "data" / "weng2021_summary.csv"
COLS = {
    "cellid": "cell_id",
    "is_baseline_formation": "is_baseline",
    "is_room_temp_aging": "is_room_temp",
    "cycles_to_70_pct": "cycles_to_70",
    "cycles_to_80_pct": "cycles_to_80",
    "dcr_10s_5_soc_at_c3": "rls_ohm",
    "form_final_discharge_capacity_ah": "cap_ah",
}


def load_public() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WENG_URL, DATA)
    d = pd.read_csv(DATA, usecols=list(COLS))
    d = d.rename(columns=COLS)
    d["protocol"] = np.where(d.is_baseline == 1, "baseline", "fast")
    d["temperature"] = np.where(d.is_room_temp == 1, "RT", "45C")
    d["rls_mohm"] = d.rls_ohm * 1000
    return d


def protocol_lifetime(d: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for temp, g in d.groupby("temperature"):
        fast = g.loc[g.protocol == "fast", "cycles_to_70"].to_numpy()
        base = g.loc[g.protocol == "baseline", "cycles_to_70"].to_numpy()
        boot = [rng.choice(fast, fast.size).mean() - rng.choice(base, base.size).mean()
                for _ in range(n_boot)]
        t = stats.ttest_ind(fast, base, equal_var=False)
        out[temp] = dict(
            n_fast=int(fast.size), n_baseline=int(base.size),
            fast_mean=float(fast.mean()), baseline_mean=float(base.mean()),
            gain_cycles=float(fast.mean() - base.mean()),
            gain_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            welch_p=float(t.pvalue),
        )
    return out


def _fit(x, y) -> dict:
    fit = stats.linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    resid_sd = float(np.sqrt((resid ** 2).sum() / (x.size - 2)))
    loo = []
    for i in range(x.size):
        m = np.ones(x.size, bool)
        m[i] = False
        f = stats.linregress(x[m], y[m])
        loo.append(abs(y[i] - (f.intercept + f.slope * x[i])))
    return dict(n=int(x.size), pearson_r=float(fit.rvalue), p_value=float(fit.pvalue),
                slope_cycles_per_mohm=float(fit.slope), intercept=float(fit.intercept),
                resid_sd_cycles=resid_sd, loo_mae_cycles=float(np.mean(loo)),
                mean_life=float(y.mean()), sd_life=float(y.std(ddof=1)))


def rls_life_model(d: pd.DataFrame, temperature: str = "RT", protocol: str | None = None) -> dict:
    """Linear fit cycles_to_70 ~ R_LS for one test temperature (and protocol)."""
    g = d[d.temperature == temperature]
    if protocol:
        g = g[g.protocol == protocol]
    out = _fit(g.rls_mohm.to_numpy(), g.cycles_to_70.to_numpy())
    out.update(temperature=temperature, protocol=protocol or "pooled")
    return out


def screening_model(d: pd.DataFrame, protocol: str, temperature: str = "RT",
                    alpha: float = 0.05) -> dict:
    """Per-cell life model used for limits and warranty.

    Uses the within-protocol R_LS slope only when it is significant at alpha.
    Otherwise R_LS carries no per-cell life information for that protocol and
    the model falls back to the protocol mean and SD (slope 0).
    """
    m = rls_life_model(d, temperature, protocol)
    if m["p_value"] < alpha:
        m["uses_rls"] = True
        return m
    m.update(uses_rls=False, slope_cycles_per_mohm=0.0, intercept=m["mean_life"],
             resid_sd_cycles=m["sd_life"])
    return m


def predicted_life(model: dict, rls_mohm):
    return model["intercept"] + model["slope_cycles_per_mohm"] * np.asarray(rls_mohm)


def p_below_spec(model: dict, rls_mohm, spec: float):
    """P(cycle life < spec | R_LS) under the fit's normal residuals."""
    mu = predicted_life(model, rls_mohm)
    return stats.norm.cdf((spec - mu) / model["resid_sd_cycles"])


def rls_limit_for_spec(model: dict, spec: float, max_claim_prob: float = 0.20) -> float | None:
    """Largest R_LS whose chance of missing the life spec stays below max_claim_prob.

    None when the model has no R_LS slope (no screening value).
    """
    if not model.get("uses_rls", True) or model["slope_cycles_per_mohm"] == 0:
        return None
    z = stats.norm.ppf(1 - max_claim_prob)
    return float((spec + z * model["resid_sd_cycles"] - model["intercept"])
                 / model["slope_cycles_per_mohm"])


def protocol_distributions(d: pd.DataFrame, temperature: str = "RT") -> dict:
    g = d[d.temperature == temperature]
    return {p: dict(cap_mean=float(s.cap_ah.mean()), cap_sd=float(s.cap_ah.std()),
                    rls_mean=float(s.rls_mohm.mean()), rls_sd=float(s.rls_mohm.std()),
                    n=int(len(s)))
            for p, s in g.groupby("protocol")}


def summarize(cost: CostConfig | None = None) -> dict:
    cost = cost or CostConfig()
    d = load_public()
    within = {f"{t}_{p}": rls_life_model(d, t, p)
              for t in ("RT", "45C") for p in ("baseline", "fast")}
    screens = {p: screening_model(d, p, "RT") for p in ("baseline", "fast")}
    return dict(
        source="Weng et al., Joule 5 (2021) 2971-2992; data doi:10.7302/pa3f-4w30",
        n_cells=int(len(d)),
        protocol_lifetime=protocol_lifetime(d),
        rls_life_pooled={t: rls_life_model(d, t) for t in ("RT", "45C")},
        rls_life_within_protocol=within,
        screening_models_RT=screens,
        distributions_RT=protocol_distributions(d, "RT"),
        rls_usl_mohm={p: rls_limit_for_spec(m, cost.life_spec_cycles_70)
                      for p, m in screens.items()},
    )
