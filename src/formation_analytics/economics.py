"""Formation duration vs cell COGS (formation cost, yield loss) vs warranty,
and the R_LS process limit seen from the cell and from the pack.

Life information is MEASURED (public data, screening_model in lifetime.py).
Costs, the life spec and the pack layout are ASSUMED and only meant to show
how the tradeoff is computed.
"""
import numpy as np
import pandas as pd

from .config import FORMATION_HOURS, CostConfig, LineConfig
from .lifetime import p_below_spec, predicted_life
from .capacity import LOAD_UNLOAD_H, channels_needed


def per_cell_cost(protocol: str, final_yield: float, model: dict, rls_shipped: np.ndarray,
                  cfg: LineConfig, cost: CostConfig) -> dict:
    ch_cost = cost.formation_cost_per_channel_hour(cfg.availability)
    formation = ch_cost * (FORMATION_HOURS[protocol] + LOAD_UNLOAD_H) + cost.formation_energy_cost
    aging = cost.aging_cost_per_cell_day * (cfg.aging_hours_min + cfg.aging_hours_max) / 48
    scrap = (1 - final_yield) / final_yield * (cost.cell_cost + formation + aging)
    claim = float(np.mean(p_below_spec(model, rls_shipped, cost.life_spec_cycles_70)))
    warranty = claim * cost.warranty_cost_per_claim_cell
    return dict(protocol=protocol, formation_hours=FORMATION_HOURS[protocol],
                formation_cost=formation, aging_cost=aging, final_yield=final_yield,
                scrap_cost_per_good_cell=scrap, warranty_claim_prob=claim,
                warranty_cost=warranty, total=formation + aging + scrap + warranty,
                formation_capex_at_end_volume=channels_needed(
                    cfg, cfg.end_cells_per_day, protocol) * cost.formation_capex_per_channel)


def duration_tradeoff(con, models: dict, cfg: LineConfig, cost: CostConfig) -> pd.DataFrame:
    """Compare protocols on cells built in the same window (pilot period)."""
    df = con.execute(
        "SELECT protocol, final_pass, rls_meas_mohm FROM cell_disposition "
        "WHERE build_day BETWEEN ? AND ?", [cfg.pilot_start_day, cfg.rollout_day - 1]).df()
    rows = []
    for p, g in df.groupby("protocol"):
        shipped = g.loc[g.final_pass, "rls_meas_mohm"].to_numpy()
        rows.append(per_cell_cost(p, float(g.final_pass.mean()), models[p], shipped, cfg, cost))
    return pd.DataFrame(rows).set_index("protocol")


def _sample_cells(dist: dict, cfg: LineConfig, n: int, rng) -> np.ndarray:
    true = rng.normal(dist["rls_mean"], dist["rls_sd"], n)
    return true + rng.normal(0, cfg.rls_noise_mohm, n)


def limit_sweep(dist: dict, model: dict, cfg: LineConfig, cost: CostConfig,
                usl_grid=np.arange(136.0, 152.01, 0.5), n: int = 400_000,
                pack_series: int = 96, pack_parallel: int = 4, n_packs: int = 2_000,
                seed: int = 1) -> pd.DataFrame:
    """Sweep the R_LS upper limit: yield, cell and pack claim risk, cost per good cell."""
    rng = np.random.default_rng(seed)
    rls = _sample_cells(dist, cfg, n, rng)
    life = predicted_life(model, rls) + rng.normal(0, model["resid_sd_cycles"], n)
    spec = cost.life_spec_cycles_70
    rows = []
    for usl in usl_grid:
        ok = rls <= usl
        y = ok.mean()
        cell_claim = float(np.mean(life[ok] < spec))
        # Pack: 96 series groups of 4 parallel cells, group life ~ mean of its
        # cells, pack life = weakest group (illustrative weakest-link model).
        pick = rng.choice(np.flatnonzero(ok), (n_packs, pack_series, pack_parallel))
        pack_life = life[pick].mean(axis=2).min(axis=1)
        pack_claim = float(np.mean(pack_life < spec))
        scrap = (1 - y) / y * cost.cell_cost
        rows.append(dict(rls_usl_mohm=float(usl), yield_rls=float(y), cell_claim_prob=cell_claim,
                         pack_claim_prob=pack_claim, median_pack_life=float(np.median(pack_life)),
                         scrap_cost=scrap, warranty_cost=cell_claim * cost.warranty_cost_per_claim_cell,
                         pack_warranty_cost=pack_claim * cost.warranty_cost_per_claim_cell))
    out = pd.DataFrame(rows)
    out["cost_cell_view"] = out.scrap_cost + out.warranty_cost
    out["cost_pack_view"] = out.scrap_cost + out.pack_warranty_cost
    return out
