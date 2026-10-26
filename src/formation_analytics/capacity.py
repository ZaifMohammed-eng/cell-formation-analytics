"""Equipment capacity, throughput, bottleneck and cycle time across the volume ramp.

A deterministic flow model: each station's capacity in cells/day is
channels x 24 h x availability / hours each cell occupies a channel. Demand
above the bottleneck's capacity becomes backlog, and Little's law turns
backlog into queue time. Equipment counts and times are ASSUMED (config.py);
formation durations are the published 56 h and 14 h.
"""
import numpy as np
import pandas as pd

from .config import FORMATION_HOURS, LineConfig
from .simulate import daily_volume

LOAD_UNLOAD_H = 1.0
GRADING_HANDLING_H = 0.25


def station_capacity(cfg: LineConfig, fast_share: float) -> dict:
    form_h = ((1 - fast_share) * FORMATION_HOURS["baseline"]
              + fast_share * FORMATION_HOURS["fast"]) + LOAD_UNLOAD_H
    aging_days = (cfg.aging_hours_min + cfg.aging_hours_max) / 2 / 24 + cfg.handling_hours / 24
    a = cfg.availability
    return {
        "formation": cfg.n_formation_racks * cfg.channels_per_formation_rack * 24 * a / form_h,
        "aging": cfg.aging_positions * a / aging_days,
        "grading": cfg.n_grading_testers * cfg.grading_channels_per_tester * 24 * a
                   / (cfg.grading_hours + GRADING_HANDLING_H),
    }


def fast_share_by_day(cfg: LineConfig, scenario: str) -> np.ndarray:
    days = np.arange(cfg.n_days)
    if scenario == "baseline_only":
        return np.zeros(cfg.n_days)
    if scenario == "fast_from_start":
        return np.ones(cfg.n_days)
    pilot = len(cfg.pilot_racks) / cfg.n_formation_racks
    return np.where(days >= cfg.rollout_day, 1.0,
                    np.where(days >= cfg.pilot_start_day, pilot, 0.0))


def ramp(cfg: LineConfig, scenario: str = "plan") -> pd.DataFrame:
    demand = daily_volume(cfg)
    share = fast_share_by_day(cfg, scenario)
    process_h = None
    backlog, rows = 0.0, []
    for d in range(cfg.n_days):
        cap = station_capacity(cfg, share[d])
        util = {k: demand[d] / v for k, v in cap.items()}
        bottleneck = max(util, key=util.get)
        c = cap[bottleneck]
        out = min(demand[d] + backlog, c)
        backlog = demand[d] + backlog - out
        form_h = (1 - share[d]) * FORMATION_HOURS["baseline"] + share[d] * FORMATION_HOURS["fast"]
        process_h = (2.0 + form_h + LOAD_UNLOAD_H + (cfg.aging_hours_min + cfg.aging_hours_max) / 2
                     + cfg.handling_hours + cfg.grading_hours + GRADING_HANDLING_H)
        rows.append(dict(day=d, demand=int(demand[d]), fast_share=float(share[d]),
                         cap_formation=cap["formation"], cap_aging=cap["aging"],
                         cap_grading=cap["grading"], bottleneck=bottleneck,
                         util_bottleneck=util[bottleneck], throughput=out, backlog=backlog,
                         cycle_time_h=process_h + backlog / c * 24))
    return pd.DataFrame(rows)


def channels_needed(cfg: LineConfig, cells_per_day: float, protocol: str,
                    target_util: float = 0.85) -> int:
    occ = FORMATION_HOURS[protocol] + LOAD_UNLOAD_H
    return int(np.ceil(cells_per_day * occ / (24 * cfg.availability * target_util)))


def summarize(cfg: LineConfig) -> dict:
    out = {}
    for sc in ("baseline_only", "plan", "fast_from_start"):
        r = ramp(cfg, sc)
        over = r[r.util_bottleneck > 1]
        out[sc] = dict(
            first_day_over_capacity=int(over.day.iloc[0]) if len(over) else None,
            bottleneck_at_end=r.bottleneck.iloc[-1],
            util_at_end=float(r.util_bottleneck.iloc[-1]),
            peak_backlog_cells=float(r.backlog.max()),
            cycle_time_end_h=float(r.cycle_time_h.iloc[-1]),
            cells_shipped=float(r.throughput.sum()),
            cells_demanded=float(r.demand.sum()),
        )
    end = cfg.end_cells_per_day
    out["formation_channels_needed_at_end_volume"] = {
        p: channels_needed(cfg, end, p) for p in ("baseline", "fast")}
    out["installed_formation_channels"] = cfg.n_formation_racks * cfg.channels_per_formation_rack
    out["station_capacity_cells_per_day"] = {
        p: {k: round(v) for k, v in station_capacity(cfg, s).items()}
        for p, s in (("baseline", 0.0), ("fast", 1.0))}
    return out
