"""Simulated high-volume formation line (cell build -> formation -> aging -> grading).

Cell-level capacity and low-SOC resistance (R_LS) are drawn from the
per-protocol distributions MEASURED in the public Weng et al. data. Line
layout, aging self-discharge, test noise and the three injected events are
ASSUMED (see config.py). Event ground truth is written to a separate file
and never read by the triage code.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FORMATION_HOURS, LineConfig

T0 = pd.Timestamp("2026-03-02 00:00:00")


def daily_volume(cfg: LineConfig) -> np.ndarray:
    ramp = np.linspace(0, 1, cfg.n_days)
    return np.round(cfg.start_cells_per_day
                    + (cfg.end_cells_per_day - cfg.start_cells_per_day) * ramp).astype(int)


def _protocol(cfg: LineConfig, day: np.ndarray, rack: np.ndarray) -> np.ndarray:
    pilot = (day >= cfg.pilot_start_day) & np.isin(rack, cfg.pilot_racks)
    fast = (day >= cfg.rollout_day) | pilot
    return np.where(fast, "fast", "baseline")


def simulate(cfg: LineConfig, dists: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg.seed)
    vol = daily_volume(cfg)
    n = int(vol.sum())
    day = np.repeat(np.arange(cfg.n_days), vol)

    build_h = day * 24 + rng.uniform(0, 24, n)
    coat_day = np.maximum(day - 2, 0)
    lot_half = rng.integers(0, 2, n)
    form_rack = rng.integers(1, cfg.n_formation_racks + 1, n)
    protocol = _protocol(cfg, day, form_rack)
    form_hours = np.where(protocol == "fast", FORMATION_HOURS["fast"], FORMATION_HOURS["baseline"])
    form_start = build_h + 2.0
    form_end = form_start + form_hours
    aging_rack = rng.integers(1, cfg.n_aging_racks + 1, n)
    aging_hours = rng.uniform(cfg.aging_hours_min, cfg.aging_hours_max, n)
    ocv1_h = form_end + cfg.handling_hours
    ocv2_h = ocv1_h + aging_hours
    grade_h = ocv2_h + cfg.handling_hours + cfg.grading_hours
    tester = rng.integers(1, cfg.n_grading_testers + 1, n)
    grade_day = np.floor(grade_h / 24).astype(int)

    # True cell properties: capacity and R_LS from the public data, by protocol.
    is_fast = protocol == "fast"
    b, f = dists["baseline"], dists["fast"]
    cap_true = np.where(is_fast, rng.normal(f["cap_mean"], f["cap_sd"], n),
                        rng.normal(b["cap_mean"], b["cap_sd"], n))
    rls_true = np.where(is_fast, rng.normal(f["rls_mean"], f["rls_sd"], n),
                        rng.normal(b["rls_mean"], b["rls_sd"], n))
    k_true = np.exp(rng.normal(np.log(cfg.k_median_mv_day), cfg.k_log_sigma, n))
    ms = rng.random(n) < cfg.microshort_rate
    k_true[ms] = np.exp(rng.normal(np.log(cfg.microshort_k_median), 0.4, ms.sum()))

    truth = {"events": [], "notes": "hidden from triage; used only for scoring"}
    ev = {e["event_id"]: e for e in cfg.events}

    # E3: contaminated anode coating lots -> extra micro-shorts and Li loss.
    e3 = ev["E3"]
    in_lot = (coat_day >= e3["day_start"]) & (coat_day <= e3["day_end"])
    hit3 = in_lot & (rng.random(n) < 0.04)
    k_true[hit3] = np.exp(rng.normal(np.log(2.5), 0.4, hit3.sum()))
    cap_true[in_lot] -= 0.003

    # Measurements on the grading testers.
    cap_meas = cap_true + rng.normal(0, cfg.cap_noise_ah, n)
    e1 = ev["E1"]
    hit1 = (tester == 7) & (grade_day >= e1["day_start"]) & (grade_day <= e1["day_end"])
    cap_meas[hit1] *= 1 - 0.006
    rls_meas = rls_true + rng.normal(0, cfg.rls_noise_mohm, n)
    ocv1 = rng.normal(3.700, 0.003, n)
    ocv2 = ocv1 - k_true * (aging_hours / 24) / 1000
    ocv1_meas = ocv1 + rng.normal(0, cfg.ocv_noise_mv / 1000, n)
    ocv2_meas = ocv2 + rng.normal(0, cfg.ocv_noise_mv / 1000, n)

    # K-value as computed by the plant ETL. E2: a deploy floors aging hours to
    # whole days for cells graded in the event window.
    e2 = ev["E2"]
    hit2 = (grade_day >= e2["day_start"]) & (grade_day <= e2["day_end"])
    aging_days_etl = np.where(hit2, np.floor(aging_hours / 24), aging_hours / 24)
    k_stored = (ocv1_meas - ocv2_meas) * 1000 / aging_days_etl

    ts = lambda h: T0 + pd.to_timedelta(h, unit="h")
    cells = pd.DataFrame({
        "cell_id": np.arange(1, n + 1),
        "build_ts": ts(build_h), "build_day": day,
        "electrode_lot": [f"AN{c:03d}{'ab'[s]}" for c, s in zip(coat_day, lot_half)],
        "electrolyte_lot": [f"EL{w:02d}" for w in day // 7],
        "formation_rack": [f"F{r:02d}" for r in form_rack],
        "protocol": protocol, "formation_hours": form_hours,
        "formation_start_ts": ts(form_start), "formation_end_ts": ts(form_end),
        "aging_rack": [f"AG{r:02d}" for r in aging_rack],
        "ocv1_ts": ts(ocv1_h), "ocv2_ts": ts(ocv2_h),
        "ocv1_v": ocv1_meas, "ocv2_v": ocv2_meas,
        "grading_tester": [f"G{t:02d}" for t in tester],
        "grade_ts": ts(grade_h), "grade_day": grade_day,
        "shift": np.array(["A", "B", "C"])[((grade_h % 24) // 8).astype(int)],
        "cap_meas_ah": cap_meas, "rls_meas_mohm": rls_meas,
        "k_stored_mv_day": k_stored,
    })

    # Retests: capacity / R_LS fails are regraded on a different tester, K
    # fails are re-aged 48 h and recomputed by the (correct) retest procedure.
    fail_cap = cap_meas < cfg.cap_lsl_ah
    fail_rls = rls_meas > cfg.rls_usl_mohm
    fail_k = k_stored > cfg.k_usl_mv_day
    idx = np.flatnonzero(fail_cap | fail_rls | fail_k)
    other = (tester[idx] + rng.integers(1, cfg.n_grading_testers, idx.size) - 1) \
        % cfg.n_grading_testers + 1
    rt_cap = cap_true[idx] + rng.normal(0, cfg.cap_noise_ah, idx.size)
    rt_rls = rls_true[idx] + rng.normal(0, cfg.rls_noise_mohm, idx.size)
    rt_k = k_true[idx] + rng.normal(0, np.sqrt(2) * cfg.ocv_noise_mv / 2.0, idx.size)
    retests = pd.DataFrame({
        "cell_id": idx + 1,
        "retest_tester": [f"G{t:02d}" for t in other],
        "retest_cap_ah": rt_cap, "retest_rls_mohm": rt_rls, "retest_k_mv_day": rt_k,
    })

    for eid, mask in (("E1", hit1), ("E2", hit2 & fail_k), ("E3", hit3)):
        e = dict(ev[eid])
        days = grade_day[mask]
        e.update(n_cells_affected=int(mask.sum()),
                 grade_day_first=int(days.min()), grade_day_last=int(days.max()))
        truth["events"].append(e)
    return cells, retests, truth


def write_truth(truth: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(truth, indent=2))
