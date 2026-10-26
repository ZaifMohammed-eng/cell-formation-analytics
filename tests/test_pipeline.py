"""End-to-end checks. Run with: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formation_analytics import capacity, change_approval, triage  # noqa: E402
from formation_analytics.config import LineConfig  # noqa: E402
from formation_analytics.lifetime import load_public, summarize  # noqa: E402
from formation_analytics.simulate import simulate  # noqa: E402
from formation_analytics.warehouse import build, query  # noqa: E402


@pytest.fixture(scope="module")
def line():
    cfg = LineConfig(n_days=120)
    life = summarize()
    cells, retests, truth = simulate(cfg, life["distributions_RT"])
    con = build(cells, retests, cfg)
    return cfg, life, con, truth


def test_public_data_matches_paper():
    life = summarize()
    assert life["n_cells"] == 39
    assert life["rls_life_pooled"]["RT"]["pearson_r"] == pytest.approx(-0.84, abs=0.01)
    assert life["protocol_lifetime"]["RT"]["gain_cycles"] > 100


def test_k_recompute_only_disagrees_in_bug_window(line):
    cfg, _, con, truth = line
    e2 = next(e for e in truth["events"] if e["event_id"] == "E2")
    k = query(con, "k_value_recompute")
    bad = k[k.mismatch_rate > 0.05].grade_day
    assert set(bad) == set(range(e2["day_start"], e2["day_end"] + 1))


def test_triage_classifies_all_injected_events(line):
    _, _, con, truth = line
    eps = triage.run(con)
    s = triage.score(eps, truth)
    assert s["n_correct"] == s["n_events"] == 3
    # A false alarm may happen at a 3-sigma threshold, but must not be
    # misclassified as a specific cause.
    assert all(fa["classification"] == "unresolved" for fa in s["false_alarm_episodes"])


def test_capacity_math():
    cfg = LineConfig()
    cap = capacity.station_capacity(cfg, 0.0)
    expected = 40 * 1024 * 24 * 0.9 / 57.0
    assert cap["formation"] == pytest.approx(expected)
    assert capacity.channels_needed(cfg, 20_000, "fast") < capacity.channels_needed(cfg, 20_000, "baseline")


def test_change_approval_flags_rls_limit(line):
    cfg, life, con, _ = line
    r = change_approval.evaluate(con, cfg, life)
    assert r["decision"] == "APPROVE"
    assert r["rls_shift_mohm"] < -5
    assert "re-derive" in r["conditions"][0]


def test_tost_rejects_real_difference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 0.1, 5000), rng.normal(1.1, 0.1, 5000)
    assert not change_approval.tost(a, b, margin=0.05)["equivalent"]
