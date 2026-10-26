"""Rerun simulation + triage across seeds so the triage score is not one lucky draw.

    python robustness.py          # about 2 min, writes results/robustness.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from formation_analytics import triage  # noqa: E402
from formation_analytics.config import LineConfig  # noqa: E402
from formation_analytics.lifetime import summarize  # noqa: E402
from formation_analytics.simulate import simulate  # noqa: E402
from formation_analytics.warehouse import build  # noqa: E402

SEEDS = range(1, 11)

if __name__ == "__main__":
    dists = summarize()["distributions_RT"]
    runs = []
    for seed in SEEDS:
        cfg = LineConfig(seed=seed)
        cells, retests, truth = simulate(cfg, dists)
        con = build(cells, retests, cfg)
        s = triage.score(triage.run(con), truth)
        con.close()
        runs.append(dict(seed=seed, correct=s["n_correct"], events=s["n_events"],
                         false_alarms=len(s["false_alarm_episodes"]),
                         false_alarm_misclassified=sum(
                             fa["classification"] != "unresolved"
                             for fa in s["false_alarm_episodes"]),
                         per_event={e["event_id"]: e["classified_as"] for e in s["events"]}))
        print(runs[-1])
    out = dict(
        seeds=list(SEEDS), runs=runs,
        events_correct=sum(r["correct"] for r in runs),
        events_total=sum(r["events"] for r in runs),
        false_alarm_episodes=sum(r["false_alarms"] for r in runs),
        false_alarms_misclassified=sum(r["false_alarm_misclassified"] for r in runs),
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if k != "runs"})
