# Cell Formation Analytics

Yield, quality and throughput analytics for a lithium-ion cell Formation line, run on 1.38 million simulated cells. The cell properties are drawn from public measured data, and the lifetime side uses the published measurements directly.

The project asks the questions a Formation data engineer answers every day:

1. **Is today's yield drop real?** Daily SQL KPIs feed a triage engine that separates calculation errors, measurement errors and true quality shifts in upstream material, process or equipment. It then writes a preliminary root-cause note for engineering review.
2. **Where is the bottleneck as volume ramps?** Equipment capacity, throughput, backlog and cycle time are computed for formation channels, aging racks and grading testers.
3. **What does formation duration cost?** A 56 h vs 14 h formation tradeoff covers formation cost, yield, capex and warranty risk, with the lifetime effect taken from public data.
4. **Should this process change be approved?** Equivalence and non-inferiority tests are applied to a fast-formation pilot.
5. **Are the process limits right?** An R_LS (low-SOC resistance) limit sweep weighs yield against life risk at the cell level and the pack level.

Run everything with `python run_all.py` (about 20 s). Then open `dashboard/index.html` and `reports/formation_report.md`.

## What is measured and what is simulated

| Source | Used for |
|---|---|
| **Measured.** Public per-cell summary from Weng et al., *Predicting the impact of formation protocols on battery lifetime immediately after manufacturing*, Joule 5 (2021) 2971-2992; data doi:10.7302/pa3f-4w30. These are 39 NCM111/graphite 2.36 Ah pouch cells with baseline (56 h) or fast (14 h) formation, aged at RT or 45 C. The file is downloaded at run time from the authors' repository at a pinned commit and is not redistributed here. | Lifetime by protocol, the R_LS-life relationship, and the per-protocol capacity and R_LS distributions that the simulated cells are drawn from. |
| **Assumed.** All values in `config.py`. | Line size and volume ramp, equipment counts, availability, aging self-discharge (K-value), test noise, defect rates, costs, the life spec, the pack layout, and the three injected events. |

The line is a simulation. The point is the analysis method and code, not a claim about any real factory.

## Results (`results/metrics.json`)

**Public data, measured**
- Fast formation adds **108 cycles** to 70% capacity at RT (95% CI 78-137, Welch p < 1e-5) and 112 cycles at 45 C.
- R_LS vs cycle life pooled across protocols gives r = -0.84 at RT, reproducing the paper. **Within a protocol the link holds only for baseline cells** (r = -0.75, p = 0.012, n = 10). For fast-formed cells it vanishes (r = -0.08, p = 0.84).
- Most of the pooled correlation therefore comes from separating the two protocols. An R_LS screen is justified only for baseline formation, so the limit and warranty analysis uses the within-protocol fits.

**Simulated line (1,380,000 cells, 120 days, ramp 3k to 20k cells/day)**
- Triage classified all three injected events correctly on the first affected day:
  - **Grading-tester current drift.** Classed as a measurement error: 78% of fails were on G07, and 83% passed on retest.
  - **ETL bug.** Classed as a calculation error: K-value was inflated by 19%, and the stored value disagreed with the SQL recompute from raw OCV in 100% of cells.
  - **Contaminated anode lots.** Classed as a true quality shift: 99% of fails fell in 8 lots across all equipment, and 1% passed on retest.
- **Across 10 random seeds, 30 of 30 events were classified correctly.** There were 11 false-alarm days in 1,200 simulated days (3-sigma tests on three failure modes daily), and all of them were left as "unresolved" rather than blamed on a cause.
- **Capacity.**
  - Baseline formation runs out of capacity on day 88 of the ramp; without a change, backlog peaks at 72k cells and cycle time reaches 289 h.
  - With the 14 h protocol the bottleneck moves to grading (98% utilization) and cycle time is 135 h.
  - Formation channels needed at 20k cells/day fall from 62,092 to 16,340.
- **Change approval: APPROVE with conditions.**
  - Capacity is equivalent within 0.5% (but 5.7 mAh lower), K-value is equivalent within 10%, and first-pass yield is non-inferior.
  - The +4 pp yield jump is mostly an artifact of an R_LS limit set on baseline cells. The approval therefore requires re-deriving that limit rather than booking the jump as a quality gain.
- **R_LS limit.**
  - At the current 145 mOhm limit, 12% of shipped baseline cells and 57% of 96s4p packs miss the (assumed) life spec under a weakest-link pack model.
  - The cost-optimal limit is about 142 mOhm judged on cells and about 141 mOhm judged on packs.

## Repository

```
run_all.py                 one command: data -> KPIs -> triage -> capacity -> economics -> outputs
robustness.py              triage across 10 seeds (results/robustness.json)
sql/                       DuckDB SQL: views, daily KPIs, K-value recompute, weekly throughput, Pareto
src/formation_analytics/
  config.py                every assumption, labeled
  lifetime.py              public-data lifetime and R_LS analysis
  simulate.py              simulated line with injected events (ground truth kept separate)
  warehouse.py             DuckDB load + views
  triage.py                detection, calculation check, localization (stats + ML), retest, classification
  capacity.py              equipment capacity, throughput, bottleneck, cycle time over the ramp
  economics.py             duration vs COGS vs warranty; R_LS limit sweep with pack view
  change_approval.py       TOST equivalence and non-inferiority for a process change
  dashboard.py             static Plotly dashboard
tests/                     pytest, including triage scored against ground truth
```

Outputs: `results/` (metrics.json, KPI and sweep CSVs), `reports/formation_report.md`, and `dashboard/index.html`. The DuckDB file `data/formation.duckdb` is rebuilt on each run and is not committed.

## Limitations

- The injected events were written alongside the triage rules, so 30/30 shows that the logic works as designed, not how it would perform on unseen failure types. Combined causes (for example, a drifting tester during a bad lot) are not tested.
- The life model rests on 10 baseline cells at one temperature. The warranty and pack numbers depend on the assumed life spec, the cost per claim, and cell independence within a pack.
- The capacity model is a deterministic flow model with fixed availability. It has no queueing variability, changeovers or downtime events.

## Setup

```
pip install -r requirements.txt
python run_all.py
python -m pytest -q
```
