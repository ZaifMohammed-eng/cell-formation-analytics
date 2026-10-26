`run_all.py` writes three files here, none of them committed:

- `weng2021_summary.csv`: public per-cell summary from Weng et al. (2021), downloaded from
  github.com/wengandrew/fast-formation at a pinned commit (see `config.py`).
  Data citation: Weng, A., Mohtat, P., Attia, P., Less, G., Lee, S., Stefanopoulou, A. (2021).
  Battery test data - fast formation study. University of Michigan Deep Blue Data.
  https://doi.org/10.7302/pa3f-4w30
- `formation.duckdb`: the simulated line warehouse.
- `ground_truth.json`: the injected events, used only to score the triage.
