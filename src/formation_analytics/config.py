"""Line, test and cost assumptions in one place.

Two kinds of numbers live here, and the README keeps them apart:

* MEASURED: taken from the public Weng et al. (2021) fast-formation dataset
  (2.36 Ah NCM111/graphite pouch cells, University of Michigan). These are
  per-protocol cell distributions measured at room temperature.
* ASSUMED: line size, equipment counts, test noise, defect rates, costs and
  the injected events. They are illustrative and chosen to be plausible, not
  taken from any real factory. Change them and rerun.
"""
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Public data (MEASURED). Pinned commit so results are reproducible.
# --------------------------------------------------------------------------
WENG_COMMIT = "50664e9a5ab3a9a86baf01566c3d0463b4171bee"
WENG_URL = (
    "https://raw.githubusercontent.com/wengandrew/fast-formation/"
    f"{WENG_COMMIT}/data/correlation_data_2022.csv"
)
# Protocol durations reported in Weng et al., Joule 5 (2021) 2971-2992.
FORMATION_HOURS = {"baseline": 56.0, "fast": 14.0}


@dataclass
class LineConfig:
    seed: int = 7
    n_days: int = 120
    start_cells_per_day: int = 3_000      # ASSUMED volume ramp start
    end_cells_per_day: int = 20_000       # ASSUMED volume ramp end

    # Equipment (ASSUMED)
    n_formation_racks: int = 40
    channels_per_formation_rack: int = 1_024
    n_aging_racks: int = 60
    n_grading_testers: int = 16
    grading_channels_per_tester: int = 192
    grading_hours: float = 3.0
    aging_positions: int = 120_000
    aging_hours_min: float = 108.0        # 4.5 days
    aging_hours_max: float = 120.0        # 5.0 days
    handling_hours: float = 1.0
    availability: float = 0.90

    # Measurement noise of the grading testers (ASSUMED)
    cap_noise_ah: float = 0.002
    rls_noise_mohm: float = 0.8
    ocv_noise_mv: float = 0.10

    # Self-discharge during aging (ASSUMED: the public data has no aging OCV)
    k_median_mv_day: float = 0.45
    k_log_sigma: float = 0.25
    microshort_rate: float = 0.0015
    microshort_k_median: float = 3.0

    # Grading limits. CAP_LSL and K_USL are ASSUMED. RLS_USL is set from the
    # lifetime regression on public data in lifetime.py (see results).
    cap_lsl_ah: float = 2.335
    k_usl_mv_day: float = 1.0
    rls_usl_mohm: float = 145.0

    # Process change: fast-formation pilot on racks F31-F40, then rollout.
    pilot_start_day: int = 95
    rollout_day: int = 110
    pilot_racks: tuple = tuple(range(31, 41))

    # Injected events (ASSUMED ground truth, hidden from the triage code)
    events: list = field(default_factory=lambda: [
        dict(event_id="E1", kind="measurement_error", factor="grading_tester",
             level="G07", mode="CAP_LOW", day_start=30, day_end=36,
             detail="current-shunt drift, capacity reads 0.6% low"),
        dict(event_id="E2", kind="calculation_error", factor="etl",
             level="k_value", mode="K_HIGH", day_start=55, day_end=58,
             detail="ETL floors aging hours to whole days in K-value"),
        dict(event_id="E3", kind="true_quality_shift", factor="electrode_lot",
             level="anode lots coated days 78-81", mode="K_HIGH",
             day_start=78, day_end=81,
             detail="contaminated anode coating, 4% micro-shorts"),
    ])


@dataclass
class CostConfig:
    """All ASSUMED, per cell unless stated."""
    cell_wh: float = 2.36 * 3.7
    cell_cost_per_kwh: float = 90.0
    formation_capex_per_channel: float = 250.0
    equipment_life_years: float = 7.0
    formation_energy_cost: float = 0.004
    aging_cost_per_cell_day: float = 0.004
    warranty_cost_per_claim_cell: float = 6.0   # pack-repair cost per cell
    life_spec_cycles_70: float = 330.0          # warranty life requirement

    @property
    def cell_cost(self) -> float:
        return self.cell_wh / 1000 * self.cell_cost_per_kwh

    def formation_cost_per_channel_hour(self, availability: float) -> float:
        hours = self.equipment_life_years * 8760 * availability
        return self.formation_capex_per_channel / hours
