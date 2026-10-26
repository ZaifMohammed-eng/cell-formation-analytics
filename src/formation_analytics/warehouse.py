"""DuckDB warehouse: load the line data, create views, run the SQL in sql/."""
from pathlib import Path

import duckdb
import pandas as pd

from .config import LineConfig

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "sql"


def build(cells: pd.DataFrame, retests: pd.DataFrame, cfg: LineConfig,
          path: Path | str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    con.register("cells_df", cells)
    con.register("retests_df", retests)
    con.execute("CREATE OR REPLACE TABLE cells AS SELECT * FROM cells_df")
    con.execute("CREATE OR REPLACE TABLE retests AS SELECT * FROM retests_df")
    con.execute(
        "CREATE OR REPLACE TABLE limits AS SELECT ? AS cap_lsl_ah, ? AS k_usl_mv_day, "
        "? AS rls_usl_mohm",
        [cfg.cap_lsl_ah, cfg.k_usl_mv_day, cfg.rls_usl_mohm],
    )
    con.unregister("cells_df")
    con.unregister("retests_df")
    con.execute((SQL / "00_views.sql").read_text())
    return con


def query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute((SQL / f"{name}.sql").read_text()).df()
