"""
verify_parquet.py

Verifies yearly Parquet exports against DuckDB.
"""

from pathlib import Path

import duckdb

from database import DB_PATH

PARQUET_ROOT = Path("parquet/archive")

TABLES = {
    "unified_market_master": "ReportDate",
    "global_assets_daily": "ReportDate",
    "global_assets_intraday": "ReportDate",
    "macro_daily_ledger": "ReportDate",
    "macro_intraday_ledger": "ReportDate",
    "institutional_ledger": "ReportDate",
    "trade_events_ledger": "ReportDate",
}


con = duckdb.connect(DB_PATH, read_only=True)

print("=" * 90)
print("VERIFYING PARQUET EXPORT")
print("=" * 90)

for table, date_col in TABLES.items():

    print(f"\n{table}")

    years = con.execute(
        f"""
        SELECT DISTINCT YEAR("{date_col}")
        FROM {table}
        ORDER BY 1
    """
    ).fetchall()

    for (year,) in years:

        db_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE YEAR("{date_col}") = {year}
        """
        ).fetchone()[0]

        parquet_file = PARQUET_ROOT / table / f"year{year}" / "data.parquet"

        if not parquet_file.exists():
            print(f"  {year} : PARQUET NOT FOUND")
            continue

        pq_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{parquet_file.as_posix()}')
        """
        ).fetchone()[0]

        if db_count == pq_count:
            print(f"  {year} : PASS ({db_count:,})")
        else:
            print(f"  {year} : FAIL " f"DB={db_count:,} " f"PARQUET={pq_count:,}")

con.close()

print("\nVerification Complete.")
