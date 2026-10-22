"""Inspect the local DuckDB relation inventory used by both dashboards.

Run: python scripts/check_market_db.py
Strict mode checks a small set of essential sources:
     python scripts/check_market_db.py --strict
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "market_data.duckdb"
EXPECTED = [
    "global_assets_daily",
    "global_assets_intraday",
    "macro_daily_ledger",
    "macro_intraday_ledger",
    "unified_market_master",
    "institutional_ledger",
    "trade_events_ledger",
    "quarterly_income_statement",
    "yearly_income_statement",
    "quarterly_balance_sheet",
    "yearly_balance_sheet",
    "quarterly_cash_flow",
    "yearly_cash_flow",
    "yearly_indirect_cash_flow",
    "mv_options_aggregates",
    "mv_spot_futures_basis",
    "mv_institutional_flow",
    "mv_unified_market_matrix",
]
CORE = {"global_assets_daily", "macro_daily_ledger", "unified_market_master"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when a core relation is missing")
    args = parser.parse_args()
    path = Path(os.getenv("OLAP_DB_PATH", str(DEFAULT_DB))).expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: DuckDB file not found: {path}", file=sys.stderr)
        print("Set OLAP_DB_PATH to the absolute path of your market_data.duckdb.", file=sys.stderr)
        return 2

    con = duckdb.connect(str(path), read_only=True)
    try:
        rows = con.execute("""
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchall()
    finally:
        con.close()

    names = {str(row[1]) for row in rows}
    print(f"Database: {path}")
    print(f"Relations found: {len(rows)}")
    print("\nExpected relation inventory:")
    for name in EXPECTED:
        print(f"  {'OK      ' if name in names else 'MISSING '} {name}")
    print("\nFirst 50 available relations:")
    for schema, name, kind in rows[:50]:
        print(f"  {schema}.{name} [{kind}]")
    missing_core = CORE - names
    if args.strict and missing_core:
        print("\nERROR: Missing core relation(s): " + ", ".join(sorted(missing_core)), file=sys.stderr)
        return 1
    if missing_core:
        print("\nNOTE: Some core relations are absent. Check ETL ingestion/schema initialization before testing the UI.")
    else:
        print("\nCore market relations are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
