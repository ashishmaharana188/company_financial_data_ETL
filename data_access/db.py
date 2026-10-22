"""Read-only connection to the shared local DuckDB OLAP database.

This is the local-development adapter. The deployment pack contains a
separate adapter that can read the same data from Supabase PostgreSQL.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "market_data.duckdb"


def get_db_path() -> Path:
    """Return the configured local OLAP path and fail clearly if absent."""
    value = os.getenv("OLAP_DB_PATH", str(DEFAULT_DB_PATH)).strip()
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".duckdb":
        raise ValueError(f"Expected a DuckDB database file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"OLAP database not found: {path}. Set OLAP_DB_PATH to your local market_data.duckdb."
        )
    return path


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a read-only connection. The ETL writer remains in scripts/database.py."""
    return duckdb.connect(database=str(get_db_path()), read_only=True)
