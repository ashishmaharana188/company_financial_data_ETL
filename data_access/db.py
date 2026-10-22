"""Read-only DuckDB connections for local OLAP or Supabase PostgreSQL.

In hosted mode, DuckDB attaches the Supabase public schema read-only and stays
responsible for analytical SQL. In local mode, it reads market_data.duckdb.
"""
from __future__ import annotations
import os
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "market_data.duckdb"


def get_db_path() -> Path | str:
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if dsn:
        return "Supabase PostgreSQL · public schema · read-only"
    path = Path(os.getenv("OLAP_DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
    if path.suffix.lower() != ".duckdb":
        raise ValueError(f"Expected a DuckDB database file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"OLAP database not found: {path}. Set OLAP_DB_PATH or SUPABASE_DB_URL."
        )
    return path


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def get_connection() -> duckdb.DuckDBPyConnection:
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if not dsn:
        path = get_db_path()
        if not isinstance(path, Path):
            raise RuntimeError("Local OLAP path did not resolve to a filesystem path.")
        return duckdb.connect(database=str(path), read_only=True)
    con = duckdb.connect(database=":memory:")
    try:
        try:
            con.execute("LOAD postgres")
        except Exception:
            con.execute("INSTALL postgres")
            con.execute("LOAD postgres")
        con.execute(
            f"ATTACH {_quote_sql_string(dsn)} AS market_db "
            "(TYPE postgres, READ_ONLY, SCHEMA 'public')"
        )
        con.execute("USE market_db")
        return con
    except Exception:
        con.close()
        raise RuntimeError(
            "Could not connect DuckDB to Supabase PostgreSQL. Check SUPABASE_DB_URL, "
            "the selected connection pooler, SSL, and the role's read permissions."
        ) from None
