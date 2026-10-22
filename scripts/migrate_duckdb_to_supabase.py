"""Stream DuckDB tables and views into Supabase PostgreSQL snapshot tables.

This is an initial/bulk migration utility, not a continuous replication service.
It copies each local DuckDB base table and view into a regular PostgreSQL table
in the Supabase public schema. Views are materialized as snapshots so the
existing TrendAnalysis data-access queries can read them as relations.

Required environment variable: SUPABASE_MIGRATION_DB_URL (or SUPABASE_DB_URL as a fallback)
Optional: OLAP_DB_PATH (defaults to repository-root market_data.duckdb)

Run first against a clean Supabase project's public schema. Use
--replace-existing only when you explicitly want to replace matching tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any

import duckdb
import psycopg2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "market_data.duckdb"
BATCH_SIZE = 10_000


def qi(identifier: str) -> str:
    """Quote a PostgreSQL/DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def qs(value: str) -> str:
    """Quote an SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def relation_sql(schema: str, name: str) -> str:
    return f"{qi(schema)}.{qi(name)}"


def postgres_type(duck_type: str) -> str:
    """Map DuckDB types to compatible PostgreSQL storage types."""
    value = duck_type.strip().upper()
    if value in {"BOOLEAN", "BOOL"}:
        return "BOOLEAN"
    if value in {"TINYINT", "UTINYINT", "SMALLINT", "USMALLINT"}:
        return "INTEGER"
    if value in {"INTEGER", "INT", "UINTEGER"}:
        return "INTEGER"
    if value in {"BIGINT", "INT8"}:
        return "BIGINT"
    if value in {"UBIGINT", "HUGEINT", "UHUGEINT"}:
        return "NUMERIC(39,0)"
    if value in {"FLOAT", "REAL", "FLOAT4"}:
        return "REAL"
    if value in {"DOUBLE", "FLOAT8", "DOUBLE PRECISION"}:
        return "DOUBLE PRECISION"
    if value.startswith("DECIMAL(") or value.startswith("NUMERIC("):
        return value.replace("DECIMAL", "NUMERIC", 1)
    if value in {"DATE"}:
        return "DATE"
    if value.startswith("TIMESTAMP WITH TIME ZONE") or value in {"TIMESTAMPTZ"}:
        return "TIMESTAMPTZ"
    if value.startswith("TIMESTAMP"):
        return "TIMESTAMP"
    if value.startswith("TIME WITH TIME ZONE") or value == "TIMETZ":
        return "TIMETZ"
    if value.startswith("TIME"):
        return "TIME"
    if value == "INTERVAL":
        return "INTERVAL"
    if value in {"UUID"}:
        return "UUID"
    if value in {"BLOB", "BYTEA"}:
        return "BYTEA"
    if value in {"JSON", "JSONB"}:
        return "JSONB"
    if value in {"VARCHAR", "TEXT", "STRING"} or value.startswith(("VARCHAR(", "CHAR(", "ENUM(")):
        return "TEXT"
    if value.startswith(("LIST(", "STRUCT(", "MAP(", "UNION(", "ARRAY(")) or value.endswith("[]"):
        return "JSONB"
    # Unknown/extension types are kept as text so the migration completes without
    # silently inventing a specialised PostgreSQL type.
    return "TEXT"


def value_to_text(value: Any, target_type: str) -> str | None:
    if value is None:
        return None
    if target_type == "JSONB" and isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return "\\x" + value.hex()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def csv_cell(value: str | None) -> str:
    # COPY CSV only treats an unquoted \N as NULL. Quote every non-null value,
    # including strings that literally equal \N, and double embedded quotes.
    if value is None:
        return r"\N"
    return '"' + value.replace('"', '""') + '"'


def get_relations(source: duckdb.DuckDBPyConnection, selected: set[str] | None):
    rows = source.execute(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_schema, table_name
        """
    ).fetchall()
    relations = [(str(schema), str(name), str(kind)) for schema, name, kind in rows]
    if selected:
        relations = [r for r in relations if r[1] in selected or f"{r[0]}.{r[1]}" in selected]
        missing = selected - {r[1] for r in relations} - {f"{r[0]}.{r[1]}" for r in relations}
        if missing:
            raise ValueError("Source relation(s) not found: " + ", ".join(sorted(missing)))
    if not relations:
        raise RuntimeError("No DuckDB tables or views were found to migrate.")
    for _, name, _ in relations:
        if len(name.encode("utf-8")) > 63:
            raise ValueError(f"Relation name exceeds PostgreSQL's 63-byte identifier limit: {name}")
    names = [name for _, name, _ in relations]
    if len(set(names)) != len(names):
        raise ValueError("Multiple source schemas contain the same relation name; flattening to public would collide.")
    return relations


def describe_relation(source: duckdb.DuckDBPyConnection, schema: str, name: str):
    rows = source.execute(f"DESCRIBE SELECT * FROM {relation_sql(schema, name)}").fetchall()
    return [(str(row[0]), str(row[1]), postgres_type(str(row[1]))) for row in rows]


def existing_public_relations(pg_conn, names: set[str]) -> set[str]:
    if not names:
        return set()
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (list(names),),
        )
        return {row[0] for row in cur.fetchall()}


def index_candidates(columns: list[str]) -> list[tuple[str, ...]]:
    existing = set(columns)
    candidates = [
        ("Ticker", "ReportDate"),
        ("IndicatorName", "ReportDate"),
        ("ClientType", "ReportDate"),
        ("ticker", "date"),
        ("ReportDate", "ClientType"),
        ("ReportDate", "Ticker"),
    ]
    matches = [candidate for candidate in candidates if set(candidate).issubset(existing)]
    if not matches:
        for single in ("ReportDate", "report_date", "date"):
            if single in existing:
                matches.append((single,))
                break
    return matches[:2]


def migrate_relation(source, pg_conn, schema: str, name: str, kind: str, replace_existing: bool, run_id: str, batch_size: int):
    source_relation = relation_sql(schema, name)
    columns = describe_relation(source, schema, name)
    if not columns:
        print(f"SKIP {schema}.{name}: no columns")
        return 0

    column_names = [c[0] for c in columns]
    pg_types = [c[2] for c in columns]
    column_definitions = ", ".join(f"{qi(col)} {typ}" for col, typ in zip(column_names, pg_types))
    digest = hashlib.sha1(f"{schema}.{name}".encode()).hexdigest()[:8]
    stage_name = f"__load_{run_id}_{digest}"[:63]
    target = f"public.{qi(name)}"
    stage = f"public.{qi(stage_name)}"
    quoted_columns = ", ".join(qi(col) for col in column_names)
    copy_sql = (
        f"COPY {stage} ({quoted_columns}) FROM STDIN "
        "WITH (FORMAT CSV, NULL '\\N', QUOTE '\"', ESCAPE '\"')"
    )
    source_cursor = source.execute(f"SELECT * FROM {source_relation}")
    row_count = 0

    try:
        with pg_conn:
            with pg_conn.cursor() as dest:
                dest.execute(f"DROP TABLE IF EXISTS {stage}")
                dest.execute(f"CREATE TABLE {stage} ({column_definitions})")
                while True:
                    batch = source_cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    buffer = StringIO()
                    for row in batch:
                        cells = []
                        for value, pg_type in zip(row, pg_types):
                            cells.append(csv_cell(value_to_text(value, pg_type)))
                        buffer.write(",".join(cells) + "\n")
                    buffer.seek(0)
                    dest.copy_expert(copy_sql, buffer)
                    row_count += len(batch)

                # Helpful indexes for the app's high-frequency date/ticker filters.
                for index_number, cols in enumerate(index_candidates(column_names), start=1):
                    index_hash = hashlib.sha1(f"{name}:{index_number}".encode()).hexdigest()[:10]
                    index_name = f"ix_mkt_{index_hash}"
                    dest.execute(
                        f"CREATE INDEX {qi(index_name)} ON {stage} "
                        f"({', '.join(qi(c) for c in cols)})"
                    )

                if replace_existing:
                    # Explicit opt-in only. DROP without CASCADE so external view/FK
                    # dependencies stop the replacement rather than being removed.
                    dest.execute(f"DROP TABLE IF EXISTS {target}")
                else:
                    dest.execute(f"ALTER TABLE {stage} RENAME TO {qi(name)}")
                    print(f"OK {schema}.{name} ({kind.lower()} snapshot): {row_count:,} rows")
                    return row_count

                dest.execute(f"ALTER TABLE {stage} RENAME TO {qi(name)}")
        print(f"OK {schema}.{name} ({kind.lower()} snapshot): {row_count:,} rows")
        return row_count
    except Exception as exc:
        # The current relation's transaction rolls back. Relation-level errors do
        # not include the connection URL, so include the database error for diagnosis.
        raise RuntimeError(
            f"Migration failed for relation {schema}.{name}; the current relation was rolled back. "
            f"Database error: {exc}. Use --replace-existing only deliberately."
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.getenv("OLAP_DB_PATH", str(DEFAULT_DB_PATH)), help="Path to source market_data.duckdb")
    parser.add_argument("--tables", help="Optional comma-separated relation names (or schema.name values)")
    parser.add_argument("--replace-existing", action="store_true", help="Replace same-named public tables after staging and loading each relation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per COPY batch; default 10000")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    migration_url = (
        os.getenv("SUPABASE_MIGRATION_DB_URL", "").strip()
        or os.getenv("SUPABASE_DB_URL", "").strip()
    )
    if not migration_url:
        print("Set SUPABASE_MIGRATION_DB_URL before running this migration.", file=sys.stderr)
        return 2

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        print(f"Source DuckDB file not found: {source_path}", file=sys.stderr)
        return 2

    # Honour optional smaller batch size for migration hosts with limited RAM.
    batch_size = args.batch_size
    selected = {item.strip() for item in args.tables.split(",") if item.strip()} if args.tables else None
    source = duckdb.connect(database=str(source_path), read_only=True)
    pg_conn = None
    try:
        relations = get_relations(source, selected)
        try:
            pg_conn = psycopg2.connect(
                migration_url,
                sslmode="require",
                connect_timeout=20,
            )
            with pg_conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT current_database(), current_user")
                db_name, db_user = cur.fetchone()
            pg_conn.commit()
        except Exception:
            print(
                "Could not connect to Supabase. Verify SUPABASE_MIGRATION_DB_URL, the database password, "
                "the pooler/host, and SSL settings. The URL has not been printed.",
                file=sys.stderr,
            )
            return 2

        relation_names = {name for _, name, _ in relations}
        existing = existing_public_relations(pg_conn, relation_names)
        if existing and not args.replace_existing:
            print(
                "The following target tables already exist in Supabase public: "
                + ", ".join(sorted(existing))
                + ". No tables were changed. Use a clean target schema or pass --replace-existing deliberately.",
                file=sys.stderr,
            )
            return 2

        run_id = hashlib.sha1(os.urandom(16)).hexdigest()[:6]
        print(f"Source: {source_path}")
        print(f"Target database: {db_name}; role: {db_user}")
        print(f"Relations to migrate: {len(relations)} (tables and views become PostgreSQL snapshot tables)")
        print("Starting migration. Each relation is staged and committed separately.")
        total_rows = 0
        for schema, name, kind in relations:
            total_rows += migrate_relation(
                source, pg_conn, schema, name, kind, args.replace_existing, run_id, batch_size
            )
        print(f"Migration complete: {len(relations)} relations, {total_rows:,} rows.")
        print("Reminder: this is a one-time snapshot. Schedule or implement a sync path for future ETL updates.")
        return 0
    finally:
        if pg_conn is not None:
            pg_conn.close()
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
