"""Incremental writer for macro and global-asset OHLC data.

Local development writes through the repository's existing DuckDB engine.
When SUPABASE_DB_URL is set (Render/hosted mode), records are written to the
existing Supabase public tables in bounded batches. The hosted path deliberately
uses UPDATE + INSERT rather than ON CONFLICT because the migrated destination
may not have primary/unique constraints matching the natural keys.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from typing import Any

import pandas as pd
from scripts.database import engine

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # Local mode does not require a live PostgreSQL client.
    psycopg2 = None
    execute_values = None

STANDARD_COLUMNS = [
    "EntityName", "ReportDate", "Timeframe", "Category", "AssetClass",
    "Open", "High", "Low", "Close", "Volume",
]

# Each entry is (destination table, destination columns, natural key columns).
TABLE_SPECS = {
    ("MACRO", "1d"): (
        "macro_daily_ledger",
        ["IndicatorName", "ReportDate", "Open", "High", "Low", "Close_Value", "Volume"],
        ["IndicatorName", "ReportDate"],
    ),
    ("MACRO", "INTRADAY"): (
        "macro_intraday_ledger",
        ["IndicatorName", "ReportDate", "Timeframe", "Open", "High", "Low", "Close_Value", "Volume"],
        ["IndicatorName", "ReportDate", "Timeframe"],
    ),
    ("ASSET", "1d"): (
        "global_assets_daily",
        ["Ticker", "ReportDate", "AssetClass", "Open", "High", "Low", "Close", "Volume"],
        ["Ticker", "ReportDate"],
    ),
    ("ASSET", "INTRADAY"): (
        "global_assets_intraday",
        ["Ticker", "ReportDate", "Timeframe", "Open", "High", "Low", "Close", "Volume"],
        ["Ticker", "ReportDate", "Timeframe"],
    ),
}


def _signed_lock_id(name: str) -> int:
    value = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return value - (1 << 64) if value >= (1 << 63) else value


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _db_value(value: Any, column: str) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if column == "ReportDate":
        # Daily batches have already been normalized to Python date objects;
        # intraday batches remain datetimes even for bars stamped at midnight.
        if isinstance(value, datetime):
            return value.replace(tzinfo=None) if value.tzinfo is not None else value
        if isinstance(value, date):
            return value
        return pd.Timestamp(value).to_pydatetime()
    if column == "Volume":
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    # Remove timezone after converting source timestamps; PostgreSQL columns are
    # TIMESTAMP (not TIMESTAMPTZ) in the current tables.
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.replace(tzinfo=None)
    return value


class MarketDataWriter:
    """Write one ticker/timeframe at a time and commit each batch."""

    def __init__(self, batch_size: int = 2_000):
        self.batch_size = max(1, int(batch_size))
        self.dsn = os.getenv("SUPABASE_DB_URL", "").strip()
        self.connection = None
        self.destination = "Supabase PostgreSQL public schema" if self.dsn else "local ETL DuckDB"
        if self.dsn:
            if psycopg2 is None or execute_values is None:
                raise RuntimeError("SUPABASE_DB_URL is set but psycopg2 is unavailable; install requirements.txt.")
            try:
                self.connection = psycopg2.connect(
                    self.dsn, connect_timeout=20, sslmode="require"
                )
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT current_database(), current_user")
                    database_name, role_name = cursor.fetchone()
                self.connection.commit()
                print(f"[MARKET_DATA] Destination verified: Supabase database={database_name}, role={role_name}.")
            except Exception as exc:
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                raise RuntimeError(
                    "Could not connect to Supabase for macro/global-asset writes. "
                    "Check SUPABASE_DB_URL and its network/SSL settings. " + str(exc)
                ) from exc
        else:
            print(f"[MARKET_DATA] Destination: {self.destination} ({os.path.abspath(engine.db_path)}).")

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def write_frame(self, frame: pd.DataFrame) -> int:
        """Persist one or more normalized data batches; returns input row count."""
        if frame is None or frame.empty:
            return 0
        missing = [column for column in STANDARD_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError("Macro batch missing required columns: " + ", ".join(missing))

        written = 0
        # Grouping here keeps the function safe for callers that still hand it a
        # combined frame. The live refresh generator only yields one group at a time.
        for (_, _, _), group in frame.groupby(
            ["Category", "EntityName", "Timeframe"], sort=False, dropna=False
        ):
            category = str(group["Category"].iloc[0]).upper()
            timeframe = str(group["Timeframe"].iloc[0])
            period = "1d" if timeframe == "1d" else "INTRADAY"
            spec = TABLE_SPECS.get((category, period))
            if spec is None:
                raise ValueError(f"Unsupported market-data destination: category={category}, timeframe={timeframe}")
            table, columns, keys = spec
            prepared = self._prepare_group(group, category, timeframe, columns, keys)
            if prepared.empty:
                continue
            if self.dsn:
                self._write_postgres(table, prepared, columns, keys)
            else:
                self._write_duckdb(table, prepared, columns, keys)
            written += len(prepared)
        return written

    @staticmethod
    def _prepare_group(
        group: pd.DataFrame,
        category: str,
        timeframe: str,
        columns: list[str],
        keys: list[str],
    ) -> pd.DataFrame:
        data = group.copy()
        if category == "MACRO":
            data = data.rename(columns={"EntityName": "IndicatorName", "Close": "Close_Value"})
        else:
            data = data.rename(columns={"EntityName": "Ticker"})

        data["ReportDate"] = pd.to_datetime(data["ReportDate"], errors="coerce")
        data = data.dropna(subset=["ReportDate", "Close"] if category == "ASSET" else ["ReportDate", "Close_Value"])
        if timeframe == "1d":
            data["ReportDate"] = data["ReportDate"].dt.date
        else:
            # The source data index is made timezone-naive before arriving here.
            data["ReportDate"] = data["ReportDate"].dt.tz_localize(None)
            data["Timeframe"] = timeframe

        if "AssetClass" in columns:
            data["AssetClass"] = data["AssetClass"].fillna("Equity")
        for col in columns:
            if col not in data.columns:
                data[col] = None
        data = data[columns]
        data = data.drop_duplicates(subset=keys, keep="last")
        return data.reset_index(drop=True)

    def _write_duckdb(self, table: str, frame: pd.DataFrame, columns: list[str], keys: list[str]) -> None:
        stage = "temp_market_batch_" + hashlib.sha1(table.encode()).hexdigest()[:10]
        cols_sql = ", ".join(f'"{column}"' for column in columns)
        keys_sql = ", ".join(f'"{column}"' for column in keys)
        updates = [column for column in columns if column not in keys]
        assignments = ", ".join(
            f'"{column}" = COALESCE(EXCLUDED."{column}", "{column}")' for column in updates
        )
        sql = (
            f'INSERT INTO "{table}" ({cols_sql}) SELECT {cols_sql} FROM "{stage}" '
            f'ON CONFLICT ({keys_sql}) DO UPDATE SET {assignments}'
        )
        registered = False
        try:
            engine.register(stage, frame)
            registered = True
            engine.execute(sql)
        except Exception as exc:
            raise RuntimeError(f"Local DuckDB write failed for {table} ({len(frame)} rows): {exc}") from exc
        finally:
            if registered:
                engine.unregister(stage)
        print(f"[MARKET_DATA] Local DuckDB upsert: {table}, {len(frame)} row(s).")

    def _write_postgres(self, table: str, frame: pd.DataFrame, columns: list[str], keys: list[str]) -> None:
        assert self.connection is not None and execute_values is not None
        stage = "__etl_stage_" + table
        qcols = ", ".join(f'"{column}"' for column in columns)
        qkeys = " AND ".join(f't."{key}" = s."{key}"' for key in keys)
        updates = [column for column in columns if column not in keys]
        set_sql = ", ".join(
            f'"{column}" = COALESCE(s."{column}", t."{column}")' for column in updates
        )
        insert_select = ", ".join(f's."{column}"' for column in columns)
        rows = [
            tuple(_db_value(value, column) for column, value in zip(columns, row))
            for row in frame.itertuples(index=False, name=None)
        ]
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_signed_lock_id(table),))
                cursor.execute(
                    f'CREATE TEMP TABLE IF NOT EXISTS "{stage}" '
                    f'(LIKE public."{table}" INCLUDING DEFAULTS) ON COMMIT DELETE ROWS'
                )
                cursor.execute(f'TRUNCATE TABLE "{stage}"')
                for start in range(0, len(rows), self.batch_size):
                    execute_values(
                        cursor,
                        f'INSERT INTO "{stage}" ({qcols}) VALUES %s',
                        rows[start : start + self.batch_size],
                        page_size=self.batch_size,
                    )
                if updates:
                    cursor.execute(
                        f'UPDATE public."{table}" AS t SET {set_sql} '
                        f'FROM "{stage}" AS s WHERE {qkeys}'
                    )
                    updated = max(cursor.rowcount, 0)
                else:
                    updated = 0
                cursor.execute(
                    f'INSERT INTO public."{table}" ({qcols}) '
                    f'SELECT {insert_select} FROM "{stage}" AS s '
                    f'WHERE NOT EXISTS (SELECT 1 FROM public."{table}" AS t WHERE {qkeys})'
                )
                inserted = max(cursor.rowcount, 0)
            self.connection.commit()
        except Exception as exc:
            self.connection.rollback()
            message = str(exc)
            if "permission denied" in message.lower():
                message += (
                    " The SUPABASE_DB_URL role needs SELECT, INSERT, and UPDATE on "
                    f"public.{table}. See SUPABASE_MACRO_WRITE_PERMISSIONS.sql in this ZIP."
                )
            raise RuntimeError(
                f"Supabase macro-data write failed for public.{table} ({len(frame)} staged rows): {message}"
            ) from exc
        print(
            f"[MARKET_DATA] Supabase public.{table}: staged={len(frame)}, "
            f"updated={updated}, inserted={inserted}."
        )
