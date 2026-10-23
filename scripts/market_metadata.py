"""Register market metadata in the active ETL destination.

Local development: use scripts.database.engine (the existing market_data.duckdb).
Hosted deployment: when SUPABASE_DB_URL is present, upsert directly into the
existing Supabase public.market_metadata table. This is necessary because the
DuckDB engine is local to the process and does not replicate writes to Supabase.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import pandas as pd
import psycopg2

from scripts.database import engine

_METADATA_COLUMNS = (
    "Ticker",
    "IndicatorName",
    "TargetTable",
    "Sector",
    "Industry",
    "AssetClass",
    "Exchange",
    "IsActive",
    "valid_data_since",
    "Description",
)

_SELECT_METADATA = '''
    SELECT "Ticker", "IndicatorName", "TargetTable", "Sector", "Industry",
           "AssetClass", "Exchange", "IsActive", "valid_data_since", "Description"
    FROM public.market_metadata
    WHERE "Ticker" = ANY(%s)
'''


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return value or None


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    ticker = _optional_text(record.get("Ticker"))
    if not ticker:
        raise ValueError("market_metadata record is missing a non-empty Ticker")

    indicator_name = _optional_text(record.get("IndicatorName"))
    target_table = _optional_text(record.get("TargetTable")) or "global_assets"
    asset_class = _optional_text(record.get("AssetClass")) or "Equity"
    active = record.get("IsActive", True)
    if isinstance(active, str):
        active = active.strip().lower() not in {"0", "false", "no", "inactive"}
    else:
        active = bool(active)

    return {
        "Ticker": ticker,
        "IndicatorName": indicator_name,
        "TargetTable": target_table,
        "Sector": _optional_text(record.get("Sector")),
        "Industry": _optional_text(record.get("Industry")),
        "AssetClass": asset_class,
        "Exchange": _optional_text(record.get("Exchange")),
        "IsActive": active,
        "valid_data_since": record.get("valid_data_since"),
        "Description": _optional_text(record.get("Description")),
    }


def _usable(value: Any) -> bool:
    text = _optional_text(value)
    return text is not None and text.upper() not in {"UNKNOWN", "N/A", "NONE"}


def _check_persisted(records: list[dict[str, Any]], rows: list[tuple[Any, ...]]) -> None:
    columns = list(_METADATA_COLUMNS)
    persisted = {str(row[0]): dict(zip(columns, row)) for row in rows}
    expected_by_ticker = {record["Ticker"]: record for record in records}
    missing = sorted(set(expected_by_ticker) - set(persisted))
    if missing:
        raise RuntimeError(
            "market_metadata write was not visible after commit for ticker(s): "
            + ", ".join(missing)
        )

    for ticker, expected in expected_by_ticker.items():
        actual = persisted[ticker]
        for field in ("TargetTable", "AssetClass", "IsActive"):
            if actual.get(field) != expected.get(field):
                raise RuntimeError(
                    f'market_metadata verification failed for {ticker}: {field} '
                    f'expected {expected.get(field)!r}, got {actual.get(field)!r}'
                )
        for field in ("Sector", "Industry", "Exchange", "Description"):
            if _usable(expected.get(field)) and actual.get(field) != expected.get(field):
                raise RuntimeError(
                    f'market_metadata verification failed for {ticker}: {field} '
                    f'expected {expected.get(field)!r}, got {actual.get(field)!r}'
                )
        indicator = _optional_text(expected.get("IndicatorName"))
        if indicator and indicator != ticker and actual.get("IndicatorName") != indicator:
            raise RuntimeError(
                f'market_metadata verification failed for {ticker}: IndicatorName '
                f'expected {indicator!r}, got {actual.get("IndicatorName")!r}'
            )
        if expected.get("valid_data_since") is not None and actual.get("valid_data_since") != expected.get("valid_data_since"):
            raise RuntimeError(
                f"market_metadata verification failed for {ticker}: valid_data_since was not persisted"
            )


def _duckdb_upsert(records: list[dict[str, Any]]) -> int:
    columns = ", ".join(f'"{column}"' for column in _METADATA_COLUMNS)
    query = f'''
        INSERT INTO market_metadata ({columns})
        SELECT {columns} FROM temp_market_metadata_upsert
        ON CONFLICT ("Ticker") DO UPDATE SET
            "IndicatorName" = CASE
                WHEN EXCLUDED."IndicatorName" IS NULL OR EXCLUDED."IndicatorName" = ''
                    THEN "IndicatorName"
                WHEN EXCLUDED."IndicatorName" = EXCLUDED."Ticker"
                    AND "IndicatorName" IS NOT NULL AND "IndicatorName" <> ''
                    THEN "IndicatorName"
                ELSE EXCLUDED."IndicatorName"
            END,
            "TargetTable" = COALESCE(NULLIF(EXCLUDED."TargetTable", ''), "TargetTable"),
            "Sector" = CASE
                WHEN EXCLUDED."Sector" IS NULL OR UPPER(TRIM(EXCLUDED."Sector")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Sector" ELSE EXCLUDED."Sector" END,
            "Industry" = CASE
                WHEN EXCLUDED."Industry" IS NULL OR UPPER(TRIM(EXCLUDED."Industry")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Industry" ELSE EXCLUDED."Industry" END,
            "AssetClass" = COALESCE(NULLIF(EXCLUDED."AssetClass", ''), "AssetClass"),
            "Exchange" = CASE
                WHEN EXCLUDED."Exchange" IS NULL OR UPPER(TRIM(EXCLUDED."Exchange")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Exchange" ELSE EXCLUDED."Exchange" END,
            "IsActive" = EXCLUDED."IsActive",
            "valid_data_since" = COALESCE(EXCLUDED."valid_data_since", "valid_data_since"),
            "Description" = COALESCE(NULLIF(EXCLUDED."Description", ''), "Description")
    '''
    frame = pd.DataFrame(records, columns=_METADATA_COLUMNS)
    registered = False
    try:
        engine.register("temp_market_metadata_upsert", frame)
        registered = True
        engine.execute(query)
    except Exception as exc:
        raise RuntimeError(
            "Failed to upsert market_metadata through the existing local DuckDB ETL engine."
        ) from exc
    finally:
        if registered:
            engine.unregister("temp_market_metadata_upsert")

    tickers = [record["Ticker"] for record in records]
    placeholders = ", ".join("?" for _ in tickers)
    readback = engine.execute(
        'SELECT ' + ", ".join(f'"{c}"' for c in _METADATA_COLUMNS)
        + f' FROM market_metadata WHERE "Ticker" IN ({placeholders})',
        tickers,
    ).df()
    _check_persisted(records, list(readback.itertuples(index=False, name=None)))
    print(
        f"[MARKET_METADATA] Upsert verified for {len(tickers)} ticker(s) in local ETL DuckDB: "
        + ", ".join(tickers[:10])
        + (" ..." if len(tickers) > 10 else "")
    )
    return len(records)


def _merge_existing_metadata(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Merge fresh metadata without erasing previously useful profile fields."""
    merged = dict(existing)
    ticker = incoming["Ticker"]

    incoming_indicator = _optional_text(incoming.get("IndicatorName"))
    existing_indicator = _optional_text(existing.get("IndicatorName"))
    if not incoming_indicator or (
        incoming_indicator == ticker
        and existing_indicator
        and existing_indicator != ticker
    ):
        merged["IndicatorName"] = existing.get("IndicatorName")
    else:
        merged["IndicatorName"] = incoming_indicator

    for field in ("TargetTable", "AssetClass"):
        merged[field] = _optional_text(incoming.get(field)) or existing.get(field)

    for field in ("Sector", "Industry", "Exchange"):
        value = incoming.get(field)
        merged[field] = value if _usable(value) else existing.get(field)

    # Activity is intentional state, so always accept its current value.
    merged["IsActive"] = incoming["IsActive"]
    merged["valid_data_since"] = (
        incoming.get("valid_data_since")
        if incoming.get("valid_data_since") is not None
        else existing.get("valid_data_since")
    )
    merged["Description"] = (
        _optional_text(incoming.get("Description"))
        or existing.get("Description")
    )
    merged["Ticker"] = ticker
    return merged


def _postgres_upsert(records: list[dict[str, Any]], dsn: str) -> int:
    """Write to Supabase without relying on a UNIQUE constraint on Ticker.

    The existing Supabase table does not have a matching unique/exclusion
    constraint for Ticker. Serialize this writer with an advisory transaction
    lock, then UPDATE matching rows or INSERT when none exists. No schema change
    is required. If legacy duplicate tickers exist, UPDATE all matching rows
    rather than creating more.
    """
    columns = ", ".join(f'"{column}"' for column in _METADATA_COLUMNS)
    value_placeholders = ", ".join(["%s"] * len(_METADATA_COLUMNS))
    update_columns = [column for column in _METADATA_COLUMNS if column != "Ticker"]
    update_assignments = ", ".join(f'"{column}" = %s' for column in update_columns)
    insert_sql = (
        f"INSERT INTO public.market_metadata ({columns}) "
        f"VALUES ({value_placeholders})"
    )
    update_sql = (
        f"UPDATE public.market_metadata SET {update_assignments} "
        'WHERE "Ticker" = %s'
    )

    tickers = [record["Ticker"] for record in records]
    inserted = 0
    updated = 0
    connection = None
    try:
        # Use the existing deployed connection string. This lock coordinates
        # this ETL's metadata writers and does not require schema/role changes.
        connection = psycopg2.connect(dsn, connect_timeout=20, sslmode="require")
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (824274110013,))

            for record in records:
                cursor.execute(
                    f'SELECT {columns} FROM public.market_metadata '
                    'WHERE "Ticker" = %s LIMIT 1',
                    (record["Ticker"],),
                )
                row = cursor.fetchone()
                if row is None:
                    values = tuple(record[column] for column in _METADATA_COLUMNS)
                    cursor.execute(insert_sql, values)
                    inserted += max(cursor.rowcount, 0)
                    continue

                existing = dict(zip(_METADATA_COLUMNS, row))
                merged = _merge_existing_metadata(existing, record)
                values = tuple(merged[column] for column in update_columns)
                cursor.execute(update_sql, values + (record["Ticker"],))
                updated += max(cursor.rowcount, 0)

        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_METADATA, (tickers,))
            rows = cursor.fetchall()
        _check_persisted(records, rows)
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        error_text = str(exc)
        if "permission denied" in error_text.lower() or "not authorized" in error_text.lower():
            hint = " The SUPABASE_DB_URL database role needs SELECT, INSERT, and UPDATE privileges on public.market_metadata."
        elif "does not exist" in error_text.lower() and "market_metadata" in error_text.lower():
            hint = " The expected existing target is public.market_metadata in the connected Supabase project."
        else:
            hint = ""
        raise RuntimeError(
            "Supabase metadata write failed for public.market_metadata. "
            "The writer uses UPDATE/INSERT and does not require a unique constraint on Ticker."
            + hint + f" Database error: {error_text}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()

    print(
        f"[MARKET_METADATA] Supabase public.market_metadata verified: "
        f"{len(tickers)} ticker(s), {inserted} inserted, {updated} updated: "
        + ", ".join(tickers[:10])
        + (" ..." if len(tickers) > 10 else "")
    )
    return len(records)


def upsert_market_metadata(records: Iterable[dict[str, Any]]) -> int:
    """Upsert and verify metadata at the destination used in this environment.

    Local: use the existing DuckDB engine. Hosted: use SUPABASE_DB_URL to write
    directly to the existing public.market_metadata table. This avoids writing
    metadata to Render's ephemeral local DuckDB while downstream apps read Supabase.
    """
    by_ticker: dict[str, dict[str, Any]] = {}
    for record in records:
        normalised = _normalise_record(record)
        by_ticker[normalised["Ticker"]] = normalised
    if not by_ticker:
        return 0

    normalised_records = list(by_ticker.values())
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if dsn:
        return _postgres_upsert(normalised_records, dsn)
    return _duckdb_upsert(normalised_records)


def fetch_active_global_assets() -> pd.DataFrame:
    """Read active assets from the same destination that receives metadata writes."""
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if dsn:
        connection = None
        try:
            connection = psycopg2.connect(dsn, connect_timeout=20, sslmode="require")
            with connection.cursor() as cursor:
                cursor.execute('''
                    SELECT "Ticker", "AssetClass", "Exchange"
                    FROM public.market_metadata
                    WHERE "IsActive" = TRUE
                    ORDER BY "Ticker"
                ''')
                rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=["Ticker", "AssetClass", "Exchange"])
        except Exception as exc:
            raise RuntimeError(
                "Failed to read active global assets from Supabase public.market_metadata: "
                + str(exc)
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    return engine.execute('''
        SELECT "Ticker", "AssetClass", "Exchange"
        FROM market_metadata
        WHERE "IsActive" = TRUE
        ORDER BY "Ticker"
    ''').df()
