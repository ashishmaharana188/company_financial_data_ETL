"""Market metadata upserts through the ETL's existing DuckDB engine.

Metadata deliberately uses the same ``scripts.database.engine`` as statement
writes. Do not open a second connection or independently resolve a database
path here: doing so can send metadata and financial statements to different
DuckDB files.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
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
    is_active = record.get("IsActive", True)
    if isinstance(is_active, str):
        is_active = is_active.strip().lower() not in {"0", "false", "no", "inactive"}
    else:
        is_active = bool(is_active)

    return {
        "Ticker": ticker,
        "IndicatorName": indicator_name,
        "TargetTable": target_table,
        "Sector": _optional_text(record.get("Sector")),
        "Industry": _optional_text(record.get("Industry")),
        "AssetClass": asset_class,
        "Exchange": _optional_text(record.get("Exchange")),
        "IsActive": is_active,
        "valid_data_since": record.get("valid_data_since"),
        "Description": _optional_text(record.get("Description")),
    }


def _metadata_upsert_sql() -> str:
    columns = ", ".join(f'"{column}"' for column in _METADATA_COLUMNS)
    # Preserve previously useful profile fields when discovery/profile APIs
    # return nulls or placeholders. Refresh activity and non-empty core fields.
    return f'''
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
                WHEN EXCLUDED."Sector" IS NULL
                    OR UPPER(TRIM(EXCLUDED."Sector")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Sector"
                ELSE EXCLUDED."Sector"
            END,
            "Industry" = CASE
                WHEN EXCLUDED."Industry" IS NULL
                    OR UPPER(TRIM(EXCLUDED."Industry")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Industry"
                ELSE EXCLUDED."Industry"
            END,
            "AssetClass" = COALESCE(NULLIF(EXCLUDED."AssetClass", ''), "AssetClass"),
            "Exchange" = CASE
                WHEN EXCLUDED."Exchange" IS NULL
                    OR UPPER(TRIM(EXCLUDED."Exchange")) IN ('', 'UNKNOWN', 'N/A', 'NONE')
                    THEN "Exchange"
                ELSE EXCLUDED."Exchange"
            END,
            "IsActive" = EXCLUDED."IsActive",
            "valid_data_since" = COALESCE(EXCLUDED."valid_data_since", "valid_data_since"),
            "Description" = COALESCE(NULLIF(EXCLUDED."Description", ''), "Description")
    '''


def _valid_update_value(value: Any) -> bool:
    text = _optional_text(value)
    return text is not None and text.strip().upper() not in {"UNKNOWN", "N/A", "NONE"}


def _verify_records(records: list[dict[str, Any]]) -> None:
    """Read back rows and check that meaningful fields really persisted."""
    by_ticker = {record["Ticker"]: record for record in records}
    tickers = list(by_ticker)
    persisted: dict[str, dict[str, Any]] = {}

    for offset in range(0, len(tickers), 400):
        chunk = tickers[offset : offset + 400]
        placeholders = ", ".join(["?"] * len(chunk))
        query = (
            'SELECT "Ticker", "IndicatorName", "TargetTable", "Sector", "Industry", '
            '"AssetClass", "Exchange", "IsActive", "valid_data_since", "Description" '
            f'FROM market_metadata WHERE "Ticker" IN ({placeholders})'
        )
        result = engine.execute(query, chunk).df()
        for row in result.to_dict("records"):
            persisted[str(row["Ticker"])] = row

    missing = set(tickers) - set(persisted)
    if missing:
        raise RuntimeError(
            "market_metadata upsert did not persist ticker(s) in the ETL database: "
            + ", ".join(sorted(missing))
        )

    # Fields explicitly sent by ETL must be visible after the write. Fields
    # omitted/Unknown are intentionally not compared because the upsert
    # preserves a previously known value instead of erasing it.
    for ticker, expected in by_ticker.items():
        actual = persisted[ticker]
        for field in ("TargetTable", "AssetClass", "IsActive"):
            if actual.get(field) != expected.get(field):
                raise RuntimeError(
                    f'market_metadata verification failed for {ticker}: '
                    f'{field} expected {expected.get(field)!r}, found {actual.get(field)!r}'
                )
        for field in ("Sector", "Industry", "Exchange", "Description"):
            if _valid_update_value(expected.get(field)) and actual.get(field) != expected.get(field):
                raise RuntimeError(
                    f'market_metadata verification failed for {ticker}: '
                    f'{field} expected {expected.get(field)!r}, found {actual.get(field)!r}'
                )
        indicator = _optional_text(expected.get("IndicatorName"))
        if indicator and indicator != ticker and actual.get("IndicatorName") != indicator:
            raise RuntimeError(
                f'market_metadata verification failed for {ticker}: '
                f'IndicatorName expected {indicator!r}, found {actual.get("IndicatorName")!r}'
            )
        if expected.get("valid_data_since") is not None and actual.get("valid_data_since") != expected.get("valid_data_since"):
            raise RuntimeError(
                f'market_metadata verification failed for {ticker}: valid_data_since was not persisted'
            )


def upsert_market_metadata(records: Iterable[dict[str, Any]]) -> int:
    """Insert/update metadata through the existing ETL engine, then verify it.

    This intentionally uses ``engine.register/execute/unregister`` just like
    ``duckdb_upsert`` in statementScrape.py. It never routes metadata based on
    environment variables or chooses a separate DuckDB file.
    """
    # De-duplicate in input order. The latest record for each ticker wins,
    # avoiding DuckDB's error when one ON CONFLICT statement repeats a key.
    by_ticker: dict[str, dict[str, Any]] = {}
    for record in records:
        normalised = _normalise_record(record)
        by_ticker[normalised["Ticker"]] = normalised

    if not by_ticker:
        return 0

    frame = pd.DataFrame(list(by_ticker.values()), columns=_METADATA_COLUMNS)
    view_name = "temp_market_metadata_upsert"
    registered = False
    try:
        engine.register(view_name, frame)
        registered = True
        engine.execute(_metadata_upsert_sql())
    except Exception as exc:
        raise RuntimeError(
            "Failed to upsert market_metadata through the same DuckDB engine used by statement tables. "
            "The ETL cannot claim metadata was updated."
        ) from exc
    finally:
        if registered:
            try:
                engine.unregister(view_name)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    "Metadata upsert ran, but the DuckDB temporary view/connection could not be closed."
                ) from cleanup_exc

    tickers = list(by_ticker)
    _verify_records(list(by_ticker.values()))
    print(
        f"[MARKET_METADATA] Upsert verified for {len(tickers)} ticker(s) "
        f"using ETL database: {', '.join(tickers[:10])}"
        + (" ..." if len(tickers) > 10 else "")
    )
    return len(tickers)


def fetch_active_global_assets() -> pd.DataFrame:
    """Read active assets through the same engine that owns the ETL database."""
    result = engine.execute(
        '''
        SELECT "Ticker", "AssetClass", "Exchange"
        FROM market_metadata
        WHERE "IsActive" = TRUE
        ORDER BY "Ticker"
        '''
    ).df()
    return result
