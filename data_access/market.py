from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pyarrow as pa

from .ticker import normalize_ticker
from .query import execute_arrow, execute_rows


def _date_value(
    value: str | date | datetime,
) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    return value


# ============================================================
# COMPANY DAILY MARKET DATA
#
# SOURCE:
#     global_assets_daily
#
# SCHEMA:
#     Ticker
#     ReportDate
#     AssetClass
#     Open
#     High
#     Low
#     Close
#     Volume
# ============================================================


def get_daily_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return daily market history using ticker fallback resolution.

    Lookup order:
        RELIANCE      -> RELIANCE, RELIANCE.NS
        RELIANCE.NS   -> RELIANCE, RELIANCE.NS

    The first candidate that returns data is used.
    """

    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            "Ticker" AS ticker,
            "ReportDate" AS report_date,
            "AssetClass" AS asset_class,
            "Open" AS open,
            "High" AS high,
            "Low" AS low,
            "Close" AS close,
            "Volume" AS volume
        FROM global_assets_daily
        WHERE "Ticker" = ?
    """

    extra_params: list[Any] = []

    if start_date is not None:
        sql += ' AND "ReportDate" >= ?'
        extra_params.append(_date_value(start_date))

    if end_date is not None:
        sql += ' AND "ReportDate" <= ?'
        extra_params.append(_date_value(end_date))

    sql += ' ORDER BY "ReportDate"'

    for candidate in candidates:
        params = [candidate, *extra_params]

        table = execute_arrow(sql, params)

        if table.num_rows > 0:
            return table

    # Return an empty table with the expected schema.
    return execute_arrow(
        sql,
        [candidates[0], *extra_params],
    )


def get_latest_daily_row(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any] | None:
    """
    Return the latest available daily market row for the ticker.
    """

    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            "Ticker" AS ticker,
            "ReportDate" AS report_date,
            "AssetClass" AS asset_class,
            "Open" AS open,
            "High" AS high,
            "Low" AS low,
            "Close" AS close,
            "Volume" AS volume
        FROM global_assets_daily
        WHERE "Ticker" = ?
          AND "Close" IS NOT NULL
    """

    extra_params: list[Any] = []

    if as_of_date is not None:
        sql += ' AND "ReportDate" <= ?'
        extra_params.append(_date_value(as_of_date))

    sql += """
        ORDER BY "ReportDate" DESC
        LIMIT 1
    """

    for candidate in candidates:
        rows = execute_rows(
            sql,
            [candidate, *extra_params],
        )

        if rows:
            columns = [
                "ticker",
                "report_date",
                "asset_class",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            return dict(zip(columns, rows[0]))

    return None


def get_next_trading_day(
    ticker: str,
    after_date: str | date | datetime,
) -> dict[str, Any] | None:
    """
    Return the first available trading observation after after_date.
    """

    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            "Ticker" AS ticker,
            "ReportDate" AS report_date,
            "AssetClass" AS asset_class,
            "Open" AS open,
            "High" AS high,
            "Low" AS low,
            "Close" AS close,
            "Volume" AS volume
        FROM global_assets_daily
        WHERE "Ticker" = ?
          AND "ReportDate" > ?
          AND "Close" IS NOT NULL
        ORDER BY "ReportDate"
        LIMIT 1
    """

    after_value = _date_value(after_date)

    for candidate in candidates:
        rows = execute_rows(
            sql,
            [candidate, after_value],
        )

        if rows:
            columns = [
                "ticker",
                "report_date",
                "asset_class",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            return dict(zip(columns, rows[0]))

    return None


def get_price_on_or_before(
    ticker: str,
    target_date: str | date | datetime,
) -> dict[str, Any] | None:
    """
    Return the latest available close on or before target_date.
    """

    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            "Ticker" AS ticker,
            "ReportDate" AS report_date,
            "Close" AS close
        FROM global_assets_daily
        WHERE "Ticker" = ?
          AND "ReportDate" <= ?
          AND "Close" IS NOT NULL
        ORDER BY "ReportDate" DESC
        LIMIT 1
    """

    target_value = _date_value(target_date)

    for candidate in candidates:
        rows = execute_rows(
            sql,
            [candidate, target_value],
        )

        if rows:
            return {
                "ticker": rows[0][0],
                "report_date": rows[0][1],
                "close": rows[0][2],
            }

    return None


def get_market_date_range(
    ticker: str,
) -> dict[str, Any] | None:
    """
    Return market-data coverage for the ticker.
    """

    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            MIN("ReportDate"),
            MAX("ReportDate"),
            COUNT(*),
            COUNT("Close")
        FROM global_assets_daily
        WHERE "Ticker" = ?
    """

    for candidate in candidates:
        rows = execute_rows(
            sql,
            [candidate],
        )

        if not rows:
            continue

        first_date = rows[0][0]

        if first_date is None:
            continue

        return {
            "first_date": first_date,
            "last_date": rows[0][1],
            "row_count": rows[0][2],
            "rows_with_close": rows[0][3],
        }

    return None