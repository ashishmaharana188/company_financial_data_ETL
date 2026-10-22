from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pyarrow as pa

from .query import execute_arrow, execute_rows


def _date_value(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    return value


# ============================================================
# MACRO DATA
# Source: macro_daily_ledger
#
# This contains:
#   Crude
#   DXY
#   VIX
#   Yields
#   CPI
#   Nifty
#   etc.
# ============================================================

def get_macro_history(
    indicator: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    sql = """
        SELECT
            IndicatorName AS indicator,
            ReportDate AS report_date,
            Open AS open,
            High AS high,
            Low AS low,
            Close_Value AS close,
            Volume AS volume
        FROM macro_daily_ledger
        WHERE IndicatorName = ?
    """

    params: list[Any] = [indicator]

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    sql += " ORDER BY ReportDate"

    return execute_arrow(sql, params)


def get_latest_macro(
    indicator: str,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any] | None:

    sql = """
        SELECT
            IndicatorName AS indicator,
            ReportDate AS report_date,
            Open AS open,
            High AS high,
            Low AS low,
            Close_Value AS close,
            Volume AS volume
        FROM macro_daily_ledger
        WHERE IndicatorName = ?
          AND Close_Value IS NOT NULL
    """

    params: list[Any] = [indicator]

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    sql += """
        ORDER BY ReportDate DESC
        LIMIT 1
    """

    rows = execute_rows(sql, params)

    if not rows:
        return None

    columns = [
        "indicator",
        "report_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    return dict(zip(columns, rows[0]))


def get_macro_indicators() -> list[str]:

    sql = """
        SELECT DISTINCT IndicatorName
        FROM macro_daily_ledger
        WHERE IndicatorName IS NOT NULL
        ORDER BY IndicatorName
    """

    rows = execute_rows(sql)

    return [row[0] for row in rows]


def get_macro_date_range(
    indicator: str,
) -> dict[str, Any] | None:

    sql = """
        SELECT
            MIN(ReportDate),
            MAX(ReportDate),
            COUNT(*),
            COUNT(Close_Value)
        FROM macro_daily_ledger
        WHERE IndicatorName = ?
    """

    rows = execute_rows(sql, [indicator])

    if not rows or rows[0][0] is None:
        return None

    return {
        "first_date": rows[0][0],
        "last_date": rows[0][1],
        "row_count": rows[0][2],
        "rows_with_close": rows[0][3],
    }


# ============================================================
# COMPANY / GLOBAL-ASSET DATA
# Source: global_assets_daily
#
# This is where your company daily price history lives.
# ============================================================

def get_global_asset_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            AssetClass AS asset_class,
            Open AS open,
            High AS high,
            Low AS low,
            Close AS close,
            Volume AS volume
        FROM global_assets_daily
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    sql += " ORDER BY ReportDate"

    return execute_arrow(sql, params)


def get_latest_global_asset(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any] | None:

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            AssetClass AS asset_class,
            Open AS open,
            High AS high,
            Low AS low,
            Close AS close,
            Volume AS volume
        FROM global_assets_daily
        WHERE Ticker = ?
          AND Close IS NOT NULL
    """

    params: list[Any] = [ticker]

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    sql += """
        ORDER BY ReportDate DESC
        LIMIT 1
    """

    rows = execute_rows(sql, params)

    if not rows:
        return None

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


def get_global_assets(
    asset_class: str | None = None,
) -> list[dict[str, Any]]:

    sql = """
        SELECT
            Ticker AS ticker,
            AssetClass AS asset_class,
            MIN(ReportDate) AS first_date,
            MAX(ReportDate) AS last_date,
            COUNT(*) AS row_count
        FROM global_assets_daily
    """

    params: list[Any] = []

    if asset_class is not None:
        sql += " WHERE AssetClass = ?"
        params.append(asset_class)

    sql += """
        GROUP BY Ticker, AssetClass
        ORDER BY Ticker
    """

    rows = execute_rows(sql, params)

    columns = [
        "ticker",
        "asset_class",
        "first_date",
        "last_date",
        "row_count",
    ]

    return [dict(zip(columns, row)) for row in rows]


def get_global_asset_date_range(
    ticker: str,
) -> dict[str, Any] | None:

    sql = """
        SELECT
            MIN(ReportDate),
            MAX(ReportDate),
            COUNT(*),
            COUNT(Close)
        FROM global_assets_daily
        WHERE Ticker = ?
    """

    rows = execute_rows(sql, [ticker])

    if not rows or rows[0][0] is None:
        return None

    return {
        "first_date": rows[0][0],
        "last_date": rows[0][1],
        "row_count": rows[0][2],
        "rows_with_close": rows[0][3],
    }