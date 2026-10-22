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
# OPTIONS
# ============================================================

def get_options_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return aggregated options data for a ticker.
    """

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            ExpiryDate AS expiry_date,
            Total_Put_OI AS total_put_oi,
            Total_Call_OI AS total_call_oi,
            OI_PCR AS oi_pcr,
            Total_Put_Volume AS total_put_volume,
            Total_Call_Volume AS total_call_volume,
            Volume_PCR AS volume_pcr
        FROM mv_options_aggregates
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    sql += """
        ORDER BY ReportDate, ExpiryDate
    """

    return execute_arrow(sql, params)


def get_latest_options(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return the latest available options observations
    for a ticker on/before as_of_date.
    """

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            ExpiryDate AS expiry_date,
            Total_Put_OI AS total_put_oi,
            Total_Call_OI AS total_call_oi,
            OI_PCR AS oi_pcr,
            Total_Put_Volume AS total_put_volume,
            Total_Call_Volume AS total_call_volume,
            Volume_PCR AS volume_pcr
        FROM mv_options_aggregates
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    sql += """
        ORDER BY ReportDate DESC, ExpiryDate
    """

    return execute_arrow(sql, params)


# ============================================================
# FUTURES BASIS
# ============================================================

def get_futures_basis_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return spot/futures basis history.
    """

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            ExpiryDate AS expiry_date,
            Spot_Price AS spot_price,
            Futures_Price AS futures_price,
            Open_Interest AS open_interest,
            Absolute_Basis AS absolute_basis,
            Basis_Percentage AS basis_percentage
        FROM mv_spot_futures_basis
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    sql += """
        ORDER BY ReportDate, ExpiryDate
    """

    return execute_arrow(sql, params)


def get_latest_futures_basis(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any] | None:
    """
    Return the highest-OI futures contract available for the
    latest date on/before as_of_date.
    """

    sql = """
        SELECT
            Ticker AS ticker,
            ReportDate AS report_date,
            ExpiryDate AS expiry_date,
            Spot_Price AS spot_price,
            Futures_Price AS futures_price,
            Open_Interest AS open_interest,
            Absolute_Basis AS absolute_basis,
            Basis_Percentage AS basis_percentage
        FROM mv_spot_futures_basis
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    sql += """
        QUALIFY
            ROW_NUMBER() OVER (
                ORDER BY ReportDate DESC, Open_Interest DESC
            ) = 1
    """

    rows = execute_rows(sql, params)

    if not rows:
        return None

    columns = [
        "ticker",
        "report_date",
        "expiry_date",
        "spot_price",
        "futures_price",
        "open_interest",
        "absolute_basis",
        "basis_percentage",
    ]

    return dict(zip(columns, rows[0]))


# ============================================================
# UNIFIED DAILY DERIVATIVES / FLOW MATRIX
# ============================================================

def get_unified_market_matrix(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return the ETL-produced company-level analytical matrix.

    This combines:
        cash market
        options PCR
        futures basis
        block/bulk trade events
    """

    sql = """
        SELECT *
        FROM mv_unified_market_matrix
        WHERE ticker = ?
    """

    params: list[Any] = [ticker]

    if start_date is not None:
        sql += " AND date >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND date <= ?"
        params.append(_date_value(end_date))

    sql += " ORDER BY date"

    return execute_arrow(sql, params)


def get_latest_unified_market_matrix(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any] | None:
    """
    Return latest matrix observation for a ticker on/before as_of_date.
    """

    sql = """
        SELECT *
        FROM mv_unified_market_matrix
        WHERE ticker = ?
    """

    params: list[Any] = [ticker]

    if as_of_date is not None:
        sql += " AND date <= ?"
        params.append(_date_value(as_of_date))

    sql += """
        ORDER BY date DESC
        LIMIT 1
    """

    rows = execute_rows(sql, params)

    if not rows:
        return None

    columns = [
        "ticker",
        "date",
        "close",
        "volume",
        "delivery_percentage",
        "daily_hl_spread",
        "daily_vwap_dev",
        "oi_pcr",
        "delta_oi_pcr",
        "futures_basis",
        "is_fo_eligible",
        "short_volume",
        "short_percentage",
        "net_block_volume",
        "has_block_deal",
        "avg_block_premium",
    ]

    return dict(zip(columns, rows[0]))