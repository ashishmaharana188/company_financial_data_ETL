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


def get_trade_events(
    ticker: str | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return block/bulk trade events.
    """

    sql = """
        SELECT
            EventID AS event_id,
            ReportDate AS report_date,
            Ticker AS ticker,
            EventType AS event_type,
            SecurityName AS security_name,
            ClientName AS client_name,
            TransactionType AS transaction_type,
            Quantity AS quantity,
            TradePrice AS trade_price,
            Remarks AS remarks
        FROM trade_events_ledger
        WHERE 1 = 1
    """

    params: list[Any] = []

    if ticker is not None:
        sql += " AND Ticker = ?"
        params.append(ticker)

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    sql += " ORDER BY ReportDate, EventID"

    return execute_arrow(sql, params)


def get_latest_trade_events(
    ticker: str,
    as_of_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Return the latest trade events available for a company.
    """

    sql = """
        SELECT
            EventID AS event_id,
            ReportDate AS report_date,
            Ticker AS ticker,
            EventType AS event_type,
            SecurityName AS security_name,
            ClientName AS client_name,
            TransactionType AS transaction_type,
            Quantity AS quantity,
            TradePrice AS trade_price,
            Remarks AS remarks
        FROM trade_events_ledger
        WHERE Ticker = ?
    """

    params: list[Any] = [ticker]

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    sql += " ORDER BY ReportDate DESC, EventID DESC"

    return execute_arrow(sql, params)