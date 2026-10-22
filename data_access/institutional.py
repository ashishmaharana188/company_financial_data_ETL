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


def get_institutional_history(
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    client_type: str | None = None,
) -> pa.Table:
    """
    Return raw institutional activity.

    Source:
        institutional_ledger
    """

    sql = """
        SELECT
            ReportDate AS report_date,
            ClientType AS client_type,
            Cash_Buy_Value AS cash_buy_value,
            Cash_Sell_Value AS cash_sell_value,
            Cash_Net_Value AS cash_net_value,
            Nifty_Close AS nifty_close,
            Future_Index_Long AS future_index_long,
            Future_Index_Short AS future_index_short,
            Future_Stock_Long AS future_stock_long,
            Future_Stock_Short AS future_stock_short,
            Option_Index_Call_Long AS option_index_call_long,
            Option_Index_Put_Long AS option_index_put_long,
            Option_Index_Call_Short AS option_index_call_short,
            Option_Index_Put_Short AS option_index_put_short,
            Option_Stock_Call_Long AS option_stock_call_long,
            Option_Stock_Put_Long AS option_stock_put_long,
            Option_Stock_Call_Short AS option_stock_call_short,
            Option_Stock_Put_Short AS option_stock_put_short,
            Total_Long_Contracts AS total_long_contracts,
            Total_Short_Contracts AS total_short_contracts
        FROM institutional_ledger
        WHERE 1 = 1
    """

    params: list[Any] = []

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    if client_type is not None:
        sql += " AND ClientType = ?"
        params.append(client_type)

    sql += " ORDER BY ReportDate, ClientType"

    return execute_arrow(sql, params)


def get_institutional_flow_history(
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    client_type: str | None = None,
) -> pa.Table:
    """
    Return derived institutional positioning.

    Source:
        mv_institutional_flow
    """

    sql = """
        SELECT
            ReportDate AS report_date,
            ClientType AS client_type,

            Future_Index_Net AS future_index_net,
            Future_Index_Net_Change AS future_index_net_change,

            Option_Index_Call_Net AS option_index_call_net,
            Option_Index_Call_Net_Change
                AS option_index_call_net_change,

            Option_Index_Put_Net AS option_index_put_net,
            Option_Index_Put_Net_Change
                AS option_index_put_net_change,

            Future_Stock_Net AS future_stock_net,
            Future_Stock_Net_Change AS future_stock_net_change,

            Option_Stock_Call_Net AS option_stock_call_net,
            Option_Stock_Call_Net_Change
                AS option_stock_call_net_change,

            Option_Stock_Put_Net AS option_stock_put_net,
            Option_Stock_Put_Net_Change
                AS option_stock_put_net_change

        FROM mv_institutional_flow
        WHERE 1 = 1
    """

    params: list[Any] = []

    if start_date is not None:
        sql += " AND ReportDate >= ?"
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(end_date))

    if client_type is not None:
        sql += " AND ClientType = ?"
        params.append(client_type)

    sql += " ORDER BY ReportDate, ClientType"

    return execute_arrow(sql, params)


def get_latest_institutional_flow(
    as_of_date: str | date | datetime | None = None,
    client_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return latest institutional-flow observation available
    on or before as_of_date.
    """

    sql = """
        SELECT
            ReportDate AS report_date,
            ClientType AS client_type,

            Future_Index_Net AS future_index_net,
            Future_Index_Net_Change AS future_index_net_change,

            Option_Index_Call_Net AS option_index_call_net,
            Option_Index_Call_Net_Change
                AS option_index_call_net_change,

            Option_Index_Put_Net AS option_index_put_net,
            Option_Index_Put_Net_Change
                AS option_index_put_net_change,

            Future_Stock_Net AS future_stock_net,
            Future_Stock_Net_Change AS future_stock_net_change,

            Option_Stock_Call_Net AS option_stock_call_net,
            Option_Stock_Call_Net_Change
                AS option_stock_call_net_change,

            Option_Stock_Put_Net AS option_stock_put_net,
            Option_Stock_Put_Net_Change
                AS option_stock_put_net_change

        FROM mv_institutional_flow
        WHERE 1 = 1
    """

    params: list[Any] = []

    if as_of_date is not None:
        sql += " AND ReportDate <= ?"
        params.append(_date_value(as_of_date))

    if client_type is not None:
        sql += " AND ClientType = ?"
        params.append(client_type)

    sql += """
        QUALIFY
            ROW_NUMBER() OVER (
                PARTITION BY ClientType
                ORDER BY ReportDate DESC
            ) = 1
        ORDER BY ClientType
    """

    rows = execute_rows(sql, params)

    columns = [
        "report_date",
        "client_type",
        "future_index_net",
        "future_index_net_change",
        "option_index_call_net",
        "option_index_call_net_change",
        "option_index_put_net",
        "option_index_put_net_change",
        "future_stock_net",
        "future_stock_net_change",
        "option_stock_call_net",
        "option_stock_call_net_change",
        "option_stock_put_net",
        "option_stock_put_net_change",
    ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]