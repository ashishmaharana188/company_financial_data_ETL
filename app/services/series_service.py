from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from data_access.derivatives import get_futures_basis_history, get_unified_market_matrix
from data_access.institutional import get_institutional_flow_history, get_institutional_history
from data_access.macro_global import get_macro_history
from data_access.query import execute_arrow

from .catalog import (
    SOURCE_BASIS, SOURCE_COMMODITY_FUTURES, SOURCE_COMMODITY_OPTIONS, SOURCE_COMMODITY_PRICES,
    SOURCE_FUTURES, SOURCE_INSTITUTIONAL, SOURCE_MACRO, SOURCE_MARKET,
    SOURCE_MATRIX, SOURCE_OPTIONS, RAW_DERIVATIVE_FILTERS, FINANCIAL_METRICS,
    SeriesRequest, get_financial_loader,
)
from .derivative_helpers import (
    add_futures_basis_columns, aggregate_option_contracts,
    select_representative_futures_contract,
)
from .market_series import load_freshest_market_history


def _to_frame(table: Any) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame(table)


def _empty_series() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})


def _col(df: pd.DataFrame, requested: str) -> str | None:
    target = requested.casefold()
    return next((column for column in df.columns if str(column).casefold() == target), None)


def _series_frame(df: pd.DataFrame, date_col: str, value_col: str, label: str) -> pd.DataFrame:
    date_name = _col(df, date_col)
    value_name = _col(df, value_col)
    if df.empty or date_name is None or value_name is None:
        return _empty_series()
    # Preserve candle fields for the terminal's fixed OHLC/date readout.
    # Not every source has OHLC (e.g. PCR, institutional flows, financials),
    # so these fields remain optional and are never manufactured.
    result = pd.DataFrame({
        "date": df[date_name].to_numpy(copy=True),
        "value": df[value_name].to_numpy(copy=True),
    })
    for field in ("open", "high", "low", "close", "volume"):
        source_column = _col(df, field)
        if source_column is not None:
            result[field] = df[source_column].to_numpy(copy=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("value", "open", "high", "low", "close", "volume"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "value"])
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    result.attrs["label"] = label
    return result


def _date_param(value: date | datetime | str | None):
    if isinstance(value, datetime):
        return value.date()
    return value


def _load_raw_derivative_contracts(
    source: str,
    ticker: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Load contracts for the exact selected ticker from unified_market_master.

    The catalogue and history loader share the same instrument predicate. The
    query never substitutes another company when the selected ticker has no data.
    """
    if source not in RAW_DERIVATIVE_FILTERS:
        raise ValueError(f"No raw derivative filter configured for {source!r}")
    predicate = RAW_DERIVATIVE_FILTERS[source]
    sql = f"""
        SELECT
            "Ticker" AS ticker, "ReportDate" AS report_date,
            "InstrumentType" AS instrument_type, "ExpiryDate" AS expiry_date,
            "StrikePrice" AS strike_price, "OptionType" AS option_type,
            "Open" AS open, "High" AS high, "Low" AS low,
            "Close" AS close, "Volume" AS volume,
            "Open_Interest" AS open_interest, "Change_In_OI" AS change_in_oi,
            "Settlement_Price" AS settlement_price, "Underlying_Price" AS underlying_price,
            "Exchange_Series" AS exchange_series
        FROM unified_market_master
        WHERE "Ticker" = ? AND {predicate}
    """
    params: list[Any] = [ticker]
    if start_date is not None:
        sql += ' AND "ReportDate" >= ?'
        params.append(_date_param(start_date))
    if end_date is not None:
        sql += ' AND "ReportDate" <= ?'
        params.append(_date_param(end_date))
    sql += ' ORDER BY "ReportDate", "ExpiryDate"'
    table = execute_arrow(sql, params)
    return table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)


def load_series(
    request: SeriesRequest,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Load one numeric series through existing read-only data_access functions.

    `end_date` is optional by design. The app often loads observations after
    the chosen event-analysis window so realized forward responses are available.
    """
    source = request.source
    identifier = request.identifier
    metric = request.metric

    if source == SOURCE_MARKET:
        frame = _to_frame(load_freshest_market_history(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_MACRO:
        frame = _to_frame(get_macro_history(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_MATRIX:
        frame = _to_frame(get_unified_market_matrix(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "date", metric, request.label)
    elif source in (SOURCE_OPTIONS, SOURCE_COMMODITY_OPTIONS):
        # Raw contract query keeps options for the selected underlying exact.
        contracts = _load_raw_derivative_contracts(source, identifier, start_date, end_date)
        frame = aggregate_option_contracts(contracts)
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source in (SOURCE_FUTURES, SOURCE_COMMODITY_FUTURES, SOURCE_COMMODITY_PRICES):
        contracts = _load_raw_derivative_contracts(source, identifier, start_date, end_date)
        frame = select_representative_futures_contract(contracts)
        frame = add_futures_basis_columns(frame)
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_BASIS:
        frame = _to_frame(get_futures_basis_history(identifier, start_date=start_date, end_date=end_date))
        if not frame.empty and "report_date" in frame.columns:
            # Legacy cash-linked view supports stock/index underlyings only.
            if "open_interest" in frame.columns:
                frame = frame.sort_values(["report_date", "open_interest"], ascending=[True, False], na_position="last")
            frame = frame.drop_duplicates("report_date", keep="first")
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source == SOURCE_INSTITUTIONAL:
        # Raw ledger measures and the pre-derived net-position measures live in
        # different existing access functions. Route each metric to its source.
        raw_ledger_metrics = {
            "cash_buy_value", "cash_sell_value", "cash_net_value", "nifty_close",
            "future_index_long", "future_index_short", "future_stock_long", "future_stock_short",
            "option_index_call_long", "option_index_put_long", "option_index_call_short",
            "option_index_put_short", "option_stock_call_long", "option_stock_put_long",
            "option_stock_call_short", "option_stock_put_short", "total_long_contracts",
            "total_short_contracts",
        }
        if metric in raw_ledger_metrics:
            frame = _to_frame(get_institutional_history(
                start_date=start_date, end_date=end_date, client_type=identifier
            ))
        else:
            frame = _to_frame(get_institutional_flow_history(
                start_date=start_date, end_date=end_date, client_type=identifier
            ))
        result = _series_frame(frame, "report_date", metric, request.label)
    elif source in FINANCIAL_METRICS:
        loader = get_financial_loader(source)
        if loader is None:
            raise ValueError(f"No financial data loader is registered for {source!r}")
        frame = _to_frame(loader(identifier, start_date=start_date, end_date=end_date))
        result = _series_frame(frame, "ReportDate", metric, request.label)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    if result.empty:
        result.attrs["label"] = request.label
        result.attrs["warning"] = "No numeric observations were returned for this series and date range."
    return result


def series_construction_note(request: SeriesRequest) -> str:
    """Explain in plain language how a plotted series is constructed."""
    if request.source == SOURCE_COMMODITY_PRICES:
        return ("MCX commodity market prices are loaded from raw unified_market_master FUTCOM rows. "
                "The chart uses the daily OHLCV fields from the highest-open-interest contract for the selected ticker on each date; "
                "the selected expiry can change over time, so inspect rollover dates before comparing price gaps.")
    if request.source == SOURCE_MARKET:
        return ("Market prices are loaded from the freshest available ticker alias "
                "(canonical ticker or .NS variant). Check the data-coverage panel for the actual last observation.")
    if request.source == SOURCE_MACRO:
        return "This series comes directly from macro_daily_ledger for the selected indicator."
    if request.source == SOURCE_MATRIX:
        return ("This is an ETL-precomputed daily field from mv_unified_market_matrix; "
                "the app plots the stored field rather than recalculating it.")
    if request.source in (SOURCE_OPTIONS, SOURCE_COMMODITY_OPTIONS):
        source_kind = "commodity" if request.source == SOURCE_COMMODITY_OPTIONS else "stock/index"
        if request.metric == "oi_pcr":
            return (f"{source_kind.title()} options for exact underlying ticker {request.identifier}: put OI divided by call OI "
                    "after summing eligible CE/PE contracts across all strikes and expiries per date. This is an underlying-level PCR, "
                    "not a single-strike or single-expiry series.")
        if request.metric == "volume_pcr":
            return (f"{source_kind.title()} options for exact underlying ticker {request.identifier}: put volume divided by call volume "
                    "after summing eligible CE/PE contracts across all strikes and expiries per date.")
        return (f"Raw option contracts are filtered to exact underlying ticker {request.identifier}; quantities are summed "
                "across available strikes and expiries per date.")
    if request.source in (SOURCE_FUTURES, SOURCE_COMMODITY_FUTURES):
        source_kind = "commodity" if request.source == SOURCE_COMMODITY_FUTURES else "stock/index"
        return (f"{source_kind.title()} futures are filtered to exact underlying ticker {request.identifier}. One contract is represented per date: "
                "highest open interest, with earlier expiry as tie-breaker. The selected expiry may change over time. Basis is calculated only "
                "where the raw record supplies an underlying price. Only contracts already present in unified_market_master appear; "
                "the app does not ingest missing commodity data.")
    if request.source == SOURCE_BASIS:
        return ("Legacy stock/index spot-linked basis view. One expiry is represented per date using highest recorded open interest. "
                "This view requires a matching cash-market row and is not the source for commodity futures.")
    if request.source == SOURCE_INSTITUTIONAL:
        raw_metrics = {
            "cash_buy_value", "cash_sell_value", "cash_net_value", "nifty_close",
            "future_index_long", "future_index_short", "future_stock_long", "future_stock_short",
            "option_index_call_long", "option_index_put_long", "option_index_call_short",
            "option_index_put_short", "option_stock_call_long", "option_stock_put_long",
            "option_stock_call_short", "option_stock_put_short", "total_long_contracts",
            "total_short_contracts",
        }
        if request.metric in raw_metrics:
            return ("This plots the selected raw institutional-ledger measure for the chosen client category. "
                    "It is not an all-client total.")
        return ("This plots a derived net-position/change field from mv_institutional_flow for the chosen "
                "client category.")
    if request.source in FINANCIAL_METRICS:
        return ("Financial observations use accounting period-end ReportDate. That date is not assumed to be "
                "the statement's public-release date.")
    return "Values are plotted from the selected source and measure as returned by data_access."
