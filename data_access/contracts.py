from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pyarrow as pa


# ============================================================
# REPOSITORY RESULT CONTRACTS
#
# These are the columns actually returned by TrendAnalysis
# data-access functions.
# ============================================================

MARKET_RESULT_COLUMNS = {
    "ticker",
    "report_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "no_of_trades",
    "delivery_qty",
    "delivery_percentage",
}

MACRO_RESULT_COLUMNS = {
    "indicator",
    "report_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

MACRO_INTRADAY_RESULT_COLUMNS = {
    "indicator",
    "report_date",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

QUARTERLY_INCOME_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "TotalRevenue",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingExpense",
    "OperatingIncome",
    "NetInterestIncome",
    "TaxProvision",
    "NetIncome",
}

YEARLY_INCOME_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "IsValid",
    "TotalRevenue",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingExpense",
    "OperatingIncome",
    "NetInterestIncome",
    "TaxProvision",
    "NetIncome",
}

QUARTERLY_BALANCE_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "CashCashEquivalentsAndShortTermInvestments",
    "Receivables",
    "Inventory",
    "CurrentAssets",
    "TotalNonCurrentAssets",
    "GrossPPE",
    "AccumulatedDepreciation",
    "NetPPE",
    "TotalAssets",
    "PayablesAndAccruedExpenses",
    "CurrentDebtAndCapitalLeaseObligation",
    "TotalTaxPayable",
    "CurrentLiabilities",
    "LongTermDebtAndCapitalLeaseObligation",
    "TotalLiabilitiesNetMinorityInterest",
    "CapitalStock",
    "RetainedEarnings",
    "StockholdersEquity",
}

YEARLY_BALANCE_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "IsValid",
    "CashCashEquivalentsAndShortTermInvestments",
    "Receivables",
    "Inventory",
    "CurrentAssets",
    "TotalNonCurrentAssets",
    "GrossPPE",
    "AccumulatedDepreciation",
    "NetPPE",
    "TotalAssets",
    "PayablesAndAccruedExpenses",
    "CurrentDebtAndCapitalLeaseObligation",
    "TotalTaxPayable",
    "CurrentLiabilities",
    "LongTermDebtAndCapitalLeaseObligation",
    "TotalLiabilitiesNetMinorityInterest",
    "CapitalStock",
    "RetainedEarnings",
    "StockholdersEquity",
}

QUARTERLY_CASH_FLOW_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "BeginningCashBalance",
    "CashReceipts",
    "CashDisbursements",
    "CashFromOperations",
    "FixedAssetPurchases",
    "NetBorrowing",
    "IncomeTaxPaid",
    "SaleOfStock",
    "EndingCashBalance",
}

YEARLY_CASH_FLOW_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "IsValid",
    "BeginningCashBalance",
    "CashReceipts",
    "CashDisbursements",
    "CashFromOperations",
    "FixedAssetPurchases",
    "NetBorrowing",
    "IncomeTaxPaid",
    "SaleOfStock",
    "EndingCashBalance",
}

YEARLY_INDIRECT_CASH_FLOW_RESULT_COLUMNS = {
    "DataSource",
    "Ticker",
    "ReportDate",
    "Currency",
    "IsValid",
    "IsSectionValid",
    "IsRollforwardValid",
    "TreasuryOpacityRatio",
    "NetIncome",
    "DepreciationAndAmortization",
    "OtherNonCashAdjustments",
    "ChangeInAccountsReceivable",
    "ChangeInInventory",
    "ChangeInAccountsPayable",
    "OtherWorkingCapitalChanges",
    "IncomeTaxPaid",
    "TotalOperatingCashFlow",
    "Unmapped_Operating",
    "CapExPurchaseOfPPE",
    "PurchaseSaleOfInvestments",
    "OtherInvestingActivities",
    "TotalInvestingCashFlow",
    "Unmapped_Investing",
    "NetDebtIssuedRepaid",
    "NetStockIssuedRepurchased",
    "DividendsPaid",
    "OtherFinancingActivities",
    "TotalFinancingCashFlow",
    "Unmapped_Financing",
    "EffectOfExchangeRates",
    "NetChangeInCash",
    "BeginningCash",
    "EndingCash",
    "Unmapped_Rollforward",
}

INSTITUTIONAL_RESULT_COLUMNS = {
    "report_date",
    "client_type",
    "cash_buy_value",
    "cash_sell_value",
    "cash_net_value",
    "nifty_close",
    "future_index_long",
    "future_index_short",
    "future_stock_long",
    "future_stock_short",
    "option_index_call_long",
    "option_index_put_long",
    "option_index_call_short",
    "option_index_put_short",
    "option_stock_call_long",
    "option_stock_put_long",
    "option_stock_call_short",
    "option_stock_put_short",
    "total_long_contracts",
    "total_short_contracts",
}

INSTITUTIONAL_FLOW_RESULT_COLUMNS = {
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
}

OPTIONS_RESULT_COLUMNS = {
    "ticker",
    "report_date",
    "expiry_date",
    "total_put_oi",
    "total_call_oi",
    "oi_pcr",
    "total_put_volume",
    "total_call_volume",
    "volume_pcr",
}

BASIS_RESULT_COLUMNS = {
    "ticker",
    "report_date",
    "expiry_date",
    "spot_price",
    "futures_price",
    "open_interest",
    "absolute_basis",
    "basis_percentage",
}

UNIFIED_MATRIX_RESULT_COLUMNS = {
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
}

TRADE_EVENTS_RESULT_COLUMNS = {
    "event_id",
    "report_date",
    "ticker",
    "event_type",
    "security_name",
    "client_name",
    "transaction_type",
    "quantity",
    "trade_price",
    "remarks",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise TypeError(
            f"Ticker must be a string, got {type(ticker).__name__}"
        )

    value = ticker.strip().upper()

    if not value:
        raise ValueError("Ticker cannot be empty")

    return value


def normalize_indicator(indicator: str) -> str:
    if not isinstance(indicator, str):
        raise TypeError(
            f"Indicator must be a string, got {type(indicator).__name__}"
        )

    value = indicator.strip()

    if not value:
        raise ValueError("Indicator cannot be empty")

    return value


def normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError("Date cannot be empty")

        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO date: {value!r}. Expected YYYY-MM-DD."
            ) from exc

    raise TypeError(
        f"Unsupported date type: {type(value).__name__}"
    )


def normalize_optional_date(
    value: str | date | datetime | None,
) -> date | None:
    if value is None:
        return None

    return normalize_date(value)


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(
    table: pa.Table,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    actual = set(table.column_names)
    required = set(required_columns)

    missing = sorted(required - actual)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def validate_not_empty(
    table: pa.Table,
    dataset_name: str,
) -> None:
    if table.num_rows == 0:
        raise ValueError(
            f"{dataset_name} returned zero rows"
        )


# ============================================================
# REPOSITORY RESULT VALIDATORS
# ============================================================

def validate_market_result(table: pa.Table) -> None:
    validate_columns(
        table,
        MARKET_RESULT_COLUMNS,
        "market repository result",
    )


def validate_macro_result(table: pa.Table) -> None:
    validate_columns(
        table,
        MACRO_RESULT_COLUMNS,
        "macro repository result",
    )


def validate_macro_intraday_result(table: pa.Table) -> None:
    validate_columns(
        table,
        MACRO_INTRADAY_RESULT_COLUMNS,
        "macro intraday repository result",
    )


def validate_quarterly_income_result(table: pa.Table) -> None:
    validate_columns(
        table,
        QUARTERLY_INCOME_RESULT_COLUMNS,
        "quarterly income result",
    )


def validate_yearly_income_result(table: pa.Table) -> None:
    validate_columns(
        table,
        YEARLY_INCOME_RESULT_COLUMNS,
        "yearly income result",
    )


def validate_quarterly_balance_result(table: pa.Table) -> None:
    validate_columns(
        table,
        QUARTERLY_BALANCE_RESULT_COLUMNS,
        "quarterly balance-sheet result",
    )


def validate_yearly_balance_result(table: pa.Table) -> None:
    validate_columns(
        table,
        YEARLY_BALANCE_RESULT_COLUMNS,
        "yearly balance-sheet result",
    )


def validate_quarterly_cash_flow_result(table: pa.Table) -> None:
    validate_columns(
        table,
        QUARTERLY_CASH_FLOW_RESULT_COLUMNS,
        "quarterly cash-flow result",
    )


def validate_yearly_cash_flow_result(table: pa.Table) -> None:
    validate_columns(
        table,
        YEARLY_CASH_FLOW_RESULT_COLUMNS,
        "yearly cash-flow result",
    )


def validate_indirect_cash_flow_result(table: pa.Table) -> None:
    validate_columns(
        table,
        YEARLY_INDIRECT_CASH_FLOW_RESULT_COLUMNS,
        "yearly indirect cash-flow result",
    )


def validate_institutional_result(table: pa.Table) -> None:
    validate_columns(
        table,
        INSTITUTIONAL_RESULT_COLUMNS,
        "institutional result",
    )


def validate_institutional_flow_result(table: pa.Table) -> None:
    validate_columns(
        table,
        INSTITUTIONAL_FLOW_RESULT_COLUMNS,
        "institutional-flow result",
    )


def validate_options_result(table: pa.Table) -> None:
    validate_columns(
        table,
        OPTIONS_RESULT_COLUMNS,
        "options result",
    )


def validate_basis_result(table: pa.Table) -> None:
    validate_columns(
        table,
        BASIS_RESULT_COLUMNS,
        "futures-basis result",
    )


def validate_unified_matrix_result(table: pa.Table) -> None:
    validate_columns(
        table,
        UNIFIED_MATRIX_RESULT_COLUMNS,
        "unified-market-matrix result",
    )


def validate_trade_events_result(table: pa.Table) -> None:
    validate_columns(
        table,
        TRADE_EVENTS_RESULT_COLUMNS,
        "trade-events result",
    )