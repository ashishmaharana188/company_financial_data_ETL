from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pyarrow as pa

from .query import execute_arrow, execute_rows


# ============================================================
# HELPERS
# ============================================================

def _date_value(
    value: str | date | datetime,
) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    return value


def _history(
    table: str,
    columns: list[str],
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:
    """
    Read financial history from one OLAP table.

    ReportDate is the accounting period-end date.
    It is NOT treated as publication/availability date.
    """

    select_columns = ",\n            ".join(
        f'"{column}"'
        for column in columns
    )

    sql = f"""
        SELECT
            {select_columns}
        FROM "{table}"
        WHERE "Ticker" = ?
    """

    params: list[Any] = [ticker]

    if start_date is not None:
        sql += ' AND "ReportDate" >= ?'
        params.append(_date_value(start_date))

    if end_date is not None:
        sql += ' AND "ReportDate" <= ?'
        params.append(_date_value(end_date))

    sql += ' ORDER BY "ReportDate"'

    return execute_arrow(sql, params)


def _latest_periods(
    table: str,
    ticker: str,
    periods: int,
) -> pa.Table:
    """
    Return the latest accounting periods.

    This uses accounting ReportDate ordering only.
    """

    if periods < 1:
        raise ValueError("periods must be >= 1")

    sql = f"""
        SELECT *
        FROM "{table}"
        WHERE "Ticker" = ?
        ORDER BY "ReportDate" DESC
        LIMIT ?
    """

    return execute_arrow(
        sql,
        [ticker, periods],
    )


# ============================================================
# QUARTERLY INCOME STATEMENT
# ============================================================

QUARTERLY_INCOME_COLUMNS = [
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
]


def get_quarterly_income_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "quarterly_income_statement",
        QUARTERLY_INCOME_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_quarterly_income(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "quarterly_income_statement",
        ticker,
        periods,
    )


# ============================================================
# YEARLY INCOME STATEMENT
# ============================================================

YEARLY_INCOME_COLUMNS = [
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
]


def get_yearly_income_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "yearly_income_statement",
        YEARLY_INCOME_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_yearly_income(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "yearly_income_statement",
        ticker,
        periods,
    )


# ============================================================
# QUARTERLY BALANCE SHEET
# ============================================================

QUARTERLY_BALANCE_COLUMNS = [
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
]


def get_quarterly_balance_sheet_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "quarterly_balance_sheet",
        QUARTERLY_BALANCE_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_quarterly_balance_sheet(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "quarterly_balance_sheet",
        ticker,
        periods,
    )


# ============================================================
# YEARLY BALANCE SHEET
# ============================================================

YEARLY_BALANCE_COLUMNS = [
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
]


def get_yearly_balance_sheet_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "yearly_balance_sheet",
        YEARLY_BALANCE_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_yearly_balance_sheet(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "yearly_balance_sheet",
        ticker,
        periods,
    )


# ============================================================
# QUARTERLY CASH FLOW
# OPTIONAL
# ============================================================

QUARTERLY_CASH_FLOW_COLUMNS = [
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
]


def get_quarterly_cash_flow_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "quarterly_cash_flow",
        QUARTERLY_CASH_FLOW_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_quarterly_cash_flow(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "quarterly_cash_flow",
        ticker,
        periods,
    )


# ============================================================
# YEARLY CASH FLOW
# MANDATORY BASELINE
# ============================================================

YEARLY_CASH_FLOW_COLUMNS = [
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
]


def get_yearly_cash_flow_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "yearly_cash_flow",
        YEARLY_CASH_FLOW_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_yearly_cash_flow(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "yearly_cash_flow",
        ticker,
        periods,
    )


# ============================================================
# YEARLY INDIRECT CASH FLOW
# MANDATORY BASELINE
# ============================================================

YEARLY_INDIRECT_CASH_FLOW_COLUMNS = [
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
]


def get_yearly_indirect_cash_flow_history(
    ticker: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> pa.Table:

    return _history(
        "yearly_indirect_cash_flow",
        YEARLY_INDIRECT_CASH_FLOW_COLUMNS,
        ticker,
        start_date,
        end_date,
    )


def get_latest_yearly_indirect_cash_flow(
    ticker: str,
    periods: int = 1,
) -> pa.Table:

    return _latest_periods(
        "yearly_indirect_cash_flow",
        ticker,
        periods,
    )


# ============================================================
# FINANCIAL COVERAGE
# ============================================================

FINANCIAL_TABLES = {
    "quarterly_income_statement":
        "quarterly_income_statement",

    "yearly_income_statement":
        "yearly_income_statement",

    "quarterly_balance_sheet":
        "quarterly_balance_sheet",

    "yearly_balance_sheet":
        "yearly_balance_sheet",

    "quarterly_cash_flow":
        "quarterly_cash_flow",

    "yearly_cash_flow":
        "yearly_cash_flow",

    "yearly_indirect_cash_flow":
        "yearly_indirect_cash_flow",
}


def get_financial_coverage(
    ticker: str,
) -> dict[str, dict[str, Any]]:

    result: dict[str, dict[str, Any]] = {}

    for label, table in FINANCIAL_TABLES.items():

        sql = f"""
            SELECT
                MIN("ReportDate"),
                MAX("ReportDate"),
                COUNT(*)
            FROM "{table}"
            WHERE "Ticker" = ?
        """

        rows = execute_rows(
            sql,
            [ticker],
        )

        if not rows or rows[0][0] is None:
            result[label] = {
                "first_report_date": None,
                "last_report_date": None,
                "row_count": 0,
            }
            continue

        first_date, last_date, row_count = rows[0]

        result[label] = {
            "first_report_date": first_date,
            "last_report_date": last_date,
            "row_count": row_count,
        }

    return result