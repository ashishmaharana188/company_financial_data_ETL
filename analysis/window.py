from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
from dateutil.relativedelta import relativedelta

from .config import AnalysisConfig

from data_access.metadata import get_company

from data_access.market import (
    get_daily_history,
    get_market_date_range,
)

from data_access.macro_global import (
    get_macro_history,
    get_macro_indicators,
    get_macro_date_range,
)

from data_access.financials import (
    get_quarterly_income_history,
    get_yearly_income_history,
    get_quarterly_balance_sheet_history,
    get_yearly_balance_sheet_history,
    get_quarterly_cash_flow_history,
    get_yearly_cash_flow_history,
    get_yearly_indirect_cash_flow_history,
)

from data_access.ticker import normalize_ticker
# ============================================================
# FINANCIAL WINDOW RULES
# ============================================================

def _quarter_count_for_timeframe(
    timeframe: str,
) -> int:

    mapping = {
        "1W": 1,
        "2W": 1,
        "1M": 1,
        "3M": 1,
        "6M": 2,
        "9M": 3,
        "1Y": 4,
        "18M": 6,
        "2Y": 8,
        "3Y": 12,
        "5Y": 20,
    }

    if timeframe not in mapping:
        raise ValueError(
            f"Unsupported financial timeframe: {timeframe}"
        )

    return mapping[timeframe]


def _quarterly_financial_start(
    config: AnalysisConfig,
):
    """
    Start date for timeframe-driven quarterly financial data.

    A small amount of extra history is retrieved so later
    comparisons have adjacent periods available.
    """

    quarter_count = _quarter_count_for_timeframe(
        config.analysis_timeframe
    )

    months = max(
        quarter_count * 3,
        3,
    )

    return config.analysis_end_date - relativedelta(
        months=months,
    )


# ============================================================
# BENCHMARK
# ============================================================

def _resolve_benchmark(
    benchmark: str,
    start_date,
    end_date,
) -> dict[str, Any]:

    macro_range = get_macro_date_range(
        benchmark
    )

    if macro_range is not None:
        history = get_macro_history(
            benchmark,
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "identifier": benchmark,
            "source": "macro_daily_ledger",
            "history": history,
        }

    market_range = get_market_date_range(
        benchmark
    )

    if market_range is not None:
        history = get_daily_history(
            benchmark,
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "identifier": benchmark,
            "source": "global_assets_daily",
            "history": history,
        }

    raise ValueError(
        f"Benchmark '{benchmark}' was not found in "
        "macro_daily_ledger or global_assets_daily."
    )


# ============================================================
# OBSERVATION WINDOW
# ============================================================

@dataclass
class ObservationWindow:

    config: AnalysisConfig

    company_metadata: dict[str, Any]

    company_market: pa.Table
    company_market_history: pa.Table

    benchmark: dict[str, Any]

    macro: dict[str, pa.Table]

    # Quarterly = timeframe-driven
    quarterly_income: pa.Table
    quarterly_balance_sheet: pa.Table
    quarterly_cash_flow: pa.Table

    # Yearly = mandatory full-history baseline
    yearly_income: pa.Table
    yearly_balance_sheet: pa.Table
    yearly_cash_flow: pa.Table
    yearly_indirect_cash_flow: pa.Table

    limitations: list[str] = field(
        default_factory=list
    )

    @property
    def ticker(self) -> str:
        return self.company_metadata["Ticker"]

    @property
    def sector(self) -> str | None:
        return self.company_metadata.get("Sector")

    @property
    def industry(self) -> str | None:
        return self.company_metadata.get("Industry")

    def summary(self) -> dict[str, Any]:

        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "industry": self.industry,

            "analysis_date":
                self.config.analysis_date,

            "data_cutoff_date":
                self.config.data_cutoff_date,

            "analysis_start_date":
                self.config.analysis_start_date,

            "analysis_end_date":
                self.config.analysis_end_date,

            "analysis_timeframe":
                self.config.analysis_timeframe,

            "holding_period_months":
                self.config.holding_period_months,

            "benchmark":
                self.config.benchmark,

            "benchmark_source":
                self.benchmark["source"],

            "company_market_rows":
                self.company_market.num_rows,

            "macro_series_count":
                len(self.macro),

            "quarterly_income_rows":
                self.quarterly_income.num_rows,

            "quarterly_balance_rows":
                self.quarterly_balance_sheet.num_rows,

            "quarterly_cash_flow_rows":
                self.quarterly_cash_flow.num_rows,

            "yearly_income_rows":
                self.yearly_income.num_rows,

            "yearly_balance_rows":
                self.yearly_balance_sheet.num_rows,

            "yearly_cash_flow_rows":
                self.yearly_cash_flow.num_rows,

            "yearly_indirect_cash_flow_rows":
                self.yearly_indirect_cash_flow.num_rows,

            "limitations":
                self.limitations,
        }


# ============================================================
# BUILDER
# ============================================================

def build_observation_window(
    ticker: str,
    config: AnalysisConfig,
) -> ObservationWindow:
   
    limitations: list[str] = []

    # ========================================================
    # 1. COMPANY METADATA
    # ========================================================

    company = get_company(ticker)

    if company is None:
        raise ValueError(
            f"Ticker '{ticker}' was not found in market_metadata."
        )

    if not company.get("IsActive", False):
        raise ValueError(
            f"Ticker '{ticker}' exists but is not active."
        )

    if not company.get("Sector"):
        limitations.append(
            "Company sector is missing."
        )

    if not company.get("Industry"):
        limitations.append(
            "Company industry is missing."
        )

    # ========================================================
    # 2. COMPANY MARKET
    # ========================================================

# ========================================================
# 2. COMPANY MARKET
# ========================================================

    company_market = get_daily_history(
        ticker,
        start_date=config.analysis_start_date,
        end_date=config.analysis_end_date,
    )

    company_market_history = get_daily_history(
        ticker,
        end_date=config.analysis_end_date,
    )

    if company_market.num_rows == 0:
        limitations.append(
            "No company market observations exist "
            "inside the selected analysis window."
        )

    if company_market_history.num_rows == 0:
        limitations.append(
            "No historical company market observations "
            "exist through the analysis cutoff."
        )

    # ========================================================
    # 3. BENCHMARK
    # ========================================================

    benchmark = _resolve_benchmark(
        config.benchmark,
        config.analysis_start_date,
        config.analysis_end_date,
    )

    if benchmark["history"].num_rows == 0:
        limitations.append(
            "Benchmark has no observations inside the "
            "selected analysis window."
        )

    # ========================================================
    # 4. MACRO
    # ========================================================

    macro_data: dict[str, pa.Table] = {}

    for indicator in get_macro_indicators():

        history = get_macro_history(
            indicator,
            start_date=config.analysis_start_date,
            end_date=config.analysis_end_date,
        )

        macro_data[indicator] = history

    # Do not flag every macro series as LIMITED merely because
    # an individual indicator lacks observations in the window.
    # Applicability/relevance is handled later.

    # ========================================================
    # 5. QUARTERLY FINANCIALS
    #
    # Timeframe-driven
    # ========================================================

    quarterly_start = _quarterly_financial_start(
        config
    )

    quarterly_income = get_quarterly_income_history(
        ticker,
        start_date=quarterly_start,
        end_date=config.analysis_end_date,
    )

    quarterly_balance_sheet = (
        get_quarterly_balance_sheet_history(
            ticker,
            start_date=quarterly_start,
            end_date=config.analysis_end_date,
        )
    )

    quarterly_cash_flow = (
        get_quarterly_cash_flow_history(
            ticker,
            start_date=quarterly_start,
            end_date=config.analysis_end_date,
        )
    )

    # ========================================================
    # 6. YEARLY FINANCIALS
    #
    # Mandatory baseline.
    #
    # IMPORTANT:
    # Do NOT restrict yearly data to analysis_start_date.
    # We need all available yearly history up to cutoff.
    # ========================================================

    yearly_income = get_yearly_income_history(
        ticker,
        start_date=None,
        end_date=config.analysis_end_date,
    )

    yearly_balance_sheet = (
        get_yearly_balance_sheet_history(
            ticker,
            start_date=None,
            end_date=config.analysis_end_date,
        )
    )

    yearly_cash_flow = (
        get_yearly_cash_flow_history(
            ticker,
            start_date=None,
            end_date=config.analysis_end_date,
        )
    )

    yearly_indirect_cash_flow = (
        get_yearly_indirect_cash_flow_history(
            ticker,
            start_date=None,
            end_date=config.analysis_end_date,
        )
    )

    # ========================================================
    # 7. MANDATORY YEARLY DATA VALIDATION
    # ========================================================

    if yearly_income.num_rows == 0:
        limitations.append(
            "Mandatory yearly income-statement data "
            "is unavailable."
        )

    if yearly_balance_sheet.num_rows == 0:
        limitations.append(
            "Mandatory yearly balance-sheet data "
            "is unavailable."
        )

    if yearly_cash_flow.num_rows == 0:
        limitations.append(
            "Mandatory yearly cash-flow data "
            "is unavailable."
        )

    if yearly_indirect_cash_flow.num_rows == 0:
        limitations.append(
            "Mandatory yearly indirect cash-flow data "
            "is unavailable."
        )

    # ========================================================
    # 8. QUARTERLY OPTIONALITY
    # ========================================================

    if quarterly_income.num_rows == 0:
        limitations.append(
            "Quarterly income-statement data is unavailable "
            "for the selected analysis timeframe."
        )

    if quarterly_balance_sheet.num_rows == 0:
        limitations.append(
            "Quarterly balance-sheet data is unavailable "
            "for the selected analysis timeframe."
        )

    # Quarterly cash flow is explicitly optional.
    # Do NOT create a limitation merely because it is absent.

    # ========================================================
    # 9. TEMPORAL FINANCIAL LIMITATION
    # ========================================================

    limitations.append(
        "Financial ReportDate represents accounting "
        "period end; verified public information "
        "availability is not stored in the OLAP schema."
    )

    return ObservationWindow(
        config=config,
        company_metadata=company,
        company_market=company_market,
        company_market_history=company_market_history,

        benchmark=benchmark,
        macro=macro_data,

        quarterly_income=quarterly_income,
        quarterly_balance_sheet=quarterly_balance_sheet,
        quarterly_cash_flow=quarterly_cash_flow,

        yearly_income=yearly_income,
        yearly_balance_sheet=yearly_balance_sheet,
        yearly_cash_flow=yearly_cash_flow,
        yearly_indirect_cash_flow=yearly_indirect_cash_flow,

        limitations=limitations,
    )