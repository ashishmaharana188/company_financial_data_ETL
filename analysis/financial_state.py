from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import math
import statistics

import pyarrow as pa


MetricType = Literal[
    "flow",
    "stock",
]


@dataclass(frozen=True)
class FinancialMetricState:
    metric: str

    metric_type: MetricType

    current_value: float | None

    recent_start_value: float | None
    recent_change: float | None
    recent_change_pct: float | None

    previous_comparable_value: float | None
    latest_comparable_value: float | None
    comparable_change: float | None
    comparable_change_pct: float | None

    historical_percentile: float | None

    historical_min: float | None
    historical_max: float | None
    historical_median: float | None

    recent_direction: str
    direction_strength: float | None

    state: str

    recent_periods: int
    historical_periods: int

    source_frequency: str

    limited: bool
    limitation: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_type": self.metric_type,
            "current_value": self.current_value,
            "recent_start_value": self.recent_start_value,
            "recent_change": self.recent_change,
            "recent_change_pct": self.recent_change_pct,
            "previous_comparable_value":
                self.previous_comparable_value,
            "latest_comparable_value":
                self.latest_comparable_value,
            "comparable_change":
                self.comparable_change,
            "comparable_change_pct":
                self.comparable_change_pct,
            "historical_percentile":
                self.historical_percentile,
            "historical_min":
                self.historical_min,
            "historical_max":
                self.historical_max,
            "historical_median":
                self.historical_median,
            "recent_direction":
                self.recent_direction,
            "direction_strength":
                self.direction_strength,
            "state": self.state,
            "recent_periods": self.recent_periods,
            "historical_periods": self.historical_periods,
            "source_frequency": self.source_frequency,
            "limited": self.limited,
            "limitation": self.limitation,
        }


# ============================================================
# CORE HELPERS
# ============================================================

def _extract_series(
    table: pa.Table,
    metric: str,
) -> list[tuple[Any, float]]:
    """
    Extract (ReportDate, numeric value) pairs.
    """

    required = {
        "ReportDate",
        metric,
    }

    missing = required - set(table.column_names)

    if missing:
        raise ValueError(
            f"Missing financial columns for {metric}: "
            f"{sorted(missing)}"
        )

    dates = table["ReportDate"].to_pylist()
    values = table[metric].to_pylist()

    result: list[tuple[Any, float]] = []

    for report_date, value in zip(dates, values):

        if report_date is None or value is None:
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(numeric):
            continue

        result.append(
            (report_date, numeric)
        )

    result.sort(
        key=lambda item: item[0]
    )

    return result


def _safe_percentage_change(
    old: float | None,
    new: float | None,
) -> float | None:
    """
    Return percentage change only when a meaningful
    non-zero denominator exists.

    This intentionally avoids misleading percentages when
    financial values cross zero.
    """

    if old is None or new is None:
        return None

    if old == 0:
        return None

    if old * new < 0:
        return None

    return (
        (new - old)
        / abs(old)
    ) * 100.0


def _percentile_rank(
    value: float,
    values: list[float],
) -> float | None:

    if not values:
        return None

    count = sum(
        item <= value
        for item in values
    )

    return (
        count / len(values)
    ) * 100.0


def _position_label(
    percentile: float | None,
) -> str:

    if percentile is None:
        return "Unknown"

    if percentile < 33.333:
        return "Low"

    if percentile > 66.667:
        return "High"

    return "Mid"


def _direction(
    values: list[float],
) -> tuple[str, float | None]:
    """
    Descriptive direction only.
    """

    if len(values) < 2:
        return "Unknown", None

    x = list(
        range(len(values))
    )

    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(values)

    numerator = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, values)
    )

    denominator = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    if denominator == 0:
        return "Unknown", None

    slope = numerator / denominator

    variance_y = sum(
        (yi - mean_y) ** 2
        for yi in values
    )

    if variance_y == 0:
        return "Stable", 0.0

    variance_x = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    correlation_denominator = math.sqrt(
        variance_x * variance_y
    )

    if correlation_denominator == 0:
        strength = 0.0
    else:
        correlation = (
            numerator
            / correlation_denominator
        )

        strength = abs(correlation)

    if slope > 0:
        return "Rising", strength

    if slope < 0:
        return "Falling", strength

    return "Stable", strength


# ============================================================
# COMPARABLE PERIOD
# ============================================================

def _period_value(
    observations: list[tuple[Any, float]],
    metric_type: MetricType,
) -> float | None:

    if not observations:
        return None

    values = [
        value
        for _, value in observations
    ]

    if metric_type == "flow":
        return sum(values)

    return values[-1]


def _comparable_periods(
    full_series: list[tuple[Any, float]],
    recent_count: int,
) -> tuple[
    list[tuple[Any, float]],
    list[tuple[Any, float]],
]:
    """
    Split the available series into:

        previous equivalent block
        latest/recent block

    Example:
        recent_count = 2

        [... previous 2 periods ...]
        [... latest 2 periods ...]
    """

    if recent_count <= 0:
        return [], []

    if len(full_series) < recent_count:
        return [], full_series

    latest = full_series[-recent_count:]

    previous_end = len(
        full_series
    ) - recent_count

    previous_start = max(
        0,
        previous_end - recent_count,
    )

    previous = full_series[
        previous_start:previous_end
    ]

    return previous, latest


# ============================================================
# FINANCIAL METRIC STATE
# ============================================================

def build_financial_metric_state(
    metric: str,
    recent_table: pa.Table,
    historical_table: pa.Table,
    metric_type: MetricType,
    source_frequency: str,
) -> FinancialMetricState:

    recent_series = _extract_series(
        recent_table,
        metric,
    )

    historical_series = _extract_series(
        historical_table,
        metric,
    )

    recent_values = [
        value
        for _, value in recent_series
    ]

    historical_values = [
        value
        for _, value in historical_series
    ]

    # --------------------------------------------------------
    # No historical observations
    # --------------------------------------------------------

    if not historical_values:

        return FinancialMetricState(
            metric=metric,
            metric_type=metric_type,

            current_value=None,

            recent_start_value=None,
            recent_change=None,
            recent_change_pct=None,

            previous_comparable_value=None,
            latest_comparable_value=None,
            comparable_change=None,
            comparable_change_pct=None,

            historical_percentile=None,

            historical_min=None,
            historical_max=None,
            historical_median=None,

            recent_direction="Unknown",
            direction_strength=None,

            state="Unavailable",

            recent_periods=0,
            historical_periods=0,

            source_frequency=source_frequency,

            limited=True,

            limitation=(
                "No usable observations exist "
                "for this financial metric."
            ),
        )

    # --------------------------------------------------------
    # Current
    # --------------------------------------------------------

    current_value = historical_values[-1]

    recent_start_value = (
        recent_values[0]
        if recent_values
        else current_value
    )

    # --------------------------------------------------------
    # Selected-period change
    # --------------------------------------------------------

    recent_change = (
        current_value
        - recent_start_value
    )

    recent_change_pct = (
        _safe_percentage_change(
            recent_start_value,
            current_value,
        )
    )

    # --------------------------------------------------------
    # Recent direction
    # --------------------------------------------------------

    direction, strength = _direction(
        recent_values
    )

    # --------------------------------------------------------
    # Previous equivalent period
    # --------------------------------------------------------

    previous_block, latest_block = (
        _comparable_periods(
            historical_series,
            max(
                len(recent_series),
                1,
            ),
        )
    )

    previous_value = _period_value(
        previous_block,
        metric_type,
    )

    latest_comparable_value = _period_value(
        latest_block,
        metric_type,
    )

    comparable_change = None
    comparable_change_pct = None

    if (
        previous_value is not None
        and latest_comparable_value is not None
    ):

        comparable_change = (
            latest_comparable_value
            - previous_value
        )

        comparable_change_pct = (
            _safe_percentage_change(
                previous_value,
                latest_comparable_value,
            )
        )

    # --------------------------------------------------------
    # Historical position
    # --------------------------------------------------------

    percentile = _percentile_rank(
        current_value,
        historical_values,
    )

    position = _position_label(
        percentile
    )

    historical_min = min(
        historical_values
    )

    historical_max = max(
        historical_values
    )

    historical_median = statistics.median(
        historical_values
    )

    if direction == "Unknown":
        state = f"Unknown / {position}"
    else:
        state = f"{direction} / {position}"

    limitation = None
    limited = False

    if len(recent_values) < 2:
        limited = True
        limitation = (
            "Only one usable recent financial period "
            "is available, so recent directional movement "
            "cannot be established."
        )

    return FinancialMetricState(
        metric=metric,
        metric_type=metric_type,

        current_value=current_value,

        recent_start_value=recent_start_value,
        recent_change=recent_change,
        recent_change_pct=recent_change_pct,

        previous_comparable_value=previous_value,
        latest_comparable_value=latest_comparable_value,
        comparable_change=comparable_change,
        comparable_change_pct=comparable_change_pct,

        historical_percentile=percentile,

        historical_min=historical_min,
        historical_max=historical_max,
        historical_median=historical_median,

        recent_direction=direction,
        direction_strength=strength,

        state=state,

        recent_periods=len(recent_values),
        historical_periods=len(historical_values),

        source_frequency=source_frequency,

        limited=limited,
        limitation=limitation,
    )


# ============================================================
# FINANCIAL METRIC REGISTRY
# ============================================================

FINANCIAL_METRICS: dict[
    str,
    dict[str, str],
] = {
    # Income statement
    "TotalRevenue": {
        "statement": "income",
        "type": "flow",
    },
    "GrossProfit": {
        "statement": "income",
        "type": "flow",
    },
    "OperatingIncome": {
        "statement": "income",
        "type": "flow",
    },
    "NetIncome": {
        "statement": "income",
        "type": "flow",
    },

    # Balance sheet
    "TotalAssets": {
        "statement": "balance_sheet",
        "type": "stock",
    },
    "CurrentAssets": {
        "statement": "balance_sheet",
        "type": "stock",
    },
    "CurrentLiabilities": {
        "statement": "balance_sheet",
        "type": "stock",
    },
    "LongTermDebtAndCapitalLeaseObligation": {
        "statement": "balance_sheet",
        "type": "stock",
    },
    "StockholdersEquity": {
        "statement": "balance_sheet",
        "type": "stock",
    },

    # Cash flow
    "CashFromOperations": {
        "statement": "cash_flow",
        "type": "flow",
    },
    "FixedAssetPurchases": {
        "statement": "cash_flow",
        "type": "flow",
    },
    "NetBorrowing": {
        "statement": "cash_flow",
        "type": "flow",
    },
    "EndingCashBalance": {
        "statement": "cash_flow",
        "type": "stock",
    },

    # Indirect cash flow
    "TotalOperatingCashFlow": {
        "statement": "indirect_cash_flow",
        "type": "flow",
    },
    "TotalInvestingCashFlow": {
        "statement": "indirect_cash_flow",
        "type": "flow",
    },
    "TotalFinancingCashFlow": {
        "statement": "indirect_cash_flow",
        "type": "flow",
    },
    "NetChangeInCash": {
        "statement": "indirect_cash_flow",
        "type": "flow",
    },
    "TreasuryOpacityRatio": {
        "statement": "indirect_cash_flow",
        "type": "stock",
    },
}


# ============================================================
# STATEMENT TABLE RESOLUTION
# ============================================================

def _tables_for_statement(
    statement: str,
    window: Any,
) -> tuple[pa.Table, pa.Table, str]:
    """
    Return:

        recent table
        yearly historical baseline
        source frequency
    """

    if statement == "income":

        if window.quarterly_income.num_rows > 0:
            return (
                window.quarterly_income,
                window.yearly_income,
                "quarterly_recent_yearly_baseline",
            )

        return (
            window.yearly_income,
            window.yearly_income,
            "yearly_only",
        )

    if statement == "balance_sheet":

        if window.quarterly_balance_sheet.num_rows > 0:
            return (
                window.quarterly_balance_sheet,
                window.yearly_balance_sheet,
                "quarterly_recent_yearly_baseline",
            )

        return (
            window.yearly_balance_sheet,
            window.yearly_balance_sheet,
            "yearly_only",
        )

    if statement == "cash_flow":

        if window.quarterly_cash_flow.num_rows > 0:
            return (
                window.quarterly_cash_flow,
                window.yearly_cash_flow,
                "quarterly_recent_yearly_baseline",
            )

        return (
            window.yearly_cash_flow,
            window.yearly_cash_flow,
            "yearly_only",
        )

    if statement == "indirect_cash_flow":

        return (
            window.yearly_indirect_cash_flow,
            window.yearly_indirect_cash_flow,
            "yearly",
        )

    raise ValueError(
        f"Unknown financial statement: {statement}"
    )


# ============================================================
# WINDOW FINANCIAL STATES
# ============================================================

def build_financial_states(
    window: Any,
) -> dict[str, FinancialMetricState]:

    result: dict[str, FinancialMetricState] = {}

    for metric, definition in FINANCIAL_METRICS.items():

        statement = definition["statement"]

        metric_type = definition["type"]

        recent_table, historical_table, frequency = (
            _tables_for_statement(
                statement,
                window,
            )
        )

        state = build_financial_metric_state(
            metric=metric,
            recent_table=recent_table,
            historical_table=historical_table,
            metric_type=metric_type,
            source_frequency=frequency,
        )

        result[metric] = state

    return result