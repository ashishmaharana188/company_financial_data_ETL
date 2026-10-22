from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from dateutil.relativedelta import relativedelta


AnalysisTimeframe = Literal[
    "1W",
    "2W",
    "1M",
    "3M",
    "6M",
    "9M",
    "1Y",
    "18M",
    "2Y",
    "3Y",
    "5Y",
]

EntryMode = Literal[
    "next_trading_day",
    "latest_known_data",
]


SUPPORTED_TIMEFRAMES: dict[str, relativedelta] = {
    "1W": relativedelta(weeks=1),
    "2W": relativedelta(weeks=2),
    "1M": relativedelta(months=1),
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "9M": relativedelta(months=9),
    "1Y": relativedelta(years=1),
    "18M": relativedelta(months=18),
    "2Y": relativedelta(years=2),
    "3Y": relativedelta(years=3),
    "5Y": relativedelta(years=5),
}


def normalize_analysis_date(
    value: str | date | datetime,
) -> date:
    """
    Convert supported date input into datetime.date.
    """

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError("analysis_date cannot be empty")

        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid analysis_date: {value!r}. "
                "Expected YYYY-MM-DD."
            ) from exc

    raise TypeError(
        f"Unsupported analysis_date type: {type(value).__name__}"
    )


def validate_timeframe(
    timeframe: str,
) -> AnalysisTimeframe:
    """
    Validate the selected analysis timeframe.
    """

    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES.keys())

        raise ValueError(
            f"Unsupported analysis timeframe: {timeframe!r}. "
            f"Supported values: {supported}"
        )

    return timeframe  # type: ignore[return-value]


def validate_holding_period(
    holding_period_months: float,
) -> float:
    """
    Validate holding period.

    Allowed range:
        0.1 <= months <= 12.0
    """

    if not isinstance(holding_period_months, (int, float)):
        raise TypeError(
            "holding_period_months must be numeric"
        )

    if not 0.1 <= holding_period_months <= 12.0:
        raise ValueError(
            "holding_period_months must be between "
            "0.1 and 12.0 months"
        )

    return float(holding_period_months)


def normalize_benchmark(benchmark: str) -> str:
    """
    Normalize the user-selected benchmark identifier.

    No benchmark mapping or interpretation occurs here.
    """

    if not isinstance(benchmark, str):
        raise TypeError(
            f"benchmark must be a string, got "
            f"{type(benchmark).__name__}"
        )

    benchmark = benchmark.strip()

    if not benchmark:
        raise ValueError("benchmark cannot be empty")

    return benchmark


def validate_entry_mode(
    entry_mode: str,
) -> EntryMode:
    if entry_mode not in {
        "next_trading_day",
        "latest_known_data",
    }:
        raise ValueError(
            "entry_mode must be either "
            "'next_trading_day' or 'latest_known_data'"
        )

    return entry_mode  # type: ignore[return-value]


def subtract_timeframe(
    analysis_date: date,
    timeframe: AnalysisTimeframe,
) -> date:
    """
    Calculate the beginning of the selected analysis window.

    This is a calendar-based boundary.
    Actual market observations will later be selected from
    available trading dates inside this window.
    """

    validate_timeframe(timeframe)

    return analysis_date - SUPPORTED_TIMEFRAMES[timeframe]


def add_holding_period(
    entry_date: date,
    holding_period_months: float,
) -> date:
    """
    Calculate the calendar target date for a holding period.

    Whole months use true calendar-month arithmetic.

    Fractional months are converted using the number of
    calendar days in the month reached after the whole-month
    component.

    Example:
        2026-10-05 + 0.1 month
        October has 31 days
        0.1 * 31 = 3.1 days
        -> 2026-10-08

    This gives deterministic support for the full requested
    0.1 to 12.0 month range.
    """

    holding_period_months = validate_holding_period(
        holding_period_months
    )

    whole_months = int(holding_period_months)
    fraction = holding_period_months - whole_months

    result = entry_date + relativedelta(
        months=whole_months
    )

    if fraction == 0:
        return result

    days_in_month = calendar.monthrange(
        result.year,
        result.month,
    )[1]

    fractional_days = int(
        days_in_month * fraction + 0.5
    )

    fractional_days = max(
        fractional_days,
        1,
    )

    return result + timedelta(
        days=fractional_days
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Immutable configuration for one TrendAnalysis evaluation.

    This class defines the user's temporal inputs.

    It does NOT retrieve data and does NOT calculate features.
    """

    analysis_date: date
    analysis_timeframe: AnalysisTimeframe
    holding_period_months: float
    benchmark: str
    entry_mode: EntryMode

    def __post_init__(self) -> None:
        normalized_date = normalize_analysis_date(
            self.analysis_date
        )

        normalized_timeframe = validate_timeframe(
            self.analysis_timeframe
        )

        normalized_holding_period = validate_holding_period(
            self.holding_period_months
        )

        normalized_benchmark = normalize_benchmark(
            self.benchmark
        )

        normalized_entry_mode = validate_entry_mode(
            self.entry_mode
        )

        object.__setattr__(
            self,
            "analysis_date",
            normalized_date,
        )

        object.__setattr__(
            self,
            "analysis_timeframe",
            normalized_timeframe,
        )

        object.__setattr__(
            self,
            "holding_period_months",
            normalized_holding_period,
        )

        object.__setattr__(
            self,
            "benchmark",
            normalized_benchmark,
        )

        object.__setattr__(
            self,
            "entry_mode",
            normalized_entry_mode,
        )

    @property
    def data_cutoff_date(self) -> date:
        """
        Last calendar date whose completed daily data may be used.

        The analysis itself occurs BEFORE analysis_date.
        """

        return self.analysis_date - timedelta(days=1)

    @property
    def analysis_start_date(self) -> date:
        """
        Calendar start of the selected historical analysis window.
        """

        return subtract_timeframe(
            self.analysis_date,
            self.analysis_timeframe,
        )

    @property
    def analysis_end_date(self) -> date:
        """
        Last calendar date allowed into the historical state window.
        """

        return self.data_cutoff_date

    def holding_target_date(
        self,
        entry_date: date,
    ) -> date:
        """
        Calendar target date used to locate the future exit
        observation.
        """

        return add_holding_period(
            entry_date,
            self.holding_period_months,
        )

    def summary(self) -> dict[str, object]:
        """
        Return the resolved temporal configuration.
        """

        return {
            "analysis_date": self.analysis_date,
            "data_cutoff_date": self.data_cutoff_date,
            "analysis_start_date": self.analysis_start_date,
            "analysis_end_date": self.analysis_end_date,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "benchmark": self.benchmark,
            "entry_mode": self.entry_mode,
        }