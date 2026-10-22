from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import math
import statistics

import pyarrow as pa

from data_access.market import get_daily_history


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class GroupMetricState:
    metric: str

    current_value: float | None
    selected_start_value: float | None

    absolute_change: float | None
    relative_change_pct: float | None

    historical_percentile: float | None

    historical_min: float | None
    historical_max: float | None
    historical_median: float | None

    recent_direction: str
    direction_strength: float | None

    state: str

    historical_observations: int

    limited: bool
    limitation: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current_value": self.current_value,
            "selected_start_value": self.selected_start_value,
            "absolute_change": self.absolute_change,
            "relative_change_pct": self.relative_change_pct,
            "historical_percentile": self.historical_percentile,
            "historical_min": self.historical_min,
            "historical_max": self.historical_max,
            "historical_median": self.historical_median,
            "recent_direction": self.recent_direction,
            "direction_strength": self.direction_strength,
            "state": self.state,
            "historical_observations": self.historical_observations,
            "limited": self.limited,
            "limitation": self.limitation,
        }


@dataclass(frozen=True)
class GroupState:
    group_type: str
    group_name: str

    constituents: list[str]

    valid_constituents: list[str]
    missing_constituents: list[str]

    market_states: dict[str, GroupMetricState]

    breadth: dict[str, float]

    benchmark_return_pct: float | None
    relative_return_pct: float | None

    financial_states: dict[str, Any] | None = None

    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_type": self.group_type,
            "group_name": self.group_name,
            "constituents": self.constituents,
            "valid_constituents": self.valid_constituents,
            "missing_constituents": self.missing_constituents,
            "market_states": {
                key: value.as_dict()
                for key, value in self.market_states.items()
            },
            "breadth": self.breadth,
            "benchmark_return_pct": self.benchmark_return_pct,
            "relative_return_pct": self.relative_return_pct,
            "financial_states": self.financial_states,
            "limitations": self.limitations,
        }


# ============================================================
# NUMERIC HELPERS
# ============================================================


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _safe_pct_change(
    start: float | None,
    current: float | None,
) -> float | None:

    if start is None or current is None:
        return None

    if start == 0:
        return None

    return (
        (current - start)
        / abs(start)
    ) * 100.0


def _percentile(
    current: float,
    historical: list[float],
) -> float | None:

    if not historical:
        return None

    return (
        sum(
            value <= current
            for value in historical
        )
        / len(historical)
    ) * 100.0


def _position(
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

    if len(values) < 2:
        return "Unknown", None

    x = list(range(len(values)))

    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(values)

    numerator = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, values)
    )

    denominator_x = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    if denominator_x == 0:
        return "Unknown", None

    slope = numerator / denominator_x

    denominator_y = sum(
        (yi - mean_y) ** 2
        for yi in values
    )

    if denominator_y == 0:
        return "Stable", 0.0

    correlation = (
        numerator
        / math.sqrt(
            denominator_x * denominator_y
        )
    )

    strength = abs(correlation)

    if slope > 0:
        return "Rising", strength

    if slope < 0:
        return "Falling", strength

    return "Stable", strength


# ============================================================
# GENERIC GROUP METRIC STATE
# ============================================================


def _build_group_metric_state(
    metric: str,
    selected_values: list[float],
    historical_values: list[float],
) -> GroupMetricState:

    historical_count = len(historical_values)

    if historical_count == 0:
        return GroupMetricState(
            metric=metric,
            current_value=None,
            selected_start_value=None,
            absolute_change=None,
            relative_change_pct=None,
            historical_percentile=None,
            historical_min=None,
            historical_max=None,
            historical_median=None,
            recent_direction="Unknown",
            direction_strength=None,
            state="Unavailable",
            historical_observations=0,
            limited=True,
            limitation="No usable historical observations.",
        )

    current = historical_values[-1]

    start = (
        selected_values[0]
        if selected_values
        else current
    )

    absolute_change = current - start

    relative_change_pct = _safe_pct_change(
        start,
        current,
    )

    percentile = _percentile(
        current,
        historical_values,
    )

    position = _position(percentile)

    direction, strength = _direction(
        selected_values
    )

    if direction == "Unknown":
        state = f"Unknown / {position}"
    else:
        state = f"{direction} / {position}"

    limited = len(selected_values) < 3

    limitation = None

    if limited:
        limitation = (
            "Fewer than three usable observations "
            "exist in the selected timeframe."
        )

    return GroupMetricState(
        metric=metric,
        current_value=current,
        selected_start_value=start,
        absolute_change=absolute_change,
        relative_change_pct=relative_change_pct,
        historical_percentile=percentile,
        historical_min=min(historical_values),
        historical_max=max(historical_values),
        historical_median=statistics.median(
            historical_values
        ),
        recent_direction=direction,
        direction_strength=strength,
        state=state,
        historical_observations=historical_count,
        limited=limited,
        limitation=limitation,
    )


# ============================================================
# TABLE → DAILY CLOSE SERIES
# ============================================================


def _close_series(
    table: pa.Table,
) -> dict[date, float]:

    if "report_date" not in table.column_names:
        raise ValueError(
            "Group market table is missing report_date."
        )

    if "close" not in table.column_names:
        raise ValueError(
            "Group market table is missing close."
        )

    result: dict[date, float] = {}

    dates = table["report_date"].to_pylist()
    closes = table["close"].to_pylist()

    for report_date, close in zip(
        dates,
        closes,
    ):

        numeric_close = _finite(close)

        if numeric_close is None:
            continue

        if numeric_close <= 0:
            continue

        if isinstance(report_date, datetime):
            report_date = report_date.date()

        elif not isinstance(report_date, date):
            report_date = date.fromisoformat(
                str(report_date)
            )

        result[report_date] = numeric_close

    return result


# ============================================================
# INDIVIDUAL RETURNS
# ============================================================


def _period_return(
    series: dict[date, float],
    start_date: date | None,
    end_date: date | None,
) -> float | None:

    observations = [
        (report_date, close)
        for report_date, close in series.items()
        if (
            (start_date is None or report_date >= start_date)
            and
            (end_date is None or report_date <= end_date)
        )
    ]

    if len(observations) < 2:
        return None

    observations.sort(
        key=lambda item: item[0]
    )

    start_close = observations[0][1]
    end_close = observations[-1][1]

    if start_close == 0:
        return None

    return (
        (end_close - start_close)
        / abs(start_close)
    ) * 100.0


# ============================================================
# GROUP DAILY RETURNS
# ============================================================


def _group_daily_returns(
    histories: dict[str, pa.Table],
) -> tuple[list[date], list[float]]:

    constituent_returns: dict[
        str,
        dict[date, float],
    ] = {}

    for ticker, table in histories.items():

        series = _close_series(table)

        if len(series) < 2:
            continue

        ordered = sorted(
            series.items(),
            key=lambda item: item[0],
        )

        returns: dict[date, float] = {}

        for (
            previous,
            current,
        ) in zip(
            ordered[:-1],
            ordered[1:],
        ):

            previous_date, previous_close = previous
            current_date, current_close = current

            if previous_close == 0:
                continue

            returns[current_date] = (
                (current_close - previous_close)
                / abs(previous_close)
            ) * 100.0

        constituent_returns[ticker] = returns

    all_dates = sorted(
        {
            report_date
            for returns in constituent_returns.values()
            for report_date in returns
        }
    )

    group_dates: list[date] = []
    group_returns: list[float] = []

    for report_date in all_dates:

        daily_values = [
            returns[report_date]
            for returns in constituent_returns.values()
            if report_date in returns
        ]

        if not daily_values:
            continue

        group_dates.append(report_date)
        group_returns.append(
            statistics.fmean(daily_values)
        )

    return group_dates, group_returns


# ============================================================
# GROUP INDEX
# ============================================================


def _group_index(
    group_returns: list[float],
) -> list[float]:

    if not group_returns:
        return []

    index_values = [100.0]

    current = 100.0

    for daily_return in group_returns:

        current *= (
            1.0 + daily_return / 100.0
        )

        index_values.append(current)

    return index_values


# ============================================================
# ROLLING VOLATILITY
# ============================================================


def _rolling_volatility(
    returns: list[float],
    window: int = 20,
) -> list[float]:

    if len(returns) < window:
        return []

    result: list[float] = []

    for index in range(
        window,
        len(returns) + 1,
    ):

        chunk = returns[
            index - window:index
        ]

        result.append(
            statistics.pstdev(chunk)
        )

    return result


# ============================================================
# BREADTH
# ============================================================


def _breadth(
    period_returns: dict[str, float | None],
) -> dict[str, float]:

    total = len(period_returns)

    if total == 0:
        return {
            "UP": 0.0,
            "SIDEWAYS": 0.0,
            "DOWN": 0.0,
        }

    up = sum(
        value is not None and value > 0
        for value in period_returns.values()
    )

    down = sum(
        value is not None and value < 0
        for value in period_returns.values()
    )

    sideways = total - up - down

    return {
        "UP": (up / total) * 100.0,
        "SIDEWAYS": (sideways / total) * 100.0,
        "DOWN": (down / total) * 100.0,
    }


# ============================================================
# FINANCIAL STATE AGGREGATION
# ============================================================


def aggregate_financial_states(
    financial_states_by_ticker: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:

    if not financial_states_by_ticker:
        return {
            "available": False,
            "metrics": {},
        }

    metric_names = sorted(
        {
            metric
            for states in financial_states_by_ticker.values()
            for metric in states
        }
    )

    aggregated: dict[str, Any] = {}

    for metric in metric_names:

        current_values: list[float] = []
        changes: list[float] = []

        state_counts: dict[str, int] = {}

        valid_constituents = 0

        for states in financial_states_by_ticker.values():

            state = states.get(metric)

            if state is None:
                continue

            current_value = _finite(
                getattr(
                    state,
                    "current_value",
                    None,
                )
            )

            change = _finite(
                getattr(
                    state,
                    "relative_change_pct",
                    None,
                )
            )

            state_name = getattr(
                state,
                "state",
                None,
            )

            if current_value is not None:
                current_values.append(
                    current_value
                )

            if change is not None:
                changes.append(change)

            if state_name:
                state_counts[state_name] = (
                    state_counts.get(
                        state_name,
                        0,
                    ) + 1
                )

            if (
                current_value is not None
                or change is not None
            ):
                valid_constituents += 1

        aggregated[metric] = {
            "current_mean": (
                statistics.fmean(current_values)
                if current_values
                else None
            ),
            "change_mean_pct": (
                statistics.fmean(changes)
                if changes
                else None
            ),
            "state_counts": state_counts,
            "valid_constituents": valid_constituents,
            "total_constituents": len(
                financial_states_by_ticker
            ),
        }

    return {
        "available": True,
        "metrics": aggregated,
    }


# ============================================================
# MAIN GROUP STATE BUILDER
# ============================================================


def build_group_state(
    group_type: str,
    group_name: str,
    constituents: list[str],
    analysis_start_date: date | str | datetime,
    analysis_end_date: date | str | datetime,
    benchmark_history: pa.Table | None = None,
    financial_states_by_ticker: dict[
        str,
        dict[str, Any],
    ] | None = None,
) -> GroupState:
    """
    Build Industry/Sector descriptive state.

    group_type:
        "industry"
        "sector"

    Constituents are supplied by market_metadata.

    Historical membership is NOT reconstructed.
    Therefore the historical group series represents the
    current constituent universe over its available history.
    """

    histories: dict[str, pa.Table] = {}

    missing_constituents: list[str] = []

    for ticker in sorted(
        set(constituents)
    ):

        table = get_daily_history(
            ticker,
            end_date=analysis_end_date,
        )

        if table.num_rows == 0:
            missing_constituents.append(
                ticker
            )
            continue

        histories[ticker] = table

    valid_constituents = sorted(
        histories.keys()
    )

    limitations: list[str] = []

    if missing_constituents:
        limitations.append(
            f"{len(missing_constituents)} constituent(s) "
            "have no usable market history."
        )

    if not valid_constituents:
        limitations.append(
            "No constituent has usable market history."
        )

        return GroupState(
            group_type=group_type,
            group_name=group_name,
            constituents=sorted(
                set(constituents)
            ),
            valid_constituents=[],
            missing_constituents=missing_constituents,
            market_states={},
            breadth={
                "UP": 0.0,
                "SIDEWAYS": 0.0,
                "DOWN": 0.0,
            },
            benchmark_return_pct=None,
            relative_return_pct=None,
            financial_states=(
                aggregate_financial_states(
                    financial_states_by_ticker
                )
                if financial_states_by_ticker is not None
                else None
            ),
            limitations=limitations,
        )

    # --------------------------------------------------------
    # Build group daily return series
    # --------------------------------------------------------

    group_dates, group_returns = (
        _group_daily_returns(histories)
    )

    if not group_returns:
        limitations.append(
            "Unable to construct a group return series."
        )

        return GroupState(
            group_type=group_type,
            group_name=group_name,
            constituents=sorted(
                set(constituents)
            ),
            valid_constituents=valid_constituents,
            missing_constituents=missing_constituents,
            market_states={},
            breadth={
                "UP": 0.0,
                "SIDEWAYS": 0.0,
                "DOWN": 0.0,
            },
            benchmark_return_pct=None,
            relative_return_pct=None,
            financial_states=(
                aggregate_financial_states(
                    financial_states_by_ticker
                )
                if financial_states_by_ticker is not None
                else None
            ),
            limitations=limitations,
        )

    group_index = _group_index(
        group_returns
    )

    # group_index has one initial value before the
    # first group daily return.
    index_dates = [
        group_dates[0]
    ] + group_dates

    # --------------------------------------------------------
    # Historical / selected slices
    # --------------------------------------------------------

    def _as_date(
        value: date | str | datetime,
    ) -> date:

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        return date.fromisoformat(
            str(value)
        )

    analysis_start = _as_date(
        analysis_start_date
    )

    analysis_end = _as_date(
        analysis_end_date
    )

    selected_values = [
        value
        for report_date, value in zip(
            index_dates,
            group_index,
        )
        if (
            report_date >= analysis_start
            and report_date <= analysis_end
        )
    ]

    historical_values = [
        value
        for report_date, value in zip(
            index_dates,
            group_index,
        )
        if report_date <= analysis_end
    ]

    market_states: dict[
        str,
        GroupMetricState,
    ] = {}

    # --------------------------------------------------------
    # Group index / price state
    # --------------------------------------------------------

    market_states["group_index"] = (
        _build_group_metric_state(
            metric="group_index",
            selected_values=selected_values,
            historical_values=historical_values,
        )
    )

    # --------------------------------------------------------
    # Daily return state
    # --------------------------------------------------------

    selected_returns = [
        value
        for report_date, value in zip(
            group_dates,
            group_returns,
        )
        if (
            report_date >= analysis_start
            and report_date <= analysis_end
        )
    ]

    market_states["daily_return"] = (
        _build_group_metric_state(
            metric="daily_return",
            selected_values=selected_returns,
            historical_values=group_returns[
                : len([
                    value
                    for value in group_dates
                    if value <= analysis_end
                ])
            ],
        )
    )

    # --------------------------------------------------------
    # Volatility state
    # --------------------------------------------------------

    historical_volatility = (
        _rolling_volatility(
            [
                value
                for report_date, value in zip(
                    group_dates,
                    group_returns,
                )
                if report_date <= analysis_end
            ]
        )
    )

    selected_volatility = []

    if historical_volatility:

        volatility_dates = group_dates[
            19:
        ]

        selected_volatility = [
            value
            for report_date, value in zip(
                volatility_dates,
                historical_volatility,
            )
            if (
                report_date >= analysis_start
                and report_date <= analysis_end
            )
        ]

    market_states["volatility_20d"] = (
        _build_group_metric_state(
            metric="volatility_20d",
            selected_values=selected_volatility,
            historical_values=historical_volatility,
        )
    )

    # --------------------------------------------------------
    # Constituent breadth
    # --------------------------------------------------------

    constituent_period_returns: dict[
        str,
        float | None,
    ] = {}

    for ticker, table in histories.items():

        series = _close_series(table)

        constituent_period_returns[ticker] = (
            _period_return(
                series,
                analysis_start,
                analysis_end,
            )
        )

    breadth = _breadth(
        constituent_period_returns
    )

    # --------------------------------------------------------
    # Group period return
    # --------------------------------------------------------

    valid_period_returns = [
        value
        for value in constituent_period_returns.values()
        if value is not None
    ]

    group_return = (
        statistics.fmean(
            valid_period_returns
        )
        if valid_period_returns
        else None
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark_return = None
    relative_return = None

    if benchmark_history is not None:

        benchmark_series = _close_series(
            benchmark_history
        )

        benchmark_return = _period_return(
            benchmark_series,
            analysis_start,
            analysis_end,
        )

        if (
            group_return is not None
            and benchmark_return is not None
        ):
            relative_return = (
                group_return
                - benchmark_return
            )

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    if len(valid_constituents) < 3:
        limitations.append(
            "Group contains fewer than three "
            "constituents with usable market history."
        )

    if not valid_period_returns:
        limitations.append(
            "No constituent has enough observations "
            "for the selected analysis timeframe."
        )

    return GroupState(
        group_type=group_type,
        group_name=group_name,
        constituents=sorted(
            set(constituents)
        ),
        valid_constituents=valid_constituents,
        missing_constituents=missing_constituents,
        market_states=market_states,
        breadth=breadth,
        benchmark_return_pct=benchmark_return,
        relative_return_pct=relative_return,
        financial_states=(
            aggregate_financial_states(
                financial_states_by_ticker
            )
            if financial_states_by_ticker is not None
            else None
        ),
        limitations=limitations,
    )