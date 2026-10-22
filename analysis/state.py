from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import statistics

import pyarrow as pa


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass(frozen=True)
class VariableState:
    """
    Descriptive state of one variable.

    No economic interpretation is included.
    """

    variable: str

    current_value: float | None

    start_value: float | None

    absolute_change: float | None

    relative_change: float | None

    historical_percentile: float | None

    historical_min: float | None

    historical_max: float | None

    historical_median: float | None

    recent_direction: str

    direction_strength: float | None

    state: str

    selected_observations: int

    historical_observations: int

    limited: bool

    limitation: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "current_value": self.current_value,
            "start_value": self.start_value,
            "absolute_change": self.absolute_change,
            "relative_change": self.relative_change,
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
            "selected_observations":
                self.selected_observations,
            "historical_observations":
                self.historical_observations,
            "limited": self.limited,
            "limitation": self.limitation,
        }


# ============================================================
# NUMERIC EXTRACTION
# ============================================================

def _extract_numeric_values(
    table: pa.Table,
    value_column: str,
) -> list[float]:
    """
    Extract valid numeric observations from an Arrow table.
    """

    if value_column not in table.column_names:
        raise ValueError(
            f"Column '{value_column}' not found. "
            f"Available columns: {table.column_names}"
        )

    values = table[value_column].to_pylist()

    result: list[float] = []

    for value in values:
        if value is None:
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(numeric):
            continue

        result.append(numeric)

    return result


# ============================================================
# HISTORICAL POSITION
# ============================================================

def _percentile_rank(
    current_value: float,
    historical_values: list[float],
) -> float:
    """
    Empirical percentile rank of current_value against the
    complete available history.

    Returns 0-100.
    """

    if not historical_values:
        return float("nan")

    less_or_equal = sum(
        value <= current_value
        for value in historical_values
    )

    return (
        less_or_equal
        / len(historical_values)
    ) * 100.0


def _position_label(
    percentile: float,
) -> str:
    """
    Distribution-relative position.

    The boundaries are based on thirds of the variable's own
    historical distribution rather than fixed numerical values.
    """

    if percentile < 33.333:
        return "Low"

    if percentile > 66.667:
        return "High"

    return "Mid"


# ============================================================
# DIRECTION
# ============================================================

def _linear_regression_strength(
    values: list[float],
) -> float | None:
    """
    Calculate absolute Pearson correlation between observation
    order and value.

    This represents how strongly the series moves in one
    direction over the selected window.

    0 = little directional structure
    1 = strong monotonic linear movement
    """

    if len(values) < 3:
        return None

    x = list(range(len(values)))

    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(values)

    numerator = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, values)
    )

    denominator_x = math.sqrt(
        sum(
            (xi - mean_x) ** 2
            for xi in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (yi - mean_y) ** 2
            for yi in values
        )
    )

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    correlation = (
        numerator
        / (denominator_x * denominator_y)
    )

    return abs(correlation)


def _direction(
    values: list[float],
) -> tuple[str, float | None]:
    """
    Determine recent directional movement.

    Direction is descriptive only.

    Rising  = regression slope > 0
    Falling = regression slope < 0
    Stable  = no measurable movement
    """

    if len(values) < 2:
        return "Unknown", None

    x = list(range(len(values)))

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

    strength = _linear_regression_strength(
        values
    )

    if slope > 0:
        return "Rising", strength

    if slope < 0:
        return "Falling", strength

    return "Stable", strength


# ============================================================
# VARIABLE STATE
# ============================================================

def build_variable_state(
    variable: str,
    selected_table: pa.Table,
    historical_table: pa.Table,
    value_column: str,
) -> VariableState:
    """
    Build descriptive state from:

        selected_table
            = selected analysis timeframe

        historical_table
            = all available history through cutoff
    """

    selected_values = _extract_numeric_values(
        selected_table,
        value_column,
    )

    historical_values = _extract_numeric_values(
        historical_table,
        value_column,
    )

    selected_count = len(selected_values)
    historical_count = len(historical_values)

    # --------------------------------------------------------
    # No historical data
    # --------------------------------------------------------

    if historical_count == 0:

        return VariableState(
            variable=variable,

            current_value=None,
            start_value=None,

            absolute_change=None,
            relative_change=None,

            historical_percentile=None,

            historical_min=None,
            historical_max=None,
            historical_median=None,

            recent_direction="Unknown",
            direction_strength=None,

            state="Unavailable",

            selected_observations=selected_count,
            historical_observations=0,

            limited=True,

            limitation=(
                "No usable historical observations."
            ),
        )

    # --------------------------------------------------------
    # Current value
    # --------------------------------------------------------

    current_value = historical_values[-1]

    # --------------------------------------------------------
    # Selected-window start
    # --------------------------------------------------------

    if selected_count >= 1:
        start_value = selected_values[0]
    else:
        start_value = None

    # --------------------------------------------------------
    # Change
    # --------------------------------------------------------

    if start_value is None:
        absolute_change = None
        relative_change = None

    else:
        absolute_change = (
            current_value - start_value
        )

        if start_value != 0:
            relative_change = (
                absolute_change
                / abs(start_value)
            ) * 100.0
        else:
            relative_change = None

    # --------------------------------------------------------
    # Historical position
    # --------------------------------------------------------

    percentile = _percentile_rank(
        current_value,
        historical_values,
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

    position = _position_label(
        percentile
    )

    # --------------------------------------------------------
    # Recent direction
    # --------------------------------------------------------

    direction, direction_strength = _direction(
        selected_values
    )

    if direction == "Unknown":
        state = f"Unknown / {position}"
    else:
        state = f"{direction} / {position}"

    limitation = None
    limited = False

    if selected_count < 3:
        limited = True
        limitation = (
            "Insufficient observations in the selected "
            "analysis timeframe to establish directional movement."
        )

    return VariableState(
        variable=variable,

        current_value=current_value,
        start_value=start_value,

        absolute_change=absolute_change,
        relative_change=relative_change,

        historical_percentile=percentile,

        historical_min=historical_min,
        historical_max=historical_max,
        historical_median=historical_median,

        recent_direction=direction,
        direction_strength=direction_strength,

        state=state,

        selected_observations=selected_count,
        historical_observations=historical_count,

        limited=limited,
        limitation=limitation,
    )


# ============================================================
# BATCH STATE BUILDER
# ============================================================

def build_daily_variable_states(
    variables: dict[str, tuple[pa.Table, pa.Table, str]],
) -> dict[str, VariableState]:
    """
    Build states for multiple variables.

    Format:

        {
            "Brent_Crude": (
                selected_table,
                historical_table,
                "close",
            ),
            ...
        }
    """

    result: dict[str, VariableState] = {}

    for variable, (
        selected_table,
        historical_table,
        value_column,
    ) in variables.items():

        result[variable] = build_variable_state(
            variable=variable,
            selected_table=selected_table,
            historical_table=historical_table,
            value_column=value_column,
        )

    return result