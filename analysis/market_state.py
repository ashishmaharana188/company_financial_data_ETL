from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import statistics

import pyarrow as pa

from data_access.ticker import normalize_ticker
# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass(frozen=True)
class MarketMetricState:
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

    selected_observations: int
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
            "selected_observations": self.selected_observations,
            "historical_observations": self.historical_observations,
            "limited": self.limited,
            "limitation": self.limitation,
        }


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _numeric_values(
    table: pa.Table,
    column: str,
) -> list[float]:

    if column not in table.column_names:
        raise ValueError(
            f"Column '{column}' not found. "
            f"Available: {table.column_names}"
        )

    result: list[float] = []

    for value in table[column].to_pylist():

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


# ============================================================
# DIRECTION
# ============================================================

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
# GENERIC MARKET METRIC STATE
# ============================================================

def build_market_metric_state(
    metric: str,
    selected_values: list[float],
    historical_values: list[float],
) -> MarketMetricState:

    selected_count = len(
        selected_values
    )

    historical_count = len(
        historical_values
    )

    if historical_count == 0:

        return MarketMetricState(
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

            selected_observations=selected_count,
            historical_observations=0,

            limited=True,

            limitation=(
                "No usable historical observations."
            ),
        )

    current = historical_values[-1]

    start = (
        selected_values[0]
        if selected_values
        else current
    )

    absolute_change = (
        current - start
    )

    relative_change_pct = (
        _safe_pct_change(
            start,
            current,
        )
    )

    percentile = _percentile(
        current,
        historical_values,
    )

    position = _position(
        percentile
    )

    direction, strength = _direction(
        selected_values
    )

    if direction == "Unknown":
        state = f"Unknown / {position}"
    else:
        state = f"{direction} / {position}"

    limited = selected_count < 3

    limitation = None

    if limited:
        limitation = (
            "Fewer than three usable observations "
            "exist in the selected timeframe."
        )

    return MarketMetricState(
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

        selected_observations=selected_count,
        historical_observations=historical_count,

        limited=limited,
        limitation=limitation,
    )


# ============================================================
# DAILY DERIVED SERIES
# ============================================================

def _daily_return_values(
    table: pa.Table,
) -> list[float]:

    closes = _numeric_values(
        table,
        "close",
    )

    if len(closes) < 2:
        return []

    returns: list[float] = []

    for previous, current in zip(
        closes[:-1],
        closes[1:],
    ):

        if previous == 0:
            continue

        returns.append(
            (
                current - previous
            )
            / abs(previous)
            * 100.0
        )

    return returns


def _daily_range_values(
    table: pa.Table,
) -> list[float]:

    required = {
        "high",
        "low",
        "close",
    }

    missing = (
        required
        - set(table.column_names)
    )

    if missing:
        raise ValueError(
            f"Missing columns for daily range: "
            f"{sorted(missing)}"
        )

    high = table["high"].to_pylist()
    low = table["low"].to_pylist()
    close = table["close"].to_pylist()

    result: list[float] = []

    for h, l, c in zip(
        high,
        low,
        close,
    ):

        if (
            h is None
            or l is None
            or c is None
        ):
            continue

        try:
            h = float(h)
            l = float(l)
            c = float(c)
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(h)
            or not math.isfinite(l)
            or not math.isfinite(c)
            or c == 0
        ):
            continue

        result.append(
            (
                (h - l)
                / abs(c)
            )
            * 100.0
        )

    return result


def _rolling_volatility_values(
    table: pa.Table,
    window: int = 20,
) -> list[float]:

    returns = _daily_return_values(
        table
    )

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
# COMPANY MARKET STATE
# ============================================================

def build_company_market_states(
    ticker: str,
    selected_table: pa.Table,
    historical_table: pa.Table,
) -> dict[str, MarketMetricState]:
    
    valid_tickers = set(normalize_ticker(ticker))
    """
    Build descriptive states for company market behavior.

    selected_table:
        Company market observations inside the selected
        analysis timeframe.

    historical_table:
        All available company market observations through
        the analysis cutoff.

    Both tables are supplied by the observation-window layer.
    """

    # --------------------------------------------------------
    # Ticker consistency
    # --------------------------------------------------------

    if "ticker" in selected_table.column_names:

        selected_tickers = {
            value
            for value in selected_table["ticker"].to_pylist()
            if value is not None
        }

        if (
            selected_tickers
            and not selected_tickers.intersection(valid_tickers)
        ):
            raise ValueError(
                f"Selected market data does not contain "
                f"{ticker!r}. Found: "
                f"{sorted(selected_tickers)}"
            )

    if "ticker" in historical_table.column_names:

        historical_tickers = {
            value
            for value in historical_table["ticker"].to_pylist()
            if value is not None
        }

        if (
            historical_tickers
            and not historical_tickers.intersection(valid_tickers)
        ):
            raise ValueError(
                f"Historical market data does not contain "
                f"{ticker!r}. Found: "
                f"{sorted(historical_tickers)}"
            )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    selected_close = _numeric_values(
        selected_table,
        "close",
    )

    historical_close = _numeric_values(
        historical_table,
        "close",
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    selected_volume = _numeric_values(
        selected_table,
        "volume",
    )

    historical_volume = _numeric_values(
        historical_table,
        "volume",
    )

    # --------------------------------------------------------
    # Daily range
    # --------------------------------------------------------

    selected_range = _daily_range_values(
        selected_table
    )

    historical_range = _daily_range_values(
        historical_table
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    selected_volatility = (
        _rolling_volatility_values(
            selected_table
        )
    )

    historical_volatility = (
        _rolling_volatility_values(
            historical_table
        )
    )

    # --------------------------------------------------------
    # Return states
    # --------------------------------------------------------

    return {
        "price": build_market_metric_state(
            metric="price",
            selected_values=selected_close,
            historical_values=historical_close,
        ),

        "volume": build_market_metric_state(
            metric="volume",
            selected_values=selected_volume,
            historical_values=historical_volume,
        ),

        "daily_range": build_market_metric_state(
            metric="daily_range",
            selected_values=selected_range,
            historical_values=historical_range,
        ),

        "volatility_20d": build_market_metric_state(
            metric="volatility_20d",
            selected_values=selected_volatility,
            historical_values=historical_volatility,
        ),
    }