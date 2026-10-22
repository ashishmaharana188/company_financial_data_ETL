from __future__ import annotations

import pandas as pd

from .statistics_service import normalize_series


def calculate_indicators(
    frame: pd.DataFrame,
    *,
    ema_periods: tuple[int, ...] = (),
    sma_periods: tuple[int, ...] = (),
    bollinger_period: int | None = None,
    bollinger_width: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Calculate moving-average overlays on the supplied observation series.

    Windows count observations, not calendar days. Each output is a clean
    {date, value} frame which can be overlaid on the underlying variable's chart.
    """
    data = normalize_series(frame)
    results: dict[str, pd.DataFrame] = {}
    if data.empty:
        return results

    values = data["value"].astype(float)
    dates = data["date"]
    for period in sorted(set(int(p) for p in ema_periods if int(p) > 0)):
        line = values.ewm(span=period, adjust=False, min_periods=period).mean()
        results[f"EMA {period}"] = pd.DataFrame({"date": dates, "value": line})
    for period in sorted(set(int(p) for p in sma_periods if int(p) > 0)):
        line = values.rolling(window=period, min_periods=period).mean()
        results[f"SMA {period}"] = pd.DataFrame({"date": dates, "value": line})
    if bollinger_period is not None and int(bollinger_period) > 1:
        period = int(bollinger_period)
        middle = values.rolling(window=period, min_periods=period).mean()
        deviation = values.rolling(window=period, min_periods=period).std(ddof=1)
        results[f"Bollinger middle {period}"] = pd.DataFrame({"date": dates, "value": middle})
        results[f"Bollinger upper {period} (+{bollinger_width:g}σ)"] = pd.DataFrame(
            {"date": dates, "value": middle + bollinger_width * deviation}
        )
        results[f"Bollinger lower {period} (-{bollinger_width:g}σ)"] = pd.DataFrame(
            {"date": dates, "value": middle - bollinger_width * deviation}
        )
    return results
