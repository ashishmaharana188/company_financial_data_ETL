from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def subtract_period(end_date: date, amount: int, unit: str) -> date:
    """Return the date `amount` units before end_date, using calendar months/years."""
    if amount < 1:
        raise ValueError("amount must be at least 1")
    normalized = unit.strip().lower()
    if normalized in {"day", "days"}:
        return end_date - timedelta(days=amount)
    if normalized in {"week", "weeks"}:
        return end_date - timedelta(weeks=amount)
    if normalized in {"month", "months"}:
        return (pd.Timestamp(end_date) - pd.DateOffset(months=amount)).date()
    if normalized in {"year", "years"}:
        return (pd.Timestamp(end_date) - pd.DateOffset(years=amount)).date()
    raise ValueError(f"Unsupported time unit: {unit!r}")


def normalize_date_range(value, default_start: date, default_end: date) -> tuple[date, date]:
    """Normalize Streamlit date_input output to a (start, end) date pair."""
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and value[0] is not None and value[1] is not None:
            start, end = value
            return start, end
        if len(value) == 1 and value[0] is not None:
            return value[0], default_end
    elif isinstance(value, date):
        return value, value
    return default_start, default_end
