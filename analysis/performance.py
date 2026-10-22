from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from datetime import date, datetime
from typing import Any, Callable, Iterator

import pyarrow as pa


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0


class DailyHistoryCache:
    """Per-validation-run cache for full daily history tables.

    Group-state construction repeatedly requests the same constituent history
    with different ``end_date`` values. We fetch each ticker once and return a
    cutoff-safe slice for later requests. No future rows are exposed to callers.
    """

    def __init__(self, loader: Callable[..., pa.Table]) -> None:
        self._loader = loader
        self._tables: dict[str, pa.Table] = {}
        self._dates: dict[str, list[date | None]] = {}
        self.stats = CacheStats()

    @staticmethod
    def _key(ticker: str) -> str:
        return str(ticker).strip().upper()

    @staticmethod
    def _as_date(value: str | date | datetime | None) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    def get_daily_history(
        self,
        ticker: str,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> pa.Table:
        self.stats.requests += 1
        key = self._key(ticker)

        if key not in self._tables:
            self.stats.misses += 1
            table = self._loader(ticker)
            self._tables[key] = table
            if "report_date" in table.column_names:
                self._dates[key] = [self._as_date(value) for value in table["report_date"].to_pylist()]
            else:
                self._dates[key] = []
        else:
            self.stats.hits += 1
            table = self._tables[key]

        if not start_date and not end_date:
            return table

        dates = self._dates.get(key, [])
        if not dates:
            return table.slice(0, 0)

        start = self._as_date(start_date)
        end = self._as_date(end_date)
        normalized_dates = [value for value in dates if value is not None]
        if not normalized_dates:
            return table.slice(0, 0)

        left = bisect_left(normalized_dates, start) if start is not None else 0
        right = bisect_right(normalized_dates, end) if end is not None else len(normalized_dates)
        if right <= left:
            return table.slice(0, 0)
        return table.slice(left, right - left)

    @property
    def unique_tickers(self) -> int:
        return len(self._tables)


@contextmanager
def patch_group_state_history_loader(
    cache: DailyHistoryCache,
) -> Iterator[None]:
    """Temporarily route group-state history reads through the run cache."""
    from analysis import group_state

    original = group_state.get_daily_history
    group_state.get_daily_history = cache.get_daily_history
    try:
        yield
    finally:
        group_state.get_daily_history = original
