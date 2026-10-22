from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pyarrow as pa

from data_access.market import (
    get_next_trading_day,
    get_price_on_or_before,
)


@dataclass(frozen=True)
class OutcomeObservation:
    """Realized forward outcome for one historical prediction date."""

    target: str
    prediction_date: date
    data_cutoff_date: date
    entry_mode: str
    holding_period_months: float

    entry_date: date | None
    exit_date: date | None

    entry_price: float | None
    exit_price: float | None

    stock_return_pct: float | None
    benchmark_return_pct: float | None
    relative_return_pct: float | None

    benchmark: str | None

    valid: bool
    limitation: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prediction_date": self.prediction_date,
            "data_cutoff_date": self.data_cutoff_date,
            "entry_mode": self.entry_mode,
            "holding_period_months": self.holding_period_months,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stock_return_pct": self.stock_return_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "relative_return_pct": self.relative_return_pct,
            "benchmark": self.benchmark,
            "valid": self.valid,
            "limitation": self.limitation,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value))


def _pct_return(
    entry: float | None,
    exit: float | None,
) -> float | None:
    if entry is None or exit is None:
        return None

    if entry == 0:
        return None

    return ((exit - entry) / abs(entry)) * 100.0


def _table_rows(
    table: pa.Table,
) -> list[tuple[date, float]]:
    if table.num_rows == 0:
        return []

    required = {"report_date", "close"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(
            f"Benchmark history is missing columns: {sorted(missing)}"
        )

    rows: list[tuple[date, float]] = []

    for report_date, close in zip(
        table["report_date"].to_pylist(),
        table["close"].to_pylist(),
    ):
        if report_date is None or close is None:
            continue

        try:
            close_value = float(close)
        except (TypeError, ValueError):
            continue

        if close_value <= 0:
            continue

        rows.append((_as_date(report_date), close_value))

    rows.sort(key=lambda item: item[0])
    return rows


def _table_price_on_or_before(
    table: pa.Table,
    target_date: date,
) -> tuple[date, float] | None:
    rows = _table_rows(table)

    for report_date, close in reversed(rows):
        if report_date <= target_date:
            return report_date, close

    return None


def _benchmark_on_entry_date(
    benchmark_history: pa.Table,
    entry_date: date,
) -> tuple[date, float] | None:
    return _table_price_on_or_before(
        benchmark_history,
        entry_date,
    )


def _benchmark_on_or_before(
    benchmark_history: pa.Table,
    target_date: date,
) -> tuple[date, float] | None:
    return _table_price_on_or_before(
        benchmark_history,
        target_date,
    )


def build_forward_outcome(
    ticker: str,
    prediction_date: str | date | datetime,
    data_cutoff_date: str | date | datetime,
    holding_period_months: float,
    entry_mode: str,
    benchmark: str | None = None,
    benchmark_history: pa.Table | None = None,
) -> OutcomeObservation:
    """
    Build the realized forward return for a historical prediction date.

    This function intentionally returns raw returns only. It does NOT
    classify UP/SIDEWAYS/DOWN because those thresholds must later be
    learned from pre-cutoff historical outcome distributions.
    """

    from dateutil.relativedelta import relativedelta

    prediction_date = _as_date(prediction_date)
    data_cutoff_date = _as_date(data_cutoff_date)

    if holding_period_months <= 0 or holding_period_months > 12:
        raise ValueError(
            "holding_period_months must be > 0 and <= 12."
        )

    if entry_mode not in {
        "next_trading_day",
        "latest_known_data",
    }:
        raise ValueError(
            "entry_mode must be 'next_trading_day' or "
            "'latest_known_data'."
        )

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    if entry_mode == "next_trading_day":
        entry = get_next_trading_day(
            ticker,
            prediction_date,
        )

    else:
        entry = get_price_on_or_before(
            ticker,
            data_cutoff_date,
        )

    if entry is None:
        return OutcomeObservation(
            target=ticker,
            prediction_date=prediction_date,
            data_cutoff_date=data_cutoff_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No valid entry market observation exists.",
        )

    entry_date = _as_date(entry["report_date"])
    entry_price = float(entry["close"])

    # --------------------------------------------------------
    # Exit target
    # --------------------------------------------------------

    # Calendar-month holding period. For fractional months,
    # convert the fractional part to days using the average
    # Gregorian month length. This is only the target date;
    # actual exit remains the latest available market observation
    # on or before that date.
    whole_months = int(holding_period_months)
    fractional_months = holding_period_months - whole_months

    target_exit_date = entry_date + relativedelta(
        months=whole_months,
    )

    if fractional_months > 0:
        extra_days = round(
            fractional_months * 30.4375
        )
        target_exit_date = target_exit_date + relativedelta(
            days=extra_days,
        )

    exit_row = get_price_on_or_before(
        ticker,
        target_exit_date,
    )

    if exit_row is None:
        return OutcomeObservation(
            target=ticker,
            prediction_date=prediction_date,
            data_cutoff_date=data_cutoff_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=entry_date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No valid exit market observation exists.",
        )

    exit_date = _as_date(exit_row["report_date"])
    exit_price = float(exit_row["close"])

    stock_return = _pct_return(
        entry_price,
        exit_price,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark_return: float | None = None

    if benchmark_history is not None:
        benchmark_entry = _benchmark_on_entry_date(
            benchmark_history,
            entry_date,
        )

        benchmark_exit = _benchmark_on_or_before(
            benchmark_history,
            exit_date,
        )

        if (
            benchmark_entry is not None
            and benchmark_exit is not None
        ):
            benchmark_return = _pct_return(
                benchmark_entry[1],
                benchmark_exit[1],
            )

    relative_return = None

    if (
        stock_return is not None
        and benchmark_return is not None
    ):
        relative_return = (
            stock_return - benchmark_return
        )

    return OutcomeObservation(
        target=ticker,
        prediction_date=prediction_date,
        data_cutoff_date=data_cutoff_date,
        entry_mode=entry_mode,
        holding_period_months=holding_period_months,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        stock_return_pct=stock_return,
        benchmark_return_pct=benchmark_return,
        relative_return_pct=relative_return,
        benchmark=benchmark,
        valid=stock_return is not None,
        limitation=(
            None
            if stock_return is not None
            else "Unable to calculate stock return."
        ),
    )
