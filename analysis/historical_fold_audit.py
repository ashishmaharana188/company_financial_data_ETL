from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any

from dateutil.relativedelta import relativedelta

from .real_olap_validation import (
    _as_date,
    _close_after,
    _close_on_or_before,
    _resolve_benchmark_history,
)
from .real_prediction_validation import (
    Phase5RealOLAPValidationResult,
    PredictionFoldResult,
)
from data_access.market import get_daily_history


@dataclass(frozen=True)
class HistoricalFoldAudit:
    prediction_date: date
    observed_trend: str
    predicted_trend: str
    trade_eligible: bool
    trade_reason: str
    method_a_trend: str | None
    method_b_trend: str | None
    conviction: str
    actual_class: str | None
    prediction_hit: bool | None

    entry_date: date | None
    exit_date: date | None
    entry_price: float | None
    exit_price: float | None
    stock_return_pct: float | None

    benchmark_entry_date: date | None
    benchmark_exit_date: date | None
    benchmark_entry_price: float | None
    benchmark_exit_price: float | None
    benchmark_return_pct: float | None
    relative_return_pct: float | None

    reported_return_pct: float
    return_difference_pct: float
    return_reconciled: bool

    entry_after_prediction: bool
    exit_after_entry: bool
    one_month_window: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _rows(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    raw = table.to_pylist() if hasattr(table, "to_pylist") else []
    result: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        report_date = row.get("report_date")
        close = row.get("close")
        if report_date is None or close is None:
            continue
        try:
            report_date = _as_date(report_date)
            close = float(close)
        except (TypeError, ValueError):
            continue
        if not isfinite(close) or close <= 0:
            continue
        result.append({"report_date": report_date, "close": close})
    result.sort(key=lambda item: item["report_date"])
    return result


def audit_prediction_fold(
    config: Any,
    prediction_date: str | date | datetime,
    validation_result: Phase5RealOLAPValidationResult,
) -> HistoricalFoldAudit:
    """Audit one historical fold without changing model parameters.

    This verifies the exact price/return arithmetic while also reporting the
    point-in-time observed trend and whether the model signal was actually
    trade-eligible. A sideways observed trend is explicitly NO TRADE.
    """
    target_date = _as_date(prediction_date)
    fold: PredictionFoldResult | None = next(
        (
            item
            for item in validation_result.prediction_folds
            if _as_date(item.prediction_date) == target_date
        ),
        None,
    )
    if fold is None:
        raise ValueError(
            f"No validated prediction fold exists for {target_date}. "
            "Run Phase 5.8 with enough folds to include this date."
        )

    company_rows = _rows(get_daily_history(config.ticker))
    if not company_rows:
        raise RuntimeError(f"No OLAP market history found for {config.ticker}.")

    benchmark_rows = _rows(
        _resolve_benchmark_history(
            config.benchmark,
            start_date=company_rows[0]["report_date"],
            end_date=company_rows[-1]["report_date"],
        )
    )
    if not benchmark_rows:
        raise RuntimeError(f"No OLAP benchmark history found for {config.benchmark}.")

    entry = (
        _close_after(company_rows, target_date)
        if config.entry_mode == "next_trading_day"
        else _close_on_or_before(company_rows, target_date)
    )
    if entry is None:
        raise RuntimeError(f"No entry price exists after {target_date}.")

    entry_date, entry_price = entry
    whole_months = int(config.holding_period_months)
    fractional_months = float(config.holding_period_months) - whole_months
    target_exit = entry_date + relativedelta(months=whole_months)
    if fractional_months > 0:
        target_exit += relativedelta(days=round(fractional_months * 30.4375))

    exit = _close_on_or_before(company_rows, target_exit)
    if exit is None:
        raise RuntimeError(f"No exit price exists by {target_exit}.")

    exit_date, exit_price = exit
    stock_return = ((exit_price - entry_price) / abs(entry_price)) * 100.0

    benchmark_entry = _close_on_or_before(benchmark_rows, entry_date)
    benchmark_exit = _close_on_or_before(benchmark_rows, exit_date)

    benchmark_return = None
    relative_return = None
    if benchmark_entry is not None and benchmark_exit is not None:
        benchmark_return = (
            (benchmark_exit[1] - benchmark_entry[1])
            / abs(benchmark_entry[1])
        ) * 100.0
        relative_return = stock_return - benchmark_return

    difference = stock_return - float(fold.actual_return_pct)
    reconciled = abs(difference) <= 1e-9

    expected_month_exit = entry_date + relativedelta(months=int(config.holding_period_months))
    one_month_window = (
        exit_date <= expected_month_exit
        and exit_date >= entry_date
    )

    return HistoricalFoldAudit(
        prediction_date=target_date,
        observed_trend=fold.observed_trend,
        predicted_trend=fold.predicted_trend,
        trade_eligible=fold.trade_eligible,
        trade_reason=fold.trade_reason,
        method_a_trend=fold.method_a_trend,
        method_b_trend=fold.method_b_trend,
        conviction=fold.conviction,
        actual_class=fold.actual_class,
        prediction_hit=fold.hit,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        stock_return_pct=stock_return,
        benchmark_entry_date=(benchmark_entry[0] if benchmark_entry is not None else None),
        benchmark_exit_date=(benchmark_exit[0] if benchmark_exit is not None else None),
        benchmark_entry_price=(benchmark_entry[1] if benchmark_entry is not None else None),
        benchmark_exit_price=(benchmark_exit[1] if benchmark_exit is not None else None),
        benchmark_return_pct=benchmark_return,
        relative_return_pct=relative_return,
        reported_return_pct=float(fold.actual_return_pct),
        return_difference_pct=difference,
        return_reconciled=reconciled,
        entry_after_prediction=entry_date > target_date,
        exit_after_entry=exit_date >= entry_date,
        one_month_window=one_month_window,
    )


def print_historical_fold_audit(audit: HistoricalFoldAudit) -> None:
    print("\nPHASE 5.8 HISTORICAL FOLD AUDIT")
    print(f"Prediction date:        {audit.prediction_date}")
    print(f"Observed trend:         {audit.observed_trend}")
    print(f"Predicted trend:        {audit.predicted_trend}")
    print(f"Trade eligible:         {audit.trade_eligible}")
    print(f"Trade reason:           {audit.trade_reason}")
    print(f"Method A trend:         {audit.method_a_trend}")
    print(f"Method B trend:         {audit.method_b_trend}")
    print(f"Conviction:             {audit.conviction}")
    print(f"Actual class:           {audit.actual_class}")
    print(f"Prediction hit:         {audit.prediction_hit}")
    print(f"Entry:                  {audit.entry_date} @ {audit.entry_price}")
    print(f"Exit:                   {audit.exit_date} @ {audit.exit_price}")
    print(f"Stock return:           {audit.stock_return_pct:.6f}%")
    print(f"Benchmark return:       {audit.benchmark_return_pct}")
    print(f"Relative return:        {audit.relative_return_pct}")
    print(f"Reported return:        {audit.reported_return_pct:.6f}%")
    print(f"Return difference:      {audit.return_difference_pct:.12f}%")
    print(f"Return reconciled:      {audit.return_reconciled}")
    print(f"Entry after prediction: {audit.entry_after_prediction}")
    print(f"Exit after entry:       {audit.exit_after_entry}")
    print(f"One-month window:       {audit.one_month_window}")
