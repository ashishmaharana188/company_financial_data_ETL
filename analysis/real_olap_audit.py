from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .real_olap_validation import (
    _as_date,
    _close_after,
    _close_on_or_before,
    _table_rows,
)
from .real_prediction_validation import Phase5RealOLAPValidationResult


@dataclass(frozen=True)
class FoldCalculationAudit:
    """Calculation-level reconciliation for one evaluated Phase 5.8 fold."""

    prediction_date: date
    reported_stock_return_pct: float
    reconstructed_stock_return_pct: float | None
    stock_return_abs_error_pct: float | None
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    reported_benchmark_return_pct: float | None
    reconstructed_benchmark_return_pct: float | None
    benchmark_return_abs_error_pct: float | None
    benchmark_entry_date: date | None
    benchmark_entry_price: float | None
    benchmark_exit_date: date | None
    benchmark_exit_price: float | None
    reported_relative_return_pct: float | None
    reconstructed_relative_return_pct: float | None
    relative_return_abs_error_pct: float | None
    actual_class: str | None
    combined_trend: str
    method_a_trend: str | None
    method_b_trend: str | None
    method_agreement: bool
    conviction: str
    up_probability_pct: float
    sideways_probability_pct: float
    down_probability_pct: float
    expected_return_pct: float | None
    training_observations: int
    state_atoms: Mapping[str, str]

    @property
    def stock_reconciled(self) -> bool:
        return (
            self.stock_return_abs_error_pct is not None
            and self.stock_return_abs_error_pct <= 1e-9
        )

    @property
    def benchmark_reconciled(self) -> bool:
        if self.reported_benchmark_return_pct is None:
            return self.reconstructed_benchmark_return_pct is None
        return (
            self.benchmark_return_abs_error_pct is not None
            and self.benchmark_return_abs_error_pct <= 1e-9
        )

    @property
    def relative_reconciled(self) -> bool:
        if self.reported_relative_return_pct is None:
            return self.reconstructed_relative_return_pct is None
        return (
            self.relative_return_abs_error_pct is not None
            and self.relative_return_abs_error_pct <= 1e-9
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("prediction_date", "entry_date", "exit_date", "benchmark_entry_date", "benchmark_exit_date"):
            value = payload.get(key)
            if value is not None:
                payload[key] = value.isoformat()
        payload["state_atoms"] = dict(self.state_atoms)
        payload["stock_reconciled"] = self.stock_reconciled
        payload["benchmark_reconciled"] = self.benchmark_reconciled
        payload["relative_reconciled"] = self.relative_reconciled
        return payload


@dataclass(frozen=True)
class RealOLAPCalculationAudit:
    ticker: str
    benchmark: str
    analysis_timeframe: str
    holding_period_months: float
    entry_mode: str
    folds: tuple[FoldCalculationAudit, ...]
    stock_mismatch_count: int
    benchmark_mismatch_count: int
    relative_mismatch_count: int
    missing_raw_input_count: int

    @property
    def all_reconciled(self) -> bool:
        return (
            self.stock_mismatch_count == 0
            and self.benchmark_mismatch_count == 0
            and self.relative_mismatch_count == 0
            and self.missing_raw_input_count == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "benchmark": self.benchmark,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "entry_mode": self.entry_mode,
            "fold_count": len(self.folds),
            "stock_mismatch_count": self.stock_mismatch_count,
            "benchmark_mismatch_count": self.benchmark_mismatch_count,
            "relative_mismatch_count": self.relative_mismatch_count,
            "missing_raw_input_count": self.missing_raw_input_count,
            "all_reconciled": self.all_reconciled,
            "folds": [fold.as_dict() for fold in self.folds],
        }


def _pct_error(reported: float | None, reconstructed: float | None) -> float | None:
    if reported is None or reconstructed is None:
        return None
    return abs(float(reported) - float(reconstructed))


def _month_exit_date(entry_date: date, holding_period_months: float) -> date:
    from dateutil.relativedelta import relativedelta

    whole_months = int(holding_period_months)
    fractional_months = holding_period_months - whole_months
    result = entry_date + relativedelta(months=whole_months)
    if fractional_months > 0:
        result += relativedelta(days=round(fractional_months * 30.4375))
    return result


def _return_pct(entry_price: float | None, exit_price: float | None) -> float | None:
    if entry_price is None or exit_price is None or entry_price == 0:
        return None
    return ((exit_price - entry_price) / abs(entry_price)) * 100.0


def reconcile_real_olap_calculations(
    result: Phase5RealOLAPValidationResult,
    config: Any,
) -> RealOLAPCalculationAudit:
    """Recompute fold outcomes from raw OLAP and compare with stored panel values."""

    from data_access.market import get_daily_history

    company_rows = _table_rows(get_daily_history(config.ticker))
    benchmark_rows = _table_rows(get_daily_history(config.benchmark))
    if not company_rows:
        raise RuntimeError(f"No OLAP market history found for {config.ticker!r}.")
    if not benchmark_rows:
        raise RuntimeError(f"No OLAP market history found for {config.benchmark!r}.")

    fold_by_date = {
        _as_date(fold.prediction_date): fold
        for fold in result.prediction_folds
    }
    panel_by_date = {
        _as_date(observation.as_of_date): observation
        for observation in result.panel_observations
    }

    folds: list[FoldCalculationAudit] = []
    missing_raw_inputs = 0

    for prediction_date in sorted(fold_by_date):
        fold = fold_by_date[prediction_date]
        observation = panel_by_date.get(prediction_date)

        if observation is None:
            missing_raw_inputs += 1
            reported_benchmark = None
            reported_relative = None
            state_atoms: dict[str, str] = {}
        else:
            reported_benchmark = (
                None
                if observation.benchmark_return_pct is None
                else float(observation.benchmark_return_pct)
            )
            reported_relative = (
                None
                if observation.relative_return_pct is None
                else float(observation.relative_return_pct)
            )
            state_atoms = {
                str(key): str(value)
                for key, value in observation.states.items()
            }

        entry = (
            _close_after(company_rows, prediction_date)
            if config.entry_mode == "next_trading_day"
            else _close_on_or_before(company_rows, prediction_date)
        )

        entry_date: date | None
        entry_price: float | None
        exit_date: date | None
        exit_price: float | None
        reconstructed_stock: float | None
        benchmark_entry = None
        benchmark_exit = None
        reconstructed_benchmark: float | None
        reconstructed_relative: float | None

        if entry is None:
            missing_raw_inputs += 1
            entry_date = entry_price = exit_date = exit_price = None
            reconstructed_stock = None
            reconstructed_benchmark = None
            reconstructed_relative = None
        else:
            entry_date, entry_price = entry
            target_exit_date = _month_exit_date(
                entry_date,
                float(config.holding_period_months),
            )
            exit_row = _close_on_or_before(company_rows, target_exit_date)

            if exit_row is None or exit_row[0] < entry_date:
                missing_raw_inputs += 1
                exit_date = exit_price = None
                reconstructed_stock = None
                reconstructed_benchmark = None
                reconstructed_relative = None
            else:
                exit_date, exit_price = exit_row
                reconstructed_stock = _return_pct(entry_price, exit_price)

                benchmark_entry = _close_on_or_before(benchmark_rows, entry_date)
                benchmark_exit = _close_on_or_before(benchmark_rows, exit_date)
                reconstructed_benchmark = _return_pct(
                    benchmark_entry[1] if benchmark_entry else None,
                    benchmark_exit[1] if benchmark_exit else None,
                )
                reconstructed_relative = (
                    reconstructed_stock - reconstructed_benchmark
                    if reconstructed_stock is not None
                    and reconstructed_benchmark is not None
                    else None
                )

        stock_error = _pct_error(
            float(fold.actual_return_pct),
            reconstructed_stock,
        )
        benchmark_error = _pct_error(
            reported_benchmark,
            reconstructed_benchmark,
        )
        relative_error = _pct_error(
            reported_relative,
            reconstructed_relative,
        )

        folds.append(
            FoldCalculationAudit(
                prediction_date=prediction_date,
                reported_stock_return_pct=float(fold.actual_return_pct),
                reconstructed_stock_return_pct=reconstructed_stock,
                stock_return_abs_error_pct=stock_error,
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                reported_benchmark_return_pct=reported_benchmark,
                reconstructed_benchmark_return_pct=reconstructed_benchmark,
                benchmark_return_abs_error_pct=benchmark_error,
                benchmark_entry_date=benchmark_entry[0] if benchmark_entry else None,
                benchmark_entry_price=benchmark_entry[1] if benchmark_entry else None,
                benchmark_exit_date=benchmark_exit[0] if benchmark_exit else None,
                benchmark_exit_price=benchmark_exit[1] if benchmark_exit else None,
                reported_relative_return_pct=reported_relative,
                reconstructed_relative_return_pct=reconstructed_relative,
                relative_return_abs_error_pct=relative_error,
                actual_class=(
                    None if fold.actual_class is None else str(fold.actual_class)
                ),
                combined_trend=str(fold.predicted_trend),
                method_a_trend=(
                    None if fold.method_a_trend is None else str(fold.method_a_trend)
                ),
                method_b_trend=(
                    None if fold.method_b_trend is None else str(fold.method_b_trend)
                ),
                method_agreement=bool(fold.method_agreement),
                conviction=str(fold.conviction),
                up_probability_pct=float(fold.probabilities_pct.get("UP", 0.0)),
                sideways_probability_pct=float(fold.probabilities_pct.get("SIDEWAYS", 0.0)),
                down_probability_pct=float(fold.probabilities_pct.get("DOWN", 0.0)),
                expected_return_pct=(
                    None
                    if fold.expected_return_pct is None
                    else float(fold.expected_return_pct)
                ),
                training_observations=int(fold.training_observations),
                state_atoms=state_atoms,
            )
        )

    stock_mismatches = sum(not fold.stock_reconciled for fold in folds)
    benchmark_mismatches = sum(not fold.benchmark_reconciled for fold in folds)
    relative_mismatches = sum(not fold.relative_reconciled for fold in folds)

    return RealOLAPCalculationAudit(
        ticker=str(config.ticker),
        benchmark=str(config.benchmark),
        analysis_timeframe=str(config.analysis_timeframe),
        holding_period_months=float(config.holding_period_months),
        entry_mode=str(config.entry_mode),
        folds=tuple(folds),
        stock_mismatch_count=stock_mismatches,
        benchmark_mismatch_count=benchmark_mismatches,
        relative_mismatch_count=relative_mismatches,
        missing_raw_input_count=missing_raw_inputs,
    )


def export_real_olap_calculation_audit(
    result: Phase5RealOLAPValidationResult,
    config: Any,
    output_dir: str | Path = "phase5_8_audit",
) -> dict[str, Path]:
    """Write JSON + CSV reconciliation artifacts and return their paths."""

    audit = reconcile_real_olap_calculations(result, config)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "phase5_8_calculation_audit.json"
    csv_path = target_dir / "phase5_8_calculation_audit.csv"

    json_path.write_text(
        json.dumps(audit.as_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    fieldnames = [
        "prediction_date",
        "reported_stock_return_pct",
        "reconstructed_stock_return_pct",
        "stock_return_abs_error_pct",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "reported_benchmark_return_pct",
        "reconstructed_benchmark_return_pct",
        "benchmark_return_abs_error_pct",
        "benchmark_entry_date",
        "benchmark_entry_price",
        "benchmark_exit_date",
        "benchmark_exit_price",
        "reported_relative_return_pct",
        "reconstructed_relative_return_pct",
        "relative_return_abs_error_pct",
        "actual_class",
        "combined_trend",
        "method_a_trend",
        "method_b_trend",
        "method_agreement",
        "conviction",
        "up_probability_pct",
        "sideways_probability_pct",
        "down_probability_pct",
        "expected_return_pct",
        "training_observations",
        "stock_reconciled",
        "benchmark_reconciled",
        "relative_reconciled",
        "state_atoms_json",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold in audit.folds:
            payload = fold.as_dict()
            row = {name: payload.get(name) for name in fieldnames if name != "state_atoms_json"}
            row["state_atoms_json"] = json.dumps(dict(fold.state_atoms), sort_keys=True)
            writer.writerow(row)

    return {"json": json_path, "csv": csv_path}
