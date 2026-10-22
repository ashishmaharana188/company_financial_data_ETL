from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite, log
from typing import Any, Callable, Iterable, Sequence
import logging
import time

from .outcome_labels import OutcomeClass, classify_return
from .prediction import PredictionEngine, PredictionResult
from .ranking import rank_relationships
from .walk_forward_relationship import validate_walk_forward_relationships_hardened
from .relationship import HistoricalRelationshipObservation

try:
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False

if TYPE_CHECKING:
    from .real_olap_validation import RealOLAPValidationConfig


PredictionTrend = str

_LOGGER = logging.getLogger("trendanalysis.phase5.real_olap")


def _object_as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "as_dict"):
        result = value.as_dict()
        return dict(result) if isinstance(result, dict) else None
    return None


def _method_selection_record(
    method_name: str,
    *,
    source: str,
    selected_relationship: Any = None,
    ranking: Any = None,
    prediction_method: Any = None,
    gate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_record = _object_as_dict(selected_relationship)
    ranking_record = _object_as_dict(ranking)
    method_record = _object_as_dict(prediction_method)
    if source == "nested_hardened_walk_forward":
        selected = selected_relationship is not None
        status = (
            "SELECTED_AFTER_ADJUSTED_P_THRESHOLD"
            if selected else "NO_RELATIONSHIP_PASSED_SELECTION_GATE"
        )
    else:
        selected = None  # Direct mode does not expose an independent nested selection result.
        status = (
            "METHOD_OUTPUT_AVAILABLE_SELECTION_STAGE_NOT_EXPOSED"
            if method_record is not None else "NO_METHOD_OUTPUT"
        )
    return {
        "method": method_name,
        "selection_source": source,
        "selection_status": status,
        "selected_after_selection_gate": selected,
        "selected_relationship": selected_record,
        "ranking": ranking_record,
        "prediction_method_output": method_record,
        "gate_metadata": dict(gate_metadata) if gate_metadata is not None else None,
    }


def _fold_selection_metadata(
    config: Any,
    *,
    nested_fold: Any = None,
    selected_a: Any = None,
    selected_b: Any = None,
    ranking_a: Any = None,
    ranking_b: Any = None,
    method_a: Any = None,
    method_b: Any = None,
    nested_run_summary: Any = None,
) -> dict[str, Any]:
    source = "nested_hardened_walk_forward" if nested_fold is not None else "prediction_engine_direct"
    nested_dict = _object_as_dict(nested_fold)
    return {
        "recording_contract_version": 2,
        "selection_mode": source,
        "selection_fraction": getattr(config, "selection_fraction", None),
        "multiple_testing_alpha": getattr(config, "multiple_testing_alpha", None),
        "purge_overlapping_labels": True if nested_fold is not None else None,
        "method_a": _method_selection_record(
            "A", source=source, selected_relationship=selected_a, ranking=ranking_a,
            prediction_method=method_a,
            gate_metadata=(nested_dict or {}).get("method_a_selection_metadata"),
        ),
        "method_b": _method_selection_record(
            "B", source=source, selected_relationship=selected_b, ranking=ranking_b,
            prediction_method=method_b,
            gate_metadata=(nested_dict or {}).get("method_b_selection_metadata"),
        ),
        "nested_selection_fold": nested_dict,
        "nested_run_summary": (
            {
                key: getattr(nested_run_summary, key)
                for key in (
                    "selection_candidate_evaluations",
                    "selection_validated_predictions",
                    "multiple_testing_controlled_folds",
                    "leakage_violations",
                    "purged_training_observations",
                    "unknown_overlap_observations",
                )
                if hasattr(nested_run_summary, key)
            }
            if nested_run_summary is not None else None
        ),
    }


def _configure_phase5_progress(enabled: bool) -> None:
    if not enabled:
        return
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[Phase 5.8] %(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def _phase5_progress(enabled: bool, message: str, *args: Any) -> None:
    if enabled:
        _LOGGER.info(message, *args)


@dataclass(frozen=True)
class PredictionFoldResult:
    prediction_date: date
    observed_trend: PredictionTrend
    actual_return_pct: float
    actual_class: OutcomeClass | None
    predicted_trend: PredictionTrend
    conviction: str
    probabilities_pct: dict[OutcomeClass, float]
    baseline_probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    training_observations: int
    method_a_trend: PredictionTrend | None
    method_b_trend: PredictionTrend | None
    method_agreement: bool
    limited: bool
    provenance_clean: bool
    hit: bool | None
    trade_eligible: bool
    trade_reason: str
    method_a_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)
    method_b_probabilities_pct: dict[OutcomeClass, float] = field(default_factory=dict)
    benchmark_return_pct: float | None = None
    relative_return_pct: float | None = None
    outcome_end_date: date | None = None
    current_states: dict[str, str] = field(default_factory=dict)
    method_a_output: dict[str, Any] | None = None
    method_b_output: dict[str, Any] | None = None
    method_a_selection_metadata: dict[str, Any] | None = None
    method_b_selection_metadata: dict[str, Any] | None = None
    selection_metadata: dict[str, Any] = field(default_factory=dict)
    outcome_thresholds: dict[str, Any] = field(default_factory=dict)
    prediction_provenance: dict[str, Any] | None = None
    prediction_result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        method_a_output = self.method_a_output if isinstance(self.method_a_output, dict) else {}
        method_b_output = self.method_b_output if isinstance(self.method_b_output, dict) else {}
        return {
            "prediction_date": self.prediction_date,
            "observed_trend": self.observed_trend,
            "actual_return_pct": self.actual_return_pct,
            "actual_class": self.actual_class,
            "predicted_trend": self.predicted_trend,
            "conviction": self.conviction,
            "probabilities_pct": dict(self.probabilities_pct),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "training_observations": self.training_observations,
            "method_a_trend": self.method_a_trend,
            "method_b_trend": self.method_b_trend,
            "method_a_probabilities_pct": dict(self.method_a_probabilities_pct),
            "method_b_probabilities_pct": dict(self.method_b_probabilities_pct),
            "method_a_output": dict(self.method_a_output) if self.method_a_output is not None else None,
            "method_b_output": dict(self.method_b_output) if self.method_b_output is not None else None,
            "method_a_evidence_score": method_a_output.get("evidence_score"),
            "method_b_evidence_score": method_b_output.get("evidence_score"),
            "method_a_effective_sample_size": method_a_output.get("effective_sample_size"),
            "method_b_effective_sample_size": method_b_output.get("effective_sample_size"),
            "method_a_sample_count": method_a_output.get("sample_count"),
            "method_b_sample_count": method_b_output.get("sample_count"),
            "method_a_variables": list(method_a_output.get("variables") or []),
            "method_b_variables": list(method_b_output.get("variables") or []),
            "method_a_condition": list(method_a_output.get("condition") or []),
            "method_b_condition": list(method_b_output.get("condition") or []),
            "method_a_selection_metadata": self.method_a_selection_metadata,
            "method_b_selection_metadata": self.method_b_selection_metadata,
            "selection_metadata": dict(self.selection_metadata),
            "outcome_thresholds": dict(self.outcome_thresholds),
            "prediction_provenance": dict(self.prediction_provenance) if self.prediction_provenance is not None else None,
            "prediction_result": dict(self.prediction_result) if self.prediction_result is not None else None,
            "benchmark_return_pct": self.benchmark_return_pct,
            "relative_return_pct": self.relative_return_pct,
            "outcome_end_date": self.outcome_end_date,
            "current_states": dict(self.current_states),
            "fold_recording_contract_version": 2,
            "method_agreement": self.method_agreement,
            "limited": self.limited,
            "provenance_clean": self.provenance_clean,
            "hit": self.hit,
            "trade_eligible": self.trade_eligible,
            "trade_reason": self.trade_reason,
        }


@dataclass(frozen=True)
class PredictionMethodSummary:
    method: str
    evaluated_predictions: int
    current_directional_opportunities: int
    trade_eligible_predictions: int
    trade_hits: int
    trade_hit_rate_pct: float
    directional_predictions: int
    directional_hits: int
    directional_hit_rate_pct: float
    clear_trend_predictions: int
    clear_trend_hits: int
    clear_trend_accuracy_pct: float
    no_clear_trend_predictions: int
    limited_predictions: int
    coverage_pct: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Phase5RealOLAPValidationResult:
    ticker: str
    benchmark: str
    analysis_timeframe: str
    holding_period_months: float
    entry_mode: str
    candidate_predictions: int
    evaluated_predictions: int
    skipped_threshold_limited: int
    skipped_provenance_failed: int
    skipped_invalid_actual: int
    prediction_folds: tuple[PredictionFoldResult, ...]
    method_a: PredictionMethodSummary
    method_b: PredictionMethodSummary
    combined: PredictionMethodSummary
    baseline_majority_accuracy_pct: float
    mean_combined_probabilities_pct: dict[OutcomeClass, float]
    mean_combined_expected_return_pct: float | None
    conviction_counts: dict[str, int]
    latest_validated_prediction_date: date | None
    latest_validated_prediction: dict[str, Any] | None
    latest_market_date: date | None
    phase4_surface_audit: dict[str, Any] | None
    selection_candidate_evaluations: int = 0
    selection_validated_predictions: int = 0
    multiple_testing_controlled_folds: int = 0
    # Exact Phase 4 panel rows used by this Phase 5.8 run. Kept out of
    # as_dict() because they can be large; the audit exporter consumes them
    # directly to reconcile raw outcomes and preserve exact state atoms.
    panel_observations: tuple[HistoricalRelationshipObservation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "benchmark": self.benchmark,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "entry_mode": self.entry_mode,
            "candidate_predictions": self.candidate_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "skipped_threshold_limited": self.skipped_threshold_limited,
            "skipped_provenance_failed": self.skipped_provenance_failed,
            "skipped_invalid_actual": self.skipped_invalid_actual,
            "prediction_folds": [fold.as_dict() for fold in self.prediction_folds],
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "baseline_majority_accuracy_pct": self.baseline_majority_accuracy_pct,
            "mean_combined_probabilities_pct": dict(self.mean_combined_probabilities_pct),
            "mean_combined_expected_return_pct": self.mean_combined_expected_return_pct,
            "conviction_counts": dict(self.conviction_counts),
            "latest_validated_prediction_date": self.latest_validated_prediction_date,
            "latest_validated_prediction": self.latest_validated_prediction,
            "latest_market_date": self.latest_market_date,
            "phase4_surface_audit": self.phase4_surface_audit,
            "selection_candidate_evaluations": self.selection_candidate_evaluations,
            "selection_validated_predictions": self.selection_validated_predictions,
            "multiple_testing_controlled_folds": self.multiple_testing_controlled_folds,
            "panel_observation_count": len(self.panel_observations),
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _percent(value: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (value / denominator) * 100.0


def _label_direction(trend: str | None) -> bool:
    return trend in {"UP", "DOWN"}


def _observed_company_trend(states: dict[str, Any]) -> PredictionTrend:
    """Map the point-in-time Phase 3 company price direction to UP/DOWN/SIDEWAYS."""
    raw = states.get("company.market.price")
    if not isinstance(raw, str):
        return "NO_CLEAR_TREND"
    direction = raw.split("/", 1)[0].strip().lower()
    return {
        "rising": "UP",
        "falling": "DOWN",
        "stable": "SIDEWAYS",
    }.get(direction, "NO_CLEAR_TREND")


def _trade_gate(
    observed_trend: PredictionTrend,
    predicted_trend: PredictionTrend,
) -> tuple[bool, str]:
    """A trade requires a directional current state and model agreement."""
    if observed_trend == "SIDEWAYS":
        return False, "SIDEWAYS_CURRENT_TREND"
    if observed_trend not in {"UP", "DOWN"}:
        return False, "CURRENT_TREND_UNAVAILABLE"
    if predicted_trend != observed_trend:
        return False, "MODEL_CONFLICT"
    return True, "DIRECTIONAL_AGREEMENT"


def _directional_hit(predicted: str | None, actual: OutcomeClass | None) -> bool | None:
    if predicted not in {"UP", "DOWN"} or actual is None:
        return None
    return predicted == actual


def _summary(
    method: str,
    folds: Sequence[PredictionFoldResult],
    trend_getter: Callable[[PredictionFoldResult], str | None],
) -> PredictionMethodSummary:
    usable = [fold for fold in folds if not fold.limited and fold.provenance_clean]
    current_directional = [fold for fold in usable if _label_direction(fold.observed_trend)]
    trade_eligible = [
        fold for fold in usable
        if _trade_gate(fold.observed_trend, trend_getter(fold))[0]
    ]
    trade_hits = [
        fold for fold in trade_eligible
        if fold.actual_class == fold.observed_trend
    ]
    directional = [fold for fold in usable if _label_direction(trend_getter(fold))]
    directional_hits = [fold for fold in directional if _directional_hit(trend_getter(fold), fold.actual_class) is True]
    clear = [fold for fold in usable if trend_getter(fold) in {"UP", "SIDEWAYS", "DOWN"}]
    clear_hits = [fold for fold in clear if trend_getter(fold) == fold.actual_class]
    no_clear = [fold for fold in usable if trend_getter(fold) == "NO_CLEAR_TREND"]
    limited = [fold for fold in folds if fold.limited or not fold.provenance_clean]

    return PredictionMethodSummary(
        method=method,
        evaluated_predictions=len(usable),
        current_directional_opportunities=len(current_directional),
        trade_eligible_predictions=len(trade_eligible),
        trade_hits=len(trade_hits),
        trade_hit_rate_pct=_percent(len(trade_hits), len(trade_eligible)),
        directional_predictions=len(directional),
        directional_hits=len(directional_hits),
        directional_hit_rate_pct=_percent(len(directional_hits), len(directional)),
        clear_trend_predictions=len(clear),
        clear_trend_hits=len(clear_hits),
        clear_trend_accuracy_pct=_percent(len(clear_hits), len(clear)),
        no_clear_trend_predictions=len(no_clear),
        limited_predictions=len(limited),
        coverage_pct=_percent(len(clear), len(folds)),
    )


def _majority_baseline_accuracy(
    folds: Sequence[PredictionFoldResult],
) -> float:
    actual = [fold.actual_class for fold in folds if fold.actual_class is not None]
    if not actual:
        return 0.0
    counts = {label: actual.count(label) for label in ("UP", "SIDEWAYS", "DOWN")}
    return _percent(max(counts.values()), len(actual))


def _mean_probabilities(folds: Sequence[PredictionFoldResult]) -> dict[OutcomeClass, float]:
    usable = [fold for fold in folds if not fold.limited and fold.provenance_clean]
    if not usable:
        return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}
    return {
        label: sum(fold.probabilities_pct.get(label, 0.0) for fold in usable) / len(usable)
        for label in ("UP", "SIDEWAYS", "DOWN")
    }


def _mean_expected_return(folds: Sequence[PredictionFoldResult]) -> float | None:
    values = [
        float(fold.expected_return_pct)
        for fold in folds
        if not fold.limited
        and fold.provenance_clean
        and fold.expected_return_pct is not None
        and isfinite(float(fold.expected_return_pct))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _assert_fold_temporal_safety(
    result: PredictionResult,
    cutoff: date,
    source: Iterable[HistoricalRelationshipObservation],
) -> None:
    """Raise if the Phase 5 output contradicts its own cutoff contract."""
    audit = result.provenance_audit
    if audit is None:
        raise AssertionError("Phase 5.7 provenance audit is missing from PredictionResult.")
    if not audit.clean:
        return
    for row in source:
        if row.target != result.target:
            continue
        state_date = _as_date(row.as_of_date)
        if state_date >= cutoff:
            continue
        if row.outcome_end_date is None:
            continue
        if _as_date(row.outcome_end_date) >= cutoff:
            continue
    if audit.threshold_fit_end_date is not None and _as_date(audit.threshold_fit_end_date) >= cutoff:
        raise AssertionError("Prediction thresholds reach or exceed the prediction cutoff.")
    if audit.latest_training_outcome_end_date is not None and _as_date(audit.latest_training_outcome_end_date) >= cutoff:
        raise AssertionError("Training outcome reaches or exceeds the prediction cutoff.")


def validate_real_olap_predictions(
    config: RealOLAPValidationConfig,
    *,
    panel_builder: Callable[[RealOLAPValidationConfig], tuple[list[HistoricalRelationshipObservation], dict[str, Any], date | None]] | None = None,
    prediction_engine: PredictionEngine | None = None,
) -> Phase5RealOLAPValidationResult:
    """Run Phase 5 predictions on the real Phase 4 OLAP state/outcome panel."""

    if panel_builder is None:
        from .real_olap_validation import build_real_olap_relationship_panel
        panel_builder = build_real_olap_relationship_panel

    observations, panel_stats, latest_market_date = panel_builder(config)
    observations = sorted(observations, key=lambda row: _as_date(row.as_of_date))
    candidate_predictions = len(observations)

    _configure_phase5_progress(getattr(config, "progress_logging", True))

    if prediction_engine is None and bool(getattr(config, "hardened_validation", True)):
        _phase5_progress(
            getattr(config, "progress_logging", True),
            "Nested selection start | exhaustive=%s | selection_fraction=%.2f | alpha=%.2f",
            not bool(getattr(config, "adaptive_relationship_search", False)),
            float(getattr(config, "selection_fraction", 0.30)),
            float(getattr(config, "multiple_testing_alpha", 0.10)),
        )
        from .relationship import RelationshipDiscoveryEngine
        selector_engine = RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=config.min_training_observations,
            adaptive_higher_order=bool(getattr(config, "adaptive_relationship_search", False)),
            progress_callback=lambda message: _phase5_progress(getattr(config, "progress_logging", True), message),
            progress_every_candidates=int(getattr(config, "relationship_progress_every_candidates", 1000)),
            candidate_batch_size=int(getattr(config, "relationship_candidate_batch_size", 512)),
            retain_supporting_data=True,
        )
        nested = validate_walk_forward_relationships_hardened(
            observations,
            selector_engine,
            min_training_observations=config.min_training_observations,
            selection_fraction=float(getattr(config, "selection_fraction", 0.30)),
            multiple_testing_alpha=float(getattr(config, "multiple_testing_alpha", 0.10)),
            purge_overlapping_labels=True,
        )
        prediction_engine = PredictionEngine(
            relationship_engine=selector_engine,
            min_threshold_observations=config.min_training_observations,
        )
        folds: list[PredictionFoldResult] = []
        skipped_threshold_limited = 0
        skipped_provenance_failed = 0
        skipped_invalid_actual = 0
        conviction_counts = {key: 0 for key in ("STRONG", "MODERATE", "LOW", "NONE")}

        nested_by_date = {fold.prediction_date: fold for fold in nested.folds}
        for fold_index, observation in enumerate(observations, start=1):
            cutoff = _as_date(observation.as_of_date)
            if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
                _phase5_progress(
                    getattr(config, "progress_logging", True),
                    "Nested fold %d/%d | prediction_date=%s",
                    fold_index, candidate_predictions, cutoff,
                )
            selected_fold = nested_by_date.get(cutoff)
            if selected_fold is None:
                continue
            selected_a = getattr(selected_fold, "selected_a_result", None)
            selected_b = getattr(selected_fold, "selected_b_result", None)
            ranking_a = rank_relationships([selected_a])[0] if selected_a is not None else None
            ranking_b = rank_relationships([selected_b])[0] if selected_b is not None else None
            result = prediction_engine.predict(
                target=observation.target,
                current_states=dict(observation.states),
                observations=observations,
                prediction_date=cutoff,
                analysis_timeframe=config.analysis_timeframe,
                holding_period_months=config.holding_period_months,
                benchmark=config.benchmark,
                entry_mode=config.entry_mode,
                validated_evidence=False,
                validation_evidence=None,
                selected_relationships={"A": ranking_a, "B": ranking_b},
            )
            observed_trend = _observed_company_trend(dict(observation.states))
            trade_eligible, trade_reason = _trade_gate(observed_trend, result.trend)
            thresholds = result.outcome_thresholds
            if thresholds.limited:
                skipped_threshold_limited += 1
                continue
            audit = result.provenance_audit
            provenance_clean = bool(audit and audit.clean)
            if not provenance_clean:
                skipped_provenance_failed += 1
            actual_numeric = observation.stock_return_pct
            if actual_numeric is None:
                skipped_invalid_actual += 1
                continue
            actual_numeric = float(actual_numeric)
            actual_class = classify_return(actual_numeric, thresholds)
            hit = _directional_hit(result.trend, actual_class)
            conviction_counts[result.conviction] = conviction_counts.get(result.conviction, 0) + 1
            selection_record = _fold_selection_metadata(
                config,
                nested_fold=selected_fold,
                selected_a=selected_a,
                selected_b=selected_b,
                ranking_a=ranking_a,
                ranking_b=ranking_b,
                method_a=result.method_a,
                method_b=result.method_b,
                nested_run_summary=nested,
            )
            folds.append(
                PredictionFoldResult(
                    prediction_date=cutoff,
                    observed_trend=observed_trend,
                    actual_return_pct=actual_numeric,
                    actual_class=actual_class,
                    predicted_trend=result.trend,
                    conviction=result.conviction,
                    probabilities_pct=dict(result.probabilities_pct),
                    baseline_probabilities_pct=dict(result.baseline_probabilities_pct),
                    expected_return_pct=result.expected_return_pct,
                    training_observations=result.training_observations,
                    method_a_trend=result.method_a.trend if result.method_a else None,
                    method_b_trend=result.method_b.trend if result.method_b else None,
                    method_a_probabilities_pct=(
                        dict(result.method_a.probabilities_pct) if result.method_a else {}
                    ),
                    method_b_probabilities_pct=(
                        dict(result.method_b.probabilities_pct) if result.method_b else {}
                    ),
                    benchmark_return_pct=observation.benchmark_return_pct,
                    relative_return_pct=observation.relative_return_pct,
                    outcome_end_date=observation.outcome_end_date,
                    current_states=dict(observation.states),
                    method_a_output=_object_as_dict(result.method_a),
                    method_b_output=_object_as_dict(result.method_b),
                    method_a_selection_metadata=selection_record["method_a"],
                    method_b_selection_metadata=selection_record["method_b"],
                    selection_metadata=selection_record,
                    outcome_thresholds=thresholds.as_dict(),
                    prediction_provenance=_object_as_dict(audit),
                    prediction_result=_object_as_dict(result),
                    method_agreement=result.method_agreement,
                    limited=result.limited,
                    provenance_clean=provenance_clean,
                    hit=hit,
                    trade_eligible=trade_eligible,
                    trade_reason=trade_reason,
                )
            )

        method_a = _summary("A", folds, lambda fold: fold.method_a_trend)
        method_b = _summary("B", folds, lambda fold: fold.method_b_trend)
        combined = _summary("COMBINED", folds, lambda fold: fold.predicted_trend)
        latest_fold = folds[-1] if folds else None
        return Phase5RealOLAPValidationResult(
            ticker=config.ticker, benchmark=config.benchmark,
            analysis_timeframe=config.analysis_timeframe,
            holding_period_months=config.holding_period_months,
            entry_mode=config.entry_mode,
            candidate_predictions=candidate_predictions,
            evaluated_predictions=len(folds),
            skipped_threshold_limited=skipped_threshold_limited,
            skipped_provenance_failed=skipped_provenance_failed,
            skipped_invalid_actual=skipped_invalid_actual,
            prediction_folds=tuple(folds),
            method_a=method_a,
            method_b=method_b,
            combined=combined,
            baseline_majority_accuracy_pct=_majority_baseline_accuracy(folds),
            mean_combined_probabilities_pct=_mean_probabilities(folds),
            mean_combined_expected_return_pct=_mean_expected_return(folds),
            conviction_counts=conviction_counts,
            latest_validated_prediction_date=latest_fold.prediction_date if latest_fold else None,
            latest_validated_prediction=latest_fold.as_dict() if latest_fold else None,
            latest_market_date=latest_market_date,
            phase4_surface_audit=panel_stats.get("state_surface_audit"),
            selection_candidate_evaluations=nested.selection_candidate_evaluations,
            selection_validated_predictions=nested.selection_validated_predictions,
            multiple_testing_controlled_folds=nested.multiple_testing_controlled_folds,
            panel_observations=tuple(observations),
        )

    if prediction_engine is None:
        progress_every_candidates = max(1, int(getattr(config, "relationship_progress_every_candidates", 500)))

        def _relationship_progress(message: str) -> None:
            _phase5_progress(getattr(config, "progress_logging", True), message)

        from .relationship import RelationshipDiscoveryEngine
        relationship_engine = RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=config.min_training_observations,
            adaptive_higher_order=bool(getattr(config, "adaptive_relationship_search", True)),
            progress_callback=_relationship_progress,
            progress_every_candidates=progress_every_candidates,
            candidate_batch_size=int(getattr(config, "relationship_candidate_batch_size", 512)),
            retain_supporting_data=False,
        )
        prediction_engine = PredictionEngine(
            relationship_engine=relationship_engine,
            min_threshold_observations=config.min_training_observations,
        )

    _phase5_progress(
        getattr(config, "progress_logging", True),
        "Prediction validation start | folds=%d | exhaustive_search=YES | batch=%d",
        candidate_predictions, int(getattr(config, "relationship_candidate_batch_size", 512)),
    )

    folds: list[PredictionFoldResult] = []
    skipped_threshold_limited = 0
    skipped_provenance_failed = 0
    skipped_invalid_actual = 0
    conviction_counts = {key: 0 for key in ("STRONG", "MODERATE", "LOW", "NONE")}

    for fold_index, observation in enumerate(observations, start=1):
        cutoff = _as_date(observation.as_of_date)
        actual_return = observation.stock_return_pct
        if actual_return is None:
            skipped_invalid_actual += 1
            continue
        try:
            actual_numeric = float(actual_return)
        except (TypeError, ValueError):
            skipped_invalid_actual += 1
            continue
        if not isfinite(actual_numeric):
            skipped_invalid_actual += 1
            continue

        fold_started = time.perf_counter()
        if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
            _phase5_progress(
                getattr(config, "progress_logging", True),
                "Fold start %d/%d | prediction_date=%s | prior_completed=%d",
                fold_index, candidate_predictions, cutoff, len(folds),
            )

        result = prediction_engine.predict(
            target=observation.target,
            current_states=dict(observation.states),
            observations=observations,
            prediction_date=cutoff,
            analysis_timeframe=config.analysis_timeframe,
            holding_period_months=config.holding_period_months,
            benchmark=config.benchmark,
            entry_mode=config.entry_mode,
            validated_evidence=False,
            validation_evidence=None,
        )

        observed_trend = _observed_company_trend(dict(observation.states))
        trade_eligible, trade_reason = _trade_gate(observed_trend, result.trend)
        _assert_fold_temporal_safety(result, cutoff, observations)
        thresholds = result.outcome_thresholds
        if thresholds.limited:
            skipped_threshold_limited += 1
            continue

        audit = result.provenance_audit
        provenance_clean = bool(audit and audit.clean)
        if not provenance_clean:
            skipped_provenance_failed += 1

        actual_class = classify_return(actual_numeric, thresholds)
        hit = _directional_hit(result.trend, actual_class)
        conviction_counts[result.conviction] = conviction_counts.get(result.conviction, 0) + 1
        selection_record = _fold_selection_metadata(
            config,
            method_a=result.method_a,
            method_b=result.method_b,
        )

        folds.append(
            PredictionFoldResult(
                prediction_date=cutoff,
                observed_trend=observed_trend,
                actual_return_pct=actual_numeric,
                actual_class=actual_class,
                predicted_trend=result.trend,
                conviction=result.conviction,
                probabilities_pct=dict(result.probabilities_pct),
                baseline_probabilities_pct=dict(result.baseline_probabilities_pct),
                expected_return_pct=result.expected_return_pct,
                training_observations=result.training_observations,
                method_a_trend=result.method_a.trend if result.method_a else None,
                method_b_trend=result.method_b.trend if result.method_b else None,
                method_a_probabilities_pct=(
                    dict(result.method_a.probabilities_pct) if result.method_a else {}
                ),
                method_b_probabilities_pct=(
                    dict(result.method_b.probabilities_pct) if result.method_b else {}
                ),
                benchmark_return_pct=observation.benchmark_return_pct,
                relative_return_pct=observation.relative_return_pct,
                outcome_end_date=observation.outcome_end_date,
                current_states=dict(observation.states),
                method_a_output=_object_as_dict(result.method_a),
                method_b_output=_object_as_dict(result.method_b),
                method_a_selection_metadata=selection_record["method_a"],
                method_b_selection_metadata=selection_record["method_b"],
                selection_metadata=selection_record,
                outcome_thresholds=thresholds.as_dict(),
                prediction_provenance=_object_as_dict(audit),
                prediction_result=_object_as_dict(result),
                method_agreement=result.method_agreement,
                limited=result.limited,
                provenance_clean=provenance_clean,
                hit=hit,
                trade_eligible=trade_eligible,
                trade_reason=trade_reason,
            )
        )

        if fold_index == 1 or fold_index % max(1, getattr(config, "progress_every", 10)) == 0 or fold_index == candidate_predictions:
            _phase5_progress(
                getattr(config, "progress_logging", True),
                "Fold complete %d/%d | elapsed=%.2fs | predicted=%s | A=%s B=%s | training=%d | candidates=%d",
                fold_index, candidate_predictions, time.perf_counter() - fold_started, result.trend,
                result.method_a.trend if result.method_a else None,
                result.method_b.trend if result.method_b else None,
                result.training_observations,
                (result.provenance_audit.relationship_candidate_count if result.provenance_audit else 0),
            )

    method_a = _summary("A", folds, lambda fold: fold.method_a_trend)
    method_b = _summary("B", folds, lambda fold: fold.method_b_trend)
    combined = _summary("COMBINED", folds, lambda fold: fold.predicted_trend)
    latest_fold = folds[-1] if folds else None
    return Phase5RealOLAPValidationResult(
        ticker=config.ticker,
        benchmark=config.benchmark,
        analysis_timeframe=config.analysis_timeframe,
        holding_period_months=config.holding_period_months,
        entry_mode=config.entry_mode,
        candidate_predictions=candidate_predictions,
        evaluated_predictions=len(folds),
        skipped_threshold_limited=skipped_threshold_limited,
        skipped_provenance_failed=skipped_provenance_failed,
        skipped_invalid_actual=skipped_invalid_actual,
        prediction_folds=tuple(folds),
        method_a=method_a,
        method_b=method_b,
        combined=combined,
        baseline_majority_accuracy_pct=_majority_baseline_accuracy(folds),
        mean_combined_probabilities_pct=_mean_probabilities(folds),
        mean_combined_expected_return_pct=_mean_expected_return(folds),
        conviction_counts=conviction_counts,
        latest_validated_prediction_date=latest_fold.prediction_date if latest_fold else None,
        latest_validated_prediction=latest_fold.as_dict() if latest_fold else None,
        latest_market_date=latest_market_date,
        phase4_surface_audit=panel_stats.get("state_surface_audit"),
        panel_observations=tuple(observations),
    )


def print_real_olap_prediction_report(result: Phase5RealOLAPValidationResult) -> None:
    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION")
    print("Ticker:", result.ticker)
    print("Benchmark:", result.benchmark)
    print("Analysis timeframe:", result.analysis_timeframe)
    print("Holding period (months):", result.holding_period_months)
    print("Entry mode:", result.entry_mode)
    print("Candidate predictions:", result.candidate_predictions)
    print("Evaluated predictions:", result.evaluated_predictions)
    print("Skipped threshold-limited:", result.skipped_threshold_limited)
    print("Skipped provenance-failed:", result.skipped_provenance_failed)
    print("Skipped invalid actual:", result.skipped_invalid_actual)
    print()
    for summary in (result.method_a, result.method_b, result.combined):
        print(f"Method {summary.method}:")
        print("  current directional opportunities:", summary.current_directional_opportunities)
        print("  trade-eligible predictions:", summary.trade_eligible_predictions)
        print("  trade hits:", summary.trade_hits)
        print("  trade hit rate:", round(summary.trade_hit_rate_pct, 2))
        print("  directional predictions:", summary.directional_predictions)
        print("  directional hits:", summary.directional_hits)
        print("  directional hit rate:", round(summary.directional_hit_rate_pct, 2))
        print("  clear-trend predictions:", summary.clear_trend_predictions)
        print("  clear-trend accuracy:", round(summary.clear_trend_accuracy_pct, 2))
        print("  NO_CLEAR_TREND:", summary.no_clear_trend_predictions)
        print("  limited:", summary.limited_predictions)
        print("  coverage:", round(summary.coverage_pct, 2))
    print()
    print("Baseline majority accuracy:", round(result.baseline_majority_accuracy_pct, 2))
    print("Mean combined probabilities:", {k: round(v, 2) for k, v in result.mean_combined_probabilities_pct.items()})
    print("Mean combined expected return:", None if result.mean_combined_expected_return_pct is None else round(result.mean_combined_expected_return_pct, 4))
    print("Conviction counts:", result.conviction_counts)
    print("Selection candidate evaluations:", result.selection_candidate_evaluations)
    print("Selection-validated predictions:", result.selection_validated_predictions)
    print("Multiple-testing-controlled folds:", result.multiple_testing_controlled_folds)
    print("Latest validated prediction date:", result.latest_validated_prediction_date)
    print("Latest market date:", result.latest_market_date)
    print("NOTE: Phase 5.8 is diagnostic. Probability calibration and full OOS performance remain Phase 6.")
