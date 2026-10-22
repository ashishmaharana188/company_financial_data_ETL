from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Any

from .hardening_4_10 import audit_parameter_timing
from .outcome_labels import OutcomeThresholds, filter_completed_outcomes
from .ranking import RelationshipRanking
from .relationship import HistoricalRelationshipObservation


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass(frozen=True)
class ValidationEvidenceAudit:
    """Explicit audit artifact required before validated evidence may be trusted."""

    source: str
    validation_cutoff_date: date | None
    evaluated_folds: int
    leakage_violations: int
    outer_test_isolated: bool
    parameter_timing_safe: bool
    multiple_testing_controlled: bool
    selection_validated_predictions: int

    @property
    def clean(self) -> bool:
        return (
            self.evaluated_folds > 0
            and self.leakage_violations == 0
            and self.outer_test_isolated
            and self.parameter_timing_safe
            and self.multiple_testing_controlled
            and self.selection_validated_predictions > 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "validation_cutoff_date": self.validation_cutoff_date,
            "evaluated_folds": self.evaluated_folds,
            "leakage_violations": self.leakage_violations,
            "outer_test_isolated": self.outer_test_isolated,
            "parameter_timing_safe": self.parameter_timing_safe,
            "multiple_testing_controlled": self.multiple_testing_controlled,
            "selection_validated_predictions": self.selection_validated_predictions,
            "clean": self.clean,
        }


@dataclass(frozen=True)
class PredictionProvenanceAudit:
    """Phase 5 end-to-end point-in-time and lineage audit."""

    target: str
    cutoff_date: date
    source_count: int
    training_count: int
    latest_training_state_date: date | None
    latest_training_outcome_end_date: date | None

    excluded_target: int
    excluded_future_state: int
    excluded_incomplete_outcome: int
    excluded_unknown_outcome_end: int
    excluded_missing_return: int
    excluded_non_finite_return: int

    threshold_sample_count: int
    threshold_fit_end_date: date | None
    threshold_temporal_violation: bool
    threshold_count_mismatch: bool

    relationship_candidate_count: int
    relationship_parameter_count: int
    relationship_parameter_missing: int
    relationship_parameter_unknown: int
    relationship_parameter_temporal_violations: int
    relationship_support_temporal_violations: int

    validated_evidence_asserted: bool
    validation_evidence_supplied: bool
    validation_evidence_clean: bool
    validation_evidence_temporal_violation: bool

    limitations: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return (
            not self.threshold_temporal_violation
            and not self.threshold_count_mismatch
            and self.relationship_parameter_missing == 0
            and self.relationship_parameter_unknown == 0
            and self.relationship_parameter_temporal_violations == 0
            and self.relationship_support_temporal_violations == 0
            and not (self.validated_evidence_asserted and not self.validation_evidence_clean)
            and not self.validation_evidence_temporal_violation
        )

    @property
    def temporal_violations(self) -> int:
        return (
            int(self.threshold_temporal_violation)
            + self.relationship_parameter_temporal_violations
            + self.relationship_support_temporal_violations
            + int(self.validation_evidence_temporal_violation)
        )

    @property
    def excluded_temporal_rows(self) -> int:
        return self.excluded_future_state + self.excluded_incomplete_outcome

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "cutoff_date": self.cutoff_date,
            "source_count": self.source_count,
            "training_count": self.training_count,
            "latest_training_state_date": self.latest_training_state_date,
            "latest_training_outcome_end_date": self.latest_training_outcome_end_date,
            "excluded_target": self.excluded_target,
            "excluded_future_state": self.excluded_future_state,
            "excluded_incomplete_outcome": self.excluded_incomplete_outcome,
            "excluded_unknown_outcome_end": self.excluded_unknown_outcome_end,
            "excluded_missing_return": self.excluded_missing_return,
            "excluded_non_finite_return": self.excluded_non_finite_return,
            "threshold_sample_count": self.threshold_sample_count,
            "threshold_fit_end_date": self.threshold_fit_end_date,
            "threshold_temporal_violation": self.threshold_temporal_violation,
            "threshold_count_mismatch": self.threshold_count_mismatch,
            "relationship_candidate_count": self.relationship_candidate_count,
            "relationship_parameter_count": self.relationship_parameter_count,
            "relationship_parameter_missing": self.relationship_parameter_missing,
            "relationship_parameter_unknown": self.relationship_parameter_unknown,
            "relationship_parameter_temporal_violations": self.relationship_parameter_temporal_violations,
            "relationship_support_temporal_violations": self.relationship_support_temporal_violations,
            "validated_evidence_asserted": self.validated_evidence_asserted,
            "validation_evidence_supplied": self.validation_evidence_supplied,
            "validation_evidence_clean": self.validation_evidence_clean,
            "validation_evidence_temporal_violation": self.validation_evidence_temporal_violation,
            "excluded_temporal_rows": self.excluded_temporal_rows,
            "temporal_violations": self.temporal_violations,
            "clean": self.clean,
            "limitations": list(self.limitations),
        }


def _training_dates(history: Iterable[HistoricalRelationshipObservation]) -> set[date]:
    return {_as_date(item.as_of_date) for item in history}


def _parameter_entries(rankings: Iterable[RelationshipRanking]) -> tuple[list[tuple[str, date | None, date | None]], int]:
    entries: list[tuple[str, date | None, date | None]] = []
    missing = 0
    for ranking in rankings:
        for result in ranking.method_results:
            provenance = tuple(result.parameter_provenance)
            if not provenance:
                missing += 1
                continue
            for name, fit_start, fit_end in provenance:
                entries.append((
                    f"{result.method}:{name}",
                    _as_date(fit_start) if fit_start is not None else None,
                    _as_date(fit_end) if fit_end is not None else None,
                ))
    return entries, missing


def audit_prediction_provenance(
    *,
    observations: Iterable[HistoricalRelationshipObservation],
    target: str,
    cutoff_date: str | date | datetime,
    training_history: Iterable[HistoricalRelationshipObservation],
    thresholds: OutcomeThresholds,
    ranked_a: Iterable[RelationshipRanking],
    ranked_b: Iterable[RelationshipRanking],
    validated_evidence: bool = False,
    validation_evidence: ValidationEvidenceAudit | None = None,
) -> PredictionProvenanceAudit:
    """Audit every learned/data-derived Phase 5 input before prediction output.

    The audit is intentionally stricter than a simple state-date cutoff:
    forward outcomes must be completed, threshold fitting must end strictly
    before the cutoff, all discovered relationship parameters need provenance,
    supporting observations must be historical training rows, and a claimed
    validation gate must carry an explicit clean validation artifact.
    """
    cutoff = _as_date(cutoff_date)
    source_rows = list(observations)
    history = list(training_history)
    ranked_a_list = list(ranked_a)
    ranked_b_list = list(ranked_b)

    _eligible, training_filter = filter_completed_outcomes(
        source_rows,
        target,
        cutoff,
    )

    training_dates = _training_dates(history)
    latest_state = max(training_dates) if training_dates else None

    outcome_end_dates = [
        _as_date(item.outcome_end_date)
        for item in history
        if item.outcome_end_date is not None
    ]
    latest_outcome_end = max(outcome_end_dates) if outcome_end_dates else None

    finite_completed_count = 0
    for item in history:
        if item.outcome_end_date is None or _as_date(item.outcome_end_date) >= cutoff:
            continue
        try:
            value = float(item.stock_return_pct)
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            finite_completed_count += 1

    threshold_fit_end = latest_outcome_end
    threshold_temporal_violation = (
        threshold_fit_end is not None and threshold_fit_end >= cutoff
    )
    threshold_count_mismatch = thresholds.sample_count != finite_completed_count

    all_rankings = ranked_a_list + ranked_b_list
    parameter_entries, missing_parameters = _parameter_entries(all_rankings)
    parameter_unknown = sum(1 for _name, _fit_start, fit_end in parameter_entries if fit_end is None)
    parameter_audits = audit_parameter_timing(parameter_entries, cutoff)
    # Phase 5 uses strict point-in-time semantics: equality with the cutoff is
    # unsafe even though the older Phase 4 helper only flags dates after it.
    parameter_temporal_violations = sum(
        1 for item in parameter_audits
        if item.fit_end_date is not None and item.fit_end_date >= cutoff
    )

    support_violations = 0
    for ranking in all_rankings:
        for result in ranking.method_results:
            for support_date, _return_pct in result.supporting_observations:
                normalized = _as_date(support_date)
                if normalized >= cutoff or normalized not in training_dates:
                    support_violations += 1

    validation_supplied = validation_evidence is not None
    validation_clean = bool(validation_evidence and validation_evidence.clean)
    validation_temporal_violation = bool(
        validation_evidence
        and validation_evidence.validation_cutoff_date is not None
        and _as_date(validation_evidence.validation_cutoff_date) >= cutoff
    )

    limitations: list[str] = []
    if training_filter.excluded_count:
        limitations.append(
            f"Training filter excluded {training_filter.excluded_count} source rows from the cutoff-safe universe."
        )
    if threshold_temporal_violation:
        limitations.append("Outcome thresholds include information at or after the prediction cutoff.")
    if threshold_count_mismatch:
        limitations.append("Outcome threshold sample count does not match the cutoff-safe finite return universe.")
    if missing_parameters:
        limitations.append(f"{missing_parameters} discovered relationship results have no parameter provenance.")
    if parameter_unknown:
        limitations.append(f"{parameter_unknown} relationship parameters have unknown fit end dates.")
    if parameter_temporal_violations:
        limitations.append("At least one relationship parameter was fit at or after the prediction cutoff.")
    if support_violations:
        limitations.append("At least one relationship support observation is outside the cutoff-safe training universe.")
    if validated_evidence and not validation_clean:
        limitations.append("Validated evidence was asserted without a clean explicit OOS validation audit.")
    if validation_temporal_violation:
        limitations.append("The supplied validation evidence reaches the prediction cutoff or later and cannot be used point-in-time.")

    return PredictionProvenanceAudit(
        target=target,
        cutoff_date=cutoff,
        source_count=len(source_rows),
        training_count=len(history),
        latest_training_state_date=latest_state,
        latest_training_outcome_end_date=latest_outcome_end,
        excluded_target=training_filter.excluded_target,
        excluded_future_state=training_filter.excluded_future_state,
        excluded_incomplete_outcome=training_filter.excluded_incomplete_outcome,
        excluded_unknown_outcome_end=training_filter.excluded_unknown_outcome_end,
        excluded_missing_return=training_filter.excluded_missing_return,
        excluded_non_finite_return=training_filter.excluded_non_finite_return,
        threshold_sample_count=thresholds.sample_count,
        threshold_fit_end_date=threshold_fit_end,
        threshold_temporal_violation=threshold_temporal_violation,
        threshold_count_mismatch=threshold_count_mismatch,
        relationship_candidate_count=len(all_rankings),
        relationship_parameter_count=len(parameter_entries),
        relationship_parameter_missing=missing_parameters,
        relationship_parameter_unknown=parameter_unknown,
        relationship_parameter_temporal_violations=parameter_temporal_violations,
        relationship_support_temporal_violations=support_violations,
        validated_evidence_asserted=validated_evidence,
        validation_evidence_supplied=validation_supplied,
        validation_evidence_clean=validation_clean,
        validation_evidence_temporal_violation=validation_temporal_violation,
        limitations=tuple(limitations),
    )
