from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from math import sqrt
from typing import Any, Iterable, Literal

from .method_a_prediction import build_method_a_probability
from .method_b_prediction import build_method_b_probability
from .outcome_labels import (
    OutcomeClass,
    OutcomeThresholds,
    classify_return,
    filter_completed_outcomes,
    learn_outcome_thresholds,
)
from .decision import DecisionResult, decide_baseline_relative
from .conviction import ConvictionResult, assess_conviction
from .method_combination import CombinationMethodInput, CombinationResult, combine_method_probabilities
from .prediction_hardening import PredictionProvenanceAudit, ValidationEvidenceAudit, audit_prediction_provenance
from .ranking import RelationshipRanking, rank_relationships
from .relationship import HistoricalRelationshipObservation, RelationshipDiscoveryEngine

PredictionTrend = Literal["UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"]
Conviction = Literal["STRONG", "MODERATE", "LOW", "NONE"]


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _empty_probabilities() -> dict[OutcomeClass, float]:
    return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def _normalize_probabilities(values: dict[OutcomeClass, float]) -> dict[OutcomeClass, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        return _empty_probabilities()
    return {
        key: max(float(value), 0.0) / total * 100.0
        for key, value in values.items()
    }


def _baseline_probabilities(
    history: list[HistoricalRelationshipObservation],
    thresholds: OutcomeThresholds,
) -> dict[OutcomeClass, float]:
    counts = _empty_probabilities()
    if thresholds.limited:
        return counts
    for observation in history:
        label = classify_return(observation.stock_return_pct, thresholds)
        counts[label] += 1.0
    return _normalize_probabilities(counts)


def _support_distribution(
    ranking: RelationshipRanking,
    thresholds: OutcomeThresholds,
) -> tuple[dict[OutcomeClass, float], float]:
    values = ranking.method_results
    if not values:
        return _empty_probabilities(), 0.0

    best = max(values, key=lambda item: abs(item.score))
    if not best.supporting_observations or thresholds.limited:
        return _empty_probabilities(), 0.0

    probabilities = _empty_probabilities()
    weights = list(best.supporting_weights)
    if len(weights) != len(best.supporting_observations):
        weights = [1.0] * len(best.supporting_observations)

    total_weight = sum(max(weight, 0.0) for weight in weights)
    if total_weight <= 0:
        return _empty_probabilities(), 0.0

    for (_support_date, return_pct), weight in zip(
        best.supporting_observations,
        weights,
    ):
        if weight <= 0:
            continue
        label = classify_return(float(return_pct), thresholds)
        probabilities[label] += float(weight)

    return _normalize_probabilities(probabilities), total_weight


def _top_direction(probabilities: dict[OutcomeClass, float]) -> OutcomeClass | None:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not ordered or ordered[0][1] <= 0:
        return None
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _directional_signal(
    probabilities: dict[OutcomeClass, float],
    baseline: dict[OutcomeClass, float],
    effective_n: float,
) -> PredictionTrend:
    """Backward-compatible adapter around the explicit Phase 5.4 decision layer."""
    return decide_baseline_relative(probabilities, baseline, effective_n).trend


@dataclass(frozen=True)
class MethodPrediction:
    method: str
    trend: PredictionTrend
    probabilities_pct: dict[OutcomeClass, float]
    baseline_probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float
    baseline_return_pct: float
    return_lift_pct: float
    evidence_score: float
    reliability: float
    sample_count: int
    effective_sample_size: float
    stable: bool
    variables: tuple[str, ...]
    condition: tuple[str, ...]
    limited: bool = False
    limitations: tuple[str, ...] = ()
    decision: DecisionResult | None = None
    combination: CombinationResult | None = None
    conviction_result: ConvictionResult | None = None
    # Source probability-builder diagnostics. These fields are audit-only;
    # they do not change probability values or decision behavior.
    class_counts: dict[OutcomeClass, int] = field(default_factory=dict)
    weighted_class_counts: dict[OutcomeClass, float] = field(default_factory=dict)
    probability_basis: str | None = None
    exact_condition_count: int | None = None
    weight_concentration: float | None = None
    stability_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trend": self.trend,
            "probabilities_pct": dict(self.probabilities_pct),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "baseline_return_pct": self.baseline_return_pct,
            "return_lift_pct": self.return_lift_pct,
            "evidence_score": self.evidence_score,
            "reliability": self.reliability,
            "sample_count": self.sample_count,
            "effective_sample_size": self.effective_sample_size,
            "stable": self.stable,
            "variables": list(self.variables),
            "condition": list(self.condition),
            "limited": self.limited,
            "limitations": list(self.limitations),
            "decision": self.decision.as_dict() if self.decision else None,
            "class_counts": dict(self.class_counts),
            "weighted_class_counts": dict(self.weighted_class_counts),
            "probability_basis": self.probability_basis,
            "exact_condition_count": self.exact_condition_count,
            "weight_concentration": self.weight_concentration,
            "stability_score": self.stability_score,
        }


@dataclass(frozen=True)
class PredictionResult:
    target: str
    prediction_date: date
    analysis_timeframe: str | None
    holding_period_months: float | None
    benchmark: str | None
    entry_mode: str | None
    training_observations: int
    outcome_thresholds: OutcomeThresholds
    baseline_probabilities_pct: dict[OutcomeClass, float]
    method_a: MethodPrediction | None
    method_b: MethodPrediction | None
    trend: PredictionTrend
    conviction: Conviction
    probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    method_agreement: bool
    limited: bool
    limitations: tuple[str, ...] = ()
    validated_evidence: bool = False
    decision: DecisionResult | None = None
    combination: CombinationResult | None = None
    conviction_result: ConvictionResult | None = None
    provenance_audit: PredictionProvenanceAudit | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prediction_date": self.prediction_date,
            "analysis_timeframe": self.analysis_timeframe,
            "holding_period_months": self.holding_period_months,
            "benchmark": self.benchmark,
            "entry_mode": self.entry_mode,
            "training_observations": self.training_observations,
            "outcome_thresholds": self.outcome_thresholds.as_dict(),
            "baseline_probabilities_pct": dict(self.baseline_probabilities_pct),
            "method_a": self.method_a.as_dict() if self.method_a else None,
            "method_b": self.method_b.as_dict() if self.method_b else None,
            "trend": self.trend,
            "conviction": self.conviction,
            "probabilities_pct": dict(self.probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "method_agreement": self.method_agreement,
            "limited": self.limited,
            "limitations": list(self.limitations),
            "validated_evidence": self.validated_evidence,
            "decision": self.decision.as_dict() if self.decision else None,
            "combination": self.combination.as_dict() if self.combination else None,
            "conviction_result": self.conviction_result.as_dict() if self.conviction_result else None,
            "provenance_audit": self.provenance_audit.as_dict() if self.provenance_audit else None,
            "probability_basis": "empirical_conditional_class_share_uncalibrated",
        }


class PredictionEngine:
    """Phase 5 prediction layer over the validated Phase 4 relationship engine."""

    def __init__(
        self,
        relationship_engine: RelationshipDiscoveryEngine | None = None,
        min_threshold_observations: int = 9,
    ) -> None:
        self.relationship_engine = relationship_engine or RelationshipDiscoveryEngine(
            max_order=3,
            min_observations=5,
        )
        self.min_threshold_observations = int(min_threshold_observations)
        if self.min_threshold_observations < 3:
            raise ValueError("min_threshold_observations must be >= 3")

    @staticmethod
    def _training_history(
        observations: Iterable[HistoricalRelationshipObservation],
        target: str,
        cutoff_date: date,
    ) -> list[HistoricalRelationshipObservation]:
        """Return only labels fully realized before the prediction cutoff.

        A historical state may predate the cutoff while its forward outcome
        still extends beyond it. Such a row cannot be used for training at
        the cutoff without looking into the future, so hardened Phase 5
        excludes it. Unknown outcome horizons are excluded as well.
        """
        history = [
            observation
            for observation in observations
            if observation.target == target
            and _as_date(observation.as_of_date) < cutoff_date
            and observation.stock_return_pct is not None
            and observation.outcome_end_date is not None
            and _as_date(observation.outcome_end_date) < cutoff_date
        ]
        return sorted(history, key=lambda item: _as_date(item.as_of_date))

    def _method_a_prediction(
        self,
        ranking: RelationshipRanking | None,
        thresholds: OutcomeThresholds,
        baseline_probabilities: dict[OutcomeClass, float],
    ) -> MethodPrediction | None:
        if ranking is None:
            return None

        best = max(
            ranking.method_results,
            key=lambda item: abs(item.score),
            default=None,
        )
        probability = build_method_a_probability(best, thresholds)

        # Phase 5.2 constructs the empirical distribution. The directional
        # decision below is retained here only as an adapter to the existing
        # MethodPrediction surface; baseline-relative direction is formally
        # treated as Phase 5.4.
        if probability.limited:
            return MethodPrediction(
                method="A",
                trend="NO_CLEAR_TREND",
                probabilities_pct=probability.probabilities_pct,
                baseline_probabilities_pct=baseline_probabilities,
                expected_return_pct=probability.expected_return_pct,
                baseline_return_pct=probability.baseline_return_pct,
                return_lift_pct=probability.return_lift_pct,
                evidence_score=probability.evidence_score,
                reliability=probability.reliability,
                sample_count=probability.sample_count,
                effective_sample_size=probability.effective_sample_size,
                stable=probability.stable,
                variables=probability.variables,
                condition=probability.condition,
                class_counts=dict(probability.class_counts),
                weighted_class_counts=dict(probability.weighted_class_counts),
                probability_basis=probability.probability_basis,
                exact_condition_count=getattr(probability, "exact_condition_count", None),
                weight_concentration=getattr(probability, "weight_concentration", None),
                stability_score=getattr(probability, "stability_score", None),
                limited=True,
                limitations=probability.limitations,
                decision=DecisionResult(
                    trend="NO_CLEAR_TREND",
                    selected_class=None,
                    probability_pct=0.0,
                    baseline_probability_pct=0.0,
                    lift_pct=0.0,
                    margin_pct=0.0,
                    uncertainty_pct=100.0,
                    effective_sample_size=0.0,
                    reason="Method A probability distribution is limited.",
                    limited=True,
                ),
            )

        decision = decide_baseline_relative(
            probability.probabilities_pct,
            baseline_probabilities,
            probability.effective_sample_size,
        )
        trend = decision.trend
        return MethodPrediction(
            method="A",
            trend=trend,
            probabilities_pct=probability.probabilities_pct,
            baseline_probabilities_pct=baseline_probabilities,
            expected_return_pct=probability.expected_return_pct,
            baseline_return_pct=probability.baseline_return_pct,
            return_lift_pct=probability.return_lift_pct,
            evidence_score=probability.evidence_score,
            reliability=probability.reliability,
            sample_count=probability.sample_count,
            effective_sample_size=probability.effective_sample_size,
            stable=probability.stable,
            variables=probability.variables,
            condition=probability.condition,
            class_counts=dict(probability.class_counts),
            weighted_class_counts=dict(probability.weighted_class_counts),
            probability_basis=probability.probability_basis,
            exact_condition_count=getattr(probability, "exact_condition_count", None),
            weight_concentration=getattr(probability, "weight_concentration", None),
            stability_score=getattr(probability, "stability_score", None),
            limited=False,
            limitations=probability.limitations,
            decision=decision,
        )

    def _method_prediction(
        self,
        ranking: RelationshipRanking | None,
        history: list[HistoricalRelationshipObservation],
        thresholds: OutcomeThresholds,
        baseline_probabilities: dict[OutcomeClass, float],
    ) -> MethodPrediction | None:
        if ranking is None:
            return None

        best = max(
            ranking.method_results,
            key=lambda item: abs(item.score),
            default=None,
        )
        if best is not None and not best.supporting_observations:
            materializer = getattr(self.relationship_engine, "materialize_support", None)
            if materializer is not None:
                best = materializer(best, history)
        probability = build_method_b_probability(best, thresholds)

        # Phase 5.3 constructs the coherent Method B weighted empirical
        # distribution. Direction selection remains a later Phase 5 stage.
        if probability.limited:
            return MethodPrediction(
                method="B",
                trend="NO_CLEAR_TREND",
                probabilities_pct=probability.probabilities_pct,
                baseline_probabilities_pct=baseline_probabilities,
                expected_return_pct=probability.expected_return_pct,
                baseline_return_pct=probability.baseline_return_pct,
                return_lift_pct=probability.return_lift_pct,
                evidence_score=probability.evidence_score,
                reliability=probability.reliability,
                sample_count=probability.sample_count,
                effective_sample_size=probability.effective_sample_size,
                stable=probability.stable,
                variables=probability.variables,
                condition=probability.condition,
                class_counts=dict(probability.class_counts),
                weighted_class_counts=dict(probability.weighted_class_counts),
                probability_basis=probability.probability_basis,
                exact_condition_count=getattr(probability, "exact_condition_count", None),
                weight_concentration=getattr(probability, "weight_concentration", None),
                stability_score=getattr(probability, "stability_score", None),
                limited=True,
                limitations=probability.limitations,
                decision=DecisionResult(
                    trend="NO_CLEAR_TREND",
                    selected_class=None,
                    probability_pct=0.0,
                    baseline_probability_pct=0.0,
                    lift_pct=0.0,
                    margin_pct=0.0,
                    uncertainty_pct=100.0,
                    effective_sample_size=0.0,
                    reason="Method B probability distribution is limited.",
                    limited=True,
                ),
            )

        effective_n = probability.effective_sample_size
        decision = decide_baseline_relative(
            probability.probabilities_pct,
            baseline_probabilities,
            effective_n,
        )
        trend = decision.trend
        return MethodPrediction(
            method="B",
            trend=trend,
            probabilities_pct=probability.probabilities_pct,
            baseline_probabilities_pct=baseline_probabilities,
            expected_return_pct=probability.expected_return_pct,
            baseline_return_pct=probability.baseline_return_pct,
            return_lift_pct=probability.return_lift_pct,
            evidence_score=probability.evidence_score,
            reliability=probability.reliability,
            sample_count=probability.sample_count,
            effective_sample_size=effective_n,
            stable=probability.stable,
            variables=probability.variables,
            condition=probability.condition,
            class_counts=dict(probability.class_counts),
            weighted_class_counts=dict(probability.weighted_class_counts),
            probability_basis=probability.probability_basis,
            exact_condition_count=getattr(probability, "exact_condition_count", None),
            weight_concentration=getattr(probability, "weight_concentration", None),
            stability_score=getattr(probability, "stability_score", None),
            decision=decision,
        )

    @staticmethod
    def _method_weight(method: MethodPrediction) -> float:
        """Backward-compatible adapter for the Phase 5.5 weighting rule."""
        return next(
            iter(
                combine_method_probabilities(
                    [
                        CombinationMethodInput(
                            method=method.method,
                            probabilities_pct=method.probabilities_pct,
                            expected_return_pct=method.expected_return_pct,
                            evidence_score=method.evidence_score,
                            reliability=method.reliability,
                            effective_sample_size=method.effective_sample_size,
                            stable=method.stable,
                            trend=method.trend,
                            limited=method.limited,
                        )
                    ]
                ).method_weights.values()
            ),
            0.0,
        )

    @staticmethod
    def _combination_inputs(
        method_a: MethodPrediction | None,
        method_b: MethodPrediction | None,
    ) -> list[CombinationMethodInput]:
        inputs: list[CombinationMethodInput] = []
        for method in (method_a, method_b):
            if method is None:
                continue
            inputs.append(
                CombinationMethodInput(
                    method=method.method,
                    probabilities_pct=method.probabilities_pct,
                    expected_return_pct=method.expected_return_pct,
                    evidence_score=method.evidence_score,
                    reliability=method.reliability,
                    effective_sample_size=method.effective_sample_size,
                    stable=method.stable,
                    trend=method.trend,
                    limited=method.limited,
                )
            )
        return inputs

    def _combine(
        self,
        method_a: MethodPrediction | None,
        method_b: MethodPrediction | None,
        baseline: dict[OutcomeClass, float],
        *,
        validated_evidence: bool = False,
    ) -> tuple[PredictionTrend, Conviction, dict[OutcomeClass, float], float | None, bool]:
        """Backward-compatible wrapper around the explicit Phase 5.5 layer."""
        combination = combine_method_probabilities(
            self._combination_inputs(method_a, method_b)
        )
        if combination.limited:
            return "NO_CLEAR_TREND", "NONE", dict(baseline), None, False

        combined_decision = decide_baseline_relative(
            combination.probabilities_pct,
            baseline,
            max(combination.effective_sample_size_sum, 1.0),
        )
        trend = combined_decision.trend
        if combination.method_conflict:
            trend = "NO_CLEAR_TREND"

        conviction_result = assess_conviction(
            trend=trend,
            combination=combination,
            decision=combined_decision,
            validated_evidence=validated_evidence,
        )
        return trend, conviction_result.conviction, combination.probabilities_pct, combination.expected_return_pct, combination.method_agreement

    def predict(
        self,
        target: str,
        current_states: dict[str, str],
        observations: Iterable[HistoricalRelationshipObservation],
        prediction_date: str | date | datetime,
        analysis_timeframe: str | None = None,
        holding_period_months: float | None = None,
        benchmark: str | None = None,
        entry_mode: str | None = None,
        validated_evidence: bool = False,
        validation_evidence: ValidationEvidenceAudit | None = None,
        selected_relationships: dict[str, RelationshipRanking | None] | None = None,
    ) -> PredictionResult:
        """
        Build a pre-calibration prediction from training observations only.

        Phase 5 produces empirical class probabilities. Probability calibration,
        cross-company robustness and trading-performance validation remain Phase 6.
        """
        cutoff = _as_date(prediction_date)
        source_observations = list(observations)
        history, training_filter = filter_completed_outcomes(
            observations=source_observations,
            target=target,
            cutoff_date=cutoff,
        )

        thresholds = learn_outcome_thresholds(
            [item.stock_return_pct for item in history],
            min_observations=self.min_threshold_observations,
        )
        baseline = _baseline_probabilities(history, thresholds)

        limitations: list[str] = []
        if training_filter.excluded_count > 0:
            limitations.append(
                f"Excluded {training_filter.excluded_count} source rows from the cutoff-safe training universe."
            )
        if thresholds.limited:
            limitations.append(thresholds.limitation or "Outcome thresholds are limited.")

        if selected_relationships is not None:
            # Phase 5.8 nested validation supplies relationships that were
            # selected only on an inner chronological holdout with FDR control.
            # Do not re-rank the complete training universe here, which would
            # reintroduce the in-sample winner's curse.
            ranked_a = [selected_relationships["A"]] if selected_relationships.get("A") is not None else []
            ranked_b = [selected_relationships["B"]] if selected_relationships.get("B") is not None else []
        else:
            discovered = self.relationship_engine.discover(
                current_states=current_states,
                observations=history,
                cutoff_date=cutoff,
            )
            ranked_a = rank_relationships(discovered.get("method_a", []))
            ranked_b = rank_relationships(discovered.get("method_b", []))

        provenance_audit = audit_prediction_provenance(
            observations=source_observations,
            target=target,
            cutoff_date=cutoff,
            training_history=history,
            thresholds=thresholds,
            ranked_a=ranked_a,
            ranked_b=ranked_b,
            validated_evidence=validated_evidence,
            validation_evidence=validation_evidence,
        )
        if not provenance_audit.clean:
            limitations.extend(provenance_audit.limitations)
            limited_decision = DecisionResult(
                trend="NO_CLEAR_TREND",
                selected_class=None,
                probability_pct=0.0,
                baseline_probability_pct=0.0,
                lift_pct=0.0,
                margin_pct=0.0,
                uncertainty_pct=100.0,
                effective_sample_size=0.0,
                reason="Phase 5 provenance hardening failed; prediction is suppressed.",
                limited=True,
            )
            return PredictionResult(
                target=target,
                prediction_date=cutoff,
                analysis_timeframe=analysis_timeframe,
                holding_period_months=holding_period_months,
                benchmark=benchmark,
                entry_mode=entry_mode,
                training_observations=len(history),
                outcome_thresholds=thresholds,
                baseline_probabilities_pct=baseline,
                method_a=None,
                method_b=None,
                trend="NO_CLEAR_TREND",
                conviction="NONE",
                probabilities_pct=dict(baseline),
                expected_return_pct=None,
                method_agreement=False,
                limited=True,
                limitations=tuple(dict.fromkeys(limitations)),
                validated_evidence=False,
                decision=limited_decision,
                combination=None,
                conviction_result=None,
                provenance_audit=provenance_audit,
            )

        method_a = self._method_a_prediction(
            ranked_a[0] if ranked_a else None,
            thresholds,
            baseline,
        )
        method_b = self._method_prediction(
            ranked_b[0] if ranked_b else None,
            history,
            thresholds,
            baseline,
        )

        if method_a is None:
            limitations.append("Method A produced no usable relationship for the current state.")
        if method_b is None:
            limitations.append("Method B produced no usable relationship for the current state.")

        combination = combine_method_probabilities(
            self._combination_inputs(method_a, method_b)
        )
        if combination.limited:
            trend = "NO_CLEAR_TREND"
            probabilities, expected_return, agreement = dict(baseline), None, False
            combined_decision = decide_baseline_relative(
                probabilities, baseline, 0.0
            )
        else:
            combined_decision = decide_baseline_relative(
                combination.probabilities_pct,
                baseline,
                max(combination.effective_sample_size_sum, 1.0),
            )
            trend = combined_decision.trend
            if combination.method_conflict:
                trend = "NO_CLEAR_TREND"
                combined_decision = DecisionResult(
                    trend="NO_CLEAR_TREND",
                    selected_class=combined_decision.selected_class,
                    probability_pct=combined_decision.probability_pct,
                    baseline_probability_pct=combined_decision.baseline_probability_pct,
                    lift_pct=combined_decision.lift_pct,
                    margin_pct=combined_decision.margin_pct,
                    uncertainty_pct=combined_decision.uncertainty_pct,
                    effective_sample_size=combined_decision.effective_sample_size,
                    reason="Method A and Method B give conflicting directional signals.",
                    limited=combined_decision.limited,
                )
            probabilities = combination.probabilities_pct
            expected_return = combination.expected_return_pct
            agreement = combination.method_agreement

        conviction_result = assess_conviction(
            trend=trend,
            combination=combination,
            decision=combined_decision,
            validated_evidence=validated_evidence,
        )
        conviction = conviction_result.conviction

        # An absent relationship is an evidence limitation, not automatically
        # a data-quality limitation. Overall LIMITED is reserved for an
        # actually unusable outcome threshold or a method result that could
        # not be converted into probabilities.
        limited = thresholds.limited or bool(
            (method_a and method_a.limited) or (method_b and method_b.limited)
        )

        trend = combined_decision.trend
        if combination.method_conflict:
            trend = "NO_CLEAR_TREND"
            conviction = "NONE"

        return PredictionResult(
            target=target,
            prediction_date=cutoff,
            analysis_timeframe=analysis_timeframe,
            holding_period_months=holding_period_months,
            benchmark=benchmark,
            entry_mode=entry_mode,
            training_observations=len(history),
            outcome_thresholds=thresholds,
            baseline_probabilities_pct=baseline,
            method_a=method_a,
            method_b=method_b,
            trend=trend,
            conviction=conviction,
            probabilities_pct=probabilities,
            expected_return_pct=expected_return,
            method_agreement=agreement,
            limited=limited,
            limitations=tuple(limitations),
            validated_evidence=validated_evidence,
            decision=combined_decision,
            combination=combination,
            conviction_result=conviction_result,
            provenance_audit=provenance_audit,
        )
