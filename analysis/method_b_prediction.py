from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .outcome_labels import OutcomeClass, OutcomeThresholds, _coerce_finite, classify_return
from .relationship import RelationshipResult


@dataclass(frozen=True)
class MethodBProbability:
    """Three-class empirical probability distribution from Phase 4 Method B.

    Method B keeps the full condition-weighted historical population. The
    probability for each class is therefore the weighted share of that same
    population. No exact-match subset or second estimator is introduced here.
    """

    probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float
    baseline_return_pct: float
    return_lift_pct: float
    evidence_score: float
    reliability: float
    sample_count: int
    exact_condition_count: int | None
    effective_sample_size: float
    weight_concentration: float
    stable: bool
    stability_score: float
    variables: tuple[str, ...]
    condition: tuple[str, ...]
    class_counts: dict[OutcomeClass, int]
    weighted_class_counts: dict[OutcomeClass, float]
    limited: bool = False
    limitations: tuple[str, ...] = ()
    probability_basis: str = "method_b_weighted_empirical_conditional_class_share"

    def as_dict(self) -> dict[str, Any]:
        return {
            "probabilities_pct": dict(self.probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "baseline_return_pct": self.baseline_return_pct,
            "return_lift_pct": self.return_lift_pct,
            "evidence_score": self.evidence_score,
            "reliability": self.reliability,
            "sample_count": self.sample_count,
            "exact_condition_count": self.exact_condition_count,
            "effective_sample_size": self.effective_sample_size,
            "weight_concentration": self.weight_concentration,
            "stable": self.stable,
            "stability_score": self.stability_score,
            "variables": list(self.variables),
            "condition": list(self.condition),
            "class_counts": dict(self.class_counts),
            "weighted_class_counts": dict(self.weighted_class_counts),
            "limited": self.limited,
            "limitations": list(self.limitations),
            "probability_basis": self.probability_basis,
        }


def _empty_counts() -> dict[OutcomeClass, int]:
    return {"UP": 0, "SIDEWAYS": 0, "DOWN": 0}


def _empty_weighted_counts() -> dict[OutcomeClass, float]:
    return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def _empty_probability() -> dict[OutcomeClass, float]:
    return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def _limited(
    relationship: RelationshipResult | None,
    *limitations: str,
) -> MethodBProbability:
    best = relationship
    return MethodBProbability(
        probabilities_pct=_empty_probability(),
        expected_return_pct=(
            best.weighted_mean_return_pct
            if best and best.weighted_mean_return_pct is not None
            else best.mean_return_pct if best else 0.0
        ),
        baseline_return_pct=best.baseline_mean_return_pct if best else 0.0,
        return_lift_pct=best.lift_pct if best else 0.0,
        evidence_score=best.score if best else 0.0,
        reliability=best.reliability if best else 0.0,
        sample_count=best.sample_count if best else 0,
        exact_condition_count=best.exact_condition_count if best else None,
        effective_sample_size=best.effective_sample_size or 0.0 if best else 0.0,
        weight_concentration=best.weight_concentration or 0.0 if best else 0.0,
        stable=best.stable if best else False,
        stability_score=best.stability_score if best else 0.0,
        variables=best.variables if best else (),
        condition=best.condition if best else (),
        class_counts=_empty_counts(),
        weighted_class_counts=_empty_weighted_counts(),
        limited=True,
        limitations=tuple(limitations) or ("Method B probability construction is limited.",),
    )


def build_method_b_probability(
    relationship: RelationshipResult | None,
    thresholds: OutcomeThresholds,
) -> MethodBProbability:
    """Construct a class distribution from the exact Phase 4 Method B support.

    Method B is a single coherent weighted estimator: probabilities, expected
    return and effective sample size all come from the retained weighted
    historical population. ``exact_condition_count`` remains diagnostic only.
    """

    if relationship is None:
        return _limited(None, "Method B produced no relationship for the current state.")

    if relationship.method != "B":
        raise ValueError(
            f"Method B probability construction received method {relationship.method!r}."
        )

    if thresholds.limited:
        return _limited(
            relationship,
            thresholds.limitation or "Outcome thresholds are limited.",
        )

    support = relationship.supporting_observations
    weights = relationship.supporting_weights

    if not support:
        return _limited(relationship, "Method B retained no supporting observations.")
    if len(weights) != len(support):
        return _limited(
            relationship,
            "Method B supporting observations and weights have mismatched lengths.",
        )
    if relationship.sample_count != len(support):
        return _limited(
            relationship,
            "Method B relationship sample count does not match retained support.",
        )

    class_counts = _empty_counts()
    weighted_counts = _empty_weighted_counts()
    weighted_return_sum = 0.0
    total_weight = 0.0
    squared_weight_sum = 0.0

    for (_support_date, return_pct), weight in zip(support, weights):
        numeric_return = _coerce_finite(return_pct)
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            numeric_weight = float("nan")

        if numeric_return is None or not isfinite(numeric_weight) or numeric_weight <= 0.0:
            return _limited(
                relationship,
                "Method B retained support contains an invalid return or weight.",
            )

        label = classify_return(numeric_return, thresholds)
        class_counts[label] += 1
        weighted_counts[label] += numeric_weight
        weighted_return_sum += numeric_return * numeric_weight
        total_weight += numeric_weight
        squared_weight_sum += numeric_weight * numeric_weight

    if total_weight <= 0.0 or squared_weight_sum <= 0.0:
        return _limited(
            relationship,
            "Method B retained support has no positive usable weight.",
        )

    probabilities_pct = {
        label: weighted_counts[label] / total_weight * 100.0
        for label in ("UP", "SIDEWAYS", "DOWN")
    }
    effective_n = (total_weight * total_weight) / squared_weight_sum
    expected_return = weighted_return_sum / total_weight

    return MethodBProbability(
        probabilities_pct=probabilities_pct,
        expected_return_pct=expected_return,
        baseline_return_pct=relationship.baseline_mean_return_pct,
        return_lift_pct=expected_return - relationship.baseline_mean_return_pct,
        evidence_score=relationship.score,
        reliability=relationship.reliability,
        sample_count=len(support),
        exact_condition_count=relationship.exact_condition_count,
        effective_sample_size=effective_n,
        weight_concentration=max(weights) / total_weight,
        stable=relationship.stable,
        stability_score=relationship.stability_score,
        variables=relationship.variables,
        condition=relationship.condition,
        class_counts=class_counts,
        weighted_class_counts=weighted_counts,
    )
