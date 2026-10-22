from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .outcome_labels import OutcomeClass, OutcomeThresholds, _coerce_finite, classify_return
from .relationship import RelationshipResult


@dataclass(frozen=True)
class MethodAProbability:
    """Three-class empirical probability distribution from Phase 4 Method A.

    Phase 5.2 deliberately stops at probability construction. It does not
    decide UP/SIDEWAYS/DOWN, compare against a baseline, combine methods, or
    assign conviction. Those decisions belong to later Phase 5 stages.
    """

    probabilities_pct: dict[OutcomeClass, float]
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
    class_counts: dict[OutcomeClass, int]
    weighted_class_counts: dict[OutcomeClass, float]
    limited: bool = False
    limitations: tuple[str, ...] = ()
    probability_basis: str = "method_a_empirical_conditional_class_share"

    def as_dict(self) -> dict[str, Any]:
        return {
            "probabilities_pct": dict(self.probabilities_pct),
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
    thresholds: OutcomeThresholds,
    *limitations: str,
) -> MethodAProbability:
    best = relationship
    return MethodAProbability(
        probabilities_pct=_empty_probability(),
        expected_return_pct=best.mean_return_pct if best else 0.0,
        baseline_return_pct=best.baseline_mean_return_pct if best else 0.0,
        return_lift_pct=best.lift_pct if best else 0.0,
        evidence_score=best.score if best else 0.0,
        reliability=best.reliability if best else 0.0,
        sample_count=best.sample_count if best else 0,
        effective_sample_size=best.effective_sample_size or 0.0 if best else 0.0,
        stable=best.stable if best else False,
        variables=best.variables if best else (),
        condition=best.condition if best else (),
        class_counts=_empty_counts(),
        weighted_class_counts=_empty_weighted_counts(),
        limited=True,
        limitations=tuple(limitations) or (
            thresholds.limitation or "Method A probability construction is limited.",
        ),
    )


def build_method_a_probability(
    relationship: RelationshipResult | None,
    thresholds: OutcomeThresholds,
) -> MethodAProbability:
    """Construct empirical class shares from the exact Method A support set.

    The function is intentionally narrow:
      1. consume the relationship's retained comparable observations;
      2. apply the already-learned Phase 5.1 return boundaries;
      3. calculate class shares from that support only.

    It fails closed on malformed support rather than silently substituting
    another population or manufacturing weights. Method A's relationship
    engine currently retains unit weights, but the exact weights are consumed
    explicitly so the probability surface remains auditable.
    """

    if relationship is None:
        return _limited(
            None,
            thresholds,
            "Method A produced no relationship for the current state.",
        )

    if relationship.method != "A":
        raise ValueError(
            f"Method A probability construction received method {relationship.method!r}."
        )

    if thresholds.limited:
        return _limited(
            relationship,
            thresholds,
            thresholds.limitation or "Outcome thresholds are limited.",
        )

    support = relationship.supporting_observations
    weights = relationship.supporting_weights

    if not support:
        return _limited(
            relationship,
            thresholds,
            "Method A retained no supporting observations.",
        )

    if len(weights) != len(support):
        return _limited(
            relationship,
            thresholds,
            "Method A supporting observations and weights have mismatched lengths.",
        )

    if relationship.sample_count != len(support):
        return _limited(
            relationship,
            thresholds,
            "Method A relationship sample count does not match retained support.",
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
                thresholds,
                "Method A retained support contains an invalid return or weight.",
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
            thresholds,
            "Method A retained support has no positive usable weight.",
        )

    probabilities_pct = {
        label: weighted_counts[label] / total_weight * 100.0
        for label in ("UP", "SIDEWAYS", "DOWN")
    }

    effective_n = (total_weight * total_weight) / squared_weight_sum
    expected_return = weighted_return_sum / total_weight

    # For Method A the retained relationship statistics are computed from the
    # same comparable population. The probability layer therefore preserves
    # the relationship metadata but derives its own weighted expected return
    # from the exact support it classified.
    return MethodAProbability(
        probabilities_pct=probabilities_pct,
        expected_return_pct=expected_return,
        baseline_return_pct=relationship.baseline_mean_return_pct,
        return_lift_pct=expected_return - relationship.baseline_mean_return_pct,
        evidence_score=relationship.score,
        reliability=relationship.reliability,
        sample_count=len(support),
        effective_sample_size=effective_n,
        stable=relationship.stable,
        variables=relationship.variables,
        condition=relationship.condition,
        class_counts=class_counts,
        weighted_class_counts=weighted_counts,
    )
