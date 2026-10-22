from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal

from .outcome_labels import OutcomeClass

PredictionTrend = Literal["UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"]


def _empty_probabilities() -> dict[OutcomeClass, float]:
    return {"UP": 0.0, "SIDEWAYS": 0.0, "DOWN": 0.0}


def _normalize_probabilities(values: dict[OutcomeClass, float]) -> dict[OutcomeClass, float]:
    cleaned = {
        key: max(float(value), 0.0)
        for key, value in values.items()
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return _empty_probabilities()
    return {key: value / total * 100.0 for key, value in cleaned.items()}


@dataclass(frozen=True)
class CombinationMethodInput:
    """Minimal adapter so Phase 5.5 does not depend on prediction.py internals."""

    method: str
    probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    evidence_score: float
    reliability: float
    effective_sample_size: float
    stable: bool
    trend: PredictionTrend = "NO_CLEAR_TREND"
    limited: bool = False


@dataclass(frozen=True)
class CombinationResult:
    """Auditable Method A/B probability combination."""

    probabilities_pct: dict[OutcomeClass, float]
    expected_return_pct: float | None
    method_weights: dict[str, float]
    method_weight_shares_pct: dict[str, float]
    usable_methods: tuple[str, ...]
    method_agreement: bool
    method_conflict: bool
    directional_methods: tuple[str, ...]
    directional_trends: tuple[PredictionTrend, ...]
    stable_methods: tuple[str, ...]
    effective_sample_size_sum: float
    weighting_rule: str
    limited: bool = False
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "probabilities_pct": dict(self.probabilities_pct),
            "expected_return_pct": self.expected_return_pct,
            "method_weights": dict(self.method_weights),
            "method_weight_shares_pct": dict(self.method_weight_shares_pct),
            "usable_methods": list(self.usable_methods),
            "method_agreement": self.method_agreement,
            "method_conflict": self.method_conflict,
            "directional_methods": list(self.directional_methods),
            "directional_trends": list(self.directional_trends),
            "stable_methods": list(self.stable_methods),
            "effective_sample_size_sum": self.effective_sample_size_sum,
            "weighting_rule": self.weighting_rule,
            "limited": self.limited,
            "limitations": list(self.limitations),
        }


def _method_weight(method: CombinationMethodInput) -> float:
    """Current analysis-time evidence weight used by Phase 5.5.

    This remains deliberately transparent and heuristic. It is not presented
    as a learned optimal weight. Phase 6 must evaluate whether this weighting
    improves out-of-sample performance and calibration.
    """
    evidence = max(float(method.evidence_score), 0.0)
    reliability = max(float(method.reliability), 0.0)
    effective_n = max(float(method.effective_sample_size), 0.0)
    if not all(isfinite(value) for value in (evidence, reliability, effective_n)):
        return 0.0
    return evidence * reliability * sqrt(max(effective_n, 1.0))


def combine_method_probabilities(
    methods: list[CombinationMethodInput],
) -> CombinationResult:
    """Combine usable Method A/B empirical distributions.

    Only non-limited methods with valid distributions contribute. There is no
    permanent 50/50 allocation. Each method's share is recomputed from its
    current evidence score, reliability and effective sample size.
    """
    usable: list[CombinationMethodInput] = []
    limitations: list[str] = []

    for method in methods:
        if method.limited:
            continue
        if method.method not in {"A", "B"}:
            limitations.append(f"Unsupported prediction method: {method.method}.")
            continue
        if set(method.probabilities_pct) != {"UP", "SIDEWAYS", "DOWN"}:
            limitations.append(f"Method {method.method} probability distribution is malformed.")
            continue
        values = [float(method.probabilities_pct[label]) for label in ("UP", "SIDEWAYS", "DOWN")]
        if any(not isfinite(value) or value < 0.0 for value in values):
            limitations.append(f"Method {method.method} probability distribution contains invalid values.")
            continue
        if abs(sum(values) - 100.0) > 1e-6:
            limitations.append(f"Method {method.method} probability distribution does not sum to 100%.")
            continue
        if method.expected_return_pct is not None and not isfinite(float(method.expected_return_pct)):
            limitations.append(f"Method {method.method} expected return is invalid.")
            continue
        usable.append(method)

    # Deduplicate by method name so a malformed caller cannot silently give
    # Method A or B multiple votes in the combination.
    by_method: dict[str, CombinationMethodInput] = {}
    for method in usable:
        if method.method in by_method:
            limitations.append(f"Duplicate usable prediction method: {method.method}.")
            continue
        by_method[method.method] = method

    usable = list(by_method.values())
    if not usable:
        return CombinationResult(
            probabilities_pct=_empty_probabilities(),
            expected_return_pct=None,
            method_weights={},
            method_weight_shares_pct={},
            usable_methods=(),
            method_agreement=False,
            method_conflict=False,
            directional_methods=(),
            directional_trends=(),
            stable_methods=(),
            effective_sample_size_sum=0.0,
            weighting_rule="evidence_score × reliability × sqrt(effective_sample_size)",
            limited=True,
            limitations=tuple(limitations) or ("No usable method distributions are available.",),
        )

    weights = {method.method: _method_weight(method) for method in usable}
    total_weight = sum(weights.values())
    if not isfinite(total_weight) or total_weight <= 0.0:
        return CombinationResult(
            probabilities_pct=_empty_probabilities(),
            expected_return_pct=None,
            method_weights=weights,
            method_weight_shares_pct={},
            usable_methods=tuple(method.method for method in usable),
            method_agreement=False,
            method_conflict=False,
            directional_methods=(),
            directional_trends=(),
            stable_methods=tuple(method.method for method in usable if method.stable),
            effective_sample_size_sum=sum(max(float(method.effective_sample_size), 0.0) for method in usable),
            weighting_rule="evidence_score × reliability × sqrt(effective_sample_size)",
            limited=True,
            limitations=tuple(limitations) + ("All usable method weights are zero.",),
        )

    shares = {name: weight / total_weight * 100.0 for name, weight in weights.items()}
    combined = _empty_probabilities()
    expected_components: list[float] = []

    for method in usable:
        share = weights[method.method] / total_weight
        for label in combined:
            combined[label] += method.probabilities_pct[label] * share
        if method.expected_return_pct is not None:
            expected_components.append(float(method.expected_return_pct) * share)

    combined = _normalize_probabilities(combined)
    expected_return = sum(expected_components) if expected_components else None

    directional = [
        method for method in usable
        if method.trend in {"UP", "DOWN"}
    ]
    directional_trends = tuple(method.trend for method in directional)
    directional_methods = tuple(method.method for method in directional)
    agreement = len(directional_trends) == 2 and directional_trends[0] == directional_trends[1]
    conflict = len(directional_trends) == 2 and directional_trends[0] != directional_trends[1]

    return CombinationResult(
        probabilities_pct=combined,
        expected_return_pct=expected_return,
        method_weights=weights,
        method_weight_shares_pct=shares,
        usable_methods=tuple(method.method for method in usable),
        method_agreement=agreement,
        method_conflict=conflict,
        directional_methods=directional_methods,
        directional_trends=directional_trends,
        stable_methods=tuple(method.method for method in usable if method.stable),
        effective_sample_size_sum=sum(max(float(method.effective_sample_size), 0.0) for method in usable),
        weighting_rule="evidence_score × reliability × sqrt(effective_sample_size)",
        limited=False,
        limitations=tuple(limitations),
    )
