from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from math import log1p, sqrt
from statistics import median
from typing import Any, Iterable


# ============================================================
# STRUCTURAL RELATIONSHIP RULES
# ============================================================

# These are structural/domain relationships, not economic claims.
# Example: industry.market may interact with macro.Brent_Crude;
# whether it is actually useful is discovered from history.

ALLOWED_RELATIONSHIP_FAMILIES = {
    frozenset({"company.financials", "company.market"}),
    frozenset({"company.financials", "industry.market"}),
    frozenset({"industry.financials", "industry.market"}),
    frozenset({"industry.market", "sector.market"}),
    frozenset({"industry.market", "macro"}),
    frozenset({"industry.market", "global"}),
    frozenset({"sector.market", "benchmark"}),
    frozenset({"sector.market", "macro"}),
    frozenset({"sector.market", "global"}),
}


@dataclass(frozen=True)
class HistoricalRelationshipObservation:
    """One historical state snapshot paired with a realized outcome."""

    as_of_date: date
    target: str
    scope: str

    states: dict[str, str]

    stock_return_pct: float
    benchmark_return_pct: float | None
    relative_return_pct: float | None


@dataclass(frozen=True)
class RelationshipResult:
    """Empirical usefulness of one state or state combination."""

    method: str
    variables: tuple[str, ...]
    condition: tuple[str, ...]

    sample_count: int

    mean_return_pct: float
    median_return_pct: float

    baseline_mean_return_pct: float
    lift_pct: float

    positive_rate_pct: float

    effect_strength: float
    reliability: float

    score: float

    stable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "variables": list(self.variables),
            "condition": list(self.condition),
            "sample_count": self.sample_count,
            "mean_return_pct": self.mean_return_pct,
            "median_return_pct": self.median_return_pct,
            "baseline_mean_return_pct": self.baseline_mean_return_pct,
            "lift_pct": self.lift_pct,
            "positive_rate_pct": self.positive_rate_pct,
            "effect_strength": self.effect_strength,
            "reliability": self.reliability,
            "score": self.score,
            "stable": self.stable,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value))


def _family(feature: str) -> str:
    """Map a state path to a structural family."""

    parts = feature.split(".")

    if len(parts) < 2:
        return parts[0]

    return f"{parts[0]}.{parts[1]}"


def _compatible_pair(
    left: str,
    right: str,
) -> bool:
    left_family = _family(left)
    right_family = _family(right)

    if left_family == right_family:
        return False

    families = frozenset({
        left_family,
        right_family,
    })

    return families in ALLOWED_RELATIONSHIP_FAMILIES


def _flatten_states(
    value: Any,
    prefix: str = "",
) -> dict[str, str]:
    """
    Flatten Phase 3-style nested state dictionaries.

    Only descriptive `state` leaves are emitted. Numeric values,
    ranges, breadth, and bookkeeping fields remain outside the
    Phase 4 categorical relationship surface for now.
    """

    result: dict[str, str] = {}

    if not isinstance(value, dict):
        return result

    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if key == "state" and isinstance(child, str):
            result[prefix] = child
            continue

        if isinstance(child, dict):
            result.update(
                _flatten_states(
                    child,
                    path,
                )
            )

    return result


def flatten_phase3_states(
    states: dict[str, Any],
) -> dict[str, str]:
    """
    Convert nested Phase 3 outputs into categorical state atoms.

    Already-flat `{feature: state}` dictionaries are also accepted.
    """

    flat: dict[str, str] = {}

    for key, value in states.items():
        if isinstance(value, str):
            flat[key] = value
            continue

        if isinstance(value, dict):
            nested = _flatten_states(
                value,
                str(key),
            )
            flat.update(nested)

    return flat


def _match_condition(
    observation: HistoricalRelationshipObservation,
    condition: tuple[str, ...],
) -> bool:
    return all(
        "=" in item
        and observation.states.get(item.split("=", 1)[0])
        == item.split("=", 1)[1]
        for item in condition
    )


def _condition_for_feature(
    observation: HistoricalRelationshipObservation,
    feature: str,
) -> tuple[str, ...]:
    return (
        f"{feature}={observation.states[feature]}",
    )


def _condition_key(
    states: dict[str, str],
    features: tuple[str, ...],
) -> tuple[str, ...] | None:
    values: list[str] = []

    for feature in features:
        state = states.get(feature)
        if state is None:
            return None

        values.append(
            f"{feature}={state}"
        )

    return tuple(values)


def _adaptive_relevance_weights(
    observations: list[HistoricalRelationshipObservation],
) -> dict[tuple[str, str], float]:
    """
    Give more weight to informative/rarer state occurrences.

    No fixed tolerance or manually assigned feature weight is used.
    """

    total = len(observations)
    frequencies: dict[tuple[str, str], int] = {}

    for observation in observations:
        for feature, state in observation.states.items():
            key = (feature, state)
            frequencies[key] = (
                frequencies.get(key, 0) + 1
            )

    weights: dict[tuple[str, str], float] = {}

    for key, frequency in frequencies.items():
        weights[key] = log1p(
            total / max(frequency, 1)
        )

    return weights


def _condition_matches_current(
    condition: tuple[str, ...],
    current_states: dict[str, str],
) -> bool:
    for item in condition:
        feature, state = item.split("=", 1)

        if current_states.get(feature) != state:
            return False

    return True


def _summary_result(
    method: str,
    variables: tuple[str, ...],
    condition: tuple[str, ...],
    matching: list[HistoricalRelationshipObservation],
    baseline: list[HistoricalRelationshipObservation],
) -> RelationshipResult | None:
    if not matching or not baseline:
        return None

    outcomes = [
        observation.stock_return_pct
        for observation in matching
    ]

    baseline_outcomes = [
        observation.stock_return_pct
        for observation in baseline
    ]

    mean_return = sum(outcomes) / len(outcomes)
    baseline_mean = (
        sum(baseline_outcomes)
        / len(baseline_outcomes)
    )

    lift = mean_return - baseline_mean

    median_return = median(outcomes)

    positive_rate = (
        sum(value > 0 for value in outcomes)
        / len(outcomes)
    ) * 100.0

    baseline_dispersion = sqrt(
        sum(
            (value - baseline_mean) ** 2
            for value in baseline_outcomes
        )
        / max(len(baseline_outcomes), 1)
    )

    effect_strength = (
        abs(lift) / baseline_dispersion
        if baseline_dispersion > 0
        else 0.0
    )

    # Data-adaptive reliability. As sample size approaches the
    # historical universe, reliability approaches 1 without using
    # a user-chosen fixed sample multiplier.
    reliability = min(
        1.0,
        sqrt(len(matching) / max(len(baseline), 1)),
    )

    score = effect_strength * reliability

    return RelationshipResult(
        method=method,
        variables=variables,
        condition=condition,
        sample_count=len(matching),
        mean_return_pct=mean_return,
        median_return_pct=median_return,
        baseline_mean_return_pct=baseline_mean,
        lift_pct=lift,
        positive_rate_pct=positive_rate,
        effect_strength=effect_strength,
        reliability=reliability,
        score=score,
        stable=False,
    )


def _candidate_feature_sets(
    features: Iterable[str],
    max_order: int,
) -> list[tuple[str, ...]]:
    features = sorted(set(features))

    candidates: list[tuple[str, ...]] = []

    # Individual variables are always allowed.
    for feature in features:
        candidates.append((feature,))

    if max_order >= 2:
        for left, right in combinations(features, 2):
            if _compatible_pair(left, right):
                candidates.append((left, right))

    # Higher-order discovery deliberately stays disabled in the
    # initial implementation. It will only be expanded after the
    # pair-search layer proves useful/stable.
    return candidates


class RelationshipDiscoveryEngine:
    """
    Phase 4 relationship discovery foundation.

    Method A:
        adaptive similarity against historically comparable states.

    Method B:
        empirical conditional effects with adaptive relevance
        weighting for the current conditions.

    This class does not produce final calibrated probabilities.
    """

    def __init__(
        self,
        max_order: int = 2,
        min_observations: int = 5,
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        if min_observations < 1:
            raise ValueError(
                "min_observations must be >= 1"
            )

        self.max_order = max_order
        self.min_observations = min_observations

    def prepare_history(
        self,
        observations: Iterable[
            HistoricalRelationshipObservation
        ],
        cutoff_date: str | date | datetime,
    ) -> list[HistoricalRelationshipObservation]:
        """Return only observations strictly before the cutoff."""

        cutoff = _as_date(cutoff_date)

        return sorted(
            [
                observation
                for observation in observations
                if (
                    _as_date(observation.as_of_date)
                    < cutoff
                )
                and (
                    observation.stock_return_pct
                    is not None
                )
            ],
            key=lambda observation: observation.as_of_date,
        )

    # --------------------------------------------------------
    # Method A
    # --------------------------------------------------------

    def method_a_similar_states(
        self,
        current_states: dict[str, str],
        history: list[HistoricalRelationshipObservation],
    ) -> list[RelationshipResult]:
        if not history:
            return []

        weights = _adaptive_relevance_weights(
            history
        )

        results: list[RelationshipResult] = []

        baseline = history

        for feature in sorted(current_states):
            current_state = current_states[feature]
            weight = weights.get(
                (feature, current_state),
                1.0,
            )

            matching = [
                observation
                for observation in history
                if observation.states.get(feature)
                == current_state
            ]

            if len(matching) < self.min_observations:
                continue

            result = _summary_result(
                method="A",
                variables=(feature,),
                condition=(
                    f"{feature}={current_state}",
                ),
                matching=matching,
                baseline=baseline,
            )

            if result is not None:
                results.append(
                    RelationshipResult(
                        **{
                            **result.__dict__,
                            "score": result.score * weight,
                        }
                    )
                )

        return sorted(
            results,
            key=lambda result: abs(result.score),
            reverse=True,
        )

    # --------------------------------------------------------
    # Method B
    # --------------------------------------------------------

    def method_b_conditioned_distribution(
        self,
        current_states: dict[str, str],
        history: list[HistoricalRelationshipObservation],
    ) -> list[RelationshipResult]:
        if not history:
            return []

        baseline = history
        features = list(current_states.keys())
        candidates = _candidate_feature_sets(
            features,
            self.max_order,
        )

        relevance = _adaptive_relevance_weights(
            history
        )

        results: list[RelationshipResult] = []

        for feature_set in candidates:
            condition = _condition_key(
                current_states,
                feature_set,
            )

            if condition is None:
                continue

            matching = [
                observation
                for observation in history
                if _match_condition(
                    observation,
                    condition,
                )
            ]

            if len(matching) < self.min_observations:
                continue

            result = _summary_result(
                method="B",
                variables=feature_set,
                condition=condition,
                matching=matching,
                baseline=baseline,
            )

            if result is None:
                continue

            condition_weight = 1.0

            for item in condition:
                feature, state = item.split("=", 1)
                condition_weight *= relevance.get(
                    (feature, state),
                    1.0,
                )

            # Combining two or more conditions naturally receives
            # a stronger evidence requirement, not a manually chosen
            # bullish/bearish weight.
            result_score = (
                result.score
                * condition_weight
            )

            results.append(
                RelationshipResult(
                    **{
                        **result.__dict__,
                        "score": result_score,
                    }
                )
            )

        return sorted(
            results,
            key=lambda result: abs(result.score),
            reverse=True,
        )

    # --------------------------------------------------------
    # Combined discovery
    # --------------------------------------------------------

    def discover(
        self,
        current_states: dict[str, str],
        observations: Iterable[
            HistoricalRelationshipObservation
        ],
        cutoff_date: str | date | datetime,
    ) -> dict[str, list[RelationshipResult]]:
        """
        Run both relationship methods using only pre-cutoff data.
        """

        history = self.prepare_history(
            observations,
            cutoff_date,
        )

        method_a = self.method_a_similar_states(
            current_states,
            history,
        )

        method_b = self.method_b_conditioned_distribution(
            current_states,
            history,
        )

        return {
            "method_a": method_a,
            "method_b": method_b,
        }
