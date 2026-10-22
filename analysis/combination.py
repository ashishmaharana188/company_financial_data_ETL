from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, sqrt
from typing import Iterable

from .ranking import RelationshipRanking, rank_relationships
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
    RelationshipResult,
    _compatible_pair,
    _family,
)


@dataclass(frozen=True)
class CombinationDiscoveryResult:
    """Auditable output from one higher-order discovery pass."""

    parent_pair_count: int
    selected_pair_count: int
    candidate_count: int
    evaluated_count: int
    rankings: tuple[RelationshipRanking, ...]
    candidates: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict:
        return {
            "parent_pair_count": self.parent_pair_count,
            "selected_pair_count": self.selected_pair_count,
            "candidate_count": self.candidate_count,
            "evaluated_count": self.evaluated_count,
            "candidates": [list(item) for item in self.candidates],
            "rankings": [item.as_dict() for item in self.rankings],
        }


def _pair_key(result: RelationshipResult) -> tuple[str, str] | None:
    if len(result.variables) != 2:
        return None
    return tuple(sorted(result.variables))  # type: ignore[return-value]


def _pair_score(result: RelationshipResult) -> float:
    return abs(result.score)


def _select_seed_pairs(
    pair_results: Iterable[RelationshipResult],
) -> list[RelationshipResult]:
    """
    Select an adaptive subset of pair relationships for expansion.

    A pair is retained when it lies on the Pareto frontier across empirical
    score, reliability, sample support, and stability. The frontier is then
    capped by sqrt(number_of_pairs), preventing uncontrolled expansion as
    the available state surface grows.
    """
    unique: dict[tuple[str, str], RelationshipResult] = {}
    for result in pair_results:
        key = _pair_key(result)
        if key is None:
            continue

        previous = unique.get(key)
        if previous is None or (
            _pair_score(result),
            result.reliability,
            result.sample_count,
            int(result.stable),
        ) > (
            _pair_score(previous),
            previous.reliability,
            previous.sample_count,
            int(previous.stable),
        ):
            unique[key] = result

    pairs = list(unique.values())
    if not pairs:
        return []

    def dominates(left: RelationshipResult, right: RelationshipResult) -> bool:
        left_vector = (
            _pair_score(left),
            left.reliability,
            left.sample_count,
            int(left.stable),
        )
        right_vector = (
            _pair_score(right),
            right.reliability,
            right.sample_count,
            int(right.stable),
        )
        return (
            all(a >= b for a, b in zip(left_vector, right_vector))
            and any(a > b for a, b in zip(left_vector, right_vector))
        )

    frontier = [
        candidate
        for candidate in pairs
        if not any(
            dominates(other, candidate)
            for other in pairs
            if other is not candidate
        )
    ]

    frontier.sort(
        key=lambda result: (
            _pair_score(result),
            result.reliability,
            result.sample_count,
            int(result.stable),
        ),
        reverse=True,
    )

    seed_count = max(
        1,
        min(
            len(frontier),
            int(ceil(sqrt(len(pairs)))),
        ),
    )
    return frontier[:seed_count]


def _candidate_triples(
    current_states: dict[str, str],
    seed_pairs: Iterable[RelationshipResult],
) -> list[tuple[str, ...]]:
    features = sorted(current_states)
    candidates: set[tuple[str, ...]] = set()

    for pair in seed_pairs:
        pair_features = tuple(sorted(pair.variables))
        pair_families = {_family(feature) for feature in pair_features}

        for feature in features:
            if feature in pair_features:
                continue

            feature_family = _family(feature)
            if feature_family in pair_families:
                continue

            # The new feature must connect structurally to at least one member
            # of the seed pair. The resulting three-family graph must remain
            # connected.
            if not any(
                _compatible_pair(feature, member)
                for member in pair_features
            ):
                continue

            candidate = tuple(sorted((*pair_features, feature)))
            if len({_family(item) for item in candidate}) != 3:
                continue

            candidates.add(candidate)

    return sorted(candidates)


def discover_higher_order_combinations(
    current_states: dict[str, str],
    history: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
) -> CombinationDiscoveryResult:
    """
    Discover three-variable relationships only after pair-level evidence
    supports expansion.

    The function deliberately uses the existing Method A and Method B
    implementations rather than creating a third prediction method.
    """
    history_list = list(history)
    if not current_states or not history_list:
        return CombinationDiscoveryResult(0, 0, 0, 0, (), ())

    pair_engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=engine.min_observations,
    )
    pair_results = pair_engine.method_b_conditioned_distribution(
        current_states=current_states,
        history=history_list,
    )
    pair_results = [
        item for item in pair_results
        if len(item.variables) == 2
    ]

    seed_pairs = _select_seed_pairs(pair_results)
    candidates = _candidate_triples(
        current_states,
        seed_pairs,
    )

    evaluated_results: list[RelationshipResult] = []

    triple_engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=engine.min_observations,
    )

    for candidate in candidates:
        subset = {
            feature: current_states[feature]
            for feature in candidate
        }

        method_a = triple_engine.method_a_similar_states(
            current_states=subset,
            history=history_list,
        )
        method_b = triple_engine.method_b_conditioned_distribution(
            current_states=subset,
            history=history_list,
        )

        evaluated_results.extend(
            item
            for item in (*method_a, *method_b)
            if len(item.variables) == 3
        )

    rankings = tuple(rank_relationships(evaluated_results))

    return CombinationDiscoveryResult(
        parent_pair_count=len(pair_results),
        selected_pair_count=len(seed_pairs),
        candidate_count=len(candidates),
        evaluated_count=len(evaluated_results),
        rankings=rankings,
        candidates=tuple(candidates),
    )
