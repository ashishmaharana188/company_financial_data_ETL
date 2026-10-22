from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, sqrt
from typing import Iterable

from .combination import _select_seed_pairs, _candidate_triples
from .ranking import RelationshipRanking, rank_relationships
from .relationship_graph import structurally_connected
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
    RelationshipResult,
    _compatible_pair,
    _family,
)


@dataclass(frozen=True)
class ExpansionStep:
    """Auditable record of one higher-order expansion level."""

    parent_order: int
    child_order: int
    parent_count: int
    seed_parent_count: int
    candidate_count: int
    evaluated_count: int
    retained_count: int
    best_score: float
    median_score: float
    best_incremental_gain: float
    median_incremental_gain: float
    stopped: bool
    stop_reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AdaptiveCombinationResult:
    """Complete adaptive higher-order discovery result."""

    rankings: tuple[RelationshipRanking, ...]
    selected_relationships: tuple[RelationshipRanking, ...]
    steps: tuple[ExpansionStep, ...]
    highest_order: int
    stopped_reason: str

    def as_dict(self) -> dict:
        return {
            "highest_order": self.highest_order,
            "stopped_reason": self.stopped_reason,
            "steps": [step.as_dict() for step in self.steps],
            "rankings": [item.as_dict() for item in self.rankings],
            "selected_relationships": [
                item.as_dict() for item in self.selected_relationships
            ],
        }


def _result_key(result: RelationshipResult) -> tuple[str, ...]:
    return tuple(sorted(result.variables))


def _score(result: RelationshipResult) -> float:
    return abs(result.score)


def _relationship_quality_vector(result: RelationshipResult) -> tuple[float, float, int, int]:
    return (
        _score(result),
        result.reliability,
        result.sample_count,
        int(result.stable),
    )


def _best_by_relationship(
    results: Iterable[RelationshipResult],
) -> dict[tuple[str, ...], RelationshipResult]:
    best: dict[tuple[str, ...], RelationshipResult] = {}
    for result in results:
        key = _result_key(result)
        previous = best.get(key)
        if previous is None or _relationship_quality_vector(result) > _relationship_quality_vector(previous):
            best[key] = result
    return best


def _parents_to_expand(
    parents: Iterable[RelationshipResult],
) -> list[RelationshipResult]:
    """Keep nondominated parents, then apply a data-scaled frontier cap."""
    best = list(_best_by_relationship(parents).values())
    if not best:
        return []

    frontier: list[RelationshipResult] = []
    for candidate in best:
        dominated = False
        vector = _relationship_quality_vector(candidate)
        for other in best:
            if other is candidate:
                continue
            other_vector = _relationship_quality_vector(other)
            if (
                all(a >= b for a, b in zip(other_vector, vector))
                and any(a > b for a, b in zip(other_vector, vector))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(key=_relationship_quality_vector, reverse=True)

    # The cap is derived from the number of currently viable parents. It is
    # not a user-defined training size or a fixed maximum combination size.
    cap = max(1, min(len(frontier), int(ceil(sqrt(len(best))))))
    return frontier[:cap]


def _candidate_children(
    current_states: dict[str, str],
    parents: Iterable[RelationshipResult],
) -> list[tuple[str, ...]]:
    features = sorted(current_states)
    result: set[tuple[str, ...]] = set()

    for parent in parents:
        parent_features = tuple(sorted(parent.variables))
        parent_families = {_family(feature) for feature in parent_features}

        for feature in features:
            if feature in parent_features:
                continue
            if _family(feature) in parent_families:
                continue
            if not any(_compatible_pair(feature, member) for member in parent_features):
                continue

            candidate = tuple(sorted((*parent_features, feature)))
            families = {_family(item) for item in candidate}
            if len(families) != len(candidate):
                continue

            if structurally_connected(candidate):
                result.add(candidate)

    return sorted(result)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _parent_map(
    rankings: Iterable[RelationshipRanking],
) -> dict[tuple[str, ...], RelationshipRanking]:
    return {
        tuple(sorted(item.variables)): item
        for item in rankings
    }


def _best_parent_for_child(
    child: RelationshipRanking,
    parents: dict[tuple[str, ...], RelationshipRanking],
) -> RelationshipRanking | None:
    child_vars = set(child.variables)
    possible = [
        parent
        for parent in parents.values()
        if len(parent.variables) + 1 == len(child.variables)
        and set(parent.variables).issubset(child_vars)
    ]
    if not possible:
        return None
    return max(possible, key=lambda item: item.best_score if item.best_score >= 0 else abs(item.best_score))


def _retain_children(
    child_rankings: list[RelationshipRanking],
    parent_rankings: list[RelationshipRanking],
) -> tuple[list[RelationshipRanking], float, float, bool, str]:
    """Retain only children that demonstrate incremental evidence over parents."""
    if not child_rankings:
        return [], 0.0, 0.0, False, "no_child_relationships"

    parent_map = _parent_map(parent_rankings)
    gains: list[float] = []
    viable: list[RelationshipRanking] = []

    for child in child_rankings:
        parent = _best_parent_for_child(child, parent_map)
        if parent is None:
            continue

        gain = child.best_score - parent.best_score
        gains.append(gain)

    if not gains:
        return [], 0.0, 0.0, False, "no_parent_comparison"

    median_gain = _median(gains)
    positive_gains = [gain for gain in gains if gain > 0]
    median_positive_gain = _median(positive_gains)

    # Adaptive stopping: a child needs to beat its parent and meet the
    # prevailing positive gain seen among the candidate children. This is a
    # relative, data-derived criterion rather than a fixed score threshold.
    threshold = median_positive_gain if positive_gains else median_gain

    for child in child_rankings:
        parent = _best_parent_for_child(child, parent_map)
        if parent is None:
            continue
        gain = child.best_score - parent.best_score
        if gain > 0 and gain >= threshold:
            viable.append(child)

    viable.sort(key=lambda item: item.rank_key, reverse=True)

    # Retain an adaptive frontier-sized subset for the next expansion level.
    cap = max(1, min(len(viable), int(ceil(sqrt(len(child_rankings)))))) if viable else 0
    retained = viable[:cap]

    best_gain = max(gains)
    stopped = not retained
    reason = (
        "no_child_improves_over_adaptive_gain"
        if stopped
        else "incremental_evidence_supports_expansion"
    )
    return retained, best_gain, median_gain, stopped, reason


def _evaluate_candidates(
    candidates: Iterable[tuple[str, ...]],
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    min_observations: int,
) -> list[RelationshipRanking]:
    candidates = list(candidates)
    if not candidates:
        return []

    order = len(candidates[0])
    discovery_engine = RelationshipDiscoveryEngine(
        max_order=order,
        min_observations=min_observations,
    )

    method_results: list[RelationshipResult] = []
    for candidate in candidates:
        subset = {feature: current_states[feature] for feature in candidate}
        method_results.extend(
            item
            for item in (
                *discovery_engine.method_a_similar_states(
                    current_states=subset,
                    history=history,
                ),
                *discovery_engine.method_b_conditioned_distribution(
                    current_states=subset,
                    history=history,
                ),
            )
            if len(item.variables) == order
        )

    return rank_relationships(method_results)


def discover_adaptive_higher_order(
    current_states: dict[str, str],
    history: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
) -> AdaptiveCombinationResult:
    """
    Discover combinations adaptively beyond triples.

    Expansion begins with evidence-supported pairs, grows to triples, and
    continues only while the next order produces incremental evidence over
    its parent relationships. There is no hard-coded maximum order.
    Structural family constraints and an adaptive frontier cap control the
    search space.
    """
    history_list = list(history)
    if not current_states or not history_list:
        return AdaptiveCombinationResult((), (), (), 0, "empty_input")

    # Pair discovery is the entry point and is deliberately shared with the
    # already validated Phase 4.5/4.6 machinery.
    pair_engine = RelationshipDiscoveryEngine(
        max_order=2,
        min_observations=engine.min_observations,
    )
    pair_methods = pair_engine.method_b_conditioned_distribution(
        current_states=current_states,
        history=history_list,
    )
    pair_methods = [item for item in pair_methods if len(item.variables) == 2]

    pair_rankings = rank_relationships(pair_methods)
    seed_pairs = _select_seed_pairs(pair_methods)
    current_parents = seed_pairs

    all_rankings: list[RelationshipRanking] = list(pair_rankings)
    steps: list[ExpansionStep] = []
    highest_order = 2 if seed_pairs else 1

    if engine.max_order < 3:
        return AdaptiveCombinationResult(
            rankings=tuple(pair_rankings),
            selected_relationships=tuple(pair_rankings),
            steps=(),
            highest_order=highest_order,
            stopped_reason="max_order_reached",
        )

    # Phase 4.6 triple generation is retained as the first expansion level.
    triple_candidates = _candidate_triples(current_states, seed_pairs)
    if not triple_candidates:
        return AdaptiveCombinationResult(
            rankings=tuple(all_rankings),
            selected_relationships=tuple(pair_rankings),
            steps=(
                ExpansionStep(
                    parent_order=2,
                    child_order=3,
                    parent_count=len(pair_methods),
                    seed_parent_count=len(seed_pairs),
                    candidate_count=0,
                    evaluated_count=0,
                    retained_count=0,
                    best_score=0.0,
                    median_score=0.0,
                    best_incremental_gain=0.0,
                    median_incremental_gain=0.0,
                    stopped=True,
                    stop_reason="no_structurally_valid_children",
                ),
            ),
            highest_order=highest_order,
            stopped_reason="no_structurally_valid_children",
        )

    child_rankings = _evaluate_candidates(
        triple_candidates,
        current_states,
        history_list,
        engine.min_observations,
    )
    retained, best_gain, median_gain, stopped, reason = _retain_children(
        child_rankings,
        pair_rankings,
    )

    triple_scores = [item.best_score for item in child_rankings]
    steps.append(
        ExpansionStep(
            parent_order=2,
            child_order=3,
            parent_count=len(pair_methods),
            seed_parent_count=len(seed_pairs),
            candidate_count=len(triple_candidates),
            evaluated_count=len(child_rankings),
            retained_count=len(retained),
            best_score=max(triple_scores, default=0.0),
            median_score=_median(triple_scores),
            best_incremental_gain=best_gain,
            median_incremental_gain=median_gain,
            stopped=stopped,
            stop_reason=reason,
        )
    )
    all_rankings.extend(child_rankings)

    if stopped:
        final_rankings = rank_relationships(
            method_result
            for ranking in all_rankings
            for method_result in ranking.method_results
        )
        selected = rank_relationships(
            method_result
            for ranking in retained
            for method_result in ranking.method_results
        )
        return AdaptiveCombinationResult(
            rankings=tuple(final_rankings),
            selected_relationships=tuple(selected),
            steps=tuple(steps),
            highest_order=3 if child_rankings else highest_order,
            stopped_reason=reason,
        )

    current_parents = [
        method_result
        for ranking in retained
        for method_result in ranking.method_results
        if len(method_result.variables) == 3
    ]

    # Continue until no structurally valid child exists or an order fails the
    # adaptive incremental-evidence gate. The available feature-family graph
    # naturally bounds the possible order.
    while current_parents:
        next_order = len(current_parents[0].variables) + 1
        if next_order > engine.max_order:
            break
        candidates = _candidate_children(current_states, current_parents)
        if not candidates:
            steps.append(
                ExpansionStep(
                    parent_order=next_order - 1,
                    child_order=next_order,
                    parent_count=len(current_parents),
                    seed_parent_count=len(current_parents),
                    candidate_count=0,
                    evaluated_count=0,
                    retained_count=0,
                    best_score=0.0,
                    median_score=0.0,
                    best_incremental_gain=0.0,
                    median_incremental_gain=0.0,
                    stopped=True,
                    stop_reason="no_structurally_valid_children",
                )
            )
            break

        rankings = _evaluate_candidates(
            candidates,
            current_states,
            history_list,
            engine.min_observations,
        )
        retained, best_gain, median_gain, stopped, reason = _retain_children(
            rankings,
            rank_relationships(
                method_result
                for parent in current_parents
                for method_result in [parent]
            ),
        )

        scores = [item.best_score for item in rankings]
        steps.append(
            ExpansionStep(
                parent_order=next_order - 1,
                child_order=next_order,
                parent_count=len(current_parents),
                seed_parent_count=len(current_parents),
                candidate_count=len(candidates),
                evaluated_count=len(rankings),
                retained_count=len(retained),
                best_score=max(scores, default=0.0),
                median_score=_median(scores),
                best_incremental_gain=best_gain,
                median_incremental_gain=median_gain,
                stopped=stopped,
                stop_reason=reason,
            )
        )
        all_rankings.extend(rankings)

        highest_order = max(highest_order, next_order if rankings else highest_order)
        if stopped:
            break

        current_parents = [
            method_result
            for ranking in retained
            for method_result in ranking.method_results
        ]

    final_rankings = rank_relationships(
        method_result
        for ranking in all_rankings
        for method_result in ranking.method_results
    )
    selected = rank_relationships(
        method_result
        for step_ranking in [item for item in all_rankings if len(item.variables) == highest_order]
        for method_result in step_ranking.method_results
    )

    final_reason = steps[-1].stop_reason if steps else "no_expansion"
    return AdaptiveCombinationResult(
        rankings=tuple(final_rankings),
        selected_relationships=tuple(selected),
        steps=tuple(steps),
        highest_order=highest_order,
        stopped_reason=final_reason,
    )
