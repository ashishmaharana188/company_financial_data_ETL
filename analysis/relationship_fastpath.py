from __future__ import annotations

"""Prepared-history execution path for Phase 5.8 relationship evaluation.

This module does not change the statistical procedure. It only avoids rebuilding
identical feature/history arrays when Method A and Method B are run over the
same fold. The public RelationshipDiscoveryEngine remains untouched.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from datetime import date
from math import log1p, sqrt
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from .hardening_4_10 import split_stability
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
    RelationshipResult,
    _adaptive_relevance_weights,
    _condition_key,
    _summary_result,
)


class PreparedRelationshipResults(list):
    """List-compatible results with optional compact permutation support."""

    def __init__(self, iterable=(), *, support_matrix=None):
        super().__init__(iterable)
        self.support_matrix = support_matrix


@dataclass(frozen=True)
class PreparedRelationshipContext:
    current_features: tuple[str, ...]
    feature_index: dict[str, int]
    observed_matrix: np.ndarray
    match_matrix: np.ndarray
    weight_vector: np.ndarray
    weighted_observed_matrix: np.ndarray
    outcomes: np.ndarray
    outcome_order: np.ndarray
    ordered_outcomes: np.ndarray
    baseline_mean: float
    baseline_dispersion: float
    fit_start: date | None
    fit_end: date | None
    date_ordinals: np.ndarray
    history: tuple[HistoricalRelationshipObservation, ...]
    history_object_id: int

    def candidate_plan(
        self,
        candidates: Iterable[tuple[str, ...]],
    ) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
        candidate_list = tuple(tuple(candidate) for candidate in candidates)
        if not candidate_list:
            return [], np.empty((0, 0), dtype=np.int32), np.empty(0, dtype=np.int16)
        indices, orders = _candidate_plan_cached(self.current_features, candidate_list)
        return list(candidate_list), indices, orders


@lru_cache(maxsize=16)
def _candidate_plan_cached(
    features: tuple[str, ...],
    candidates: tuple[tuple[str, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Cache candidate->feature-index encoding independent of history values."""
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    if not candidates:
        return (
            np.empty((0, 0), dtype=np.int32),
            np.empty(0, dtype=np.int16),
        )
    max_order = max(len(candidate) for candidate in candidates)
    indices = np.full((len(candidates), max_order), -1, dtype=np.int32)
    orders = np.empty(len(candidates), dtype=np.int16)
    for row, candidate in enumerate(candidates):
        orders[row] = len(candidate)
        for col, feature in enumerate(candidate):
            indices[row, col] = feature_index[feature]
    return indices, orders


def prepare_relationship_context(
    current_states: dict[str, str],
    history: Sequence[HistoricalRelationshipObservation],
) -> PreparedRelationshipContext:
    """Materialize feature/history arrays once for an outer or inner fold."""
    if not current_states or not history:
        raise ValueError("current_states and history must be non-empty")

    history_tuple = tuple(history)
    features = tuple(current_states.keys())
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    relevance = _adaptive_relevance_weights(history_tuple)

    n = len(history_tuple)
    observed_matrix = np.zeros((len(features), n), dtype=np.uint8)
    match_matrix = np.zeros((len(features), n), dtype=np.uint8)
    weight_vector = np.empty(len(features), dtype=np.float64)

    for feature, idx in feature_index.items():
        current_state = current_states[feature]
        weight_vector[idx] = max(
            float(relevance.get((feature, current_state), 1.0)),
            0.0,
        )
        observed_matrix[idx] = np.fromiter(
            (1 if observation.states.get(feature) is not None else 0 for observation in history_tuple),
            dtype=np.uint8,
            count=n,
        )
        match_matrix[idx] = np.fromiter(
            (1 if observation.states.get(feature) == current_state else 0 for observation in history_tuple),
            dtype=np.uint8,
            count=n,
        )

    outcomes = np.asarray(
        [float(observation.stock_return_pct) for observation in history_tuple],
        dtype=np.float64,
    )
    outcome_order = np.argsort(outcomes, kind="stable")
    ordered_outcomes = outcomes[outcome_order]
    baseline_mean = float(np.mean(outcomes))
    baseline_dispersion = float(np.sqrt(np.mean((outcomes - baseline_mean) ** 2)))
    dates = [observation.as_of_date for observation in history_tuple]
    fit_start = min(dates) if dates else None
    fit_end = max(dates) if dates else None
    date_ordinals = np.asarray([day.toordinal() for day in dates], dtype=np.int64)

    weighted_observed_matrix = observed_matrix.astype(np.float64) * weight_vector[:, None]

    return PreparedRelationshipContext(
        current_features=features,
        feature_index=feature_index,
        observed_matrix=observed_matrix,
        match_matrix=match_matrix,
        weight_vector=weight_vector,
        weighted_observed_matrix=weighted_observed_matrix,
        outcomes=outcomes,
        outcome_order=outcome_order,
        ordered_outcomes=ordered_outcomes,
        baseline_mean=baseline_mean,
        baseline_dispersion=baseline_dispersion,
        fit_start=fit_start,
        fit_end=fit_end,
        date_ordinals=date_ordinals,
        history=history_tuple,
        history_object_id=id(history),
    )


def _validate_context(
    context: PreparedRelationshipContext,
    current_states: dict[str, str],
    history: Sequence[HistoricalRelationshipObservation],
) -> None:
    if tuple(current_states.keys()) != context.current_features:
        raise ValueError("Prepared relationship context does not match current_states")
    if len(history) != len(context.history):
        raise ValueError("Prepared relationship context does not match history")
    if history and id(history) != context.history_object_id:
        # The normal walk-forward path supplies the exact history object. This is
        # only a defensive guard against silently reusing a context for another
        # sample with coincidentally equal length.
        raise ValueError("Prepared relationship context must use the same history object")


def method_a_prepared(
    engine: RelationshipDiscoveryEngine,
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    candidate_sets: Iterable[tuple[str, ...]],
    context: PreparedRelationshipContext,
    *,
    compact_support: bool = False,
) -> list[RelationshipResult]:
    """Exact Method-A compatibility path for the active repository engine.

    The repository's public Method-A implementation is itself candidate-batched.
    Because local Phase 5.8 work may legitimately evolve that implementation,
    duplicating its mathematics here is unsafe: a future change in
    ``relationship.py`` can silently make the optimized kernel numerically
    different.

    Therefore Method A delegates the candidate evaluation to the active engine,
    which guarantees exact equivalence. Method B and the permutation gate remain
    on the prepared/vectorized hot paths. When compact support is requested, only
    the selected Method-A support is converted to the compact matrix after the
    exact engine calculation has completed.

    ``context`` is retained in the interface so the walk-forward orchestration
    can share one prepared context across Method A and Method B without changing
    call sites.
    """
    if not current_states or not history:
        return []
    _validate_context(context, current_states, history)

    candidates = [tuple(candidate) for candidate in candidate_sets]
    if not candidates:
        return []

    # Use the active engine's exact implementation rather than maintaining a
    # second copy of Method-A mathematics in this module.
    exact_results = engine.method_a_similar_states(
        current_states,
        history,
        candidate_sets=candidates,
    )

    if not compact_support:
        return PreparedRelationshipResults(exact_results)

    n = len(history)
    support_matrix = np.zeros((len(exact_results), n), dtype=np.float64)

    # Build a multimap so duplicate (date, return) observations are handled
    # deterministically rather than collapsing onto one history row.
    positions: dict[tuple[date, float], list[int]] = {}
    for index, observation in enumerate(context.history):
        key = (observation.as_of_date, float(observation.stock_return_pct))
        positions.setdefault(key, []).append(index)

    compact_results: list[RelationshipResult] = []
    for row, result in enumerate(exact_results):
        # Support is candidate-local. The same historical observation may be
        # valid support for many different candidate relationships, so a global
        # consumed counter would incorrectly make later candidates unable to
        # reuse that observation.
        consumed: dict[tuple[date, float], int] = {}
        support_observations = tuple(result.supporting_observations)
        support_weights = tuple(result.supporting_weights)
        if not support_observations or not support_weights:
            raise ValueError(
                "Method A compact support requires supporting data from the selection engine"
            )
        if len(support_observations) != len(support_weights):
            raise ValueError("Method A supporting observations/weights are misaligned")

        for observation, weight in zip(support_observations, support_weights):
            key = (observation[0], float(observation[1]))
            available = positions.get(key, ())
            cursor = consumed.get(key, 0)
            if cursor >= len(available):
                raise ValueError(
                    f"Method A support observation {key!r} cannot be located in prepared history"
                )
            support_matrix[row, available[cursor]] = float(weight)
            consumed[key] = cursor + 1

        compact_results.append(
            replace(
                result,
                supporting_observations=(),
                supporting_weights=(),
            )
        )

    return PreparedRelationshipResults(compact_results, support_matrix=support_matrix)


def method_b_prepared(
    engine: RelationshipDiscoveryEngine,
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    candidate_sets: Iterable[tuple[str, ...]],
    context: PreparedRelationshipContext,
    *,
    compact_support: bool = False,
) -> list[RelationshipResult]:
    """Prepared equivalent of the current vectorized Method B implementation."""
    if not current_states or not history:
        return []
    _validate_context(context, current_states, history)

    candidates, index_matrix_all, orders = context.candidate_plan(candidate_sets)
    if not candidates:
        return []

    engine._progress(
        f"Prepared Method B search start | features={len(current_states)} | candidates={len(candidates)} | history={len(history)} | exhaustive=YES | batch={engine.candidate_batch_size}"
    )

    n = len(history)
    batch_size = engine.candidate_batch_size
    min_obs = engine.min_observations
    results: list[RelationshipResult] = []
    result_positions: list[int] = []
    support_matrix_full = (
        np.zeros((len(candidates), len(context.history)), dtype=np.float64)
        if compact_support
        else None
    )
    processed_candidates = 0
    last_progress = 0

    for order in sorted(set(int(value) for value in orders.tolist())):
        candidate_rows = np.flatnonzero(orders == order)
        order_candidates = [candidates[int(row)] for row in candidate_rows]
        order_indices_all = index_matrix_all[candidate_rows]

        for batch_start in range(0, len(order_candidates), batch_size):
            batch_slice = slice(batch_start, batch_start + batch_size)
            batch = order_candidates[batch_slice]
            # candidate_plan pads rows to the maximum candidate order.
            # Slice back to the actual order for this batch so padded -1
            # feature indices never participate in the statistics.
            batch_indices = order_indices_all[batch_slice, :order]
            batch_len = len(batch)
            match_counts = np.sum(
                context.match_matrix[batch_indices], axis=1, dtype=np.float64
            )
            observed_counts = np.sum(
                context.observed_matrix[batch_indices], axis=1, dtype=np.float64
            )
            similarity = match_counts / float(order)
            compared_weight = np.sum(
                context.weighted_observed_matrix[batch_indices],
                axis=1,
                dtype=np.float64,
            )
            family_coverage_pct = observed_counts / float(order) * 100.0
            total_candidate_weight = np.sum(context.weight_vector[batch_indices], axis=1)
            state_coverage_pct = np.divide(
                compared_weight,
                total_candidate_weight[:, None],
                out=np.zeros_like(compared_weight),
                where=total_candidate_weight[:, None] > 0.0,
            ) * 100.0
            valid_mask = (
                (state_coverage_pct >= engine.min_state_coverage_pct)
                & (family_coverage_pct >= engine.min_family_coverage_pct)
                & (total_candidate_weight[:, None] > 0.0)
            )
            valid_counts = np.count_nonzero(valid_mask, axis=1)
            eligible_rows = valid_counts >= min_obs
            if not np.any(eligible_rows):
                processed_candidates += batch_len
                if (
                    processed_candidates - last_progress >= engine.progress_every_candidates
                    or processed_candidates == len(candidates)
                ):
                    engine._progress(
                        f"Prepared Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                    )
                    last_progress = processed_candidates
                continue

            finite_similarity = np.isfinite(similarity)
            finite_valid_mask = valid_mask & finite_similarity
            positive_mask = finite_valid_mask & (similarity > 0.0)
            positive = np.where(positive_mask, similarity, np.nan)
            has_positive = np.any(positive_mask, axis=1)
            scales = np.ones(batch_len, dtype=np.float64)
            if np.any(has_positive):
                scales[has_positive] = np.nanmedian(positive[has_positive], axis=1)
            scales = np.where(np.isfinite(scales), np.maximum(scales, 1e-9), 1.0)

            valid_max = np.max(
                np.where(finite_valid_mask, similarity, -np.inf),
                axis=1,
            )
            weights_full = np.zeros_like(similarity, dtype=np.float64)
            valid_rows_for_weights = np.isfinite(valid_max)
            if np.any(valid_rows_for_weights):
                row_ids = np.flatnonzero(valid_rows_for_weights)
                safe_similarity = similarity[row_ids]
                exponent = (
                    safe_similarity - valid_max[row_ids, None]
                ) / scales[row_ids, None]
                finite_positions = finite_valid_mask[row_ids]
                weights = np.zeros_like(safe_similarity, dtype=np.float64)
                weights[finite_positions] = np.exp(exponent[finite_positions])
                weights_full[row_ids] = weights

            total_weighted = np.sum(weights_full, axis=1)
            usable = eligible_rows & (total_weighted > 0.0)
            if not np.any(usable):
                processed_candidates += batch_len
                if (
                    processed_candidates - last_progress >= engine.progress_every_candidates
                    or processed_candidates == len(candidates)
                ):
                    engine._progress(
                        f"Prepared Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                    )
                    last_progress = processed_candidates
                continue

            weighted_mean = np.divide(
                weights_full @ context.outcomes,
                total_weighted,
                out=np.zeros_like(total_weighted),
                where=total_weighted > 0.0,
            )
            weighted_positive_rate = np.divide(
                weights_full @ (context.outcomes > 0.0).astype(np.float64),
                total_weighted,
                out=np.zeros_like(total_weighted),
                where=total_weighted > 0.0,
            ) * 100.0
            weight_square_sum = np.sum(weights_full * weights_full, axis=1)
            effective_sample_size = np.divide(
                total_weighted * total_weighted,
                weight_square_sum,
                out=np.zeros_like(total_weighted),
                where=weight_square_sum > 0.0,
            )
            weight_concentration = np.divide(
                np.max(weights_full, axis=1),
                total_weighted,
                out=np.zeros_like(total_weighted),
                where=total_weighted > 0.0,
            )

            ordered_weights = weights_full[:, context.outcome_order]
            cumulative = np.cumsum(ordered_weights, axis=1)
            half_threshold = total_weighted / 2.0
            median_reached = cumulative >= half_threshold[:, None]
            median_positions = np.argmax(median_reached, axis=1)
            weighted_median_values = context.ordered_outcomes[median_positions]

            weighted_lift = weighted_mean - context.baseline_mean
            weighted_effect_strength = np.divide(
                np.abs(weighted_lift),
                context.baseline_dispersion,
                out=np.zeros_like(weighted_lift),
                where=context.baseline_dispersion > 0.0,
            )
            weighted_reliability = np.minimum(
                1.0,
                np.sqrt(effective_sample_size / max(len(context.history), 1)),
            )
            weighted_score = weighted_effect_strength * weighted_reliability

            exact_matches = np.count_nonzero(
                match_counts == float(order), axis=1
            ).astype(np.int64)

            midpoint = n // 2
            first_mask = valid_mask[:, :midpoint]
            second_mask = valid_mask[:, midpoint:]
            first_count = first_mask.sum(axis=1).astype(np.float64)
            second_count = second_mask.sum(axis=1).astype(np.float64)
            first_values = context.outcomes[:midpoint][None, :]
            second_values = context.outcomes[midpoint:][None, :]
            first_sum = np.sum(np.where(first_mask, first_values, 0.0), axis=1)
            second_sum = np.sum(np.where(second_mask, second_values, 0.0), axis=1)
            first_mean = np.divide(first_sum, first_count, out=np.zeros_like(first_sum), where=first_count > 0.0)
            second_mean = np.divide(second_sum, second_count, out=np.zeros_like(second_sum), where=second_count > 0.0)
            first_diff = np.where(first_mask, first_values - first_mean[:, None], 0.0)
            second_diff = np.where(second_mask, second_values - second_mean[:, None], 0.0)
            first_var = np.divide(
                np.sum(first_diff * first_diff, axis=1),
                np.maximum(first_count - 1.0, 1.0),
                out=np.zeros_like(first_count),
                where=first_count >= 2.0,
            )
            second_var = np.divide(
                np.sum(second_diff * second_diff, axis=1),
                np.maximum(second_count - 1.0, 1.0),
                out=np.zeros_like(second_count),
                where=second_count >= 2.0,
            )
            pooled_se = np.sqrt(
                first_var / np.maximum(first_count, 1.0)
                + second_var / np.maximum(second_count, 1.0)
            )
            mean_difference = np.abs(first_mean - second_mean)
            sign_agreement = np.where(
                (first_mean == 0.0) | (second_mean == 0.0),
                first_mean == second_mean,
                (first_mean > 0.0) == (second_mean > 0.0),
            )
            consistency = np.clip(
                1.0 - np.divide(
                    mean_difference,
                    np.maximum(engine.stability_sem_multiplier * pooled_se, 1e-12),
                ),
                0.0,
                1.0,
            )
            stability_scores = np.where(sign_agreement, consistency, 0.0)
            stable_flags = (
                sign_agreement
                & (mean_difference <= engine.stability_sem_multiplier * pooled_se)
                & (first_count >= 2.0)
                & (second_count >= 2.0)
            )

            mean_coverage = np.divide(
                np.sum(np.where(valid_mask, state_coverage_pct, 0.0), axis=1),
                np.maximum(valid_counts, 1),
            )
            mean_family_coverage = np.divide(
                np.sum(np.where(valid_mask, family_coverage_pct, 0.0), axis=1),
                np.maximum(valid_counts, 1),
            )

            for row_index, candidate in enumerate(batch):
                if not usable[row_index]:
                    continue
                valid_indices = np.flatnonzero(valid_mask[row_index])
                condition = _condition_key(current_states, candidate) or ()
                if compact_support:
                    candidate_global_position = int(candidate_rows[batch_start + row_index])
                    support_matrix_full[candidate_global_position] = weights_full[row_index]
                    result_positions.append(candidate_global_position)
                results.append(
                    RelationshipResult(
                        method="B",
                        variables=candidate,
                        condition=condition,
                        sample_count=int(valid_counts[row_index]),
                        mean_return_pct=float(weighted_mean[row_index]),
                        median_return_pct=float(weighted_median_values[row_index]),
                        baseline_mean_return_pct=context.baseline_mean,
                        lift_pct=float(weighted_lift[row_index]),
                        positive_rate_pct=float(weighted_positive_rate[row_index]),
                        effect_strength=float(weighted_effect_strength[row_index]),
                        reliability=float(weighted_reliability[row_index]),
                        score=float(weighted_score[row_index]),
                        stable=bool(stable_flags[row_index]),
                        weighted_mean_return_pct=float(weighted_mean[row_index]),
                        weighted_positive_rate_pct=float(weighted_positive_rate[row_index]),
                        effective_sample_size=float(effective_sample_size[row_index]),
                        weight_concentration=float(weight_concentration[row_index]),
                        supporting_observations=(
                            tuple(
                                (context.history[index].as_of_date, float(context.outcomes[index]))
                                for index in valid_indices
                            )
                            if engine.retain_supporting_data and not compact_support
                            else ()
                        ),
                        supporting_weights=(
                            tuple(
                                float(weight)
                                for weight in weights_full[row_index, valid_indices]
                            )
                            if engine.retain_supporting_data and not compact_support
                            else ()
                        ),
                        state_coverage_pct=float(mean_coverage[row_index]),
                        family_coverage_pct=float(mean_family_coverage[row_index]),
                        stability_score=float(stability_scores[row_index]),
                        exact_condition_count=int(exact_matches[row_index]),
                        parameter_provenance=(
                            ("state_relevance_frequency", context.fit_start, context.fit_end),
                            ("method_b_similarity_scale", context.fit_start, context.fit_end),
                        ),
                    )
                )

            processed_candidates += batch_len
            if (
                processed_candidates - last_progress >= engine.progress_every_candidates
                or processed_candidates == len(candidates)
            ):
                engine._progress(
                    f"Prepared Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                )
                last_progress = processed_candidates

    if compact_support:
        ordered_support = support_matrix_full[np.asarray(result_positions, dtype=np.int64)]
        return PreparedRelationshipResults(results, support_matrix=ordered_support)
    return results


def materialize_selected_support(
    result: RelationshipResult,
    current_states: dict[str, str],
    context: PreparedRelationshipContext,
    support_row: np.ndarray,
) -> RelationshipResult:
    """Materialize the selected relationship's exact support after gate selection."""
    support = np.asarray(support_row, dtype=np.float64)
    if support.ndim != 1 or support.shape[0] != len(context.history):
        raise ValueError("selected support row does not match prepared history")

    indices = np.flatnonzero(support > 0.0)
    if result.method == "A":
        # Compact Method-A support records the exact selected neighbours but not
        # their historical ordering. Recreate the established similarity/date
        # ordering for the one relationship that actually survived the gate.
        feature_indices = [context.feature_index[feature] for feature in result.variables]
        order = len(feature_indices)
        similarities = (
            np.sum(context.match_matrix[feature_indices], axis=0, dtype=np.float64)
            / float(order)
        )
        sort_indices = np.lexsort((
            -context.date_ordinals[indices],
            -similarities[indices],
        ))
        indices = indices[sort_indices]

    observations = tuple(
        (context.history[index].as_of_date, float(context.outcomes[index]))
        for index in indices
    )
    weights = tuple(float(support[index]) for index in indices)
    return replace(
        result,
        supporting_observations=observations,
        supporting_weights=weights,
    )
