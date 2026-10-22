from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from math import exp, log1p, sqrt
from statistics import median
from typing import Any, Callable, Iterable

import numpy as np

from .hardening_4_10 import (
    categorical_similarity,
    split_stability,
    weighted_median,
)


# ============================================================
# STRUCTURAL RELATIONSHIP RULES
# ============================================================

from .relationship_graph import (
    ALLOWED_RELATIONSHIP_FAMILIES,
    candidate_feature_sets,
    compatible_pair,
    family_for_feature,
    structurally_connected,
)


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
    outcome_end_date: date | None = None


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

    weighted_mean_return_pct: float | None = None
    weighted_positive_rate_pct: float | None = None
    effective_sample_size: float | None = None
    weight_concentration: float | None = None

    # Raw support retained for Phase 5 so the prediction layer can convert
    # the exact Method A/B evidence into three-class probabilities without
    # duplicating the Phase 4 algorithms.
    supporting_observations: tuple[tuple[date, float], ...] = ()
    supporting_weights: tuple[float, ...] = ()

    # Phase 4.10 methodological diagnostics.
    state_coverage_pct: float = 100.0
    family_coverage_pct: float = 100.0
    stability_score: float = 0.0
    exact_condition_count: int | None = None
    parameter_provenance: tuple[tuple[str, date | None, date | None], ...] = ()

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
            "weighted_mean_return_pct": self.weighted_mean_return_pct,
            "weighted_positive_rate_pct": self.weighted_positive_rate_pct,
            "effective_sample_size": self.effective_sample_size,
            "weight_concentration": self.weight_concentration,
            "supporting_observation_count": len(self.supporting_observations),
            "supporting_weight_count": len(self.supporting_weights),
            "state_coverage_pct": self.state_coverage_pct,
            "family_coverage_pct": self.family_coverage_pct,
            "stability_score": self.stability_score,
            "exact_condition_count": self.exact_condition_count,
            "parameter_provenance": [
                {
                    "name": name,
                    "fit_start_date": fit_start,
                    "fit_end_date": fit_end,
                }
                for name, fit_start, fit_end in self.parameter_provenance
            ],
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value))


def _family(feature: str) -> str:
    """Compatibility wrapper for the canonical family registry."""
    return family_for_feature(feature)


def _compatible_pair(
    left: str,
    right: str,
) -> bool:
    """Compatibility wrapper around the canonical graph implementation."""
    return compatible_pair(left, right)


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




def _fast_candidate_similarity(
    feature_set: tuple[str, ...],
    current_states: dict[str, str],
    observed_states: dict[str, str],
    feature_weights: dict[str, float],
) -> tuple[float, float, float]:
    """Fast equivalent of ``categorical_similarity`` for graph-valid candidates.

    The canonical relationship graph guarantees that a valid candidate contains
    at most one feature from each structural family. In that case the generic
    family-normalized similarity reduces exactly to: matched family count /
    candidate family count, with coverage computed from the observed members.
    Keeping this path local to candidate evaluation removes repeated allocation of
    tiny dictionaries inside the Phase 5 walk-forward loop without changing the
    mathematical definition.
    """
    family_count = len(feature_set)
    if family_count == 0:
        return 0.0, 0.0, 0.0

    total_weight = 0.0
    compared_weight = 0.0
    matched_families = 0
    observed_families = 0

    for feature in feature_set:
        weight = max(float(feature_weights.get(feature, 1.0)), 0.0)
        total_weight += weight
        observed = observed_states.get(feature)
        if observed is None:
            continue
        compared_weight += weight
        observed_families += 1
        if observed == current_states[feature]:
            matched_families += 1

    if total_weight <= 0.0:
        return 0.0, 0.0, 0.0

    score = matched_families / family_count
    state_coverage_pct = compared_weight / total_weight * 100.0
    family_coverage_pct = observed_families / family_count * 100.0
    return score, state_coverage_pct, family_coverage_pct


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


def _structurally_connected(features: tuple[str, ...]) -> bool:
    """Compatibility wrapper around the canonical structural graph."""
    return structurally_connected(features)


def _candidate_feature_sets(
    features: Iterable[str],
    max_order: int,
) -> list[tuple[str, ...]]:
    """Compatibility wrapper around the single canonical candidate graph."""
    return candidate_feature_sets(features, max_order)


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
        min_state_coverage_pct: float = 75.0,
        min_family_coverage_pct: float = 75.0,
        stability_sem_multiplier: float = 2.0,
        adaptive_higher_order: bool = False,
        progress_callback: Callable[[str], None] | None = None,
        progress_every_candidates: int = 1000,
        candidate_batch_size: int = 512,
        retain_supporting_data: bool = True,
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        if min_observations < 1:
            raise ValueError(
                "min_observations must be >= 1"
            )

        if not 0.0 < min_state_coverage_pct <= 100.0:
            raise ValueError("min_state_coverage_pct must be in (0, 100]")
        if not 0.0 < min_family_coverage_pct <= 100.0:
            raise ValueError("min_family_coverage_pct must be in (0, 100]")
        if stability_sem_multiplier <= 0.0:
            raise ValueError("stability_sem_multiplier must be > 0")

        self.max_order = max_order
        self.min_observations = min_observations
        self.min_state_coverage_pct = float(min_state_coverage_pct)
        self.min_family_coverage_pct = float(min_family_coverage_pct)
        self.stability_sem_multiplier = float(stability_sem_multiplier)
        self.adaptive_higher_order = bool(adaptive_higher_order)
        self.progress_callback = progress_callback
        self.progress_every_candidates = max(1, int(progress_every_candidates))
        self.candidate_batch_size = max(32, int(candidate_batch_size))
        self.retain_supporting_data = bool(retain_supporting_data)

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

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
        candidate_sets: Iterable[tuple[str, ...]] | None = None,
    ) -> list[RelationshipResult]:
        """Evaluate Method A for an explicit candidate universe.

        When ``candidate_sets`` is supplied, the candidate universe is unchanged
        but feature/state comparisons are materialized once per call and reused
        across candidates. This is a performance optimization only.
        """
        if not current_states or not history:
            return []

        baseline = history
        features = list(current_states.keys())
        candidates = (
            [tuple(candidate) for candidate in candidate_sets]
            if candidate_sets is not None
            else _candidate_feature_sets(features, self.max_order)
        )
        if not candidates:
            return []

        state_weights = _adaptive_relevance_weights(history)
        feature_index = {feature: idx for idx, feature in enumerate(features)}
        feature_weight_vector = np.asarray(
            [
                max(float(state_weights.get((feature, current_states[feature]), 1.0)), 0.0)
                for feature in features
            ],
            dtype=np.float64,
        )
        n = len(history)
        match_matrix = np.zeros((len(features), n), dtype=np.uint8)
        observed_matrix = np.zeros((len(features), n), dtype=np.uint8)
        for feature, idx in feature_index.items():
            current_state = current_states[feature]
            observed_matrix[idx] = np.fromiter(
                (1 if observation.states.get(feature) is not None else 0 for observation in history),
                dtype=np.uint8, count=n,
            )
            match_matrix[idx] = np.fromiter(
                (1 if observation.states.get(feature) == current_state else 0 for observation in history),
                dtype=np.uint8, count=n,
            )

        outcomes = np.asarray([float(item.stock_return_pct) for item in history], dtype=np.float64)
        baseline_mean = float(np.mean(outcomes))
        baseline_dispersion = float(np.sqrt(np.mean((outcomes - baseline_mean) ** 2)))
        fit_dates = [item.as_of_date for item in history]
        fit_start = min(fit_dates) if fit_dates else None
        fit_end = max(fit_dates) if fit_dates else None
        date_ordinals = np.asarray([item.as_of_date.toordinal() for item in history], dtype=np.int64)
        min_obs = self.min_observations
        results: list[RelationshipResult] = []

        by_order: dict[int, list[tuple[str, ...]]] = {}
        for candidate in candidates:
            by_order.setdefault(len(candidate), []).append(candidate)

        for order in sorted(by_order):
            order_candidates = by_order[order]
            batch_size = self.candidate_batch_size
            for batch_start in range(0, len(order_candidates), batch_size):
                batch = order_candidates[batch_start:batch_start + batch_size]
                index_matrix = np.asarray(
                    [[feature_index[feature] for feature in candidate] for candidate in batch],
                    dtype=np.int32,
                )
                candidate_weights = feature_weight_vector[index_matrix]
                observed = observed_matrix[index_matrix]
                matches = match_matrix[index_matrix]
                similarity = np.sum(matches, axis=1, dtype=np.float64) / float(order)
                compared_weight = np.sum(
                    candidate_weights[:, :, None] * observed, axis=1, dtype=np.float64
                )
                total_candidate_weight = np.sum(candidate_weights, axis=1)
                state_coverage_pct = np.divide(
                    compared_weight,
                    total_candidate_weight[:, None],
                    out=np.zeros_like(compared_weight),
                    where=total_candidate_weight[:, None] > 0.0,
                ) * 100.0
                family_coverage_pct = np.sum(observed, axis=1, dtype=np.float64) / float(order) * 100.0
                valid_mask = (
                    (state_coverage_pct >= self.min_state_coverage_pct)
                    & (family_coverage_pct >= self.min_family_coverage_pct)
                    & (total_candidate_weight[:, None] > 0.0)
                )
                valid_counts = np.count_nonzero(valid_mask, axis=1)

                for row_index, candidate in enumerate(batch):
                    if valid_counts[row_index] < min_obs:
                        continue
                    valid_indices = np.flatnonzero(valid_mask[row_index] & (similarity[row_index] > 0.0))
                    if len(valid_indices) < min_obs:
                        continue
                    neighbour_count = min(
                        len(valid_indices),
                        max(min_obs, int(max(1, round(sqrt(len(valid_indices)))))),
                    )
                    sort_indices = np.lexsort((
                        -date_ordinals[valid_indices],
                        -similarity[row_index, valid_indices],
                    ))
                    selected_indices = valid_indices[sort_indices[:neighbour_count]]
                    selected = [history[index] for index in selected_indices]
                    result = _summary_result(
                        method='A',
                        variables=candidate,
                        condition=tuple(f'{feature}~={current_states[feature]}' for feature in candidate),
                        matching=selected,
                        baseline=baseline,
                    )
                    if result is None:
                        continue
                    ordered_neighbours = sorted(selected, key=lambda observation: observation.as_of_date)
                    stability = split_stability(
                        [item.stock_return_pct for item in ordered_neighbours],
                        sem_multiplier=self.stability_sem_multiplier,
                    )
                    mean_similarity = float(np.mean(similarity[row_index, selected_indices]))
                    results.append(
                        RelationshipResult(
                            **{
                                **result.__dict__,
                                'score': result.score * mean_similarity,
                                'stable': stability.stable,
                                'supporting_observations': (
                                    tuple((item.as_of_date, float(item.stock_return_pct)) for item in selected)
                                    if self.retain_supporting_data else ()
                                ),
                                'supporting_weights': (
                                    tuple(1.0 for _ in selected)
                                    if self.retain_supporting_data else ()
                                ),
                                'state_coverage_pct': float(np.mean(state_coverage_pct[row_index, selected_indices])),
                                'family_coverage_pct': float(np.mean(family_coverage_pct[row_index, selected_indices])),
                                'stability_score': stability.stability_score,
                                'parameter_provenance': (
                                    ('state_relevance_frequency', fit_start, fit_end),
                                    ('method_a_adaptive_neighbor_count', fit_start, fit_end),
                                ),
                            }
                        )
                    )

                if self.progress_callback is not None:
                    processed = min(batch_start + len(batch), len(order_candidates))
                    self._progress(
                        f'Method A candidate progress {processed}/{len(order_candidates)} | results={len(results)}'
                    )

        return results

    # --------------------------------------------------------
    # Method B
    # --------------------------------------------------------

    def method_b_conditioned_distribution(
        self,
        current_states: dict[str, str],
        history: list[HistoricalRelationshipObservation],
        candidate_sets: Iterable[tuple[str, ...]] | None = None,
    ) -> list[RelationshipResult]:
        """
        Method B: weighted historical distribution conditioned on the current
        state.

        The candidate universe remains exhaustive for ``max_order``. The
        performance implementation only changes *how* the exact statistics are
        calculated: feature/state matches are materialized once, candidates are
        evaluated in vectorized batches, and the outcome ordering is reused.
        No candidate is pruned for performance.
        """
        if not current_states or not history:
            return []

        baseline = history
        features = list(current_states.keys())
        relevance = _adaptive_relevance_weights(history)
        candidates = (
            [tuple(candidate) for candidate in candidate_sets]
            if candidate_sets is not None
            else _candidate_feature_sets(features, self.max_order)
        )
        self._progress(
            f"Method B search start | features={len(features)} | candidates={len(candidates)} | history={len(history)} | exhaustive=YES | batch={self.candidate_batch_size}"
        )
        if not candidates:
            return []

        n = len(history)
        feature_index = {feature: idx for idx, feature in enumerate(features)}
        match_matrix = np.zeros((len(features), n), dtype=np.uint8)
        observed_matrix = np.zeros((len(features), n), dtype=np.uint8)
        weight_vector = np.empty(len(features), dtype=np.float64)

        # This is the only feature x history pass. All subsequent candidate
        # calculations reuse these arrays.
        for feature, idx in feature_index.items():
            current_state = current_states[feature]
            weight_vector[idx] = max(
                float(relevance.get((feature, current_state), 1.0)),
                0.0,
            )
            observed = np.fromiter(
                (1 if observation.states.get(feature) is not None else 0 for observation in history),
                dtype=np.uint8,
                count=n,
            )
            matches = np.fromiter(
                (1 if observation.states.get(feature) == current_state else 0 for observation in history),
                dtype=np.uint8,
                count=n,
            )
            observed_matrix[idx] = observed
            match_matrix[idx] = matches

        outcomes = np.asarray(
            [float(observation.stock_return_pct) for observation in history],
            dtype=np.float64,
        )
        outcome_order = np.argsort(outcomes, kind="stable")
        ordered_outcomes = outcomes[outcome_order]
        baseline_mean = float(np.mean(outcomes))
        baseline_dispersion = float(np.sqrt(np.mean((outcomes - baseline_mean) ** 2)))
        fit_dates = [item.as_of_date for item in history]
        fit_start = min(fit_dates) if fit_dates else None
        fit_end = max(fit_dates) if fit_dates else None
        min_obs = self.min_observations
        batch_size = self.candidate_batch_size
        results: list[RelationshipResult] = []
        last_progress = 0

        processed_candidates = 0
        orders = sorted({len(candidate) for candidate in candidates})
        for order in orders:
            order_candidates = [candidate for candidate in candidates if len(candidate) == order]
            for batch_start in range(0, len(order_candidates), batch_size):
                batch = order_candidates[batch_start:batch_start + batch_size]
                batch_len = len(batch)

                index_matrix = np.asarray(
                    [[feature_index[feature] for feature in candidate] for candidate in batch],
                    dtype=np.int32,
                )
                candidate_weights = weight_vector[index_matrix]
    
                # Shape: batch x observations. This exactly reproduces the scalar
                # _fast_candidate_similarity calculations for graph-valid candidates.
                similarity = (
                    np.sum(match_matrix[index_matrix], axis=1, dtype=np.float64)
                    / float(order)
                )
                compared_weight = np.sum(
                    candidate_weights[:, :, None] * observed_matrix[index_matrix],
                    axis=1,
                    dtype=np.float64,
                )
                family_coverage_pct = (
                    np.sum(observed_matrix[index_matrix], axis=1, dtype=np.float64)
                    / float(order)
                    * 100.0
                )
                total_candidate_weight = np.sum(candidate_weights, axis=1)
                state_coverage_pct = np.divide(
                    compared_weight,
                    total_candidate_weight[:, None],
                    out=np.zeros_like(compared_weight),
                    where=total_candidate_weight[:, None] > 0.0,
                ) * 100.0
                valid_mask = (
                    (state_coverage_pct >= self.min_state_coverage_pct)
                    & (family_coverage_pct >= self.min_family_coverage_pct)
                    & (total_candidate_weight[:, None] > 0.0)
                )
                valid_counts = np.count_nonzero(valid_mask, axis=1)
                eligible_rows = valid_counts >= min_obs
                if not np.any(eligible_rows):
                    processed_candidates += batch_len
                    if processed_candidates - last_progress >= self.progress_every_candidates or processed_candidates == len(candidates):
                        self._progress(
                            f"Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                        )
                        last_progress = processed_candidates
                    continue
    
                # Similarity can be zero for every valid historical row, and in
                # defensive cases it can contain non-finite values. Do not build an
                # all-NaN row and do not rely on NaN * 0 cleanup: IEEE arithmetic
                # keeps NaN alive, which then contaminates the weighted statistics.
                finite_similarity = np.isfinite(similarity)
                finite_valid_mask = valid_mask & finite_similarity
                positive_mask = finite_valid_mask & (similarity > 0.0)
                positive = np.where(positive_mask, similarity, np.nan)
                has_positive = np.any(positive_mask, axis=1)
                scales = np.ones(batch_len, dtype=np.float64)
                if np.any(has_positive):
                    # Only rows with at least one positive value reach nanmedian,
                    # so the reduction cannot emit an all-NaN warning.
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
                    if processed_candidates - last_progress >= self.progress_every_candidates or processed_candidates == len(candidates):
                        self._progress(
                            f"Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                        )
                        last_progress = processed_candidates
                    continue
    
                weighted_mean = np.divide(
                    weights_full @ outcomes,
                    total_weighted,
                    out=np.zeros_like(total_weighted),
                    where=total_weighted > 0.0,
                )
                weighted_positive_rate = np.divide(
                    weights_full @ (outcomes > 0.0).astype(np.float64),
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
    
                ordered_weights = weights_full[:, outcome_order]
                cumulative = np.cumsum(ordered_weights, axis=1)
                half_threshold = total_weighted / 2.0
                median_reached = cumulative >= half_threshold[:, None]
                median_positions = np.argmax(median_reached, axis=1)
                weighted_median_values = ordered_outcomes[median_positions]
    
                weighted_lift = weighted_mean - baseline_mean
                weighted_effect_strength = np.divide(
                    np.abs(weighted_lift),
                    baseline_dispersion,
                    out=np.zeros_like(weighted_lift),
                    where=baseline_dispersion > 0.0,
                )
                weighted_reliability = np.minimum(
                    1.0,
                    np.sqrt(effective_sample_size / max(len(baseline), 1)),
                )
                weighted_score = weighted_effect_strength * weighted_reliability
    
                # Exact-condition counts use the comparable row mask, not positive
                # floating weights, matching the scalar implementation.
                exact_matches = np.all(
                    match_matrix[index_matrix].astype(bool),
                    axis=1,
                ).sum(axis=1).astype(np.int64)
    
                # Vectorized split-half stability. This is exactly the same
                # chronological split calculation as split_stability(), expressed
                # on the batch masks to avoid another observation loop.
                midpoint = n // 2
                first_mask = valid_mask[:, :midpoint]
                second_mask = valid_mask[:, midpoint:]
                first_count = first_mask.sum(axis=1).astype(np.float64)
                second_count = second_mask.sum(axis=1).astype(np.float64)
                first_values = outcomes[:midpoint][None, :]
                second_values = outcomes[midpoint:][None, :]
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
                    (first_mean == second_mean),
                    (first_mean > 0.0) == (second_mean > 0.0),
                )
                consistency = np.clip(
                    1.0 - np.divide(
                        mean_difference,
                        np.maximum(self.stability_sem_multiplier * pooled_se, 1e-12),
                    ),
                    0.0,
                    1.0,
                )
                stability_scores = np.where(sign_agreement, consistency, 0.0)
                stable_flags = (
                    sign_agreement
                    & (mean_difference <= self.stability_sem_multiplier * pooled_se)
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
                    results.append(
                        RelationshipResult(
                            method="B",
                            variables=candidate,
                            condition=condition,
                            sample_count=int(valid_counts[row_index]),
                            mean_return_pct=float(weighted_mean[row_index]),
                            median_return_pct=float(weighted_median_values[row_index]),
                            baseline_mean_return_pct=baseline_mean,
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
                                    (history[index].as_of_date, float(outcomes[index]))
                                    for index in valid_indices
                                )
                                if self.retain_supporting_data
                                else ()
                            ),
                            supporting_weights=(
                                tuple(
                                    float(weight)
                                    for weight in weights_full[row_index, valid_indices]
                                )
                                if self.retain_supporting_data
                                else ()
                            ),
                            state_coverage_pct=float(mean_coverage[row_index]),
                            family_coverage_pct=float(mean_family_coverage[row_index]),
                            stability_score=float(stability_scores[row_index]),
                            exact_condition_count=int(exact_matches[row_index]),
                            parameter_provenance=(
                                ("state_relevance_frequency", fit_start, fit_end),
                                ("method_b_similarity_scale", fit_start, fit_end),
                            ),
                        )
                    )
    
                processed_candidates += batch_len
                if processed_candidates - last_progress >= self.progress_every_candidates or processed_candidates == len(candidates):
                    self._progress(
                        f"Method B candidate progress {processed_candidates}/{len(candidates)} | results={len(results)}"
                    )
                    last_progress = processed_candidates
    
        return results

    def materialize_support(
        self,
        result: RelationshipResult,
        history: list[HistoricalRelationshipObservation],
    ) -> RelationshipResult:
        """Materialize exact support for one selected Method-B relationship."""
        if result.method != "B" or result.supporting_observations:
            return result
        if not history or not result.variables:
            return result

        current_states: dict[str, str] = {}
        for item in result.condition:
            if "=" not in item:
                continue
            feature, state = item.split("=", 1)
            current_states[feature] = state
        if len(current_states) != len(result.variables):
            return result

        relevance = _adaptive_relevance_weights(history)
        feature_weights = {
            feature: max(
                float(relevance.get((feature, current_states[feature]), 1.0)),
                0.0,
            )
            for feature in result.variables
        }
        total_weight = sum(feature_weights.values())
        if total_weight <= 0.0:
            return result

        scored: list[tuple[HistoricalRelationshipObservation, float]] = []
        for observation in history:
            matched = 0
            observed_count = 0
            compared_weight = 0.0
            for feature in result.variables:
                observed = observation.states.get(feature)
                weight = feature_weights[feature]
                if observed is None:
                    continue
                observed_count += 1
                compared_weight += weight
                if observed == current_states[feature]:
                    matched += 1
            state_coverage = compared_weight / total_weight * 100.0
            family_coverage = observed_count / len(result.variables) * 100.0
            if (
                state_coverage >= self.min_state_coverage_pct
                and family_coverage >= self.min_family_coverage_pct
            ):
                scored.append((observation, matched / len(result.variables)))

        if len(scored) < self.min_observations:
            return result

        similarities = [similarity for _observation, similarity in scored]
        positive_similarity = [value for value in similarities if value > 0.0]
        similarity_scale = median(positive_similarity) if positive_similarity else 1.0
        similarity_scale = max(similarity_scale, 1e-9)
        max_similarity = max(similarities)
        weights = [
            exp((similarity - max_similarity) / similarity_scale)
            for _observation, similarity in scored
        ]

        return replace(
            result,
            supporting_observations=tuple(
                (observation.as_of_date, float(observation.stock_return_pct))
                for observation, _weight in scored
            ),
            supporting_weights=tuple(float(weight) for weight in weights),
        )

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
