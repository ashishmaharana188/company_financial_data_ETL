from __future__ import annotations

"""Allocation-light implementation of the existing max-statistic permutation gate.

The statistical contract is unchanged. Candidate/date/value validation is kept;
only the internal representation changes from one Python dict per candidate to a
single dense weight matrix assembled from validated support arrays.
"""

from datetime import date
from random import Random
from typing import Sequence

import numpy as np

from .hardening_4_10 import as_date
from .relationship import RelationshipResult


def search_adjusted_permutation_p_values_fast(
    candidates: Sequence[RelationshipResult],
    baseline_mean: float,
    *,
    family_id: str,
    universe_observations: Sequence[tuple[date, float]],
    permutations: int = 499,
    seed: int = 0,
) -> tuple[float, ...]:
    """Return the same search-wide max-statistic p-values as the current gate."""
    if not candidates:
        return ()
    if permutations < 1:
        raise ValueError("permutations must be >= 1")

    universe: dict[date, float] = {}
    for day, value in universe_observations:
        normalized_day = as_date(day)
        normalized_value = float(value)
        existing = universe.get(normalized_day)
        if existing is not None and abs(existing - normalized_value) > 1e-12:
            raise ValueError(f"conflicting outcomes for shared date {normalized_day}")
        universe[normalized_day] = normalized_value
    if not universe:
        raise ValueError("universe_observations must be non-empty")

    dates = sorted(universe)
    date_ordinals = np.asarray([day.toordinal() for day in dates], dtype=np.int64)
    values = np.asarray([universe[day] for day in dates], dtype=np.float64)
    date_to_index = {day: index for index, day in enumerate(dates)}

    candidate_count = len(candidates)
    weight_matrix = np.zeros((candidate_count, len(dates)), dtype=np.float64)
    total_weights = np.zeros(candidate_count, dtype=np.float64)
    observed_stats = np.zeros(candidate_count, dtype=np.float64)

    for row_index, result in enumerate(candidates):
        observations = result.supporting_observations
        weights = result.supporting_weights
        if len(observations) != len(weights) or not observations:
            raise ValueError("candidate observations and weights must be aligned and non-empty")

        support_dates = np.fromiter(
            (as_date(day).toordinal() for day, _value in observations),
            dtype=np.int64,
            count=len(observations),
        )
        support_weights = np.asarray(
            [max(float(weight), 0.0) for weight in weights],
            dtype=np.float64,
        )
        support_values = np.asarray(
            [float(value) for _day, value in observations],
            dtype=np.float64,
        )

        positive = support_weights > 0.0
        if not np.any(positive):
            raise ValueError("candidate must have positive support weight")

        support_dates = support_dates[positive]
        support_weights = support_weights[positive]
        support_values = support_values[positive]

        positions = np.searchsorted(date_ordinals, support_dates, side="left")
        in_range = positions < len(date_ordinals)
        if not np.all(in_range) or not np.array_equal(date_ordinals[positions], support_dates):
            bad = int(np.flatnonzero(~in_range)[0]) if np.any(~in_range) else 0
            raise ValueError(
                f"candidate date {date.fromordinal(int(support_dates[bad]))} is outside the testing universe"
            )

        universe_values = values[positions]
        if np.any(np.abs(universe_values - support_values) > 1e-12):
            bad = int(np.flatnonzero(np.abs(universe_values - support_values) > 1e-12)[0])
            raise ValueError(
                f"candidate outcome conflicts with testing universe on {date.fromordinal(int(support_dates[bad]))}"
            )

        if len(np.unique(positions)) != len(positions):
            raise ValueError("candidate contains duplicate support dates")

        weight_matrix[row_index, positions] = support_weights
        total = float(np.sum(support_weights))
        total_weights[row_index] = total
        observed_mean = float(np.sum(support_values * support_weights) / total)
        observed_stats[row_index] = abs(observed_mean - float(baseline_mean))

    exceed = np.ones(candidate_count, dtype=np.int64)
    rng = Random(seed)
    permutation_indices = np.arange(len(values), dtype=np.int64)

    for _ in range(permutations):
        # Keep Python Random.shuffle so the pseudorandom stream remains the same
        # family of permutations as the established implementation.
        shuffled_indices = permutation_indices.copy()
        rng.shuffle(shuffled_indices)
        shuffled = values[shuffled_indices]
        permuted_means = (weight_matrix @ shuffled) / total_weights
        max_stat = float(np.max(np.abs(permuted_means - float(baseline_mean))))
        exceed += max_stat >= observed_stats

    return tuple(np.minimum(1.0, exceed / (permutations + 1)).tolist())


def search_adjusted_permutation_p_values_matrix(
    support_matrix: np.ndarray,
    outcomes: np.ndarray,
    baseline_mean: float,
    *,
    permutations: int = 499,
    seed: int = 0,
) -> tuple[float, ...]:
    """Compute the same max-statistic p-values from an already-aligned weight matrix.

    This internal form avoids rebuilding Python date/value dictionaries and support
    tuples during inner selection. Rows correspond exactly to the candidate result
    order and columns correspond exactly to ``outcomes``.
    """
    support = np.asarray(support_matrix, dtype=np.float64)
    values = np.asarray(outcomes, dtype=np.float64)
    if support.ndim != 2:
        raise ValueError("support_matrix must be 2-dimensional")
    if values.ndim != 1 or support.shape[1] != len(values):
        raise ValueError("support_matrix columns must match outcomes")
    if support.shape[0] == 0:
        return ()
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(values)):
        raise ValueError("support_matrix and outcomes must be finite")

    total_weights = np.sum(support, axis=1, dtype=np.float64)
    if np.any(total_weights <= 0.0):
        raise ValueError("every candidate must have positive support weight")

    observed_means = (support @ values) / total_weights
    observed_stats = np.abs(observed_means - float(baseline_mean))
    exceed = np.ones(support.shape[0], dtype=np.int64)

    rng = Random(seed)
    permutation_indices = np.arange(len(values), dtype=np.int64)
    for _ in range(permutations):
        shuffled_indices = permutation_indices.copy()
        rng.shuffle(shuffled_indices)
        shuffled = values[shuffled_indices]
        permuted_means = (support @ shuffled) / total_weights
        max_stat = float(np.max(np.abs(permuted_means - float(baseline_mean))))
        exceed += max_stat >= observed_stats

    return tuple(np.minimum(1.0, exceed / (permutations + 1)).tolist())
