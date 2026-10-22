from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import sqrt
from random import Random
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np


def as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass(frozen=True)
class SimilarityDiagnostics:
    score: float
    state_coverage_pct: float
    family_coverage_pct: float
    compared_weight: float
    total_weight: float


def categorical_similarity(
    current_states: dict[str, str],
    observed_states: dict[str, str],
    feature_weights: dict[str, float],
    *,
    family_lookup: Any,
) -> SimilarityDiagnostics:
    """
    Compute coverage-aware categorical similarity.

    Missing observed states are treated as unavailable evidence, not a match.
    They also contribute zero agreement to the full current-state denominator,
    so missingness cannot increase similarity.

    Family scores are normalized by the full family weight, then averaged so a
    family with more variables does not dominate mechanically.
    """
    if not current_states:
        return SimilarityDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    families: dict[str, list[str]] = {}
    for feature in current_states:
        families.setdefault(str(family_lookup(feature)), []).append(feature)

    total_weight = sum(
        max(float(feature_weights.get(feature, 1.0)), 0.0)
        for feature in current_states
    )
    if total_weight <= 0.0:
        return SimilarityDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    compared_weight = 0.0
    family_scores: list[float] = []
    observed_families = 0

    for family_features in families.values():
        family_weight = sum(
            max(float(feature_weights.get(feature, 1.0)), 0.0)
            for feature in family_features
        )
        if family_weight <= 0.0:
            continue

        matched_weight = 0.0
        family_observed = False
        for feature in family_features:
            observed = observed_states.get(feature)
            weight = max(float(feature_weights.get(feature, 1.0)), 0.0)
            if observed is None:
                continue
            family_observed = True
            compared_weight += weight
            if observed == current_states[feature]:
                matched_weight += weight

        # Denominator is the complete family weight. Missing variables therefore
        # lower the family's maximum achievable similarity rather than helping it.
        family_scores.append(matched_weight / family_weight)
        if family_observed:
            observed_families += 1

    state_coverage_pct = compared_weight / total_weight * 100.0
    family_coverage_pct = (
        observed_families / len(families) * 100.0
        if families else 0.0
    )
    score = sum(family_scores) / len(family_scores) if family_scores else 0.0

    return SimilarityDiagnostics(
        score=max(0.0, min(score, 1.0)),
        state_coverage_pct=max(0.0, min(state_coverage_pct, 100.0)),
        family_coverage_pct=max(0.0, min(family_coverage_pct, 100.0)),
        compared_weight=compared_weight,
        total_weight=total_weight,
    )


@dataclass(frozen=True)
class StabilityDiagnostics:
    stable: bool
    stability_score: float
    first_mean: float
    second_mean: float
    mean_difference: float
    pooled_standard_error: float
    sign_agreement: bool


def split_stability(
    outcomes: Sequence[float],
    *,
    sem_multiplier: float = 2.0,
) -> StabilityDiagnostics:
    """
    Evaluate chronological split-half stability.

    Stability now reflects direction plus magnitude consistency. The two-half
    mean difference is compared with the pooled standard error. The returned
    score is continuous in [0, 1]; the boolean flag is a conservative
    diagnostic, not a claim of statistical proof.
    """
    values = [float(value) for value in outcomes]
    if len(values) < 4:
        return StabilityDiagnostics(False, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    midpoint = len(values) // 2
    first = values[:midpoint]
    second = values[midpoint:]

    first_mean = mean(first)
    second_mean = mean(second)

    def variance(sample: list[float], sample_mean: float) -> float:
        if len(sample) < 2:
            return 0.0
        return sum((value - sample_mean) ** 2 for value in sample) / (len(sample) - 1)

    first_var = variance(first, first_mean)
    second_var = variance(second, second_mean)
    pooled_se = sqrt(
        first_var / max(len(first), 1)
        + second_var / max(len(second), 1)
    )

    difference = abs(first_mean - second_mean)
    scale = max(pooled_se, 1e-12)

    if first_mean == 0.0 or second_mean == 0.0:
        sign_agreement = first_mean == second_mean == 0.0
    else:
        sign_agreement = (first_mean > 0.0) == (second_mean > 0.0)

    consistency = max(0.0, min(1.0, 1.0 - difference / (sem_multiplier * scale)))
    stability_score = consistency if sign_agreement else 0.0
    stable = bool(sign_agreement and difference <= sem_multiplier * scale)

    return StabilityDiagnostics(
        stable=stable,
        stability_score=stability_score,
        first_mean=first_mean,
        second_mean=second_mean,
        mean_difference=difference,
        pooled_standard_error=pooled_se,
        sign_agreement=sign_agreement,
    )


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the weighted median of finite observations."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and aligned")

    rows = sorted(
        (float(value), max(float(weight), 0.0))
        for value, weight in zip(values, weights)
        if weight > 0.0
    )
    if not rows:
        return 0.0

    total = sum(weight for _, weight in rows)
    threshold = total / 2.0
    running = 0.0
    for value, weight in rows:
        running += weight
        if running >= threshold:
            return value
    return rows[-1][0]


@dataclass(frozen=True)
class MultipleTestingResult:
    candidate_count: int
    tested_count: int
    alpha: float
    adjusted_alpha: float
    raw_p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    accepted_indices: tuple[int, ...]


def empirical_permutation_p_value(
    outcomes: Sequence[float],
    weights: Sequence[float],
    baseline_mean: float,
    *,
    permutations: int = 499,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for weighted mean lift versus baseline."""
    if len(outcomes) != len(weights) or not outcomes:
        return 1.0
    if permutations < 1:
        raise ValueError("permutations must be >= 1")

    clean = [float(value) for value in outcomes]
    clean_weights = [max(float(weight), 0.0) for weight in weights]
    total_weight = sum(clean_weights)
    if total_weight <= 0.0:
        return 1.0

    observed = abs(
        sum(value * weight for value, weight in zip(clean, clean_weights))
        / total_weight
        - float(baseline_mean)
    )

    rng = Random(seed)
    exceed = 1
    for _ in range(permutations):
        shuffled = clean[:]
        rng.shuffle(shuffled)
        perm_mean = sum(
            value * weight for value, weight in zip(shuffled, clean_weights)
        ) / total_weight
        if abs(perm_mean - float(baseline_mean)) >= observed:
            exceed += 1

    return min(1.0, exceed / (permutations + 1))


@dataclass(frozen=True)
class SearchAdjustedPermutationResult:
    """Permutation evidence adjusted for the complete candidate search family."""

    family_id: str
    candidate_count: int
    permutation_count: int
    raw_p_values: tuple[float, ...]
    max_statistic_p_values: tuple[float, ...]


def search_adjusted_permutation_p_values(
    candidates: Sequence[tuple[Sequence[tuple[date, float]], Sequence[float]]],
    baseline_mean: float,
    *,
    family_id: str,
    universe_observations: Sequence[tuple[date, float]],
    permutations: int = 499,
    seed: int = 0,
    compute_raw_p_values: bool = True,
) -> SearchAdjustedPermutationResult:
    """
    Compute max-statistic permutation p-values across a complete candidate family.

    Candidate supports are keyed by observation date. Outcomes are permuted once
    across the shared union of historical dates, then the same permutation is
    evaluated against every candidate's fixed state-derived weights. The maximum
    absolute lift across the candidate search is therefore part of the null
    distribution. This directly addresses search/selection inflation.
    """
    if not candidates:
        return SearchAdjustedPermutationResult(
            family_id=str(family_id),
            candidate_count=0,
            permutation_count=0,
            raw_p_values=(),
            max_statistic_p_values=(),
        )
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

    candidate_maps: list[dict[date, tuple[float, float]]] = []
    raw_p_values: list[float] = []
    observed_stats: list[float] = []

    for observations, weights in candidates:
        if len(observations) != len(weights) or not observations:
            raise ValueError("candidate observations and weights must be aligned and non-empty")
        mapping: dict[date, tuple[float, float]] = {}
        for (day, value), weight in zip(observations, weights):
            normalized_day = as_date(day)
            normalized_value = float(value)
            normalized_weight = max(float(weight), 0.0)
            if normalized_day not in universe:
                raise ValueError(f"candidate date {normalized_day} is outside the testing universe")
            if normalized_weight <= 0.0:
                continue
            expected_value = universe[normalized_day]
            if abs(expected_value - normalized_value) > 1e-12:
                raise ValueError(f"candidate outcome conflicts with testing universe on {normalized_day}")
            mapping[normalized_day] = (normalized_value, normalized_weight)

        if not mapping:
            raise ValueError("candidate must have positive support weight")
        candidate_maps.append(mapping)

    dates = sorted(universe)
    values = [universe[day] for day in dates]
    for mapping in candidate_maps:
        total_weight = sum(weight for _value, weight in mapping.values())
        observed_mean = sum(value * weight for value, weight in mapping.values()) / total_weight
        observed = abs(observed_mean - float(baseline_mean))
        observed_stats.append(observed)
        candidate_weights = [mapping.get(day, (universe[day], 0.0))[1] for day in dates]
        if compute_raw_p_values:
            raw_p_values.append(
                empirical_permutation_p_value(
                    values,
                    candidate_weights,
                    baseline_mean,
                    permutations=permutations,
                    seed=seed + len(raw_p_values) + 1,
                )
            )
        else:
            raw_p_values.append(float("nan"))

    exceed = np.ones(len(candidate_maps), dtype=np.int64)
    rng = Random(seed)

    if not compute_raw_p_values:
        # The FDR gate consumes only the search-wide max-statistic p-values.
        # Build one dense candidate-weight matrix and evaluate all candidates
        # for each shared permutation with a BLAS-backed matrix multiply.
        # This preserves the exact candidate universe and permutation count;
        # only the implementation of the same statistic changes.
        candidate_count = len(candidate_maps)
        date_index = {day: index for index, day in enumerate(dates)}
        weight_matrix = np.zeros((candidate_count, len(dates)), dtype=np.float64)
        total_weights = np.zeros(candidate_count, dtype=np.float64)
        for candidate_index, mapping in enumerate(candidate_maps):
            total = 0.0
            for day, (_value, weight) in mapping.items():
                column = date_index[day]
                weight_matrix[candidate_index, column] = float(weight)
                total += float(weight)
            total_weights[candidate_index] = total

        observed_array = np.asarray(observed_stats, dtype=np.float64)
        base_values = np.asarray(values, dtype=np.float64)
        for _ in range(permutations):
            permutation_indices = list(range(len(base_values)))
            rng.shuffle(permutation_indices)
            shuffled = base_values[np.asarray(permutation_indices, dtype=np.int64)]
            permuted_means = (weight_matrix @ shuffled) / total_weights
            max_stat = float(np.max(np.abs(permuted_means - float(baseline_mean))))
            exceed += (max_stat >= observed_array)
    else:
        # Keep the original scalar path for callers that explicitly request
        # per-candidate raw permutation p-values.
        exceed = exceed.tolist()
        rng = Random(seed)
        for _ in range(permutations):
            shuffled = list(values)
            rng.shuffle(shuffled)
            permuted_values = dict(zip(dates, shuffled))
            max_stat = 0.0
            for mapping in candidate_maps:
                total_weight = sum(weight for _value, weight in mapping.values())
                perm_mean = sum(
                    permuted_values[day] * weight
                    for day, (_value, weight) in mapping.items()
                ) / total_weight
                max_stat = max(max_stat, abs(perm_mean - float(baseline_mean)))
            for index, observed in enumerate(observed_stats):
                if max_stat >= observed:
                    exceed[index] += 1

    adjusted = tuple(min(1.0, count / (permutations + 1)) for count in exceed)
    return SearchAdjustedPermutationResult(
        family_id=str(family_id),
        candidate_count=len(candidate_maps),
        permutation_count=permutations,
        raw_p_values=tuple(raw_p_values),
        max_statistic_p_values=adjusted,
    )


@dataclass(frozen=True)
class SearchProcedurePermutationResult:
    """Permutation result for the complete discovery/search procedure."""

    family_id: str
    method: str
    observed_statistic: float
    permutation_count: int
    successful_permutations: int
    empty_search_permutations: int
    empirical_p_value: float
    null_max_statistics: tuple[float, ...]


def search_procedure_permutation_p_value(
    current_states: dict[str, str],
    history: Sequence[Any],
    engine: Any,
    *,
    method: str = "COMBINED",
    family_id: str = "",
    permutations: int = 199,
    seed: int = 0,
) -> SearchProcedurePermutationResult:
    """
    Run the full relationship discovery/search procedure under permutation.

    Unlike a fixed-candidate permutation test, each permutation shuffles the
    historical outcomes, rebuilds the historical observations, reruns Method A
    and/or Method B discovery, reruns the relationship ranking, and records the
    selected relationship's absolute score. The null therefore includes the
    model-search step itself.

    This function is a methodological audit. In the nested outer validator,
    candidate selection remains chronological and the outer outcome remains
    untouched; this function is used to quantify search inflation separately.
    """
    if not current_states or not history:
        return SearchProcedurePermutationResult(
            family_id=str(family_id),
            method=str(method),
            observed_statistic=0.0,
            permutation_count=0,
            successful_permutations=0,
            empty_search_permutations=0,
            empirical_p_value=1.0,
            null_max_statistics=(),
        )
    if method not in {"A", "B", "COMBINED"}:
        raise ValueError("method must be 'A', 'B', or 'COMBINED'")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")

    from dataclasses import replace
    from .ranking import rank_relationships

    clean_history = [
        item for item in history
        if getattr(item, "stock_return_pct", None) is not None
    ]
    if not clean_history:
        return SearchProcedurePermutationResult(
            family_id=str(family_id),
            method=str(method),
            observed_statistic=0.0,
            permutation_count=0,
            successful_permutations=0,
            empty_search_permutations=0,
            empirical_p_value=1.0,
            null_max_statistics=(),
        )

    def discover(rows: Sequence[Any]) -> list[Any]:
        results: list[Any] = []
        if method in {"A", "COMBINED"}:
            results.extend(engine.method_a_similar_states(current_states, list(rows)))
        if method in {"B", "COMBINED"}:
            results.extend(engine.method_b_conditioned_distribution(current_states, list(rows)))
        return results

    observed_rankings = rank_relationships(discover(clean_history))
    observed_statistic = (
        abs(float(observed_rankings[0].best_score))
        if observed_rankings
        else 0.0
    )

    original_outcomes = [float(item.stock_return_pct) for item in clean_history]
    rng = Random(seed)
    null_statistics: list[float] = []
    successful = 0
    empty = 0

    for _ in range(permutations):
        shuffled = original_outcomes[:]
        rng.shuffle(shuffled)
        permuted_rows = [
            replace(item, stock_return_pct=shuffled[index])
            for index, item in enumerate(clean_history)
        ]
        rankings = rank_relationships(discover(permuted_rows))
        if not rankings:
            null_statistics.append(0.0)
            empty += 1
            continue
        successful += 1
        null_statistics.append(abs(float(rankings[0].best_score)))

    exceed = sum(
        statistic >= observed_statistic
        for statistic in null_statistics
    )
    empirical = (
        (1 + exceed) / (len(null_statistics) + 1)
        if null_statistics
        else 1.0
    )

    return SearchProcedurePermutationResult(
        family_id=str(family_id),
        method=str(method),
        observed_statistic=observed_statistic,
        permutation_count=len(null_statistics),
        successful_permutations=successful,
        empty_search_permutations=empty,
        empirical_p_value=min(1.0, empirical),
        null_max_statistics=tuple(null_statistics),
    )


@dataclass(frozen=True)
class TestingFamily:
    """Explicit statistical testing family for one chronological selection event."""

    target: str
    scope: str
    method: str
    prediction_date: date
    analysis_id: str = ""

    @property
    def family_id(self) -> str:
        parts = [self.target, self.scope, self.method, self.prediction_date.isoformat()]
        if self.analysis_id:
            parts.append(self.analysis_id)
        return "|".join(parts)


def control_multiple_testing_by_family(
    families: dict[str, Sequence[float]],
    *,
    alpha: float = 0.10,
) -> dict[str, MultipleTestingResult]:
    """Apply BH-FDR independently to each explicitly defined testing family."""
    return {
        family_id: control_multiple_testing(
            p_values,
            candidate_count=len(p_values),
            alpha=alpha,
        )
        for family_id, p_values in families.items()
    }


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.10,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Benjamini-Hochberg FDR correction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not p_values:
        return (), ()

    m = len(p_values)
    indexed = sorted(
        enumerate(max(0.0, min(1.0, float(p))) for p in p_values),
        key=lambda item: item[1],
    )
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        candidate = p_value * m / rank
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)

    accepted = tuple(
        index for index, value in enumerate(adjusted)
        if value <= alpha
    )
    return tuple(adjusted), accepted


def control_multiple_testing(
    raw_p_values: Sequence[float],
    *,
    candidate_count: int,
    alpha: float = 0.10,
) -> MultipleTestingResult:
    """Apply FDR control while preserving the total searched candidate count."""
    adjusted, accepted = benjamini_hochberg(raw_p_values, alpha=alpha)
    return MultipleTestingResult(
        candidate_count=max(int(candidate_count), len(raw_p_values)),
        tested_count=len(raw_p_values),
        alpha=alpha,
        adjusted_alpha=alpha,
        raw_p_values=tuple(float(p) for p in raw_p_values),
        adjusted_p_values=adjusted,
        accepted_indices=accepted,
    )


@dataclass(frozen=True)
class ParameterTimingAudit:
    name: str
    fit_start_date: date | None
    fit_end_date: date | None
    cutoff_date: date
    temporal_violation: bool


def audit_parameter_timing(
    parameters: Iterable[tuple[str, date | None, date | None]],
    cutoff_date: str | date | datetime,
) -> tuple[ParameterTimingAudit, ...]:
    """
    Audit fitted parameter provenance separately from source-row provenance.

    Each parameter tuple is (name, fit_start_date, fit_end_date). A fitted
    parameter is temporally unsafe when its fit_end_date is after the state
    cutoff. Unknown provenance is retained as an explicit audit limitation.
    """
    cutoff = as_date(cutoff_date)
    audits: list[ParameterTimingAudit] = []
    for name, fit_start, fit_end in parameters:
        normalized_end = as_date(fit_end) if fit_end is not None else None
        audits.append(
            ParameterTimingAudit(
                name=str(name),
                fit_start_date=as_date(fit_start) if fit_start is not None else None,
                fit_end_date=normalized_end,
                cutoff_date=cutoff,
                temporal_violation=(normalized_end is not None and normalized_end > cutoff),
            )
        )
    return tuple(audits)



@dataclass(frozen=True)
class ParameterManifestAudit:
    """Verify presence and temporal safety of every required learned parameter."""

    required_names: tuple[str, ...]
    observed_names: tuple[str, ...]
    missing_names: tuple[str, ...]
    audits: tuple[ParameterTimingAudit, ...]

    @property
    def temporal_violation(self) -> bool:
        return any(item.temporal_violation for item in self.audits)

    @property
    def complete(self) -> bool:
        return not self.missing_names


def audit_parameter_manifest(
    required_names: Iterable[str],
    parameters: Iterable[tuple[str, date | None, date | None]],
    cutoff_date: str | date | datetime,
) -> ParameterManifestAudit:
    """Audit presence and temporal timing of a declared learned-parameter manifest."""
    required = tuple(sorted(set(str(name) for name in required_names)))
    audits = audit_parameter_timing(parameters, cutoff_date)
    observed = tuple(sorted({item.name for item in audits}))
    missing = tuple(name for name in required if name not in observed)
    return ParameterManifestAudit(
        required_names=required,
        observed_names=observed,
        missing_names=missing,
        audits=audits,
    )


@dataclass(frozen=True)
class StateSurfaceAudit:
    expected_families: tuple[str, ...]
    observed_families: tuple[str, ...]
    missing_families: tuple[str, ...]
    coverage_pct: float


def audit_state_surface(
    states: dict[str, str],
    expected_families: Iterable[str],
) -> StateSurfaceAudit:
    expected = tuple(sorted(set(str(item) for item in expected_families)))
    observed = tuple(sorted({str(feature).rsplit(".", 1)[0] for feature in states}))
    missing = tuple(item for item in expected if item not in observed)
    coverage = (
        (len(expected) - len(missing)) / len(expected) * 100.0
        if expected else 100.0
    )
    return StateSurfaceAudit(
        expected_families=expected,
        observed_families=observed,
        missing_families=missing,
        coverage_pct=coverage,
    )
