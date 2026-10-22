from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np

from .hardening_4_10 import (
    audit_parameter_manifest,
    TestingFamily,
)

from .ranking import RelationshipRanking, rank_relationships
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
    RelationshipResult,
)
from .relationship_graph import candidate_feature_sets
from .relationship_fastpath import (
    method_a_prepared,
    method_b_prepared,
    prepare_relationship_context,
    materialize_selected_support,
)
from .fast_permutation import search_adjusted_permutation_p_values_matrix


@dataclass(frozen=True)
class MethodValidationSummary:
    method: str
    evaluated_folds: int
    directional_predictions: int
    directional_hit_rate_pct: float
    coverage_pct: float
    mean_signed_return_pct: float
    mean_realized_return_pct: float
    selected_relationships: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class WalkForwardFold:
    prediction_date: date
    training_observations: int
    actual_return_pct: float
    method_a: dict | None
    method_b: dict | None
    combined: dict | None
    # Internal Phase 5 hand-off: the relationship that survived the inner
    # discovery -> selection -> FDR gate. These are never outer-test data.
    selected_a_result: RelationshipResult | None = None
    selected_b_result: RelationshipResult | None = None
    method_a_selection_metadata: dict[str, Any] | None = None
    method_b_selection_metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict:
        return {
            "prediction_date": self.prediction_date,
            "training_observations": self.training_observations,
            "actual_return_pct": self.actual_return_pct,
            "method_a": self.method_a,
            "method_b": self.method_b,
            "combined": self.combined,
            "selected_a_result": self.selected_a_result.as_dict() if self.selected_a_result else None,
            "selected_b_result": self.selected_b_result.as_dict() if self.selected_b_result else None,
            "method_a_selection_metadata": self.method_a_selection_metadata,
            "method_b_selection_metadata": self.method_b_selection_metadata,
        }


@dataclass(frozen=True)
class WalkForwardValidationResult:
    folds: tuple[WalkForwardFold, ...]
    method_a: MethodValidationSummary
    method_b: MethodValidationSummary
    combined: MethodValidationSummary
    leakage_violations: int
    skipped_insufficient_history: int
    skipped_no_relationship: int
    purged_training_observations: int = 0
    unknown_overlap_observations: int = 0
    selection_candidate_evaluations: int = 0
    selection_validated_predictions: int = 0
    multiple_testing_controlled_folds: int = 0

    def as_dict(self) -> dict:
        return {
            "folds": [fold.as_dict() for fold in self.folds],
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "leakage_violations": self.leakage_violations,
            "skipped_insufficient_history": self.skipped_insufficient_history,
            "skipped_no_relationship": self.skipped_no_relationship,
            "purged_training_observations": self.purged_training_observations,
            "unknown_overlap_observations": self.unknown_overlap_observations,
            "selection_candidate_evaluations": self.selection_candidate_evaluations,
            "selection_validated_predictions": self.selection_validated_predictions,
            "multiple_testing_controlled_folds": self.multiple_testing_controlled_folds,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _direction(value: float) -> str:
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def _predictive_direction(ranking: RelationshipRanking | None) -> str | None:
    if ranking is None:
        return None
    if ranking.direction not in {"POSITIVE", "NEGATIVE"}:
        return None
    return ranking.direction


def _fold_result(
    ranking: RelationshipRanking | None,
    actual_return_pct: float,
) -> dict | None:
    predicted = _predictive_direction(ranking)
    if predicted is None:
        return None

    actual = _direction(actual_return_pct)
    if actual == "NEUTRAL":
        signed = 0.0
        hit = False
    elif predicted == "POSITIVE":
        signed = actual_return_pct
        hit = actual == "POSITIVE"
    else:
        signed = -actual_return_pct
        hit = actual == "NEGATIVE"

    return {
        "variables": list(ranking.variables),
        "condition": list(ranking.condition),
        "direction": predicted,
        "score": ranking.best_score,
        "reliability": ranking.reliability,
        "sample_count": ranking.sample_count,
        "method_agreement": ranking.method_agreement,
        "stable_method_count": ranking.stable_method_count,
        "actual_direction": actual,
        "hit": hit,
        "signed_return_pct": signed,
        "realized_return_pct": actual_return_pct,
    }


def _summary(
    method: str,
    rows: list[dict],
    evaluated_folds: int,
) -> MethodValidationSummary:
    directional_predictions = len(rows)
    hits = sum(1 for row in rows if row["hit"])

    coverage = (
        directional_predictions / evaluated_folds * 100.0
        if evaluated_folds
        else 0.0
    )
    hit_rate = (
        hits / directional_predictions * 100.0
        if directional_predictions
        else 0.0
    )
    mean_signed = (
        sum(row["signed_return_pct"] for row in rows)
        / directional_predictions
        if directional_predictions
        else 0.0
    )
    mean_realized = (
        sum(row["realized_return_pct"] for row in rows)
        / directional_predictions
        if directional_predictions
        else 0.0
    )

    return MethodValidationSummary(
        method=method,
        evaluated_folds=evaluated_folds,
        directional_predictions=directional_predictions,
        directional_hit_rate_pct=hit_rate,
        coverage_pct=coverage,
        mean_signed_return_pct=mean_signed,
        mean_realized_return_pct=mean_realized,
        selected_relationships=directional_predictions,
    )


def _top_ranking(
    results: Iterable,
) -> RelationshipRanking | None:
    rankings = rank_relationships(results)
    if not rankings:
        return None
    return rankings[0]


def validate_walk_forward_relationships(
    observations: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    min_training_observations: int | None = None,
) -> WalkForwardValidationResult:
    """
    Validate relationship discovery in strict chronological walk-forward order.

    For every prediction date T:
        train = observations strictly before T
        discover relationships using only train
        select the top relationship independently for Method A and Method B
        score the directional prediction against the held-out return at T

    The held-out observation is never passed into relationship discovery.
    This validates relationship direction and usefulness before later
    probability calibration creates the final UP/SIDEWAYS/DOWN labels.
    """
    rows = sorted(
        list(observations),
        key=lambda item: _as_date(item.as_of_date),
    )
    if not rows:
        empty = _summary("A", [], 0)
        return WalkForwardValidationResult(
            folds=(),
            method_a=empty,
            method_b=_summary("B", [], 0),
            combined=_summary("COMBINED", [], 0),
            leakage_violations=0,
            skipped_insufficient_history=0,
            skipped_no_relationship=0,
        )

    minimum = (
        engine.min_observations
        if min_training_observations is None
        else int(min_training_observations)
    )
    if minimum < engine.min_observations:
        raise ValueError(
            "min_training_observations cannot be below engine.min_observations"
        )

    folds: list[WalkForwardFold] = []
    method_a_rows: list[dict] = []
    method_b_rows: list[dict] = []
    combined_rows: list[dict] = []

    leakage_violations = 0
    skipped_insufficient_history = 0
    skipped_no_relationship = 0

    discovery_engine = RelationshipDiscoveryEngine(
        max_order=engine.max_order,
        min_observations=engine.min_observations,
    )

    for index, test_observation in enumerate(rows):
        prediction_date = _as_date(test_observation.as_of_date)
        training = [
            observation
            for observation in rows[:index]
            if _as_date(observation.as_of_date) < prediction_date
            and observation.stock_return_pct is not None
        ]

        if len(training) < minimum:
            skipped_insufficient_history += 1
            continue

        if any(
            _as_date(observation.as_of_date) >= prediction_date
            for observation in training
        ):
            leakage_violations += 1
            continue

        discovered_results, _candidate_counts = _discover_all_candidate_relationships(
            test_observation.states,
            training,
            discovery_engine,
        )
        method_a = [item for item in discovered_results if item.method == "A"]
        method_b = [item for item in discovered_results if item.method == "B"]

        top_a = _top_ranking(method_a)
        top_b = _top_ranking(method_b)

        row_a = _fold_result(top_a, test_observation.stock_return_pct)
        row_b = _fold_result(top_b, test_observation.stock_return_pct)

        combined_ranking = _top_ranking([
            *method_a,
            *method_b,
        ])
        row_combined = _fold_result(
            combined_ranking,
            test_observation.stock_return_pct,
        )

        if row_a is None and row_b is None and row_combined is None:
            skipped_no_relationship += 1

        if row_a is not None:
            method_a_rows.append(row_a)
        if row_b is not None:
            method_b_rows.append(row_b)
        if row_combined is not None:
            combined_rows.append(row_combined)

        folds.append(
            WalkForwardFold(
                prediction_date=prediction_date,
                training_observations=len(training),
                actual_return_pct=float(test_observation.stock_return_pct),
                method_a=row_a,
                method_b=row_b,
                combined=row_combined,
            )
        )

    evaluated = len(folds)

    return WalkForwardValidationResult(
        folds=tuple(folds),
        method_a=_summary("A", method_a_rows, evaluated),
        method_b=_summary("B", method_b_rows, evaluated),
        combined=_summary("COMBINED", combined_rows, evaluated),
        leakage_violations=leakage_violations,
        skipped_insufficient_history=skipped_insufficient_history,
        skipped_no_relationship=skipped_no_relationship,
    )


def _discover_all_candidate_relationships(
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
) -> tuple[list[RelationshipResult], dict[int, int]]:
    """Discover the complete configured candidate universe.

    ``adaptive_higher_order=True`` enables the exploratory Phase 4.6/4.7
    frontier. In the production Phase 5.8 validation, the default is now
    exhaustive: every structurally valid level 1..N candidate is evaluated.
    Performance optimizations belong inside candidate evaluation, not in the
    candidate universe.
    """
    if not current_states or not history:
        return [], {}

    discovered: dict[tuple[str, tuple[str, ...]], RelationshipResult] = {}

    if engine.adaptive_higher_order:
        base_order = min(engine.max_order, 2)
        for candidate in candidate_feature_sets(current_states.keys(), base_order):
            subset = {feature: current_states[feature] for feature in candidate}
            candidate_engine = RelationshipDiscoveryEngine(
                max_order=len(candidate),
                min_observations=engine.min_observations,
                min_state_coverage_pct=engine.min_state_coverage_pct,
                min_family_coverage_pct=engine.min_family_coverage_pct,
                stability_sem_multiplier=engine.stability_sem_multiplier,
            )
            results = (
                candidate_engine.method_a_similar_states(subset, history)
                + candidate_engine.method_b_conditioned_distribution(subset, history)
            )
            for result in results:
                if tuple(sorted(result.variables)) != tuple(sorted(candidate)):
                    continue
                discovered[(result.method, tuple(sorted(result.variables)))] = result

        if engine.max_order >= 3:
            from .adaptive_combination import discover_adaptive_higher_order
            adaptive = discover_adaptive_higher_order(
                current_states=current_states,
                history=history,
                engine=engine,
            )
            for ranking in adaptive.rankings:
                for result in ranking.method_results:
                    if len(result.variables) < 3:
                        continue
                    discovered[(result.method, tuple(sorted(result.variables)))] = result
    else:
        # Exact exhaustive universe. No performance-driven candidate pruning.
        candidates = candidate_feature_sets(current_states.keys(), engine.max_order)
        by_order: dict[int, list[tuple[str, ...]]] = {}
        for candidate in candidates:
            by_order.setdefault(len(candidate), []).append(candidate)

        total_candidates = len(candidates)
        processed_total = 0
        progress = engine.progress_callback
        prepared_context = prepare_relationship_context(current_states, history)
        if progress is not None:
            progress(
                f"Nested discovery start | features={len(current_states)} | "
                f"history={len(history)} | candidates={total_candidates} | exhaustive=YES"
            )

        for order in sorted(by_order):
            order_candidates = by_order[order]
            if progress is not None:
                progress(
                    f"Nested discovery L{order} start | candidates={len(order_candidates)} | "
                    f"processed={processed_total}/{total_candidates}"
                )

            # Evaluate the entire order in vectorized batches. The previous
            # implementation called the vectorized engine once per candidate,
            # reducing the optimization to batch_size=1 and repeating the
            # feature/history matrix construction tens of thousands of times.
            candidate_engine = RelationshipDiscoveryEngine(
                max_order=order,
                min_observations=engine.min_observations,
                min_state_coverage_pct=engine.min_state_coverage_pct,
                min_family_coverage_pct=engine.min_family_coverage_pct,
                stability_sem_multiplier=engine.stability_sem_multiplier,
                progress_callback=None,
                progress_every_candidates=engine.progress_every_candidates,
                candidate_batch_size=engine.candidate_batch_size,
                # Discovery results are only candidate definitions/provenance
                # for the inner holdout. Their support is recomputed there.
                retain_supporting_data=False,
            )
            results_a = method_a_prepared(
                candidate_engine, current_states, history, order_candidates, prepared_context
            )
            results_b = method_b_prepared(
                candidate_engine, current_states, history, order_candidates, prepared_context
            )
            for result in (*results_a, *results_b):
                key = (result.method, tuple(sorted(result.variables)))
                discovered[key] = result

            processed_total += len(order_candidates)
            if progress is not None:
                progress(
                    f"Nested discovery L{order} progress "
                    f"{len(order_candidates)}/{len(order_candidates)} | "
                    f"total={processed_total}/{total_candidates} | "
                    f"results={len(discovered)}"
                )

    counts: dict[int, int] = {}
    for _method, variables in discovered:
        counts[len(variables)] = counts.get(len(variables), 0) + 1
    return list(discovered.values()), counts


def _method_results_for_candidate(
    method: str,
    candidate_variables: tuple[str, ...],
    current_states: dict[str, str],
    history: list[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
) -> object | None:
    subset = {feature: current_states[feature] for feature in candidate_variables if feature in current_states}
    if len(subset) != len(candidate_variables):
        return None
    if method == "A":
        results = engine.method_a_similar_states(subset, history)
    else:
        results = engine.method_b_conditioned_distribution(subset, history)
    return next((item for item in results if tuple(sorted(item.variables)) == tuple(sorted(candidate_variables))), None)


def _holdout_select(
    method: str,
    discovery_results: list,
    current_states: dict[str, str],
    selection_history: list[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    family: TestingFamily,
    *,
    alpha: float,
    prepared_context=None,
) -> tuple[object | None, int, bool, dict[str, Any]]:
    """
    Select a discovered relationship on a chronological inner holdout.

    Discovery candidates are fixed before the selection set is touched. Each
    candidate is independently re-evaluated on the selection set, then the
    complete candidate family is evaluated with a search-wide max-statistic
    permutation test. The returned p-values are already adjusted for the
    candidate search family, so they are thresholded directly at ``alpha``.
    Applying BH again to these already search-adjusted p-values would double-
    penalize the same search and can make the gate effectively impossible to
    pass when the exhaustive universe contains hundreds of thousands of
    candidates.
    """
    configured_permutations = 199
    selection_seed = sum(ord(ch) for ch in family.family_id) + len(selection_history)

    def audit_record(
        status: str,
        *,
        evaluated_count: int = 0,
        adjusted_p_values: tuple[float, ...] = (),
        accepted_count: int = 0,
        selected_relationship: object | None = None,
        selected_index: int | None = None,
        selected_adjusted_p_value: float | None = None,
        permutations_run: bool = False,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "selection_status": status,
            "discovery_candidate_count": len(discovery_results),
            "selection_history_count": len(selection_history),
            "candidate_evaluation_count": evaluated_count,
            "multiple_testing": {
                "procedure": "search_adjusted_max_statistic_permutation",
                "permutations_configured": configured_permutations,
                "permutations_run": configured_permutations if permutations_run else 0,
                "alpha": float(alpha),
                "seed": selection_seed,
                "minimum_adjusted_p_value": min(adjusted_p_values) if adjusted_p_values else None,
                "median_adjusted_p_value": float(np.median(adjusted_p_values)) if adjusted_p_values else None,
                "maximum_adjusted_p_value": max(adjusted_p_values) if adjusted_p_values else None,
                "accepted_candidate_count": accepted_count,
                "selected_adjusted_p_value": selected_adjusted_p_value,
                "selected_evaluated_candidate_index": selected_index,
            },
            "selected_relationship": (
                selected_relationship.as_dict()
                if selected_relationship is not None and hasattr(selected_relationship, "as_dict")
                else None
            ),
        }

    if not discovery_results or not selection_history:
        return (
            None,
            0,
            False,
            audit_record("NOT_RUN_EMPTY_DISCOVERY_OR_SELECTION_HISTORY"),
        )

    if not engine.retain_supporting_data:
        raise RuntimeError(
            "Nested selection requires retain_supporting_data=True because "
            "search-wide permutation testing consumes each candidate's exact "
            "supporting observations and weights."
        )

    evaluations: list[object] = []
    baseline_mean = sum(item.stock_return_pct for item in selection_history) / len(selection_history)

    progress = engine.progress_callback
    total_candidates = len(discovery_results)
    if progress is not None:
        progress(
            f"Nested selection {method} start | candidates={total_candidates} | "
            f"history={len(selection_history)}"
        )

    # Re-evaluate all discovered candidates in vectorized batches on the inner
    # holdout. This is the same chronological selection test, but avoids invoking
    # a 1-candidate search for every relationship.
    candidate_variables = [tuple(candidate.variables) for candidate in discovery_results]
    evaluated_results: list[RelationshipResult] = []
    if candidate_variables:
        if prepared_context is None:
            prepared_context = prepare_relationship_context(current_states, selection_history)
        evaluated_results = (
            method_a_prepared(
                engine, current_states, selection_history, candidate_variables, prepared_context,
                compact_support=True,
            )
            if method == "A"
            else method_b_prepared(
                engine, current_states, selection_history, candidate_variables, prepared_context,
                compact_support=True,
            )
        )

    # Prepared evaluation preserves the discovery candidate ordering. Invalid
    # candidates simply produce no RelationshipResult and therefore no support row;
    # the compact result list and support matrix remain aligned one-to-one.
    evaluations = list(evaluated_results)

    if progress is not None:
        progress(
            f"Nested selection {method} progress "
            f"{total_candidates}/{total_candidates} | "
            f"re_evaluated={len(evaluated_results)} | "
            f"support_mode=COMPACT | evaluations={len(evaluations)}"
        )

    if not evaluations:
        if progress is not None:
            progress(f"Nested selection {method} complete | evaluations=0 | accepted=0")
        return (
            None,
            0,
            False,
            audit_record("NOT_RUN_NO_VALID_CANDIDATE_EVALUATIONS"),
        )

    compact_support = getattr(evaluated_results, "support_matrix", None)
    if compact_support is None or len(compact_support) != len(evaluations):
        raise RuntimeError("Compact selection support/result alignment is invalid")

    if progress is not None:
        progress(
            f"Nested selection {method} search-adjusted gate start | "
            f"evaluations={len(evaluations)} | permutations=199 | alpha={alpha:.3f} | support=COMPACT"
        )

    permutation = search_adjusted_permutation_p_values_matrix(
        compact_support,
        np.asarray([float(item.stock_return_pct) for item in selection_history], dtype=np.float64),
        baseline_mean,
        permutations=configured_permutations,
        seed=selection_seed,
    )
    adjusted = tuple(float(value) for value in permutation)
    accepted = tuple(
        index
        for index, p_value in enumerate(adjusted)
        if p_value <= float(alpha)
    )
    if not accepted:
        if progress is not None:
            progress(
                f"Nested selection {method} search-adjusted gate complete | "
                f"evaluations={len(evaluations)} | accepted=0 | "
                f"min_adjusted_p={min(adjusted):.6f}"
            )
        return (
            None,
            len(evaluations),
            True,
            audit_record(
                "REJECTED_NO_CANDIDATE_PASSED_ADJUSTED_P_THRESHOLD",
                evaluated_count=len(evaluations),
                adjusted_p_values=adjusted,
                accepted_count=0,
                permutations_run=True,
            ),
        )

    if progress is not None:
        progress(
            f"Nested selection {method} search-adjusted gate complete | "
            f"evaluations={len(evaluations)} | accepted={len(accepted)} | "
            f"min_adjusted_p={min(adjusted):.6f}"
        )

    accepted_rows = [
        (evaluations[index], adjusted[index])
        for index in accepted
    ]
    selected, _q = min(
        accepted_rows,
        key=lambda row: (
            row[1],
            -abs(float(row[0].score)),
            -float(row[0].reliability),
        ),
    )
    selected_index = evaluations.index(selected)
    selected_adjusted_p_value = float(_q)
    selected = materialize_selected_support(
        selected, current_states, prepared_context, compact_support[selected_index]
    )
    if progress is not None:
        progress(
            f"Nested selection {method} complete | evaluations={len(evaluations)} | "
            f"accepted={len(accepted)} | selected_order={len(selected.variables)} | "
            f"selected_adjusted_p={_q:.6f}"
        )
    return (
        selected,
        len(evaluations),
        True,
        audit_record(
            "SELECTED_AFTER_ADJUSTED_P_THRESHOLD",
            evaluated_count=len(evaluations),
            adjusted_p_values=adjusted,
            accepted_count=len(accepted),
            selected_relationship=selected,
            selected_index=selected_index,
            selected_adjusted_p_value=selected_adjusted_p_value,
            permutations_run=True,
        ),
    )


def _run_hardened_fold(
    fold_index: int,
    total_folds: int,
    test_observation: HistoricalRelationshipObservation,
    training: list[HistoricalRelationshipObservation],
    discovery_history: list[HistoricalRelationshipObservation],
    selection_history: list[HistoricalRelationshipObservation],
    engine_config: dict[str, object],
    multiple_testing_alpha: float,
    progress_enabled: bool,
) -> dict[str, object]:
    """Execute one hardened outer fold.

    The fold is self-contained: it only reads the supplied chronological
    histories and test observation. This makes folds safe to execute concurrently
    without changing the walk-forward information boundary.
    """
    prediction_date = _as_date(test_observation.as_of_date)

    def progress(message: str) -> None:
        if progress_enabled:
            print(f"[Phase 5.8] Fold {fold_index}/{total_folds} | {message}", flush=True)

    progress(
        f"start | prediction_date={prediction_date} | training={len(training)} | "
        f"discovery={len(discovery_history)} | selection={len(selection_history)}"
    )

    discovery_engine = RelationshipDiscoveryEngine(
        max_order=int(engine_config["max_order"]),
        min_observations=int(engine_config["min_observations"]),
        min_state_coverage_pct=float(engine_config["min_state_coverage_pct"]),
        min_family_coverage_pct=float(engine_config["min_family_coverage_pct"]),
        stability_sem_multiplier=float(engine_config["stability_sem_multiplier"]),
        adaptive_higher_order=bool(engine_config["adaptive_higher_order"]),
        progress_callback=progress,
        progress_every_candidates=int(engine_config["progress_every_candidates"]),
        candidate_batch_size=int(engine_config["candidate_batch_size"]),
        # Discovery support is not consumed by inner selection. Re-materializing
        # it there avoids millions of Python tuples per outer fold.
        retain_supporting_data=False,
    )

    progress(
        f"discovery input | discovery_history={len(discovery_history)} | "
        f"selection_history={len(selection_history)} | features={len(test_observation.states)}"
    )
    discovered_results, candidate_counts = _discover_all_candidate_relationships(
        test_observation.states,
        discovery_history,
        discovery_engine,
    )
    progress(
        f"discovery complete | relationships={len(discovered_results)} | counts={candidate_counts}"
    )

    discovery_a = [item for item in discovered_results if item.method == "A"]
    discovery_b = [item for item in discovered_results if item.method == "B"]
    required_a = ("state_relevance_frequency", "method_a_adaptive_neighbor_count")
    required_b = ("state_relevance_frequency", "method_b_similarity_scale")
    audits_a = [
        audit_parameter_manifest(required_a, result.parameter_provenance, prediction_date)
        for result in discovery_a
    ]
    audits_b = [
        audit_parameter_manifest(required_b, result.parameter_provenance, prediction_date)
        for result in discovery_b
    ]
    if any(audit.temporal_violation for audit in (*audits_a, *audits_b)):
        return {
            "status": "leakage",
            "fold_index": fold_index,
            "leakage_violations": 1,
            "purged_training": 0,
            "unknown_overlap": 0,
        }
    if any(not audit.complete for audit in (*audits_a, *audits_b)):
        return {
            "status": "insufficient",
            "fold_index": fold_index,
            "leakage_violations": 0,
            "purged_training": 0,
            "unknown_overlap": 0,
        }

    # Discovery remains memory-light. Inner selection needs exact support for
    # search-wide permutation testing, so it uses a separate engine.
    selection_engine = RelationshipDiscoveryEngine(
        max_order=int(engine_config["max_order"]),
        min_observations=int(engine_config["min_observations"]),
        min_state_coverage_pct=float(engine_config["min_state_coverage_pct"]),
        min_family_coverage_pct=float(engine_config["min_family_coverage_pct"]),
        stability_sem_multiplier=float(engine_config["stability_sem_multiplier"]),
        adaptive_higher_order=bool(engine_config["adaptive_higher_order"]),
        progress_callback=progress,
        progress_every_candidates=int(engine_config["progress_every_candidates"]),
        candidate_batch_size=int(engine_config["candidate_batch_size"]),
        retain_supporting_data=True,
    )

    family_a = TestingFamily(
        target=test_observation.target,
        scope=test_observation.scope,
        method="A",
        prediction_date=prediction_date,
    )
    family_b = TestingFamily(
        target=test_observation.target,
        scope=test_observation.scope,
        method="B",
        prediction_date=prediction_date,
    )

    progress(
        f"selection input | A_candidates={len(discovery_a)} | "
        f"B_candidates={len(discovery_b)} | support_retention=ON"
    )
    selection_context = (
        prepare_relationship_context(test_observation.states, selection_history)
        if (discovery_a or discovery_b) and selection_history
        else None
    )
    selected_a, count_a, controlled_a, selection_metadata_a = _holdout_select(
        "A", discovery_a, test_observation.states, selection_history,
        selection_engine, family_a, alpha=multiple_testing_alpha,
        prepared_context=selection_context,
    )
    selected_b, count_b, controlled_b, selection_metadata_b = _holdout_select(
        "B", discovery_b, test_observation.states, selection_history,
        selection_engine, family_b, alpha=multiple_testing_alpha,
        prepared_context=selection_context,
    )

    ranking_a = _top_ranking([selected_a] if selected_a is not None else [])
    ranking_b = _top_ranking([selected_b] if selected_b is not None else [])
    actual_return = float(test_observation.stock_return_pct)
    row_a = _fold_result(ranking_a, actual_return)
    row_b = _fold_result(ranking_b, actual_return)
    common_ranking = (
        ranking_a
        if ranking_a is not None and ranking_b is not None and ranking_a.direction == ranking_b.direction
        else None
    )
    row_combined = _fold_result(common_ranking, actual_return)

    progress(
        f"complete | selected_A={selected_a is not None} | selected_B={selected_b is not None} | "
        f"A_evals={count_a} | B_evals={count_b}"
    )
    return {
        "status": "complete",
        "fold_index": fold_index,
        "fold": WalkForwardFold(
            prediction_date=prediction_date,
            training_observations=len(training),
            actual_return_pct=actual_return,
            method_a=row_a,
            method_b=row_b,
            combined=row_combined,
            selected_a_result=selected_a,
            selected_b_result=selected_b,
            method_a_selection_metadata=selection_metadata_a,
            method_b_selection_metadata=selection_metadata_b,
        ),
        "row_a": row_a,
        "row_b": row_b,
        "row_combined": row_combined,
        "count_a": count_a,
        "count_b": count_b,
        "controlled": int(controlled_a or controlled_b),
        "selection_validated": int(selected_a is not None or selected_b is not None),
        "leakage_violations": 0,
        "purged_training": 0,
        "unknown_overlap": 0,
    }


def validate_walk_forward_relationships_hardened(
    observations: Iterable[HistoricalRelationshipObservation],
    engine: RelationshipDiscoveryEngine,
    min_training_observations: int | None = None,
    *,
    selection_fraction: float = 0.30,
    multiple_testing_alpha: float = 0.10,
    purge_overlapping_labels: bool = True,
    parallel_workers: int = 1,
) -> WalkForwardValidationResult:
    """
    Methodological-hardening validator for Phase 4.10.

    Outer test observations are never used for discovery or selection. The
    training sample is split chronologically into discovery and selection
    segments. Candidate relationships are discovered in the earlier segment,
    selected on the later segment with FDR control, then scored on the held-out
    outer observation. When outcome_end_date is known, training labels whose
    realized horizons overlap the outer prediction date are purged. Unknown
    horizons are conservatively excluded in hardened mode.
    """
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be in (0, 1)")
    if not 0.0 < multiple_testing_alpha < 1.0:
        raise ValueError("multiple_testing_alpha must be in (0, 1)")

    rows = sorted(list(observations), key=lambda item: _as_date(item.as_of_date))
    if not rows:
        return validate_walk_forward_relationships(rows, engine, min_training_observations)

    minimum = engine.min_observations if min_training_observations is None else int(min_training_observations)
    if minimum < engine.min_observations:
        raise ValueError("min_training_observations cannot be below engine.min_observations")

    folds: list[WalkForwardFold] = []
    method_a_rows: list[dict] = []
    method_b_rows: list[dict] = []
    combined_rows: list[dict] = []
    leakage_violations = 0
    skipped_insufficient_history = 0
    skipped_no_relationship = 0
    purged_training = 0
    unknown_overlap = 0
    selection_evaluations = 0
    selection_validated = 0
    controlled_folds = 0

    worker_count = max(1, int(parallel_workers))
    if worker_count > 1:
        worker_count = min(worker_count, max(1, int(os.cpu_count() or 1)))

    engine_config = {
        "max_order": engine.max_order,
        "min_observations": engine.min_observations,
        "min_state_coverage_pct": engine.min_state_coverage_pct,
        "min_family_coverage_pct": engine.min_family_coverage_pct,
        "stability_sem_multiplier": engine.stability_sem_multiplier,
        "adaptive_higher_order": engine.adaptive_higher_order,
        "progress_every_candidates": engine.progress_every_candidates,
        "candidate_batch_size": engine.candidate_batch_size,
    }

    jobs: list[tuple[int, HistoricalRelationshipObservation, list[HistoricalRelationshipObservation], list[HistoricalRelationshipObservation], list[HistoricalRelationshipObservation]]] = []
    for index, test_observation in enumerate(rows):
        prediction_date = _as_date(test_observation.as_of_date)
        raw_training = [
            item for item in rows[:index]
            if _as_date(item.as_of_date) < prediction_date
            and item.stock_return_pct is not None
        ]
        if len(raw_training) < minimum:
            skipped_insufficient_history += 1
            continue

        training: list[HistoricalRelationshipObservation] = []
        for item in raw_training:
            end_date = item.outcome_end_date
            if purge_overlapping_labels:
                if end_date is None:
                    unknown_overlap += 1
                    continue
                if _as_date(end_date) >= prediction_date:
                    purged_training += 1
                    continue
            training.append(item)

        if len(training) < minimum:
            skipped_insufficient_history += 1
            continue
        if any(_as_date(item.as_of_date) >= prediction_date for item in training):
            leakage_violations += 1
            continue

        split_index = int(len(training) * (1.0 - selection_fraction))
        split_index = min(max(split_index, engine.min_observations), len(training) - 1)
        discovery_history = training[:split_index]
        selection_history = training[split_index:]
        if len(discovery_history) < minimum or len(selection_history) < engine.min_observations:
            skipped_insufficient_history += 1
            continue
        jobs.append((index, test_observation, training, discovery_history, selection_history))

    def submit_args(job):
        index, observation, training, discovery_history, selection_history = job
        return _run_hardened_fold(
            index + 1,
            len(rows),
            observation,
            training,
            discovery_history,
            selection_history,
            engine_config,
            multiple_testing_alpha,
            bool(engine.progress_callback is not None),
        )

    completed_results: list[dict[str, object]] = []
    if worker_count == 1:
        for job in jobs:
            completed_results.append(submit_args(job))
    else:
        print(f"[Phase 5.8] Parallel fold execution | workers={worker_count} | runnable_folds={len(jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="phase5-fold") as executor:
            future_map = {executor.submit(submit_args, job): job[0] + 1 for job in jobs}
            for future in as_completed(future_map):
                result = future.result()
                completed_results.append(result)
                print(
                    f"[Phase 5.8] Parallel fold complete | fold={result.get('fold_index')} | "
                    f"completed={len(completed_results)}/{len(jobs)}",
                    flush=True,
                )

    completed_results.sort(key=lambda item: int(item["fold_index"]))
    for result in completed_results:
        status = result.get("status")
        if status == "leakage":
            leakage_violations += 1
            continue
        if status == "insufficient":
            skipped_insufficient_history += 1
            continue
        fold = result["fold"]
        folds.append(fold)  # type: ignore[arg-type]
        row_a = result.get("row_a")
        row_b = result.get("row_b")
        row_combined = result.get("row_combined")
        if row_a is not None:
            method_a_rows.append(row_a)  # type: ignore[arg-type]
        if row_b is not None:
            method_b_rows.append(row_b)  # type: ignore[arg-type]
        if row_combined is not None:
            combined_rows.append(row_combined)  # type: ignore[arg-type]
        selection_evaluations += int(result.get("count_a", 0)) + int(result.get("count_b", 0))
        controlled_folds += int(result.get("controlled", 0))
        selection_validated += int(result.get("selection_validated", 0))

    evaluated = len(folds)
    return WalkForwardValidationResult(
        folds=tuple(folds),
        method_a=_summary("A", method_a_rows, evaluated),
        method_b=_summary("B", method_b_rows, evaluated),
        combined=_summary("COMBINED", combined_rows, evaluated),
        leakage_violations=leakage_violations,
        skipped_insufficient_history=skipped_insufficient_history,
        skipped_no_relationship=skipped_no_relationship,
        purged_training_observations=purged_training,
        unknown_overlap_observations=unknown_overlap,
        selection_candidate_evaluations=selection_evaluations,
        selection_validated_predictions=selection_validated,
        multiple_testing_controlled_folds=controlled_folds,
    )
