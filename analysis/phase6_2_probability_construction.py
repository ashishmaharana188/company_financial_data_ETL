from __future__ import annotations

"""Read-only diagnostics for Phase 5.8 empirical class-share construction.

The Phase 5.8 probability builders construct Method A/B distributions from
class-specific support weights. The current frozen schema records the final
method vectors and some method metadata, but it does not necessarily retain
the probability-builder's class_counts / weighted_class_counts object. This
module reports that gap explicitly and never infers missing counts from a
probability vector.
"""

from collections import Counter
from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable

CLASSES = ("UP", "SIDEWAYS", "DOWN")


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _distribution(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    values: dict[str, float] = {}
    for label in CLASSES:
        value = _as_float(raw.get(label))
        if value is None or value < 0.0:
            return None
        values[label] = value
    total = sum(values.values())
    if total <= 0.0 or not isfinite(total):
        return None
    return {label: values[label] / total * 100.0 for label in CLASSES}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_distribution(*candidates: Any) -> dict[str, float] | None:
    for candidate in candidates:
        distribution = _distribution(candidate)
        if distribution is not None:
            return distribution
    return None


def _method_output(row: dict[str, Any], method: str) -> dict[str, Any]:
    key = f"method_{method.lower()}_output"
    prediction = _mapping(row.get("prediction_result"))
    nested_method = _mapping(prediction.get(f"method_{method.lower()}"))
    selection = _mapping(row.get(f"method_{method.lower()}_selection_metadata"))
    selected_prediction = _mapping(selection.get("prediction_method_output"))
    return _mapping(row.get(key)) or nested_method or selected_prediction


def _method_selection(row: dict[str, Any], method: str) -> dict[str, Any]:
    key = f"method_{method.lower()}_selection_metadata"
    direct = _mapping(row.get(key))
    selection = _mapping(row.get("selection_metadata"))
    nested = _mapping(selection.get(f"method_{method.lower()}"))
    return direct or nested


def _method_probability(row: dict[str, Any], method: str) -> dict[str, float] | None:
    output = _method_output(row, method)
    prediction = _mapping(row.get("prediction_result"))
    nested_method = _mapping(prediction.get(f"method_{method.lower()}"))
    return _first_distribution(
        row.get(f"method_{method.lower()}_probabilities_pct"),
        output.get("probabilities_pct"),
        nested_method.get("probabilities_pct"),
    )


def _find_nested_mapping_with_key(root: Any, key: str, max_depth: int = 5) -> dict[str, Any] | None:
    """Find a mapping owning key without treating a parent-level unrelated key as method metadata."""
    if max_depth < 0 or not isinstance(root, dict):
        return None
    if key in root:
        return root
    for child in root.values():
        if isinstance(child, dict):
            found = _find_nested_mapping_with_key(child, key, max_depth - 1)
            if found is not None:
                return found
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    found = _find_nested_mapping_with_key(item, key, max_depth - 1)
                    if found is not None:
                        return found
    return None


def _source_trace(row: dict[str, Any], method: str, distribution: dict[str, float] | None) -> dict[str, Any]:
    output = _method_output(row, method)
    selection = _method_selection(row, method)
    prediction = _mapping(row.get("prediction_result"))
    nested_method = _mapping(prediction.get(f"method_{method.lower()}"))
    roots = [output, nested_method, selection]
    counts_owner = next((root for root in roots if isinstance(root.get("class_counts"), dict)), None)
    weighted_owner = next((root for root in roots if isinstance(root.get("weighted_class_counts"), dict)), None)
    basis_owner = next((root for root in roots if root.get("probability_basis") is not None), None)
    limitations = output.get("limitations") or nested_method.get("limitations") or []
    return {
        "class_counts_recorded": counts_owner is not None,
        "class_counts": counts_owner.get("class_counts") if counts_owner else None,
        "weighted_class_counts_recorded": weighted_owner is not None,
        "weighted_class_counts": weighted_owner.get("weighted_class_counts") if weighted_owner else None,
        "probability_basis_recorded": basis_owner is not None,
        "probability_basis": basis_owner.get("probability_basis") if basis_owner else None,
        "sample_count": output.get("sample_count", nested_method.get("sample_count")),
        "effective_sample_size": output.get("effective_sample_size", nested_method.get("effective_sample_size")),
        "limited": output.get("limited", nested_method.get("limited")),
        "limitations": list(limitations) if isinstance(limitations, (tuple, list)) else [str(limitations)],
        "selection_status": selection.get("selection_status", "MISSING"),
        "selected_relationship_support_count": (
            _mapping(selection.get("selected_relationship")).get("supporting_observation_count")
        ),
        "distribution_available": distribution is not None,
        "distribution_zero_classes": (
            [label for label, value in distribution.items() if value <= 0.0]
            if distribution is not None else []
        ),
        "source_class_counts_status": (
            "RECORDED" if counts_owner else "NOT_RETAINED_IN_FROZEN_FOLD"
        ),
    }


def _actual_class(row: dict[str, Any]) -> str | None:
    actual = row.get("actual_class")
    return actual if actual in CLASSES else None


def _zero_summary(rows: list[dict[str, Any]], field: str, *, top_cases: int) -> dict[str, Any]:
    total_entries = 0
    zero_entries = 0
    zeros_by_class = Counter()
    actual_zero_rows = []
    usable_rows = 0
    invalid_rows = 0
    near_zero_counts = {"1e-12": 0, "1e-8": 0, "1e-6": 0, "1e-4": 0, "1e-3": 0, "1e-2": 0}

    for row in rows:
        actual = _actual_class(row)
        dist = _distribution(row.get(field))
        if dist is None:
            invalid_rows += 1
            continue
        usable_rows += 1
        for label in CLASSES:
            value = dist[label] / 100.0
            total_entries += 1
            if value == 0.0:
                zero_entries += 1
                zeros_by_class[label] += 1
            for threshold_str in near_zero_counts:
                if value <= float(threshold_str):
                    near_zero_counts[threshold_str] += 1
        if actual:
            p_actual = dist[actual] / 100.0
            if p_actual == 0.0:
                actual_zero_rows.append({
                    "prediction_date": str(row.get("prediction_date")),
                    "actual_class": actual,
                    "predicted_class": max(CLASSES, key=lambda label: dist[label]),
                    "p_actual": 0.0,
                    "probability_basis": _mapping(row.get("prediction_result")).get("probability_basis")
                    if field == "probabilities_pct" else None,
                })

    return {
        "available_rows": usable_rows,
        "missing_or_invalid_rows": invalid_rows,
        "total_class_probability_entries": total_entries,
        "exact_zero_entries": zero_entries,
        "exact_zero_entries_by_class": {label: zeros_by_class[label] for label in CLASSES},
        "true_class_exact_zero_rows": len(actual_zero_rows),
        "true_class_exact_zero_rate_pct": (len(actual_zero_rows) / usable_rows * 100.0) if usable_rows else None,
        "near_zero_entry_counts": near_zero_counts,
        "lowest_true_class_probability_cases": actual_zero_rows[:max(0, top_cases)],
    }


def _combined_reconstruction(rows: list[dict[str, Any]], tolerance_pct_points: float = 1e-5) -> dict[str, Any]:
    """Verify the stored combined vector against its recorded method vectors and combination weights."""
    checked = 0
    mismatches = 0
    skipped_no_weights = 0
    skipped_missing_methods = 0
    skipped_invalid_combined = 0
    max_diffs: list[float] = []
    mismatch_examples: list[dict[str, Any]] = []

    for row in rows:
        stored = _distribution(row.get("probabilities_pct"))
        if stored is None:
            skipped_invalid_combined += 1
            continue
        result = _mapping(row.get("prediction_result"))
        combination = _mapping(result.get("combination"))
        weights = combination.get("method_weights")
        if not isinstance(weights, dict) or not weights:
            shares = combination.get("method_weight_shares_pct")
            weights = shares if isinstance(shares, dict) and shares else None
        if not isinstance(weights, dict) or not weights:
            skipped_no_weights += 1
            continue

        raw_vectors = {
            method: _method_probability(row, method)
            for method in ("A", "B")
        }
        usable: list[tuple[str, dict[str, float], float]] = []
        missing_positive_weight_vector = False
        for method in ("A", "B"):
            weight = _as_float(weights.get(method))
            vector = raw_vectors[method]
            if weight is None or weight <= 0.0:
                continue
            if vector is None:
                missing_positive_weight_vector = True
                break
            usable.append((method, vector, weight))
        # Rebuilding from the surviving methods alone would silently change
        # the original combination. If a positively weighted method's vector
        # is unavailable, mark the fold uncheckable instead.
        if missing_positive_weight_vector or not usable:
            skipped_missing_methods += 1
            continue

        total_weight = sum(item[2] for item in usable)
        if total_weight <= 0.0:
            skipped_missing_methods += 1
            continue
        rebuilt = {
            label: sum(vector[label] * weight for _, vector, weight in usable) / total_weight
            for label in CLASSES
        }
        rebuilt = _distribution(rebuilt)
        if rebuilt is None:
            skipped_missing_methods += 1
            continue
        checked += 1
        max_diff = max(abs(rebuilt[label] - stored[label]) for label in CLASSES)
        max_diffs.append(max_diff)
        if max_diff > tolerance_pct_points:
            mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append({
                    "prediction_date": str(row.get("prediction_date")),
                    "max_abs_difference_pct_points": max_diff,
                    "stored": stored,
                    "reconstructed": rebuilt,
                    "active_methods": [method for method, _, _ in usable],
                })

    return {
        "folds_checked": checked,
        "folds_mismatched": mismatches,
        "mismatch_rate_pct": (mismatches / checked * 100.0) if checked else None,
        "max_abs_difference_pct_points": max(max_diffs) if max_diffs else None,
        "mean_abs_difference_pct_points": (sum(max_diffs) / len(max_diffs)) if max_diffs else None,
        "tolerance_pct_points": tolerance_pct_points,
        "skipped_missing_or_invalid_combined": skipped_invalid_combined,
        "skipped_without_recorded_weights": skipped_no_weights,
        "skipped_without_usable_method_vectors": skipped_missing_methods,
        "mismatch_examples": mismatch_examples,
        "status": "CHECKED" if checked and not mismatches else "MISMATCHES_FOUND" if mismatches else "NOT_ENOUGH_RECORDED_COMPONENTS",
    }


def audit_probability_construction(rows: Iterable[dict[str, Any]], *, top_cases: int = 10) -> dict[str, Any]:
    """Audit stored construction traces without fabricating missing source counts."""
    materialized = [row for row in rows if isinstance(row, dict)]
    report: dict[str, Any] = {
        "fold_count": len(materialized),
        "methods": {},
        "combined_reconstruction": _combined_reconstruction(materialized),
        "source_contract": {
            "required_method_builder_fields": ["class_counts", "weighted_class_counts", "probability_basis"],
            "note": (
                "The Phase 5.2 builders derive class probabilities from weighted class shares. "
                "The frozen Phase 5.8 fold contract keeps the resulting MethodPrediction vectors, "
                "sample counts, effective sample sizes and selected-relationship metadata; it does "
                "not necessarily retain the builder-level class_counts / weighted_class_counts. "
                "Missing counts are reported as missing and are never reconstructed from percentages."
            ),
        },
    }

    for name, field in (("Combined", "probabilities_pct"), ("Method A", "method_a_probabilities_pct"), ("Method B", "method_b_probabilities_pct")):
        traces: list[dict[str, Any]] = []
        zero_rows: list[dict[str, Any]] = []
        distribution_rows: list[dict[str, Any]] = []
        selection_statuses = Counter()
        basis_status = Counter()
        counts_trace = 0
        weighted_counts_trace = 0
        basis_trace = 0
        vector_output_mismatch = 0
        weighted_count_vector_mismatch = 0
        support_sample_count_mismatches = 0
        support_count_comparisons = 0
        sample_counts: list[float] = []
        effective_sizes: list[float] = []

        method_letter = "A" if name == "Method A" else "B" if name == "Method B" else None
        for row in materialized:
            if field == "probabilities_pct":
                raw = row.get(field)
                dist = _distribution(raw)
            else:
                dist = _method_probability(row, method_letter or "A")
            if dist is not None:
                distribution_rows.append({
                    "prediction_date": row.get("prediction_date"),
                    "actual_class": _actual_class(row),
                    "probabilities_pct": dist,
                })
            if field == "probabilities_pct":
                if dist is not None:
                    builder_basis = _mapping(row.get("prediction_result")).get("probability_basis")
                    basis_status[builder_basis or "NOT_RECORDED"] += 1
                continue

            trace = _source_trace(row, method_letter or "A", dist)
            traces.append(trace)
            status = trace["selection_status"] or "MISSING"
            selection_statuses[status] += 1
            basis_status[trace["probability_basis"] or "NOT_RECORDED"] += 1
            counts_trace += int(trace["class_counts_recorded"])
            weighted_counts_trace += int(trace["weighted_class_counts_recorded"])
            basis_trace += int(trace["probability_basis_recorded"])
            if trace["sample_count"] is not None:
                val = _as_float(trace["sample_count"])
                if val is not None:
                    sample_counts.append(val)
            if trace["effective_sample_size"] is not None:
                val = _as_float(trace["effective_sample_size"])
                if val is not None:
                    effective_sizes.append(val)
            support_count = _as_float(trace["selected_relationship_support_count"])
            method_count = _as_float(trace["sample_count"])
            if support_count is not None and method_count is not None:
                support_count_comparisons += 1
                if abs(support_count - method_count) > 1e-6:
                    support_sample_count_mismatches += 1
            weighted_counts = trace.get("weighted_class_counts")
            if dist is not None and isinstance(weighted_counts, dict):
                weighted = {label: _as_float(weighted_counts.get(label)) for label in CLASSES}
                if all(value is not None and value >= 0.0 for value in weighted.values()):
                    mass = sum(float(value) for value in weighted.values() if value is not None)
                    if mass > 0.0:
                        expected = {label: float(weighted[label]) / mass * 100.0 for label in CLASSES}
                        if max(abs(expected[label] - dist[label]) for label in CLASSES) > 1e-5:
                            weighted_count_vector_mismatch += 1
            output = _method_output(row, method_letter or "A")
            output_dist = _distribution(output.get("probabilities_pct"))
            if output_dist is not None and dist is not None:
                if max(abs(output_dist[label] - dist[label]) for label in CLASSES) > 1e-5:
                    vector_output_mismatch += 1

        zeros = _zero_summary(
            distribution_rows,
            "probabilities_pct",
            top_cases=top_cases,
        )
        if name == "Combined":
            status = "COMPLETE" if len(distribution_rows) == len(materialized) and materialized else "PARTIAL" if distribution_rows else "UNAVAILABLE"
            report["methods"][name] = {
                "availability_status": status,
                "zero_audit": zeros,
                "probability_basis_counts": dict(basis_status),
            }
        else:
            status = "COMPLETE" if len(distribution_rows) == len(materialized) and materialized else "PARTIAL" if distribution_rows else "UNAVAILABLE"
            report["methods"][name] = {
                "availability_status": status,
                "zero_audit": zeros,
                "selection_status_counts": dict(selection_statuses),
                "class_counts_trace_rows": counts_trace,
                "weighted_class_counts_trace_rows": weighted_counts_trace,
                "probability_basis_trace_rows": basis_trace,
                "probability_basis_counts": dict(basis_status),
                "stored_method_output_vector_mismatches": vector_output_mismatch,
                "weighted_count_vector_mismatches": weighted_count_vector_mismatch,
                "selected_support_sample_count_comparisons": support_count_comparisons,
                "selected_support_sample_count_mismatches": support_sample_count_mismatches,
                "sample_count_mean": (sum(sample_counts) / len(sample_counts)) if sample_counts else None,
                "sample_count_min": min(sample_counts) if sample_counts else None,
                "sample_count_max": max(sample_counts) if sample_counts else None,
                "effective_sample_size_mean": (sum(effective_sizes) / len(effective_sizes)) if effective_sizes else None,
                "effective_sample_size_min": min(effective_sizes) if effective_sizes else None,
                "effective_sample_size_max": max(effective_sizes) if effective_sizes else None,
                "source_count_trace_status": "RECORDED" if counts_trace and weighted_counts_trace else "MISSING_FROM_CURRENT_FROZEN_ARTIFACT",
            }

    combined = report["methods"].get("Combined", {})
    combined_zeros = combined.get("zero_audit", {}).get("true_class_exact_zero_rows", 0)
    report["interpretation"] = {
        "combined_true_class_zero_rows": combined_zeros,
        "combined_reconstruction_status": report["combined_reconstruction"]["status"],
        "builder_level_count_trace_available": all(
            report["methods"].get(name, {}).get("source_count_trace_status") == "RECORDED"
            for name in ("Method A", "Method B")
        ),
        "root_cause_limit": (
            "A zero probability is consistent with zero positive-weight support in that class under the empirical class-share formula. "
            "Because raw builder-level class counts are not recorded on every frozen fold, this audit does not claim to verify the exact "+
            "support count behind each zero. It separately checks whether the stored combined forecast can be rebuilt from recorded method vectors and weights."
        ),
    }
    return report


def format_probability_construction_report(report: dict[str, Any], *, snapshot: str, top_cases: int = 10) -> str:
    lines = [
        "PHASE 6.2 PROBABILITY-CONSTRUCTION TRACE AUDIT",
        f"Snapshot: {snapshot}",
        f"Frozen folds: {report.get('fold_count', 0)}",
        "Protocol: READ_ONLY_FROZEN_ARTIFACT; NO_OLAP_RECOMPUTATION",
        "",
    ]
    for name in ("Combined", "Method A", "Method B"):
        item = report.get("methods", {}).get(name, {})
        zero = item.get("zero_audit", {})
        lines.append(name)
        lines.append(f"  availability: {item.get('availability_status', 'MISSING')}")
        lines.append(
            f"  exact zero class-probability entries: {zero.get('exact_zero_entries', 0)}/"
            f"{zero.get('total_class_probability_entries', 0)}"
        )
        lines.append(
            f"  actual-class exact-zero rows: {zero.get('true_class_exact_zero_rows', 0)} "
            f"({zero.get('true_class_exact_zero_rate_pct') if zero.get('true_class_exact_zero_rate_pct') is not None else 'n/a'}%)"
        )
        lines.append(f"  zeros by class: {zero.get('exact_zero_entries_by_class', {})}")
        lines.append(f"  near-zero counts: {zero.get('near_zero_entry_counts', {})}")
        if name != "Combined":
            lines.append(f"  selection statuses: {item.get('selection_status_counts', {})}")
            lines.append(
                "  builder class-count traces: "
                f"{item.get('class_counts_trace_rows', 0)}/{report.get('fold_count', 0)}"
            )
            lines.append(
                "  weighted class-count traces: "
                f"{item.get('weighted_class_counts_trace_rows', 0)}/{report.get('fold_count', 0)}"
            )
            lines.append(
                f"  sample count mean/min/max: {item.get('sample_count_mean')} / "
                f"{item.get('sample_count_min')} / {item.get('sample_count_max')}"
            )
            lines.append(
                "  effective sample size mean/min/max: "
                f"{item.get('effective_sample_size_mean')} / "
                f"{item.get('effective_sample_size_min')} / {item.get('effective_sample_size_max')}"
            )
            lines.append(f"  stored method-output/vector mismatches: {item.get('stored_method_output_vector_mismatches', 0)}")
            lines.append(f"  weighted-count/vector mismatches: {item.get('weighted_count_vector_mismatches', 0)}")
            lines.append(
                "  selected support vs method sample count mismatches: "
                f"{item.get('selected_support_sample_count_mismatches', 0)}/"
                f"{item.get('selected_support_sample_count_comparisons', 0)}"
            )
            lines.append(f"  source-count trace status: {item.get('source_count_trace_status')}")
        cases = zero.get("lowest_true_class_probability_cases", [])[:max(0, top_cases)]
        if cases:
            lines.append("  first folds with actual-class probability exactly zero:")
            for case in cases:
                lines.append(
                    f"    {case.get('prediction_date')} actual={case.get('actual_class')} "
                    f"predicted={case.get('predicted_class')} p(actual)=0"
                )
        lines.append("")

    reconstruction = report.get("combined_reconstruction", {})
    lines.extend([
        "Stored combined-vector reconstruction from method vectors and recorded weights:",
        f"  status: {reconstruction.get('status')}",
        f"  checked folds: {reconstruction.get('folds_checked')}",
        f"  mismatches: {reconstruction.get('folds_mismatched')}",
        f"  maximum absolute difference: {reconstruction.get('max_abs_difference_pct_points')} percentage points",
        f"  skipped (no recorded weights): {reconstruction.get('skipped_without_recorded_weights')}",
        f"  skipped (no usable method vectors): {reconstruction.get('skipped_without_usable_method_vectors')}",
        "",
        "Source trace limitation:",
        "  Phase 5.2 Method A/B builders construct probabilities from weighted class shares.",
        "  This frozen artifact retains the output vector, but builder-level class_counts and weighted_class_counts are not present on all folds.",
        "  The audit does not reconstruct those counts from percentages or claim that all zero probabilities are proven source-count zeros.",
        "",
        "Interpretation guardrails:",
        "  - A probability of zero is consistent with zero positive-weight support for that class.",
        "  - Method-level counts missing from the frozen record cannot be recovered without retained source-count fields.",
        "  - Missing method vectors remain missing; they are not inferred from trends.",
        "  - No snapshot file, Phase 5.8 calculation, relationship search or OLAP query is changed.",
    ])
    return "\n".join(lines) + "\n"
