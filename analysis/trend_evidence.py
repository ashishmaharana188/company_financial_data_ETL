from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


TREND_CLASSES = ("UP", "SIDEWAYS", "DOWN")
ALL_SIGNAL_CLASSES = (*TREND_CLASSES, "NO_CLEAR_TREND")


@dataclass(frozen=True)
class TrendEvidenceMethodSummary:
    """Historical trend-evidence diagnostics for one method.

    These metrics are deliberately framed as trend-detection diagnostics.
    They do not represent a future trading prediction claim.
    """

    method: str
    evaluated_folds: int
    clear_trend_folds: int
    coverage_pct: float
    no_clear_trend_folds: int
    directional_folds: int
    up_signals: int
    down_signals: int
    sideways_signals: int
    trend_counts: Mapping[str, int]
    clear_trend_accuracy_pct: float
    matched_majority_baseline_pct: float
    accuracy_lift_vs_matched_baseline_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "evaluated_folds": self.evaluated_folds,
            "clear_trend_folds": self.clear_trend_folds,
            "coverage_pct": self.coverage_pct,
            "no_clear_trend_folds": self.no_clear_trend_folds,
            "directional_folds": self.directional_folds,
            "up_signals": self.up_signals,
            "down_signals": self.down_signals,
            "sideways_signals": self.sideways_signals,
            "trend_counts": dict(self.trend_counts),
            "clear_trend_accuracy_pct": self.clear_trend_accuracy_pct,
            "matched_majority_baseline_pct": self.matched_majority_baseline_pct,
            "accuracy_lift_vs_matched_baseline_pct": self.accuracy_lift_vs_matched_baseline_pct,
        }


@dataclass(frozen=True)
class TrendEvidenceReport:
    """Trend-focused view over an already validated Phase 5.8 result."""

    method_a: TrendEvidenceMethodSummary
    method_b: TrendEvidenceMethodSummary
    combined: TrendEvidenceMethodSummary
    both_methods_clear_folds: int
    method_agreement_folds: int
    method_conflict_folds: int
    method_agreement_rate_pct: float
    recent_window_size: int
    recent_combined_trend_counts: Mapping[str, int]
    latest_validated_trend: str | None
    latest_method_a_trend: str | None
    latest_method_b_trend: str | None
    latest_conviction: str | None
    latest_probabilities_pct: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_a": self.method_a.as_dict(),
            "method_b": self.method_b.as_dict(),
            "combined": self.combined.as_dict(),
            "both_methods_clear_folds": self.both_methods_clear_folds,
            "method_agreement_folds": self.method_agreement_folds,
            "method_conflict_folds": self.method_conflict_folds,
            "method_agreement_rate_pct": self.method_agreement_rate_pct,
            "recent_window_size": self.recent_window_size,
            "recent_combined_trend_counts": dict(self.recent_combined_trend_counts),
            "latest_validated_trend": self.latest_validated_trend,
            "latest_method_a_trend": self.latest_method_a_trend,
            "latest_method_b_trend": self.latest_method_b_trend,
            "latest_conviction": self.latest_conviction,
            "latest_probabilities_pct": dict(self.latest_probabilities_pct),
        }


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


def _majority_accuracy(actual_classes: Sequence[str | None]) -> float:
    usable = [label for label in actual_classes if label in TREND_CLASSES]
    if not usable:
        return 0.0
    counts = {label: usable.count(label) for label in TREND_CLASSES}
    return _percent(max(counts.values()), len(usable))


def _trend_counts(
    folds: Sequence[Any],
    trend_getter: Callable[[Any], str | None],
) -> dict[str, int]:
    counts = {label: 0 for label in ALL_SIGNAL_CLASSES}
    for fold in folds:
        trend = trend_getter(fold)
        if trend in counts:
            counts[trend] += 1
    return counts


def _method_summary(
    method: str,
    folds: Sequence[Any],
    trend_getter: Callable[[Any], str | None],
) -> TrendEvidenceMethodSummary:
    usable = [fold for fold in folds if not getattr(fold, "limited", False) and getattr(fold, "provenance_clean", False)]
    clear = [fold for fold in usable if trend_getter(fold) in TREND_CLASSES]
    directional = [fold for fold in usable if trend_getter(fold) in {"UP", "DOWN"}]
    hits = [fold for fold in clear if trend_getter(fold) == getattr(fold, "actual_class", None)]
    baseline = _majority_accuracy([getattr(fold, "actual_class", None) for fold in clear])
    accuracy = _percent(len(hits), len(clear))
    counts = _trend_counts(usable, trend_getter)

    return TrendEvidenceMethodSummary(
        method=method,
        evaluated_folds=len(usable),
        clear_trend_folds=len(clear),
        coverage_pct=_percent(len(clear), len(folds)),
        no_clear_trend_folds=counts["NO_CLEAR_TREND"],
        directional_folds=len(directional),
        up_signals=counts["UP"],
        down_signals=counts["DOWN"],
        sideways_signals=counts["SIDEWAYS"],
        trend_counts=counts,
        clear_trend_accuracy_pct=accuracy,
        matched_majority_baseline_pct=baseline,
        accuracy_lift_vs_matched_baseline_pct=accuracy - baseline,
    )


def build_trend_evidence_report(
    result: Any,
    *,
    recent_window: int = 20,
) -> TrendEvidenceReport:
    """Build trend-focused diagnostics from a completed Phase 5.8 result.

    The result is expected to expose ``prediction_folds`` with the existing
    ``PredictionFoldResult`` interface. No new prediction is performed here.
    """

    if recent_window < 1:
        raise ValueError("recent_window must be >= 1")

    folds = tuple(getattr(result, "prediction_folds", ()))
    method_a = _method_summary("A", folds, lambda fold: getattr(fold, "method_a_trend", None))
    method_b = _method_summary("B", folds, lambda fold: getattr(fold, "method_b_trend", None))
    combined = _method_summary("COMBINED", folds, lambda fold: getattr(fold, "predicted_trend", None))

    both_clear = [
        fold
        for fold in folds
        if getattr(fold, "method_a_trend", None) in TREND_CLASSES
        and getattr(fold, "method_b_trend", None) in TREND_CLASSES
    ]
    agreement = [
        fold
        for fold in both_clear
        if getattr(fold, "method_a_trend", None) == getattr(fold, "method_b_trend", None)
    ]
    conflicts = len(both_clear) - len(agreement)

    recent = folds[-recent_window:]
    recent_counts = _trend_counts(
        recent,
        lambda fold: getattr(fold, "predicted_trend", None),
    )

    latest = folds[-1] if folds else None
    latest_probabilities = getattr(latest, "probabilities_pct", {}) if latest is not None else {}
    if not isinstance(latest_probabilities, Mapping):
        latest_probabilities = {}

    return TrendEvidenceReport(
        method_a=method_a,
        method_b=method_b,
        combined=combined,
        both_methods_clear_folds=len(both_clear),
        method_agreement_folds=len(agreement),
        method_conflict_folds=conflicts,
        method_agreement_rate_pct=_percent(len(agreement), len(both_clear)),
        recent_window_size=len(recent),
        recent_combined_trend_counts=recent_counts,
        latest_validated_trend=(getattr(latest, "predicted_trend", None) if latest is not None else None),
        latest_method_a_trend=(getattr(latest, "method_a_trend", None) if latest is not None else None),
        latest_method_b_trend=(getattr(latest, "method_b_trend", None) if latest is not None else None),
        latest_conviction=(getattr(latest, "conviction", None) if latest is not None else None),
        latest_probabilities_pct={
            str(key): float(value)
            for key, value in latest_probabilities.items()
        },
    )


def print_trend_evidence_report(
    report: TrendEvidenceReport,
) -> None:
    """Print the trend-analysis interpretation layer for Phase 5.8."""

    print()
    print("PHASE 5.8 TREND EVIDENCE DIAGNOSTICS")
    print("Purpose: validate whether the analysis is producing repeatable trend signals, not trading predictions.")
    print()

    for summary in (report.method_a, report.method_b, report.combined):
        print(f"Method {summary.method} trend evidence:")
        print("  evaluated folds:", summary.evaluated_folds)
        print("  clear-trend folds:", summary.clear_trend_folds)
        print("  trend coverage:", round(summary.coverage_pct, 2))
        print("  UP / SIDEWAYS / DOWN:", summary.up_signals, "/", summary.sideways_signals, "/", summary.down_signals)
        print("  NO_CLEAR_TREND:", summary.no_clear_trend_folds)
        print("  historical clear-trend agreement with realized class:", round(summary.clear_trend_accuracy_pct, 2))
        print("  matched majority baseline:", round(summary.matched_majority_baseline_pct, 2))
        print("  lift vs matched baseline:", round(summary.accuracy_lift_vs_matched_baseline_pct, 2))
        print()

    print("Method A/B cross-method evidence:")
    print("  both methods clear:", report.both_methods_clear_folds)
    print("  agreement:", report.method_agreement_folds)
    print("  conflict:", report.method_conflict_folds)
    print("  agreement rate:", round(report.method_agreement_rate_pct, 2))
    print()

    print(f"Recent combined trend window ({report.recent_window_size} folds):")
    print(" ", dict(report.recent_combined_trend_counts))
    print()

    print("Latest historically validated trend state:")
    print("  combined trend:", report.latest_validated_trend)
    print("  Method A:", report.latest_method_a_trend)
    print("  Method B:", report.latest_method_b_trend)
    print("  conviction:", report.latest_conviction)
    print("  probabilities:", {k: round(v, 2) for k, v in report.latest_probabilities_pct.items()})
    print()
    print("NOTE: The latest historically validated state is not the current 2026-10-07 state. It is the latest fold whose forward outcome was available for validation.")
    print("NOTE: Historical hit-rate metrics are sanity checks for trend detection. They are not being used as the project's primary success criterion.")
