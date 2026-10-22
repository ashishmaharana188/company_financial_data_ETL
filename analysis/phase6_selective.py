from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import numpy as np

from .outcome_labels import OUTCOME_CLASSES
from .phase6_calibration import CalibratedPrediction

_EPS = 1e-12


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))




def _normalise(values: dict[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(values, dict):
        vector = np.asarray(
            [float(values[label]) for label in OUTCOME_CLASSES],
            dtype=np.float64,
        )
    else:
        vector = np.asarray(list(values), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("Probability vector must contain exactly three classes.")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("Probability vector must contain finite non-negative values.")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("Probability vector must have positive total mass.")
    return vector / total

def _log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, _EPS, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _baseline_accuracy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=len(OUTCOME_CLASSES))
    return float(np.max(counts) / len(labels) * 100.0)


@dataclass(frozen=True)
class SelectiveThresholdResult:
    threshold_pct: float
    total_observations: int
    selected_observations: int
    coverage_pct: float
    accuracy_pct: float
    baseline_accuracy_pct: float
    accuracy_lift_pct_points: float
    mean_confidence_pct: float
    confidence_minus_accuracy_pct_points: float
    log_loss: float | None
    brier_score: float | None

    @property
    def improves_over_baseline(self) -> bool:
        return self.accuracy_pct > self.baseline_accuracy_pct

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold_pct": self.threshold_pct,
            "total_observations": self.total_observations,
            "selected_observations": self.selected_observations,
            "coverage_pct": self.coverage_pct,
            "accuracy_pct": self.accuracy_pct,
            "baseline_accuracy_pct": self.baseline_accuracy_pct,
            "accuracy_lift_pct_points": self.accuracy_lift_pct_points,
            "mean_confidence_pct": self.mean_confidence_pct,
            "confidence_minus_accuracy_pct_points": self.confidence_minus_accuracy_pct_points,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "improves_over_baseline": self.improves_over_baseline,
        }


@dataclass(frozen=True)
class ConfidenceBucketResult:
    lower_bound_pct: float
    upper_bound_pct: float
    observations: int
    coverage_pct: float
    accuracy_pct: float
    mean_confidence_pct: float
    confidence_minus_accuracy_pct_points: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower_bound_pct": self.lower_bound_pct,
            "upper_bound_pct": self.upper_bound_pct,
            "observations": self.observations,
            "coverage_pct": self.coverage_pct,
            "accuracy_pct": self.accuracy_pct,
            "mean_confidence_pct": self.mean_confidence_pct,
            "confidence_minus_accuracy_pct_points": self.confidence_minus_accuracy_pct_points,
        }


@dataclass(frozen=True)
class SelectedThresholdEvaluation:
    threshold_pct: float | None
    development_observations: int
    development_coverage_pct: float
    development_accuracy_pct: float | None
    development_baseline_accuracy_pct: float
    development_accuracy_lift_pct_points: float | None
    final_observations: int
    final_coverage_pct: float
    final_accuracy_pct: float | None
    final_baseline_accuracy_pct: float
    final_accuracy_lift_pct_points: float | None
    selection_rule: str
    selection_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold_pct": self.threshold_pct,
            "development_observations": self.development_observations,
            "development_coverage_pct": self.development_coverage_pct,
            "development_accuracy_pct": self.development_accuracy_pct,
            "development_baseline_accuracy_pct": self.development_baseline_accuracy_pct,
            "development_accuracy_lift_pct_points": self.development_accuracy_lift_pct_points,
            "final_observations": self.final_observations,
            "final_coverage_pct": self.final_coverage_pct,
            "final_accuracy_pct": self.final_accuracy_pct,
            "final_baseline_accuracy_pct": self.final_baseline_accuracy_pct,
            "final_accuracy_lift_pct_points": self.final_accuracy_lift_pct_points,
            "selection_rule": self.selection_rule,
            "selection_status": self.selection_status,
        }


@dataclass(frozen=True)
class Phase6SelectiveResult:
    observations: int
    baseline_accuracy_pct: float
    thresholds: tuple[SelectiveThresholdResult, ...]
    confidence_buckets: tuple[ConfidenceBucketResult, ...]
    selected_threshold: SelectedThresholdEvaluation
    protocol: str

    @property
    def improving_thresholds(self) -> tuple[SelectiveThresholdResult, ...]:
        return tuple(item for item in self.thresholds if item.improves_over_baseline)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "baseline_accuracy_pct": self.baseline_accuracy_pct,
            "thresholds": [item.as_dict() for item in self.thresholds],
            "confidence_buckets": [item.as_dict() for item in self.confidence_buckets],
            "selected_threshold": self.selected_threshold.as_dict(),
            "improving_thresholds": [item.as_dict() for item in self.improving_thresholds],
            "protocol": self.protocol,
        }


def evaluate_selective_predictions(
    predictions: Sequence[CalibratedPrediction],
    *,
    thresholds_pct: Sequence[float] = (34.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0),
    confidence_buckets_pct: Sequence[tuple[float, float]] = (
        (33.3333333333, 40.0),
        (40.0, 50.0),
        (50.0, 60.0),
        (60.0, 70.0),
        (70.0, 100.0000001),
    ),
    development_fraction: float = 0.80,
    minimum_development_coverage_pct: float = 10.0,
) -> Phase6SelectiveResult:
    """Evaluate selective classification using fixed, predeclared thresholds.

    Thresholds are not fitted on the evaluation observations. This is a
    diagnostic gate for Phase 6.2, not a trading-policy optimization step.
    """

    ordered = sorted(predictions, key=lambda item: _as_date(item.prediction_date))
    if not ordered:
        raise ValueError("At least one calibrated prediction is required.")
    if not 0.5 <= development_fraction < 1.0:
        raise ValueError("development_fraction must be in [0.5, 1.0).")
    if not 0.0 <= minimum_development_coverage_pct <= 100.0:
        raise ValueError("minimum_development_coverage_pct must be between 0 and 100.")

    labels = np.asarray(
        [OUTCOME_CLASSES.index(item.actual_class) for item in ordered],
        dtype=np.int64,
    )
    probabilities = np.vstack(
        [_normalise(item.calibrated_probabilities_pct) for item in ordered]
    )
    predicted = np.argmax(probabilities, axis=1)
    confidence_pct = np.max(probabilities, axis=1) * 100.0
    baseline = _baseline_accuracy(labels)

    clean_thresholds = sorted({float(threshold) for threshold in thresholds_pct})
    if any(not 0.0 <= threshold <= 100.0 for threshold in clean_thresholds):
        raise ValueError("Selective thresholds must be between 0 and 100 percent.")

    threshold_results: list[SelectiveThresholdResult] = []
    for threshold in clean_thresholds:
        mask = confidence_pct >= threshold
        selected = int(mask.sum())
        coverage = selected / len(ordered) * 100.0

        if selected:
            accuracy = float(np.mean(predicted[mask] == labels[mask]) * 100.0)
            mean_confidence = float(np.mean(confidence_pct[mask]))
            gap = mean_confidence - accuracy
            selected_probabilities = probabilities[mask]
            selected_labels = labels[mask]
            selected_log_loss = _log_loss(selected_probabilities, selected_labels)
            selected_brier = _brier(selected_probabilities, selected_labels)
        else:
            accuracy = 0.0
            mean_confidence = 0.0
            gap = 0.0
            selected_log_loss = None
            selected_brier = None

        threshold_results.append(
            SelectiveThresholdResult(
                threshold_pct=threshold,
                total_observations=len(ordered),
                selected_observations=selected,
                coverage_pct=coverage,
                accuracy_pct=accuracy,
                baseline_accuracy_pct=baseline,
                accuracy_lift_pct_points=accuracy - baseline,
                mean_confidence_pct=mean_confidence,
                confidence_minus_accuracy_pct_points=gap,
                log_loss=selected_log_loss,
                brier_score=selected_brier,
            )
        )

    bucket_results: list[ConfidenceBucketResult] = []
    for bucket_index, (lower, upper) in enumerate(confidence_buckets_pct):
        lower = float(lower)
        upper = float(upper)
        if not 0.0 <= lower < upper <= 100.0000001:
            raise ValueError("Confidence bucket bounds are invalid.")
        if upper > 100.0:
            upper = 100.0
        if bucket_index == 0:
            mask = (confidence_pct >= lower) & (confidence_pct <= upper)
        else:
            mask = (confidence_pct > lower) & (confidence_pct <= upper)
        count = int(mask.sum())
        if count:
            accuracy = float(np.mean(predicted[mask] == labels[mask]) * 100.0)
            mean_confidence = float(np.mean(confidence_pct[mask]))
        else:
            accuracy = 0.0
            mean_confidence = 0.0
        bucket_results.append(
            ConfidenceBucketResult(
                lower_bound_pct=lower,
                upper_bound_pct=upper,
                observations=count,
                coverage_pct=count / len(ordered) * 100.0,
                accuracy_pct=accuracy,
                mean_confidence_pct=mean_confidence,
                confidence_minus_accuracy_pct_points=mean_confidence - accuracy,
            )
        )

    # Chronological threshold selection: choose only from the development
    # period, then evaluate that one selected threshold on the later final
    # period. This prevents threshold choice from consuming the final OOS data.
    split_index = int(np.floor(len(ordered) * development_fraction))
    split_index = min(max(split_index, 1), len(ordered) - 1)

    dev_probs = probabilities[:split_index]
    dev_labels = labels[:split_index]
    dev_confidence = confidence_pct[:split_index]
    dev_predicted = predicted[:split_index]
    final_probs = probabilities[split_index:]
    final_labels = labels[split_index:]
    final_confidence = confidence_pct[split_index:]
    final_predicted = predicted[split_index:]

    dev_baseline = _baseline_accuracy(dev_labels)
    final_baseline = _baseline_accuracy(final_labels)

    candidates: list[tuple[float, float, float]] = []
    for threshold in clean_thresholds:
        dev_mask = dev_confidence >= threshold
        dev_selected = int(dev_mask.sum())
        dev_coverage = dev_selected / len(dev_labels) * 100.0
        if dev_selected == 0 or dev_coverage < minimum_development_coverage_pct:
            continue
        dev_accuracy = float(np.mean(dev_predicted[dev_mask] == dev_labels[dev_mask]) * 100.0)
        lift = dev_accuracy - dev_baseline
        if dev_accuracy > dev_baseline:
            # Sort by development lift, then coverage, then lower threshold.
            candidates.append((lift, dev_coverage, -threshold))

    if candidates:
        _, _, negative_threshold = max(candidates)
        selected_threshold = -negative_threshold
        dev_mask = dev_confidence >= selected_threshold
        final_mask = final_confidence >= selected_threshold
        dev_accuracy = float(np.mean(dev_predicted[dev_mask] == dev_labels[dev_mask]) * 100.0)
        final_selected = int(final_mask.sum())
        final_accuracy = (
            float(np.mean(final_predicted[final_mask] == final_labels[final_mask]) * 100.0)
            if final_selected
            else None
        )
        selected_evaluation = SelectedThresholdEvaluation(
            threshold_pct=selected_threshold,
            development_observations=len(dev_labels),
            development_coverage_pct=float(dev_mask.mean() * 100.0),
            development_accuracy_pct=dev_accuracy,
            development_baseline_accuracy_pct=dev_baseline,
            development_accuracy_lift_pct_points=dev_accuracy - dev_baseline,
            final_observations=len(final_labels),
            final_coverage_pct=float(final_mask.mean() * 100.0),
            final_accuracy_pct=final_accuracy,
            final_baseline_accuracy_pct=final_baseline,
            final_accuracy_lift_pct_points=(final_accuracy - final_baseline) if final_accuracy is not None else None,
            selection_rule=(
                "Select highest development accuracy lift among predeclared thresholds "
                f"with >= {minimum_development_coverage_pct:.1f}% development coverage; "
                "tie-break by higher coverage, then lower threshold."
            ),
            selection_status="SELECTED_FROM_DEVELOPMENT",
        )
    else:
        selected_evaluation = SelectedThresholdEvaluation(
            threshold_pct=None,
            development_observations=len(dev_labels),
            development_coverage_pct=100.0,
            development_accuracy_pct=None,
            development_baseline_accuracy_pct=dev_baseline,
            development_accuracy_lift_pct_points=None,
            final_observations=len(final_labels),
            final_coverage_pct=0.0,
            final_accuracy_pct=None,
            final_baseline_accuracy_pct=final_baseline,
            final_accuracy_lift_pct_points=None,
            selection_rule=(
                "Select highest development accuracy lift among predeclared thresholds "
                f"with >= {minimum_development_coverage_pct:.1f}% development coverage."
            ),
            selection_status="NO_DEVELOPMENT_THRESHOLD_BEATS_BASELINE",
        )

    return Phase6SelectiveResult(
        observations=len(ordered),
        baseline_accuracy_pct=baseline,
        thresholds=tuple(threshold_results),
        confidence_buckets=tuple(bucket_results),
        selected_threshold=selected_evaluation,
        protocol="CHRONOLOGICAL_DEVELOPMENT_SELECTION_FINAL_OOS",
    )
