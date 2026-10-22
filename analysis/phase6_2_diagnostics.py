from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

import numpy as np

from .outcome_labels import OUTCOME_CLASSES
from .phase6_calibration import CalibratedPrediction


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
    if total <= 0:
        raise ValueError("Probability vector must have positive total mass.")
    return vector / total


def _accuracy(mask: np.ndarray, predicted: np.ndarray, labels: np.ndarray) -> float | None:
    n = int(mask.sum())
    if not n:
        return None
    return float(np.mean(predicted[mask] == labels[mask]) * 100.0)


def _majority_accuracy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=3)
    return float(np.max(counts) / len(labels) * 100.0)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    sx = float(rx.std())
    sy = float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class ConfidenceRankPoint:
    population_fraction_pct: float
    observations: int
    accuracy_pct: float | None
    baseline_accuracy_pct: float
    lift_pct_points: float | None
    mean_confidence_pct: float | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ClassDiagnostic:
    predicted_class: str
    observations: int
    accuracy_pct: float | None
    mean_confidence_pct: float | None
    actual_up_pct: float
    actual_sideways_pct: float
    actual_down_pct: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Phase62Diagnostics:
    observations: int
    overall_accuracy_pct: float
    baseline_majority_accuracy_pct: float
    confidence_accuracy_spearman: float | None
    incorrect_confidence_mean_pct: float
    correct_confidence_mean_pct: float
    confidence_gap_correct_minus_incorrect_pct_points: float
    high_confidence_is_better: bool
    confidence_rank_curve: tuple[ConfidenceRankPoint, ...]
    class_diagnostics: tuple[ClassDiagnostic, ...]
    protocol: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "overall_accuracy_pct": self.overall_accuracy_pct,
            "baseline_majority_accuracy_pct": self.baseline_majority_accuracy_pct,
            "confidence_accuracy_spearman": self.confidence_accuracy_spearman,
            "incorrect_confidence_mean_pct": self.incorrect_confidence_mean_pct,
            "correct_confidence_mean_pct": self.correct_confidence_mean_pct,
            "confidence_gap_correct_minus_incorrect_pct_points": self.confidence_gap_correct_minus_incorrect_pct_points,
            "high_confidence_is_better": self.high_confidence_is_better,
            "confidence_rank_curve": [item.as_dict() for item in self.confidence_rank_curve],
            "class_diagnostics": [item.as_dict() for item in self.class_diagnostics],
            "protocol": self.protocol,
        }


def diagnose_confidence(
    predictions: Sequence[CalibratedPrediction],
    *,
    top_fractions_pct: Sequence[float] = (10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0),
) -> Phase62Diagnostics:
    ordered = sorted(predictions, key=lambda item: _as_date(item.prediction_date))
    if not ordered:
        raise ValueError("At least one calibrated prediction is required.")

    probabilities = np.vstack([
        _normalise(item.calibrated_probabilities_pct) for item in ordered
    ])
    labels = np.asarray(
        [OUTCOME_CLASSES.index(item.actual_class) for item in ordered],
        dtype=np.int64,
    )
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1) * 100.0
    correct = (predicted == labels)

    baseline = _majority_accuracy(labels)
    correct_mean = float(np.mean(confidence[correct])) if np.any(correct) else 0.0
    incorrect_mean = float(np.mean(confidence[~correct])) if np.any(~correct) else 0.0

    # Higher confidence should normally imply a higher probability of being
    # correct. A negative Spearman correlation is direct evidence of the
    # opposite ordering, while a non-positive top-half vs bottom-half lift
    # indicates no useful confidence ranking.
    rank_corr = _spearman(confidence, correct.astype(np.float64))

    order_desc = np.argsort(-confidence, kind="mergesort")
    rank_points: list[ConfidenceRankPoint] = []
    for fraction in sorted({float(v) for v in top_fractions_pct}):
        if not 0.0 < fraction <= 100.0:
            raise ValueError("Top fractions must be within (0, 100].")
        n = max(1, int(np.ceil(len(ordered) * fraction / 100.0)))
        idx = order_desc[:n]
        mask = np.zeros(len(ordered), dtype=bool)
        mask[idx] = True
        acc = _accuracy(mask, predicted, labels)
        rank_points.append(
            ConfidenceRankPoint(
                population_fraction_pct=fraction,
                observations=n,
                accuracy_pct=acc,
                baseline_accuracy_pct=baseline,
                lift_pct_points=(acc - baseline) if acc is not None else None,
                mean_confidence_pct=float(np.mean(confidence[mask])),
            )
        )

    class_results: list[ClassDiagnostic] = []
    for class_index, class_name in enumerate(OUTCOME_CLASSES):
        mask = predicted == class_index
        n = int(mask.sum())
        if n:
            actual_counts = np.bincount(labels[mask], minlength=3) / n * 100.0
            class_results.append(
                ClassDiagnostic(
                    predicted_class=class_name,
                    observations=n,
                    accuracy_pct=float(np.mean(labels[mask] == class_index) * 100.0),
                    mean_confidence_pct=float(np.mean(confidence[mask])),
                    actual_up_pct=float(actual_counts[0]),
                    actual_sideways_pct=float(actual_counts[1]),
                    actual_down_pct=float(actual_counts[2]),
                )
            )
        else:
            class_results.append(
                ClassDiagnostic(
                    predicted_class=class_name,
                    observations=0,
                    accuracy_pct=None,
                    mean_confidence_pct=None,
                    actual_up_pct=0.0,
                    actual_sideways_pct=0.0,
                    actual_down_pct=0.0,
                )
            )

    top_half = next(
        (p for p in rank_points if abs(p.population_fraction_pct - 50.0) < 1e-9),
        None,
    )
    high_confidence_is_better = bool(
        rank_corr is not None and rank_corr > 0.0 and
        top_half is not None and
        (top_half.accuracy_pct or 0.0) > baseline
    )

    return Phase62Diagnostics(
        observations=len(ordered),
        overall_accuracy_pct=float(correct.mean() * 100.0),
        baseline_majority_accuracy_pct=baseline,
        confidence_accuracy_spearman=rank_corr,
        incorrect_confidence_mean_pct=incorrect_mean,
        correct_confidence_mean_pct=correct_mean,
        confidence_gap_correct_minus_incorrect_pct_points=correct_mean - incorrect_mean,
        high_confidence_is_better=high_confidence_is_better,
        confidence_rank_curve=tuple(rank_points),
        class_diagnostics=tuple(class_results),
        protocol="PHASE_6_2_CONFIDENCE_RANK_AND_CLASS_DIAGNOSTICS",
    )
