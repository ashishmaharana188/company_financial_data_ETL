from __future__ import annotations

"""Shared, read-only Phase 6 probability scoring and mature-label calibration.

The evaluator uses only fold-stored forecasts and labels. A previous label is
eligible for calibrator fitting only when its forward-outcome end date is
strictly earlier than the current prediction date. This matters for rolling,
overlapping holding periods: chronological prediction order alone is not
sufficient to prevent outcome leakage.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

import numpy as np

from .outcome_labels import OUTCOME_CLASSES
from .phase6_calibration import CalibrationObservation, fit_temperature

_CLASSES = tuple(OUTCOME_CLASSES)
_EPS = 1e-15


def as_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def normalize_distribution(raw: Any) -> dict[str, float] | None:
    """Normalize a three-class map to percentages; reject invalid mass."""
    if not isinstance(raw, dict):
        return None
    try:
        values = np.asarray([float(raw.get(label, 0.0)) for label in _CLASSES], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        return None
    total = float(values.sum())
    if total <= 0.0:
        return None
    values = values / total * 100.0
    return {label: float(values[i]) for i, label in enumerate(_CLASSES)}


@dataclass(frozen=True)
class PrequentialCalibration:
    probabilities_pct: tuple[dict[str, float] | None, ...]
    statuses: tuple[str, ...]
    temperatures: tuple[float | None, ...]
    mature_training_observations: tuple[int, ...]


def calibrate_with_mature_labels(
    prediction_dates: Sequence[date | datetime | str],
    outcome_end_dates: Sequence[date | datetime | str | None],
    actual_classes: Sequence[str],
    raw_probabilities_pct: Sequence[dict[str, float] | None],
    *,
    min_calibration_observations: int = 30,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    temperature_grid_size: int = 161,
) -> PrequentialCalibration:
    """Apply point-in-time temperature scaling, respecting label maturity.

    For prediction i, calibration history contains earlier usable forecasts
    whose outcomes had completed strictly before prediction_dates[i]. The
    current fold is calibrated before its own outcome can enter history.
    Forecasts with missing outcome-end dates can be scored, but are never used
    to fit a later calibrator because their label availability is unknown.
    """
    lengths = {
        len(prediction_dates), len(outcome_end_dates), len(actual_classes),
        len(raw_probabilities_pct),
    }
    if len(lengths) != 1:
        raise ValueError("Dates, outcome-end dates, actual classes and forecasts must have equal lengths.")
    if min_calibration_observations < 1:
        raise ValueError("min_calibration_observations must be >= 1.")
    if not np.isfinite(temperature_min) or not np.isfinite(temperature_max) or not 0.0 < temperature_min < temperature_max:
        raise ValueError("Temperature bounds must satisfy 0 < temperature_min < temperature_max.")
    if temperature_grid_size < 3:
        raise ValueError("temperature_grid_size must be >= 3.")

    dates = [as_date(value) for value in prediction_dates]
    end_dates = [as_date(value) for value in outcome_end_dates]
    if any(value is None for value in dates):
        raise ValueError("Every prediction date is required.")
    typed_dates = [value for value in dates if value is not None]
    if typed_dates != sorted(typed_dates):
        raise ValueError("Input rows must be sorted by prediction date.")

    normalized = [normalize_distribution(value) for value in raw_probabilities_pct]
    calibrated: list[dict[str, float] | None] = []
    statuses: list[str] = []
    temperatures: list[float | None] = []
    training_counts: list[int] = []

    for index, prediction_date in enumerate(typed_dates):
        history: list[CalibrationObservation] = []
        for previous in range(index):
            previous_date = typed_dates[previous]
            outcome_end = end_dates[previous]
            previous_forecast = normalized[previous]
            previous_actual = actual_classes[previous]
            if previous_forecast is None or previous_actual not in _CLASSES:
                continue
            if outcome_end is None or outcome_end >= prediction_date:
                continue
            history.append(
                CalibrationObservation(
                    prediction_date=previous_date,
                    probabilities_pct=previous_forecast,
                    actual_class=previous_actual,
                )
            )

        training_counts.append(len(history))
        current = normalized[index]
        if current is None:
            calibrated.append(None)
            statuses.append("METHOD_UNAVAILABLE")
            temperatures.append(None)
            continue

        fitted = fit_temperature(
            history,
            min_observations=min_calibration_observations,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            grid_size=temperature_grid_size,
        )
        calibrated.append(fitted.apply(current))
        statuses.append(fitted.status)
        temperatures.append(float(fitted.temperature))

    return PrequentialCalibration(
        probabilities_pct=tuple(calibrated),
        statuses=tuple(statuses),
        temperatures=tuple(temperatures),
        mature_training_observations=tuple(training_counts),
    )


def summarize_temperature_boundaries(
    temperatures: Sequence[float | None],
    statuses: Sequence[str],
    *,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize fitted temperatures that land on either search boundary.

    Only rows whose calibration status is ``FITTED`` enter the denominator.
    No-calibration and insufficient-history rows are intentionally excluded.
    """
    if len(temperatures) != len(statuses):
        raise ValueError("Temperatures and calibration statuses must have equal lengths.")
    if not 0.0 < temperature_min < temperature_max:
        raise ValueError("Temperature bounds must satisfy 0 < minimum < maximum.")
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative.")

    fitted: list[float] = []
    for temperature, status in zip(temperatures, statuses):
        if status != "FITTED" or temperature is None:
            continue
        numeric = float(temperature)
        if not np.isfinite(numeric):
            continue
        fitted.append(numeric)

    lower_tolerance = max(abs(temperature_min) * relative_tolerance, 1e-12)
    upper_tolerance = max(abs(temperature_max) * relative_tolerance, 1e-12)
    lower_hits = sum(value <= temperature_min + lower_tolerance for value in fitted)
    upper_hits = sum(value >= temperature_max - upper_tolerance for value in fitted)
    count = len(fitted)

    return {
        "fitted_observations": count,
        "lower_boundary_hits": int(lower_hits),
        "upper_boundary_hits": int(upper_hits),
        "lower_boundary_hit_rate_pct": (lower_hits / count * 100.0) if count else None,
        "upper_boundary_hit_rate_pct": (upper_hits / count * 100.0) if count else None,
        "any_boundary_hits": int(lower_hits + upper_hits),
        "any_boundary_hit_rate_pct": ((lower_hits + upper_hits) / count * 100.0) if count else None,
        "temperature_mean": float(np.mean(fitted)) if count else None,
        "temperature_median": float(np.median(fitted)) if count else None,
        "temperature_minimum_fitted": float(min(fitted)) if count else None,
        "temperature_maximum_fitted": float(max(fitted)) if count else None,
        "temperature_search_min": float(temperature_min),
        "temperature_search_max": float(temperature_max),
        "boundary_status": "NO_FITTED_CALIBRATORS" if not count else "SUMMARIZED",
    }


def _matrix(
    actual_classes: Sequence[str],
    probabilities_pct: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(actual_classes) != len(probabilities_pct):
        raise ValueError("Actual-class and probability lengths must match.")
    if not actual_classes:
        raise ValueError("At least one scored prediction is required.")
    if any(value not in _CLASSES for value in actual_classes):
        raise ValueError("Actual classes must be valid UP/SIDEWAYS/DOWN labels.")
    distributions = [normalize_distribution(value) for value in probabilities_pct]
    if any(value is None for value in distributions):
        raise ValueError("Scored probability rows must contain usable distributions.")
    matrix = np.asarray(
        [[value[label] / 100.0 for label in _CLASSES] for value in distributions if value is not None],
        dtype=np.float64,
    )
    labels = np.asarray([_CLASSES.index(value) for value in actual_classes], dtype=np.int64)
    return matrix, labels


def per_observation_losses(
    actual_classes: Sequence[str],
    probabilities_pct: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row log loss, multiclass Brier loss, and correctness."""
    probabilities, labels = _matrix(actual_classes, probabilities_pct)
    selected = probabilities[np.arange(len(labels)), labels]
    log_loss = -np.log(np.clip(selected, _EPS, 1.0))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    correct = (np.argmax(probabilities, axis=1) == labels).astype(np.float64)
    return log_loss, brier, correct


def score_probabilities(
    actual_classes: Sequence[str],
    probabilities_pct: Sequence[dict[str, float]],
    *,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Return proper scores, class bias, a confusion matrix and top-label ECE."""
    if ece_bins < 1:
        raise ValueError("ece_bins must be >= 1.")
    probabilities, labels = _matrix(actual_classes, probabilities_pct)
    predicted = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correct = (predicted == labels).astype(np.float64)
    log_loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], _EPS, 1.0)))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for bin_index in range(ece_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if bin_index == ece_bins - 1 else confidences < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))

    predicted_mean = probabilities.mean(axis=0) * 100.0
    observed = np.bincount(labels, minlength=len(_CLASSES)) / len(labels) * 100.0
    confusion = {
        actual_label: {
            predicted_label: int(np.sum((labels == actual_index) & (predicted == predicted_index)))
            for predicted_index, predicted_label in enumerate(_CLASSES)
        }
        for actual_index, actual_label in enumerate(_CLASSES)
    }
    return {
        "observations": int(len(labels)),
        "accuracy_pct": float(correct.mean() * 100.0),
        "log_loss": float(log_loss),
        "brier_score": brier,
        "top_label_ece_pct": float(ece * 100.0),
        "mean_confidence_pct": float(confidences.mean() * 100.0),
        "mean_probability_pct": {label: float(predicted_mean[i]) for i, label in enumerate(_CLASSES)},
        "observed_class_frequency_pct": {label: float(observed[i]) for i, label in enumerate(_CLASSES)},
        "probability_bias_pct_points": {
            label: float(predicted_mean[i] - observed[i]) for i, label in enumerate(_CLASSES)
        },
        "predicted_class_counts": {
            label: int(np.sum(predicted == class_index)) for class_index, label in enumerate(_CLASSES)
        },
        "confusion_matrix_actual_rows_predicted_columns": confusion,
    }


def moving_block_bootstrap_ci(
    values: Sequence[float],
    *,
    block_length: int = 5,
    replicates: int = 1000,
    seed: int = 20261010,
    minimum_observations: int = 10,
) -> dict[str, Any]:
    """Circular moving-block bootstrap CI for a mean paired difference.

    Returns no interval when there are fewer than ``minimum_observations``;
    a tiny sample should not be dressed up as a stable uncertainty estimate.
    """
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    estimate = float(array.mean()) if n else None
    if n < minimum_observations:
        return {
            "observations": n,
            "estimate": estimate,
            "ci95_lower": None,
            "ci95_upper": None,
            "status": f"NOT_REPORTED_FEWER_THAN_{minimum_observations}_OBSERVATIONS",
            "block_length": int(block_length),
            "replicates": int(replicates),
        }
    if not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap values must be finite.")
    if block_length < 1 or replicates < 100:
        raise ValueError("block_length must be >= 1 and replicates must be >= 100.")

    length = min(int(block_length), n)
    blocks_per_sample = int(np.ceil(n / length))
    offsets = np.arange(length)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, n, size=blocks_per_sample)
        indices = np.concatenate([(start + offsets) % n for start in starts])[:n]
        means[replicate] = float(array[indices].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "observations": n,
        "estimate": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "status": "ESTIMATED_CIRCULAR_MOVING_BLOCK_BOOTSTRAP",
        "block_length": length,
        "replicates": int(replicates),
    }


def compare_probabilities_paired(
    actual_classes: Sequence[str],
    candidate_probabilities_pct: Sequence[dict[str, float]],
    reference_probabilities_pct: Sequence[dict[str, float]],
    *,
    block_length: int = 5,
    replicates: int = 1000,
    seed: int = 20261010,
    minimum_observations: int = 10,
) -> dict[str, Any]:
    """Paired model-vs-reference differences; positive gains favor candidate.

    Log-loss and Brier gains are reference loss minus candidate loss.
    Accuracy gain is candidate correctness minus reference correctness, in
    percentage points. Both forecasts must be evaluated on the same rows.
    """
    candidate_log, candidate_brier, candidate_correct = per_observation_losses(
        actual_classes, candidate_probabilities_pct
    )
    reference_log, reference_brier, reference_correct = per_observation_losses(
        actual_classes, reference_probabilities_pct
    )
    differences = {
        "log_loss_improvement": reference_log - candidate_log,
        "brier_score_improvement": reference_brier - candidate_brier,
        "accuracy_delta_pct_points": (candidate_correct - reference_correct) * 100.0,
    }
    return {
        name: moving_block_bootstrap_ci(
            values,
            block_length=block_length,
            replicates=replicates,
            seed=seed + index,
            minimum_observations=minimum_observations,
        )
        for index, (name, values) in enumerate(differences.items())
    }


def print_score_line(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"  {name}: n={metrics['observations']} accuracy={metrics['accuracy_pct']:.2f}% "
        f"log_loss={metrics['log_loss']:.5f} brier={metrics['brier_score']:.5f} "
        f"top-label ECE={metrics['top_label_ece_pct']:.2f}% "
        f"mean_confidence={metrics['mean_confidence_pct']:.2f}%"
    )


def print_paired_comparison(name: str, comparison: dict[str, Any]) -> None:
    print(f"  {name} (positive values favor the candidate model):")
    labels = (
        ("log_loss_improvement", "log-loss gain"),
        ("brier_score_improvement", "Brier gain"),
        ("accuracy_delta_pct_points", "accuracy delta"),
    )
    for key, label in labels:
        item = comparison[key]
        estimate = item["estimate"]
        if item["ci95_lower"] is None:
            interval = f"CI n/a ({item['status']})"
        else:
            interval = f"95% block-bootstrap CI [{item['ci95_lower']:+.5f}, {item['ci95_upper']:+.5f}]"
        print(f"    {label}: {estimate:+.5f}; {interval}; n={item['observations']}")
