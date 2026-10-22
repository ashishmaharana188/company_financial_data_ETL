from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, log
from typing import Any, Iterable, Sequence

import numpy as np

from .outcome_labels import OUTCOME_CLASSES, OutcomeClass

_EPS = 1e-12


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalise(values: dict[OutcomeClass, float] | Sequence[float]) -> np.ndarray:
    if isinstance(values, dict):
        vector = np.asarray([float(values[label]) for label in OUTCOME_CLASSES], dtype=np.float64)
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
    probabilities = np.clip(probabilities, _EPS, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(probabilities[np.arange(len(labels)), labels])))


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(labels) == 0:
        return 0.0
    confidence = np.max(probabilities, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if np.any(mask):
            error += (float(mask.sum()) / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error * 100.0


@dataclass(frozen=True)
class CalibrationObservation:
    prediction_date: date
    probabilities_pct: dict[OutcomeClass, float]
    actual_class: OutcomeClass

    def vector(self) -> np.ndarray:
        return _normalise(self.probabilities_pct)


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    fitted_observations: int
    fitted_log_loss: float
    status: str
    fit_start_date: date | None
    fit_end_date: date | None

    def apply(self, probabilities_pct: dict[OutcomeClass, float] | Sequence[float]) -> dict[OutcomeClass, float]:
        p = _normalise(probabilities_pct)
        logits = np.log(np.clip(p, _EPS, 1.0)) / float(self.temperature)
        logits -= np.max(logits)
        e = np.exp(logits)
        calibrated = e / e.sum()
        return {label: float(calibrated[i] * 100.0) for i, label in enumerate(OUTCOME_CLASSES)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "fitted_observations": self.fitted_observations,
            "fitted_log_loss": self.fitted_log_loss,
            "status": self.status,
            "fit_start_date": self.fit_start_date,
            "fit_end_date": self.fit_end_date,
        }


@dataclass(frozen=True)
class CalibratedPrediction:
    prediction_date: date
    raw_probabilities_pct: dict[OutcomeClass, float]
    calibrated_probabilities_pct: dict[OutcomeClass, float]
    actual_class: OutcomeClass
    temperature: float
    calibration_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_date": self.prediction_date,
            "raw_probabilities_pct": dict(self.raw_probabilities_pct),
            "calibrated_probabilities_pct": dict(self.calibrated_probabilities_pct),
            "actual_class": self.actual_class,
            "temperature": self.temperature,
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True)
class CalibrationMetrics:
    observations: int
    log_loss: float
    brier_score: float
    accuracy_pct: float
    expected_calibration_error_pct: float
    raw_log_loss: float
    raw_brier_score: float
    raw_accuracy_pct: float
    raw_expected_calibration_error_pct: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Phase6CalibrationResult:
    predictions: tuple[CalibratedPrediction, ...]
    metrics: CalibrationMetrics
    calibration_window_minimum: int
    protocol: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions": [item.as_dict() for item in self.predictions],
            "metrics": self.metrics.as_dict(),
            "calibration_window_minimum": self.calibration_window_minimum,
            "protocol": self.protocol,
        }


def fit_temperature(
    observations: Sequence[CalibrationObservation],
    *,
    min_observations: int = 30,
    temperature_min: float = 0.25,
    temperature_max: float = 4.0,
    grid_size: int = 161,
) -> TemperatureCalibration:
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1.")
    if not 0.0 < temperature_min < temperature_max:
        raise ValueError("Temperature bounds are invalid.")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3.")

    ordered = sorted(observations, key=lambda item: item.prediction_date)
    if len(ordered) < min_observations:
        return TemperatureCalibration(
            temperature=1.0,
            fitted_observations=len(ordered),
            fitted_log_loss=float("nan"),
            status="NO_CALIBRATION_DATA" if not ordered else "INSUFFICIENT_CALIBRATION_DATA",
            fit_start_date=ordered[0].prediction_date if ordered else None,
            fit_end_date=ordered[-1].prediction_date if ordered else None,
        )

    p = np.vstack([item.vector() for item in ordered])
    labels = np.asarray([OUTCOME_CLASSES.index(item.actual_class) for item in ordered], dtype=np.int64)

    best_temperature = 1.0
    best_loss = float("inf")
    for log_temperature in np.linspace(
        log(temperature_min), log(temperature_max), grid_size, dtype=np.float64
    ):
        temperature = float(np.exp(log_temperature))
        logits = np.log(np.clip(p, _EPS, 1.0)) / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits)
        calibrated = e / e.sum(axis=1, keepdims=True)
        loss = _log_loss(calibrated, labels)
        if loss < best_loss - 1e-15:
            best_loss = loss
            best_temperature = temperature

    return TemperatureCalibration(
        temperature=best_temperature,
        fitted_observations=len(ordered),
        fitted_log_loss=best_loss,
        status="FITTED",
        fit_start_date=ordered[0].prediction_date,
        fit_end_date=ordered[-1].prediction_date,
    )


def evaluate_calibration(
    observations: Sequence[CalibrationObservation],
    calibrated_predictions: Sequence[CalibratedPrediction],
) -> CalibrationMetrics:
    if not observations:
        raise ValueError("At least one observation is required.")
    if len(observations) != len(calibrated_predictions):
        raise ValueError("Observation/prediction lengths must match.")

    ordered = sorted(observations, key=lambda item: item.prediction_date)
    raw = np.vstack([item.vector() for item in ordered])
    labels = np.asarray([OUTCOME_CLASSES.index(item.actual_class) for item in ordered], dtype=np.int64)
    lookup = {item.prediction_date: item for item in calibrated_predictions}
    calibrated = np.vstack([
        _normalise(lookup[item.prediction_date].calibrated_probabilities_pct) for item in ordered
    ])

    return CalibrationMetrics(
        observations=len(ordered),
        log_loss=_log_loss(calibrated, labels),
        brier_score=_brier(calibrated, labels),
        accuracy_pct=float(np.mean(np.argmax(calibrated, axis=1) == labels) * 100.0),
        expected_calibration_error_pct=_ece(calibrated, labels),
        raw_log_loss=_log_loss(raw, labels),
        raw_brier_score=_brier(raw, labels),
        raw_accuracy_pct=float(np.mean(np.argmax(raw, axis=1) == labels) * 100.0),
        raw_expected_calibration_error_pct=_ece(raw, labels),
    )


def calibrate_phase5_folds(
    folds: Iterable[Any],
    *,
    min_calibration_observations: int = 30,
) -> Phase6CalibrationResult:
    ordered = sorted(
        [
            fold for fold in folds
            if getattr(fold, "actual_class", None) in OUTCOME_CLASSES
            and getattr(fold, "probabilities_pct", None) is not None
        ],
        key=lambda fold: _as_date(fold.prediction_date),
    )
    if not ordered:
        raise ValueError("No valid Phase 5 prediction folds were supplied.")

    history: list[CalibrationObservation] = []
    predictions: list[CalibratedPrediction] = []
    observations: list[CalibrationObservation] = []

    for fold in ordered:
        current_date = _as_date(fold.prediction_date)
        calibration = fit_temperature(
            history,
            min_observations=min_calibration_observations,
        )
        raw = {label: float(fold.probabilities_pct.get(label, 0.0)) for label in OUTCOME_CLASSES}
        calibrated = calibration.apply(raw)
        observation = CalibrationObservation(
            prediction_date=current_date,
            probabilities_pct=raw,
            actual_class=fold.actual_class,
        )
        predictions.append(
            CalibratedPrediction(
                prediction_date=current_date,
                raw_probabilities_pct=raw,
                calibrated_probabilities_pct=calibrated,
                actual_class=fold.actual_class,
                temperature=calibration.temperature,
                calibration_status=calibration.status,
            )
        )
        observations.append(observation)
        history.append(observation)

    return Phase6CalibrationResult(
        predictions=tuple(predictions),
        metrics=evaluate_calibration(observations, predictions),
        calibration_window_minimum=int(min_calibration_observations),
        protocol="PREQUENTIAL_OOS_TEMPERATURE_SCALING",
    )
