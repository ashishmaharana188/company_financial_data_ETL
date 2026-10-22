from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

from .outcome_labels import OutcomeClass, OUTCOME_CLASSES

PredictionTrend = Literal["UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"]


def _standard_error_pct(probability_pct: float, effective_n: float) -> float:
    """Approximate sampling uncertainty for an empirical class share.

    This is intentionally an uncertainty diagnostic, not a calibrated
    probability interval. Phase 6 is responsible for calibration and robust
    out-of-sample evaluation.
    """
    if effective_n <= 0:
        return 100.0
    p = max(0.0, min(1.0, probability_pct / 100.0))
    return sqrt(max(p * (1.0 - p), 0.0) / effective_n) * 100.0


def _valid_distribution(values: dict[OutcomeClass, float]) -> bool:
    if set(values) != set(OUTCOME_CLASSES):
        return False
    numbers = [float(values[label]) for label in OUTCOME_CLASSES]
    if any(not isfinite(value) or value < 0.0 for value in numbers):
        return False
    return abs(sum(numbers) - 100.0) <= 1e-6


@dataclass(frozen=True)
class DecisionResult:
    """Auditable Phase 5.4 baseline-relative decision."""

    trend: PredictionTrend
    selected_class: OutcomeClass | None
    probability_pct: float
    baseline_probability_pct: float
    lift_pct: float
    margin_pct: float
    uncertainty_pct: float
    effective_sample_size: float
    reason: str
    limited: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "trend": self.trend,
            "selected_class": self.selected_class,
            "probability_pct": self.probability_pct,
            "baseline_probability_pct": self.baseline_probability_pct,
            "lift_pct": self.lift_pct,
            "margin_pct": self.margin_pct,
            "uncertainty_pct": self.uncertainty_pct,
            "effective_sample_size": self.effective_sample_size,
            "reason": self.reason,
            "limited": self.limited,
        }


def decide_baseline_relative(
    probabilities: dict[OutcomeClass, float],
    baseline: dict[OutcomeClass, float],
    effective_sample_size: float,
) -> DecisionResult:
    """Convert an empirical class distribution into a cautious decision.

    Rules:
    1. A directional class must beat its own historical baseline share.
    2. When neither directional class does, SIDEWAYS is allowed only when
       SIDEWAYS itself beats baseline and clears the uncertainty margin.
    3. The selected class must beat the second-highest class by more than the
       approximate sampling uncertainty of the selected class.
    4. Ties, malformed distributions, and insufficient support fail closed.
    """
    if not _valid_distribution(probabilities) or not _valid_distribution(baseline):
        return DecisionResult(
            trend="NO_CLEAR_TREND",
            selected_class=None,
            probability_pct=0.0,
            baseline_probability_pct=0.0,
            lift_pct=0.0,
            margin_pct=0.0,
            uncertainty_pct=100.0,
            effective_sample_size=float(max(effective_sample_size, 0.0)),
            reason="Probability or baseline distribution is invalid.",
            limited=True,
        )

    effective_n = float(effective_sample_size)
    if not isfinite(effective_n) or effective_n <= 0.0:
        return DecisionResult(
            trend="NO_CLEAR_TREND",
            selected_class=None,
            probability_pct=0.0,
            baseline_probability_pct=0.0,
            lift_pct=0.0,
            margin_pct=0.0,
            uncertainty_pct=100.0,
            effective_sample_size=0.0,
            reason="No effective historical support is available for a decision.",
            limited=True,
        )

    lifts = {
        label: float(probabilities[label]) - float(baseline[label])
        for label in OUTCOME_CLASSES
    }

    directional_candidates = [
        label for label in ("UP", "DOWN") if lifts[label] > 0.0
    ]

    if directional_candidates:
        selected = max(directional_candidates, key=lambda label: probabilities[label])
    elif lifts["SIDEWAYS"] > 0.0:
        selected = "SIDEWAYS"
    else:
        return DecisionResult(
            trend="NO_CLEAR_TREND",
            selected_class=None,
            probability_pct=max(probabilities.values()),
            baseline_probability_pct=0.0,
            lift_pct=max(lifts.values()),
            margin_pct=0.0,
            uncertainty_pct=_standard_error_pct(max(probabilities.values()), effective_n),
            effective_sample_size=effective_n,
            reason="No outcome class beats its historical baseline share.",
        )

    selected_probability = float(probabilities[selected])
    selected_baseline = float(baseline[selected])
    selected_lift = lifts[selected]
    alternatives = [
        float(probabilities[label])
        for label in OUTCOME_CLASSES
        if label != selected
    ]
    second_probability = max(alternatives, default=0.0)
    margin = selected_probability - second_probability
    uncertainty = _standard_error_pct(selected_probability, effective_n)

    if margin <= 0.0:
        trend: PredictionTrend = "NO_CLEAR_TREND"
        reason = "The leading class is tied with another class."
    elif margin <= uncertainty:
        trend = "NO_CLEAR_TREND"
        reason = (
            "The leading class beats baseline, but its probability margin does not "
            "clear the estimated sampling uncertainty."
        )
    else:
        trend = selected  # type: ignore[assignment]
        reason = (
            f"{selected} beats its historical baseline by {selected_lift:.2f} percentage points "
            f"and clears the estimated sampling uncertainty by margin."
        )

    return DecisionResult(
        trend=trend,
        selected_class=selected,
        probability_pct=selected_probability,
        baseline_probability_pct=selected_baseline,
        lift_pct=selected_lift,
        margin_pct=margin,
        uncertainty_pct=uncertainty,
        effective_sample_size=effective_n,
        reason=reason,
    )
