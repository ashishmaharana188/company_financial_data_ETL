from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Literal, Sequence

from .relationship import HistoricalRelationshipObservation

OutcomeClass = Literal["UP", "SIDEWAYS", "DOWN"]
OUTCOME_CLASSES: tuple[OutcomeClass, ...] = ("UP", "SIDEWAYS", "DOWN")


@dataclass(frozen=True)
class OutcomeTrainingFilter:
    """Audit information for the point-in-time outcome-training filter."""

    target: str
    cutoff_date: date
    source_count: int
    eligible_count: int
    excluded_target: int
    excluded_future_state: int
    excluded_incomplete_outcome: int
    excluded_unknown_outcome_end: int
    excluded_missing_return: int
    excluded_non_finite_return: int

    @property
    def excluded_count(self) -> int:
        return self.source_count - self.eligible_count

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "cutoff_date": self.cutoff_date,
            "source_count": self.source_count,
            "eligible_count": self.eligible_count,
            "excluded_target": self.excluded_target,
            "excluded_future_state": self.excluded_future_state,
            "excluded_incomplete_outcome": self.excluded_incomplete_outcome,
            "excluded_unknown_outcome_end": self.excluded_unknown_outcome_end,
            "excluded_missing_return": self.excluded_missing_return,
            "excluded_non_finite_return": self.excluded_non_finite_return,
        }


@dataclass(frozen=True)
class OutcomeThresholds:
    """Empirically learned return boundaries for one holding-period distribution.

    The boundaries are distribution-relative tertiles. They deliberately do
    not claim that ``SIDEWAYS`` means a small absolute return. That economic
    interpretation belongs to a later calibration/utility layer.
    """

    lower_pct: float
    upper_pct: float
    sample_count: int
    lower_quantile: float = 1.0 / 3.0
    upper_quantile: float = 2.0 / 3.0
    method: str = "empirical_tertiles"
    limited: bool = False
    limitation: str | None = None
    invalid_count: int = 0

    def as_dict(self) -> dict:
        return {
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "sample_count": self.sample_count,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "method": self.method,
            "limited": self.limited,
            "limitation": self.limitation,
            "invalid_count": self.invalid_count,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _coerce_finite(value: object) -> float | None:
    """Return a finite numeric return, otherwise None."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile from an empty series.")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * q
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    fraction = position - left
    return float(
        sorted_values[left]
        + (sorted_values[right] - sorted_values[left]) * fraction
    )


def filter_completed_outcomes(
    observations: Iterable[HistoricalRelationshipObservation],
    target: str,
    cutoff_date: str | date | datetime,
) -> tuple[list[HistoricalRelationshipObservation], OutcomeTrainingFilter]:
    """Return only labels fully knowable before ``cutoff_date``.

    This is the point-in-time gate for Phase 5.1. A row is eligible only when:

    - it belongs to the requested target;
    - its state date is strictly before the prediction cutoff;
    - its forward outcome exists and has a known end date; and
    - that outcome ended strictly before the cutoff.

    Missing or non-finite returns are excluded rather than converted to zero.
    """

    cutoff = _as_date(cutoff_date)
    rows = list(observations)
    eligible: list[HistoricalRelationshipObservation] = []

    excluded_target = 0
    excluded_future_state = 0
    excluded_incomplete = 0
    excluded_unknown_end = 0
    excluded_missing_return = 0
    excluded_non_finite = 0

    for observation in rows:
        if observation.target != target:
            excluded_target += 1
            continue

        state_date = _as_date(observation.as_of_date)
        if state_date >= cutoff:
            excluded_future_state += 1
            continue

        if observation.outcome_end_date is None:
            excluded_unknown_end += 1
            continue

        outcome_end = _as_date(observation.outcome_end_date)
        if outcome_end >= cutoff:
            excluded_incomplete += 1
            continue

        if observation.stock_return_pct is None:
            excluded_missing_return += 1
            continue

        try:
            numeric_return = float(observation.stock_return_pct)
        except (TypeError, ValueError):
            excluded_missing_return += 1
            continue

        if not isfinite(numeric_return):
            excluded_non_finite += 1
            continue

        eligible.append(observation)

    eligible.sort(key=lambda item: _as_date(item.as_of_date))

    audit = OutcomeTrainingFilter(
        target=target,
        cutoff_date=cutoff,
        source_count=len(rows),
        eligible_count=len(eligible),
        excluded_target=excluded_target,
        excluded_future_state=excluded_future_state,
        excluded_incomplete_outcome=excluded_incomplete,
        excluded_unknown_outcome_end=excluded_unknown_end,
        excluded_missing_return=excluded_missing_return,
        excluded_non_finite_return=excluded_non_finite,
    )
    return eligible, audit


def learn_outcome_thresholds(
    returns_pct: Iterable[float | int | None],
    min_observations: int = 9,
) -> OutcomeThresholds:
    """Learn UP/SIDEWAYS/DOWN boundaries from a completed return distribution.

    The supplied values are assumed to have already passed the point-in-time
    outcome filter when they originate from historical observations. Invalid
    numeric values are ignored and reported through ``invalid_count``.
    """

    if min_observations < 3:
        raise ValueError("min_observations must be at least 3.")

    values: list[float] = []
    invalid_count = 0
    for value in returns_pct:
        numeric = _coerce_finite(value)
        if numeric is None:
            invalid_count += 1
        else:
            values.append(numeric)

    values.sort()

    if len(values) < min_observations:
        return OutcomeThresholds(
            lower_pct=0.0,
            upper_pct=0.0,
            sample_count=len(values),
            limited=True,
            limitation=(
                "Insufficient completed historical forward returns to learn "
                "three-class outcome thresholds."
            ),
            invalid_count=invalid_count,
        )

    lower = _quantile(values, 1.0 / 3.0)
    upper = _quantile(values, 2.0 / 3.0)

    if lower >= upper:
        return OutcomeThresholds(
            lower_pct=lower,
            upper_pct=upper,
            sample_count=len(values),
            limited=True,
            limitation=(
                "Historical forward-return distribution does not provide "
                "distinct three-class boundaries."
            ),
            invalid_count=invalid_count,
        )

    return OutcomeThresholds(
        lower_pct=lower,
        upper_pct=upper,
        sample_count=len(values),
        invalid_count=invalid_count,
    )


def learn_thresholds_from_observations(
    observations: Iterable[HistoricalRelationshipObservation],
    target: str,
    cutoff_date: str | date | datetime,
    min_observations: int = 9,
) -> tuple[OutcomeThresholds, OutcomeTrainingFilter]:
    """Learn thresholds from observations that are fully known at cutoff time."""

    eligible, audit = filter_completed_outcomes(
        observations=observations,
        target=target,
        cutoff_date=cutoff_date,
    )
    thresholds = learn_outcome_thresholds(
        [row.stock_return_pct for row in eligible],
        min_observations=min_observations,
    )
    return thresholds, audit


def classify_return(
    return_pct: float,
    thresholds: OutcomeThresholds,
) -> OutcomeClass:
    if thresholds.limited:
        raise ValueError(
            "Cannot classify returns with limited outcome thresholds."
        )

    numeric_return = _coerce_finite(return_pct)
    if numeric_return is None:
        raise ValueError("return_pct must be a finite numeric value.")

    if numeric_return < thresholds.lower_pct:
        return "DOWN"
    if numeric_return > thresholds.upper_pct:
        return "UP"
    return "SIDEWAYS"


def classify_returns(
    returns_pct: Iterable[float],
    thresholds: OutcomeThresholds,
) -> list[OutcomeClass]:
    return [classify_return(float(value), thresholds) for value in returns_pct]


def class_counts(
    returns_pct: Iterable[float],
    thresholds: OutcomeThresholds,
) -> dict[OutcomeClass, int]:
    """Return deterministic counts for audit/reporting of the label split."""

    counts: dict[OutcomeClass, int] = {
        "UP": 0,
        "SIDEWAYS": 0,
        "DOWN": 0,
    }
    for label in classify_returns(returns_pct, thresholds):
        counts[label] += 1
    return counts
