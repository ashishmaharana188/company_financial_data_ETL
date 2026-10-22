from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Protocol

from .relationship import HistoricalRelationshipObservation


class OutcomeLike(Protocol):
    """Runtime-neutral subset of OutcomeObservation required by Phase 4.2."""

    prediction_date: date
    data_cutoff_date: date
    target: str
    entry_mode: str
    entry_date: date | None
    exit_date: date | None
    stock_return_pct: float | None
    benchmark_return_pct: float | None
    relative_return_pct: float | None
    valid: bool


# ============================================================
# PHASE 4.2 — HISTORICAL STATE / OUTCOME PANEL
# ============================================================


@dataclass(frozen=True)
class HistoricalStateSnapshot:
    """One time-bounded descriptive Phase 3 state snapshot."""

    as_of_date: date
    data_cutoff_date: date
    target: str
    scope: str
    states: dict[str, str]

    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "data_cutoff_date": self.data_cutoff_date,
            "target": self.target,
            "scope": self.scope,
            "states": dict(self.states),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class PanelBuildStats:
    """Audit counts for one panel build."""

    state_snapshots_received: int
    valid_pairs: int
    skipped_missing_outcome: int
    skipped_cutoff_violation: int
    skipped_identity_mismatch: int
    skipped_contract_violation: int

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_missing_outcome
            + self.skipped_cutoff_violation
            + self.skipped_identity_mismatch
            + self.skipped_contract_violation
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "state_snapshots_received": self.state_snapshots_received,
            "valid_pairs": self.valid_pairs,
            "skipped_missing_outcome": self.skipped_missing_outcome,
            "skipped_cutoff_violation": self.skipped_cutoff_violation,
            "skipped_identity_mismatch": self.skipped_identity_mismatch,
            "skipped_contract_violation": self.skipped_contract_violation,
        }


@dataclass(frozen=True)
class HistoricalStateOutcomePanel:
    """
    Historical Phase 3 state snapshots paired with raw Phase 4 outcomes.

    Every emitted relationship observation satisfies the temporal contract:

        state information <= prediction cutoff < realized outcome horizon

    UP/SIDEWAYS/DOWN labels are deliberately absent here.
    """

    observations: tuple[HistoricalRelationshipObservation, ...]
    stats: PanelBuildStats
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": [
                {
                    "as_of_date": item.as_of_date,
                    "target": item.target,
                    "scope": item.scope,
                    "states": dict(item.states),
                    "stock_return_pct": item.stock_return_pct,
                    "benchmark_return_pct": item.benchmark_return_pct,
                    "relative_return_pct": item.relative_return_pct,
                    "outcome_end_date": item.outcome_end_date,
                }
                for item in self.observations
            ],
            "stats": self.stats.as_dict(),
            "limitations": list(self.limitations),
        }


# ============================================================
# NORMALIZATION / VALIDATION
# ============================================================


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_states(
    states: dict[str, Any],
) -> dict[str, str]:
    """Keep only categorical state leaves accepted by Phase 4."""

    normalized: dict[str, str] = {}

    for feature, value in states.items():
        if not isinstance(feature, str) or not feature.strip():
            continue

        if value is None:
            continue

        if not isinstance(value, str):
            raise ValueError(
                f"State '{feature}' must be categorical text; "
                f"got {type(value).__name__}."
            )

        state = value.strip()
        if not state:
            continue

        normalized[feature.strip()] = state

    return normalized


def _entry_contract_ok(
    snapshot: HistoricalStateSnapshot,
    outcome: OutcomeLike,
) -> bool:
    """Validate the entry date against the state information boundary."""

    if outcome.entry_date is None:
        return False

    if outcome.entry_mode == "next_trading_day":
        # Entry may occur after the prediction information cutoff.
        return outcome.entry_date > snapshot.data_cutoff_date

    if outcome.entry_mode == "latest_known_data":
        # Entry must be known by the cutoff used to build the state.
        return outcome.entry_date <= snapshot.data_cutoff_date

    return False


def validate_state_outcome_pair(
    snapshot: HistoricalStateSnapshot,
    outcome: OutcomeLike,
) -> list[str]:
    """
    Return contract violations for one state/outcome pair.

    No silent repair is performed. Historical backtesting is exactly the
    place where quietly 'fixing' dates would manufacture look-ahead bias.
    """

    violations: list[str] = []

    if snapshot.as_of_date != outcome.prediction_date:
        violations.append(
            "State and outcome prediction dates do not match."
        )

    if snapshot.data_cutoff_date != outcome.data_cutoff_date:
        violations.append(
            "State and outcome data cutoffs do not match."
        )

    if snapshot.target != outcome.target:
        violations.append(
            "State and outcome targets do not match."
        )

    if not snapshot.states:
        violations.append(
            "State snapshot contains no usable categorical states."
        )

    if not outcome.valid or outcome.stock_return_pct is None:
        violations.append(
            "Outcome is not a valid realized stock return."
        )

    if snapshot.data_cutoff_date > snapshot.as_of_date:
        violations.append(
            "State data cutoff occurs after the state as-of date."
        )

    if not _entry_contract_ok(snapshot, outcome):
        violations.append(
            "Outcome entry date violates the entry information boundary."
        )

    if outcome.exit_date is None or outcome.entry_date is None:
        violations.append(
            "Outcome is missing an entry or exit date."
        )
    elif outcome.exit_date <= outcome.entry_date:
        violations.append(
            "Outcome exit date must be after entry date."
        )

    return violations


# ============================================================
# PANEL BUILDER
# ============================================================


def build_historical_state_outcome_panel(
    snapshots: Iterable[HistoricalStateSnapshot],
    outcomes: Iterable[OutcomeLike],
    cutoff_date: str | date | datetime | None = None,
    strict: bool = True,
) -> HistoricalStateOutcomePanel:
    """
    Pair Phase 3 historical states with realized Phase 4 outcomes.

    Matching key:

        (target, prediction_date)

    Rules:

    - only pre-cutoff prediction snapshots are eligible
    - exactly one outcome must exist for each eligible snapshot
    - invalid or temporally inconsistent outcomes are skipped
    - no missing values are converted into synthetic states
    - raw absolute, benchmark, and relative returns are preserved

    When strict=True, contract violations raise ValueError instead of being
    admitted into the relationship dataset.
    """

    snapshot_list = list(snapshots)
    outcome_map: dict[tuple[str, date], OutcomeLike] = {}

    for outcome in outcomes:
        key = (
            outcome.target,
            _as_date(outcome.prediction_date),
        )

        if key in outcome_map:
            raise ValueError(
                "Duplicate outcome for target/date: "
                f"{key}."
            )

        outcome_map[key] = outcome

    cutoff = (
        _as_date(cutoff_date)
        if cutoff_date is not None
        else None
    )

    observations: list[HistoricalRelationshipObservation] = []
    limitations: list[str] = []

    skipped_missing_outcome = 0
    skipped_cutoff_violation = 0
    skipped_identity_mismatch = 0
    skipped_contract_violation = 0

    seen_snapshot_keys: set[tuple[str, date]] = set()

    for raw_snapshot in snapshot_list:
        snapshot = HistoricalStateSnapshot(
            as_of_date=_as_date(raw_snapshot.as_of_date),
            data_cutoff_date=_as_date(raw_snapshot.data_cutoff_date),
            target=str(raw_snapshot.target),
            scope=str(raw_snapshot.scope),
            states=_normalize_states(raw_snapshot.states),
            limitations=tuple(raw_snapshot.limitations),
        )

        key = (snapshot.target, snapshot.as_of_date)

        if key in seen_snapshot_keys:
            raise ValueError(
                "Duplicate state snapshot for target/date: "
                f"{key}."
            )
        seen_snapshot_keys.add(key)

        if cutoff is not None and snapshot.as_of_date >= cutoff:
            skipped_cutoff_violation += 1
            continue

        outcome = outcome_map.get(key)

        if outcome is None:
            skipped_missing_outcome += 1
            limitations.append(
                f"Missing outcome for {snapshot.target} "
                f"on {snapshot.as_of_date}."
            )
            continue

        violations = validate_state_outcome_pair(
            snapshot,
            outcome,
        )

        identity_violations = {
            "State and outcome prediction dates do not match.",
            "State and outcome targets do not match.",
        }
        if any(message in identity_violations for message in violations):
            skipped_identity_mismatch += 1

        if violations:
            message = (
                f"Invalid state/outcome pair for {key}: "
                + " | ".join(violations)
            )

            if strict:
                raise ValueError(message)

            skipped_contract_violation += 1
            limitations.append(message)
            continue

        observations.append(
            HistoricalRelationshipObservation(
                as_of_date=snapshot.as_of_date,
                target=snapshot.target,
                scope=snapshot.scope,
                states=dict(snapshot.states),
                stock_return_pct=float(outcome.stock_return_pct),
                benchmark_return_pct=(
                    None
                    if outcome.benchmark_return_pct is None
                    else float(outcome.benchmark_return_pct)
                ),
                relative_return_pct=(
                    None
                    if outcome.relative_return_pct is None
                    else float(outcome.relative_return_pct)
                ),
                outcome_end_date=(
                    None
                    if outcome.exit_date is None
                    else _as_date(outcome.exit_date)
                ),
            )
        )

    observations.sort(
        key=lambda item: (
            item.as_of_date,
            item.target,
        )
    )

    # Snapshot-level limitations are inherited as audit information,
    # but they do not automatically invalidate a usable market outcome.
    for snapshot in snapshot_list:
        limitations.extend(snapshot.limitations)

    # Preserve order while removing repeated messages.
    limitations = list(dict.fromkeys(limitations))

    stats = PanelBuildStats(
        state_snapshots_received=len(snapshot_list),
        valid_pairs=len(observations),
        skipped_missing_outcome=skipped_missing_outcome,
        skipped_cutoff_violation=skipped_cutoff_violation,
        skipped_identity_mismatch=skipped_identity_mismatch,
        skipped_contract_violation=skipped_contract_violation,
    )

    return HistoricalStateOutcomePanel(
        observations=tuple(observations),
        stats=stats,
        limitations=tuple(limitations),
    )
