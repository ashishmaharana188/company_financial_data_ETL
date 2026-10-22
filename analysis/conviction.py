from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .method_combination import CombinationResult, PredictionTrend

Conviction = Literal["STRONG", "MODERATE", "LOW", "NONE"]


@dataclass(frozen=True)
class ConvictionResult:
    """Auditable Phase 5.6 evidential-strength assessment.

    Conviction is a qualitative assessment of evidence strength. It is not a
    probability of correctness and it is not a calibrated confidence level.
    """

    conviction: Conviction
    trend: PredictionTrend
    method_count: int
    directional_method_count: int
    method_agreement: bool
    method_conflict: bool
    all_usable_methods_stable: bool
    validated_evidence: bool
    decision_cleared: bool
    combined_lift_pct: float
    combined_margin_pct: float
    effective_sample_size: float
    reason: str
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "conviction": self.conviction,
            "trend": self.trend,
            "method_count": self.method_count,
            "directional_method_count": self.directional_method_count,
            "method_agreement": self.method_agreement,
            "method_conflict": self.method_conflict,
            "all_usable_methods_stable": self.all_usable_methods_stable,
            "validated_evidence": self.validated_evidence,
            "decision_cleared": self.decision_cleared,
            "combined_lift_pct": self.combined_lift_pct,
            "combined_margin_pct": self.combined_margin_pct,
            "effective_sample_size": self.effective_sample_size,
            "reason": self.reason,
            "limitations": list(self.limitations),
        }



def combination_direction(combination: CombinationResult, fallback: PredictionTrend) -> PredictionTrend:
    """Return the externally supplied final combination direction.

    Phase 5.5 itself stores method evidence, not the final baseline-relative
    decision. The prediction layer supplies the Phase 5.4 result separately.
    """
    if fallback in {"UP", "DOWN", "SIDEWAYS", "NO_CLEAR_TREND"}:
        return fallback
    return "NO_CLEAR_TREND"


def assess_conviction(
    *,
    trend: PredictionTrend,
    combination: CombinationResult,
    decision: object,
    validated_evidence: bool = False,
) -> ConvictionResult:
    """Assign Phase 5.6 conviction using explicit evidence gates.

    Rules are intentionally conservative:

    - NO_CLEAR_TREND, limited combinations, or directional method conflict =>
      NONE.
    - SIDEWAYS => LOW because it is a neutral outcome class rather than a
      directional conviction.
    - A single usable directional method => at most MODERATE.
    - Two agreeing directional methods => MODERATE unless every usable method
      is stable and validated evidence is explicitly supplied.
    - STRONG therefore requires consensus + stability + explicit validation.

    ``validated_evidence`` is an externally supplied audit flag. This function
    never treats it as a calibrated probability guarantee. Phase 6 remains
    responsible for final prediction backtesting and probability calibration.
    """
    usable = tuple(combination.usable_methods)
    directional_count = len(combination.directional_methods)
    all_stable = bool(usable) and len(combination.stable_methods) == len(usable)
    decision_trend = getattr(decision, "trend", "NO_CLEAR_TREND")
    decision_limited = bool(getattr(decision, "limited", True))
    decision_lift = float(getattr(decision, "lift_pct", 0.0) or 0.0)
    decision_margin = float(getattr(decision, "margin_pct", 0.0) or 0.0)
    effective_n = float(getattr(decision, "effective_sample_size", combination.effective_sample_size_sum) or 0.0)

    if combination.limited or decision_limited or not usable:
        return ConvictionResult(
            conviction="NONE",
            trend="NO_CLEAR_TREND",
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=combination.method_agreement,
            method_conflict=combination.method_conflict,
            all_usable_methods_stable=all_stable,
            validated_evidence=validated_evidence,
            decision_cleared=False,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="Prediction evidence is limited or the Phase 5.4 decision did not clear its gates.",
        )

    if combination.method_conflict:
        return ConvictionResult(
            conviction="NONE",
            trend="NO_CLEAR_TREND",
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=False,
            method_conflict=True,
            all_usable_methods_stable=all_stable,
            validated_evidence=validated_evidence,
            decision_cleared=False,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="Method A and Method B provide conflicting directional evidence; conviction is suppressed.",
        )

    if trend not in {"UP", "DOWN", "SIDEWAYS"} or decision_trend != trend:
        return ConvictionResult(
            conviction="NONE",
            trend="NO_CLEAR_TREND",
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=combination.method_agreement,
            method_conflict=combination.method_conflict,
            all_usable_methods_stable=all_stable,
            validated_evidence=validated_evidence,
            decision_cleared=False,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="Final trend does not agree with the cleared Phase 5.4 decision.",
        )

    if trend == "SIDEWAYS":
        return ConvictionResult(
            conviction="LOW",
            trend=trend,
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=combination.method_agreement,
            method_conflict=False,
            all_usable_methods_stable=all_stable,
            validated_evidence=validated_evidence,
            decision_cleared=True,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="SIDEWAYS is a neutral class; the decision is retained with LOW conviction rather than treating neutrality as directional strength.",
        )

    # Directional result from here onward.
    if directional_count == 1:
        return ConvictionResult(
            conviction="MODERATE",
            trend=trend,
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=False,
            method_conflict=False,
            all_usable_methods_stable=all_stable,
            validated_evidence=validated_evidence,
            decision_cleared=True,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="A single method provides the directional evidence; conviction is capped at MODERATE.",
        )

    if combination.method_agreement and all_stable and validated_evidence:
        return ConvictionResult(
            conviction="STRONG",
            trend=trend,
            method_count=len(usable),
            directional_method_count=directional_count,
            method_agreement=True,
            method_conflict=False,
            all_usable_methods_stable=True,
            validated_evidence=True,
            decision_cleared=True,
            combined_lift_pct=decision_lift,
            combined_margin_pct=decision_margin,
            effective_sample_size=effective_n,
            reason="Both methods agree directionally, all usable methods are stable, and explicit validated evidence is available.",
        )

    return ConvictionResult(
        conviction="MODERATE",
        trend=trend,
        method_count=len(usable),
        directional_method_count=directional_count,
        method_agreement=combination.method_agreement,
        method_conflict=False,
        all_usable_methods_stable=all_stable,
        validated_evidence=validated_evidence,
        decision_cleared=True,
        combined_lift_pct=decision_lift,
        combined_margin_pct=decision_margin,
        effective_sample_size=effective_n,
        reason=(
            "Directional evidence cleared the Phase 5.4 decision, but the full "
            "consensus + stability + validation gates required for STRONG conviction are not all satisfied."
        ),
    )
