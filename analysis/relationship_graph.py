from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Iterable

# Structural families are intentionally narrower than feature paths.
# company.market.price           -> company.market
# institutional.FII_FPI.x        -> institutional
# derivatives.options.oi_pcr     -> derivatives
# microstructure.delivery_pct    -> microstructure
# trade_events.net_event_volume  -> trade_events

ROOT_FAMILIES = frozenset({
    "institutional",
    "derivatives",
    "microstructure",
    "trade_events",
    "macro",
    "global",
    "benchmark",
})

TWO_LEVEL_ROOTS = frozenset({
    "company",
    "industry",
    "sector",
})


# Structural relationship graph. These edges say only that the two domains may
# be tested together. They do NOT encode economic direction or predictive value.
ALLOWED_RELATIONSHIP_FAMILIES = {
    frozenset({"company.financials", "company.market"}),
    frozenset({"company.financials", "industry.market"}),
    frozenset({"industry.financials", "industry.market"}),
    frozenset({"industry.market", "sector.market"}),
    frozenset({"industry.market", "macro"}),
    frozenset({"industry.market", "global"}),
    frozenset({"sector.market", "benchmark"}),
    frozenset({"sector.financials", "sector.market"}),
    frozenset({"sector.market", "macro"}),
    frozenset({"sector.market", "global"}),
    # Flow domains.
    frozenset({"institutional", "company.market"}),
    frozenset({"institutional", "industry.market"}),
    frozenset({"institutional", "sector.market"}),
    frozenset({"institutional", "macro"}),
    frozenset({"institutional", "global"}),
    frozenset({"institutional", "benchmark"}),
    frozenset({"institutional", "derivatives"}),
    frozenset({"institutional", "microstructure"}),
    frozenset({"institutional", "trade_events"}),
    frozenset({"derivatives", "company.market"}),
    frozenset({"derivatives", "industry.market"}),
    frozenset({"derivatives", "sector.market"}),
    frozenset({"derivatives", "macro"}),
    frozenset({"derivatives", "global"}),
    frozenset({"derivatives", "benchmark"}),
    frozenset({"derivatives", "microstructure"}),
    frozenset({"derivatives", "trade_events"}),
    frozenset({"microstructure", "company.market"}),
    frozenset({"microstructure", "industry.market"}),
    frozenset({"microstructure", "sector.market"}),
    frozenset({"microstructure", "macro"}),
    frozenset({"microstructure", "global"}),
    frozenset({"microstructure", "benchmark"}),
    frozenset({"microstructure", "trade_events"}),
    frozenset({"trade_events", "company.market"}),
    frozenset({"trade_events", "industry.market"}),
    frozenset({"trade_events", "sector.market"}),
    frozenset({"trade_events", "macro"}),
    frozenset({"trade_events", "global"}),
    frozenset({"trade_events", "benchmark"}),
}

# Exact source-preserved duplicates from mv_unified_market_matrix versus the
# dedicated derivative views. These should never be counted as two independent
# pieces of evidence inside the same candidate.
_FEATURE_LINEAGE_OVERRIDES = {
    "derivatives.options.oi_pcr": "options.oi_pcr",
    "microstructure.oi_pcr": "options.oi_pcr",
    "derivatives.futures.basis_percentage": "futures.basis_percentage",
    "microstructure.futures_basis": "futures.basis_percentage",
}


def family_for_feature(feature: str) -> str:
    """Resolve a feature path to its canonical structural family."""
    parts = [part for part in str(feature).split(".") if part]
    if not parts:
        return ""

    root = parts[0]
    if root in TWO_LEVEL_ROOTS and len(parts) >= 2:
        return f"{root}.{parts[1]}"
    if root in ROOT_FAMILIES:
        return root
    if len(parts) >= 2:
        # Preserve the historical two-level convention for unknown domains.
        return f"{root}.{parts[1]}"
    return root


def feature_lineage(feature: str) -> str:
    """Return the canonical information lineage for duplicate-data control."""
    feature = str(feature)
    return _FEATURE_LINEAGE_OVERRIDES.get(feature, feature)


def lineages_are_distinct(features: Iterable[str]) -> bool:
    lineages = [feature_lineage(feature) for feature in features]
    return len(lineages) == len(set(lineages))


def compatible_pair(left: str, right: str) -> bool:
    left_family = family_for_feature(left)
    right_family = family_for_feature(right)

    if not left_family or not right_family or left_family == right_family:
        return False
    if not lineages_are_distinct((left, right)):
        return False

    return frozenset({left_family, right_family}) in ALLOWED_RELATIONSHIP_FAMILIES


def structurally_connected(features: Iterable[str]) -> bool:
    """Return True when feature families form a connected structural graph."""
    features = tuple(features)
    if not features:
        return False

    families = {family_for_feature(feature) for feature in features}
    if "" in families:
        return False
    if len(families) != len(features):
        return False
    if not lineages_are_distinct(features):
        return False
    if len(families) <= 1:
        return False

    remaining = set(families)
    visited = {next(iter(remaining))}

    changed = True
    while changed:
        changed = False
        for left in tuple(visited):
            for right in tuple(remaining - visited):
                if frozenset({left, right}) in ALLOWED_RELATIONSHIP_FAMILIES:
                    visited.add(right)
                    changed = True

    return visited == families


@lru_cache(maxsize=128)
def _candidate_feature_sets_cached(features: tuple[str, ...], max_order: int) -> tuple[tuple[str, ...], ...]:
    if max_order < 1:
        raise ValueError("max_order must be >= 1")

    candidates: list[tuple[str, ...]] = [(feature,) for feature in features]
    if max_order >= 2:
        for left, right in combinations(features, 2):
            if compatible_pair(left, right):
                candidates.append((left, right))
    for order in range(3, max_order + 1):
        for candidate in combinations(features, order):
            if structurally_connected(candidate):
                candidates.append(candidate)
    return tuple(candidates)


def candidate_feature_sets(features: Iterable[str], max_order: int) -> list[tuple[str, ...]]:
    """Generate structurally valid 1..N feature candidates.

    The immutable candidate universe is cached by feature set/order. This is a
    pure performance optimization and cannot change which candidates are valid.
    """
    normalized = tuple(sorted(set(str(feature) for feature in features)))
    return list(_candidate_feature_sets_cached(normalized, int(max_order)))
