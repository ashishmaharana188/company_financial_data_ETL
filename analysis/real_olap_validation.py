from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any

import pyarrow as pa
import logging
import time
from dateutil.relativedelta import relativedelta

from .history_panel import (
    HistoricalStateSnapshot,
    build_historical_state_outcome_panel,
)
from .performance import DailyHistoryCache, patch_group_state_history_loader
from .relationship import (
    HistoricalRelationshipObservation,
    RelationshipDiscoveryEngine,
)


TIMEFRAME_MONTHS: dict[str, float] = {
    "1W": 7 / 30.4375,
    "2W": 14 / 30.4375,
    "1M": 1.0,
    "3M": 3.0,
    "6M": 6.0,
    "9M": 9.0,
    "1Y": 12.0,
    "18M": 18.0,
    "2Y": 24.0,
    "3Y": 36.0,
    "5Y": 60.0,
}


# Phase 4 covers every daily evidence family in the intended initial TrendAnalysis
# surface. Financial families remain timing-limited because OLAP ReportDate is
# an accounting period end rather than a verified information-availability date.
EXPECTED_PHASE4_STATE_FAMILIES = (
    "company.market",
    "company.financials",
    "industry.market",
    "industry.financials",
    "sector.market",
    "sector.financials",
    "macro",
    "global",
    "benchmark",
    "institutional",
    "derivatives",
    "microstructure",
    "trade_events",
)
FINANCIAL_TIMING_LIMITED_FAMILIES = (
    "company.financials",
    "industry.financials",
    "sector.financials",
)

# Domain classification taken from the existing upstream ETL fetch taxonomy.
# These are global-context series, not company/industry relationship mappings.
# They remain in the global family even though they physically live in
# macro_daily_ledger in the current OLAP schema.
UPSTREAM_GLOBAL_CONTEXT_SERIES = frozenset({
    "US_10Y_Yield",
    "Brent_Crude",
    "USD_INR",
    "US_Dollar_Index",
    "Broad_Commodity",
    "US_VIX",
})

TIME_SAFE_PHASE4_STATE_FAMILIES = tuple(
    family
    for family in EXPECTED_PHASE4_STATE_FAMILIES
    if family not in FINANCIAL_TIMING_LIMITED_FAMILIES
)

FLOW_STATE_FAMILIES = (
    "institutional",
    "derivatives",
    "microstructure",
    "trade_events",
)


@dataclass(frozen=True)
class RealOLAPValidationConfig:
    ticker: str = "RELIANCE"
    benchmark: str = "Nifty_50"
    analysis_timeframe: str = "6M"
    holding_period_months: float = 1.0
    entry_mode: str = "next_trading_day"
    max_folds: int = 250
    step_trading_days: int = 5
    min_training_observations: int = 12
    hardened_validation: bool = True
    performance_cache: bool = True
    progress_logging: bool = True
    progress_every: int = 10
    adaptive_relationship_search: bool = False
    selection_fraction: float = 0.30
    multiple_testing_alpha: float = 0.10
    relationship_progress_every_candidates: int = 1000
    relationship_candidate_batch_size: int = 512
    phase5_parallel_workers: int = 1


@dataclass(frozen=True)
class RealOLAPValidationResult:
    ticker: str
    benchmark: str
    analysis_timeframe: str
    holding_period_months: float
    entry_mode: str
    panel_rows: int
    skipped_no_outcome: int
    skipped_no_state: int
    walk_forward_folds: int
    method_a_hit_rate_pct: float
    method_b_hit_rate_pct: float
    combined_hit_rate_pct: float
    leakage_violations: int
    state_future_violations: int
    outcome_temporal_violations: int
    latest_market_date: date | None
    first_prediction_date: date | None
    state_surface_coverage_pct: float = 0.0
    time_safe_state_surface_coverage_pct: float = 0.0
    missing_state_families: tuple[str, ...] = ()
    timing_limited_state_families: tuple[str, ...] = ()
    broad_relationship_surface_validated: bool = False
    time_safe_relationship_surface_validated: bool = False
    purged_training_observations: int = 0
    unknown_overlap_observations: int = 0
    selection_candidate_evaluations: int = 0
    selection_validated_predictions: int = 0
    multiple_testing_controlled_folds: int = 0
    method_a_directional_predictions: int = 0
    method_b_directional_predictions: int = 0
    combined_directional_predictions: int = 0
    candidate_observations: int = 0
    valid_state_observations: int = 0
    financial_timing_limited_observations: int = 0
    industry_constituents: int = 0
    sector_constituents: int = 0
    macro_series: int = 0
    global_series: int = 0
    institutional_rows: int = 0
    options_rows: int = 0
    basis_rows: int = 0
    trade_event_rows: int = 0
    microstructure_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_LOGGER = logging.getLogger("trendanalysis.phase4.real_olap")


def _configure_progress_logging(enabled: bool) -> None:
    if not enabled:
        return
    if not _LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[Phase 4] %(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def _progress(enabled: bool, message: str, *args: Any) -> None:
    if enabled:
        _LOGGER.info(message, *args)


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _table_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        report_date = row.get("report_date")
        close = row.get("close")
        if report_date is None or close is None:
            continue
        try:
            numeric_close = float(close)
        except (TypeError, ValueError):
            continue
        if not isfinite(numeric_close) or numeric_close <= 0:
            continue
        normalized = dict(row)
        normalized["report_date"] = _as_date(report_date)
        normalized["close"] = numeric_close
        cleaned.append(normalized)
    cleaned.sort(key=lambda row: row["report_date"])
    return cleaned


def _subset_table(
    rows: list[dict[str, Any]],
    start_date: date,
    end_date: date,
) -> pa.Table:
    selected = [
        row
        for row in rows
        if start_date <= row["report_date"] <= end_date
    ]
    if not selected:
        return pa.table({
            "report_date": pa.array([], type=pa.date32()),
            "close": pa.array([], type=pa.float64()),
            "volume": pa.array([], type=pa.float64()),
        })
    return pa.Table.from_pylist(selected)


def _close_on_or_before(
    rows: list[dict[str, Any]],
    target_date: date,
) -> tuple[date, float] | None:
    for row in reversed(rows):
        if row["report_date"] <= target_date:
            return row["report_date"], float(row["close"])
    return None


def _close_after(
    rows: list[dict[str, Any]],
    target_date: date,
) -> tuple[date, float] | None:
    for row in rows:
        if row["report_date"] > target_date:
            return row["report_date"], float(row["close"])
    return None


def _state_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        label = value.get("state")
        return label if isinstance(label, str) else None
    label = getattr(value, "state", None)
    return label if isinstance(label, str) else None


def _metric_state_atoms(
    family: str,
    state_objects: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for metric, state_object in state_objects.items():
        label = _state_label(state_object)
        if label is None or not label.strip() or label.startswith("Unknown"):
            continue
        result[f"{family}.{metric}"] = label
    return result


def _build_company_market_atoms(
    ticker: str,
    selected_table: pa.Table,
    historical_table: pa.Table,
) -> dict[str, str]:
    """Delegate company market-state construction to Phase 3.5."""
    from analysis.market_state import build_company_market_states

    states = build_company_market_states(
        ticker=ticker,
        selected_table=selected_table,
        historical_table=historical_table,
    )
    return _metric_state_atoms("company.market", states)


def _build_generic_market_atoms(
    family: str,
    identifier: str,
    selected_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Use the exact Phase 3.5 market-state engine for contextual market data."""
    from analysis.market_state import build_company_market_states
    from data_access.ticker import normalize_ticker

    if not historical_rows:
        return {}

    # Context instruments (benchmark/global/group series) may use mixed-case
    # source identifiers such as ``Nifty_50``.  The Phase 3 market-state
    # engine validates against its canonical normalized ticker, so the adapter
    # must canonicalize both the ticker argument and the synthetic ticker
    # column consistently.  This is an adapter concern, not a change to the
    # underlying OLAP values.
    canonical_identifier = normalize_ticker(identifier)[0]

    def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "ticker": canonical_identifier,
                "report_date": row["report_date"],
                "asset_class": row.get("asset_class", "context"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            }
            for row in rows
        ]

    selected_canonical = canonical_rows(selected_rows)
    historical_canonical = canonical_rows(historical_rows)

    selected_table = pa.Table.from_pylist(selected_canonical)
    historical_table = pa.Table.from_pylist(historical_canonical)

    states = build_company_market_states(
        ticker=canonical_identifier,
        selected_table=selected_table,
        historical_table=historical_table,
    )
    return _metric_state_atoms(family, states)


def _build_macro_atoms(
    macro_histories: dict[str, list[dict[str, Any]]],
    analysis_start: date,
    prediction_date: date,
) -> dict[str, str]:
    from analysis.state import build_variable_state

    atoms: dict[str, str] = {}
    for indicator, rows in macro_histories.items():
        selected_rows = [
            row for row in rows
            if analysis_start <= row["report_date"] <= prediction_date
        ]
        historical_rows = [
            row for row in rows
            if row["report_date"] <= prediction_date
        ]
        if not historical_rows:
            continue
        selected_table = pa.Table.from_pylist(selected_rows) if selected_rows else pa.table({
            "report_date": pa.array([], type=pa.date32()),
            "close": pa.array([], type=pa.float64()),
        })
        historical_table = pa.Table.from_pylist(historical_rows)
        state = build_variable_state(
            variable=indicator,
            selected_table=selected_table,
            historical_table=historical_table,
            value_column="close",
        )
        label = _state_label(state)
        if label and not label.startswith("Unknown"):
            atoms[f"macro.{indicator}"] = label
    return atoms


def _build_global_atoms(
    global_histories: dict[str, list[dict[str, Any]]],
    analysis_start: date,
    prediction_date: date,
) -> dict[str, str]:
    from analysis.state import build_variable_state

    atoms: dict[str, str] = {}
    for ticker, rows in global_histories.items():
        selected_rows = [
            row for row in rows
            if analysis_start <= row["report_date"] <= prediction_date
        ]
        historical_rows = [
            row for row in rows
            if row["report_date"] <= prediction_date
        ]
        if not historical_rows:
            continue
        selected_table = pa.Table.from_pylist(selected_rows) if selected_rows else pa.table({
            "report_date": pa.array([], type=pa.date32()),
            "close": pa.array([], type=pa.float64()),
        })
        historical_table = pa.Table.from_pylist(historical_rows)
        state = build_variable_state(
            variable=ticker,
            selected_table=selected_table,
            historical_table=historical_table,
            value_column="close",
        )
        label = _state_label(state)
        if label and not label.startswith("Unknown"):
            atoms[f"global.{ticker}"] = label
    return atoms


def _group_financial_majority_atoms(
    family: str,
    financial_states: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Convert the Phase 3 group financial aggregation into descriptive majority
    state atoms. This helper is used for current-surface inspection only; the
    historical prediction panel deliberately excludes these atoms because
    financial publication timing is not known.
    """
    if not financial_states:
        return {}

    result: dict[str, str] = {}
    metrics = financial_states.get("metrics", {})
    for metric, payload in metrics.items():
        counts = payload.get("state_counts") if isinstance(payload, dict) else None
        if not isinstance(counts, dict) or not counts:
            continue
        ranked = sorted(
            ((str(state), int(count)) for state, count in counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        top_state, top_count = ranked[0]
        tied = [state for state, count in ranked if count == top_count]
        if len(tied) != 1:
            continue
        result[f"{family}.{metric}"] = top_state
    return result


def _resolve_benchmark_history(
    benchmark: str,
    start_date: date,
    end_date: date,
) -> pa.Table:
    """Load the benchmark through Phase 2 repositories, never raw SQL."""
    try:
        from data_access.macro_global import get_macro_date_range, get_macro_history

        if get_macro_date_range(benchmark) is not None:
            return get_macro_history(
                benchmark,
                start_date=start_date,
                end_date=end_date,
            )
    except ImportError:
        pass

    from data_access.market import get_daily_history

    return get_daily_history(
        benchmark,
        start_date=start_date,
        end_date=end_date,
    )


def _load_all_macro_histories(
    start_date: date,
    end_date: date,
    benchmark: str,
    progress_logging: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    from data_access.macro_global import get_macro_history, get_macro_indicators

    result: dict[str, list[dict[str, Any]]] = {}
    indicators = get_macro_indicators()
    _progress(progress_logging, "Macro load start | indicators=%d", len(indicators))
    for index, indicator in enumerate(indicators, start=1):
        # The selected benchmark already gets its own benchmark family.
        if indicator == benchmark:
            continue
        # Global-context series are loaded separately into the global family.
        if indicator in UPSTREAM_GLOBAL_CONTEXT_SERIES:
            continue
        rows = _table_rows(
            get_macro_history(
                indicator,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if rows:
            result[indicator] = rows
        if index == 1 or index % 10 == 0 or index == len(indicators):
            _progress(progress_logging, "Macro load progress %d/%d | usable=%d", index, len(indicators), len(result))
    return result


def _load_global_histories(
    start_date: date,
    end_date: date,
    benchmark: str,
    target: str,
    progress_logging: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    from data_access.macro_global import (
        get_global_asset_history,
        get_global_assets,
        get_macro_history,
        get_macro_indicators,
    )
    from data_access.ticker import normalize_ticker

    result: dict[str, list[dict[str, Any]]] = {}

    # 1. True non-equity rows in global_assets_daily.
    excluded = set(normalize_ticker(target)) | set(normalize_ticker(benchmark))
    assets = get_global_assets()
    equity_candidates = 0
    global_asset_candidates = 0
    global_asset_rows = 0

    _progress(progress_logging, "Global load start | global_assets_daily candidates=%d", len(assets))
    for index, asset in enumerate(assets, start=1):
        ticker = str(asset.get("ticker", "")).strip()
        if not ticker or ticker in excluded:
            continue
        asset_class = str(asset.get("asset_class", "")).strip().lower()
        if asset_class in {"equity", "stock", "company", "share"}:
            equity_candidates += 1
            continue
        global_asset_candidates += 1
        rows = _table_rows(
            get_global_asset_history(
                ticker,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if rows:
            result[ticker] = rows
            global_asset_rows += 1
        if index == 1 or index % 10 == 0 or index == len(assets):
            _progress(
                progress_logging,
                "Global asset progress %d/%d | usable=%d equity_skipped=%d non_equity=%d",
                index, len(assets), len(result), equity_candidates, global_asset_candidates,
            )

    # 2. Global-context series that upstream ETL stores in macro_daily_ledger.
    indicators = get_macro_indicators()
    macro_global_rows = 0
    for indicator in indicators:
        if indicator == benchmark or indicator not in UPSTREAM_GLOBAL_CONTEXT_SERIES:
            continue
        rows = _table_rows(
            get_macro_history(
                indicator,
                start_date=start_date,
                end_date=end_date,
            )
        )
        if rows:
            result[indicator] = rows
            macro_global_rows += 1

    _progress(
        progress_logging,
        "Global load complete | usable_series=%d non_equity_assets=%d macro_global_series=%d | source_rows=%d+%d",
        len(result), global_asset_rows, macro_global_rows, global_asset_candidates, macro_global_rows,
    )
    return result


def _build_table_outcome(
    company_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    target: str,
    prediction_date: date,
    holding_period_months: float,
    entry_mode: str,
    benchmark: str,
):
    """Apply the Phase 4.2 outcome contract against preloaded real OLAP rows."""
    from .outcome import OutcomeObservation

    if entry_mode == "next_trading_day":
        entry = _close_after(company_rows, prediction_date)
    else:
        entry = _close_on_or_before(company_rows, prediction_date)

    if entry is None:
        return OutcomeObservation(
            target=target,
            prediction_date=prediction_date,
            data_cutoff_date=prediction_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=None,
            exit_date=None,
            entry_price=None,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No real-OLAP entry observation exists.",
        )

    entry_date, entry_price = entry
    whole_months = int(holding_period_months)
    fractional_months = holding_period_months - whole_months

    target_exit_date = entry_date + relativedelta(months=whole_months)
    if fractional_months > 0:
        target_exit_date += relativedelta(
            days=round(fractional_months * 30.4375)
        )

    exit_row = _close_on_or_before(company_rows, target_exit_date)
    if exit_row is None or exit_row[0] < entry_date:
        return OutcomeObservation(
            target=target,
            prediction_date=prediction_date,
            data_cutoff_date=prediction_date,
            entry_mode=entry_mode,
            holding_period_months=holding_period_months,
            entry_date=entry_date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            stock_return_pct=None,
            benchmark_return_pct=None,
            relative_return_pct=None,
            benchmark=benchmark,
            valid=False,
            limitation="No real-OLAP exit observation exists.",
        )

    exit_date, exit_price = exit_row
    stock_return = ((exit_price - entry_price) / abs(entry_price)) * 100.0

    benchmark_return = None
    benchmark_entry = _close_on_or_before(benchmark_rows, entry_date)
    benchmark_exit = _close_on_or_before(benchmark_rows, exit_date)
    if benchmark_entry is not None and benchmark_exit is not None:
        benchmark_return = (
            (benchmark_exit[1] - benchmark_entry[1])
            / abs(benchmark_entry[1])
        ) * 100.0

    relative_return = (
        stock_return - benchmark_return
        if benchmark_return is not None
        else None
    )

    return OutcomeObservation(
        target=target,
        prediction_date=prediction_date,
        data_cutoff_date=prediction_date,
        entry_mode=entry_mode,
        holding_period_months=holding_period_months,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        stock_return_pct=stock_return,
        benchmark_return_pct=benchmark_return,
        relative_return_pct=relative_return,
        benchmark=benchmark,
        valid=isfinite(stock_return),
        limitation=None if isfinite(stock_return) else "Non-finite return.",
    )


def _prediction_dates(
    rows: list[dict[str, Any]],
    analysis_timeframe: str,
    holding_period_months: float,
    step_trading_days: int,
    max_folds: int,
) -> list[date]:
    if analysis_timeframe not in TIMEFRAME_MONTHS:
        raise ValueError(
            f"Unsupported analysis timeframe: {analysis_timeframe}"
        )
    if step_trading_days < 1:
        raise ValueError("step_trading_days must be >= 1")
    if max_folds < 1:
        raise ValueError("max_folds must be >= 1")

    timeframe_months = TIMEFRAME_MONTHS[analysis_timeframe]
    min_start = rows[0]["report_date"] + relativedelta(
        days=max(1, round(timeframe_months * 30.4375))
    )
    latest_usable = rows[-1]["report_date"] - relativedelta(
        days=max(1, round(holding_period_months * 30.4375))
    )

    candidates = [
        row["report_date"]
        for index, row in enumerate(rows)
        if row["report_date"] >= min_start
        and row["report_date"] <= latest_usable
        and index >= step_trading_days
    ]

    if not candidates:
        return []

    stepped = candidates[::max(1, step_trading_days)]
    return stepped[-max_folds:]


def _current_membership(
    ticker: str,
    as_of_date: date | None = None,
) -> tuple[str | None, str | None, list[str], list[str]]:
    from data_access.metadata import (
        get_companies_by_industry,
        get_companies_by_sector,
        get_company,
    )

    company = get_company(ticker)
    if company is None:
        raise RuntimeError(f"Ticker {ticker!r} not found in market_metadata.")

    industry = company.get("Industry")
    sector = company.get("Sector")

    industry_constituents = [
        str(row.get("Ticker"))
        for row in get_companies_by_industry(industry, as_of_date=as_of_date)
        if row.get("Ticker")
    ] if industry else []

    sector_constituents = [
        str(row.get("Ticker"))
        for row in get_companies_by_sector(sector, as_of_date=as_of_date)
        if row.get("Ticker")
    ] if sector else []

    return industry, sector, sorted(set(industry_constituents)), sorted(set(sector_constituents))


def _call_history_loader(loader: Any, *, ticker: str | None, start_date: date, end_date: date):
    """Call a Phase 2 repository using its declared parameter names."""
    import inspect

    params = inspect.signature(loader).parameters
    kwargs: dict[str, Any] = {}
    for name in params:
        lowered = name.lower()
        if lowered in {"ticker", "symbol", "identifier", "instrument", "security"} and ticker is not None:
            kwargs[name] = ticker
        elif lowered in {"start_date", "from_date", "begin_date", "date_from"}:
            kwargs[name] = start_date
        elif lowered in {"end_date", "to_date", "until_date", "date_to"}:
            kwargs[name] = end_date
    try:
        return loader(**kwargs)
    except TypeError:
        if ticker is not None:
            return loader(ticker, start_date=start_date, end_date=end_date)
        return loader(start_date=start_date, end_date=end_date)


def _generic_rows(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_pylist"):
        raw = table.to_pylist()
    elif isinstance(table, list):
        raw = table
    else:
        return []
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        raw_date = normalized.get("report_date") or normalized.get("date") or normalized.get("event_date")
        if raw_date is None:
            continue
        try:
            normalized["report_date"] = _as_date(raw_date)
        except (TypeError, ValueError):
            continue
        rows.append(normalized)
    return rows


def _load_flow_histories(
    start_date: date,
    end_date: date,
    ticker: str,
    progress_logging: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load institutional, option, futures-basis, event and matrix histories via repositories."""
    institutional_rows: list[dict[str, Any]] = []
    options_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    microstructure_rows: list[dict[str, Any]] = []

    try:
        from data_access import institutional as institutional_repo
        keyed: dict[tuple[date, str], dict[str, Any]] = {}
        for loader_name in ("get_institutional_history", "get_institutional_flow_history"):
            loader = getattr(institutional_repo, loader_name, None)
            if loader is None:
                continue
            for row in _generic_rows(
                _call_history_loader(loader, ticker=None, start_date=start_date, end_date=end_date)
            ):
                client_type = str(row.get("client_type", "ALL"))
                key = (row["report_date"], client_type)
                keyed.setdefault(key, {}).update(row)
        institutional_rows = [row for _, row in sorted(keyed.items(), key=lambda item: item[0])]
    except (ImportError, AttributeError):
        pass

    try:
        from data_access.derivatives import get_options_history
        options_rows = _generic_rows(_call_history_loader(get_options_history, ticker=ticker, start_date=start_date, end_date=end_date))
    except (ImportError, AttributeError):
        pass

    try:
        from data_access.derivatives import get_futures_basis_history
        basis_rows = _generic_rows(_call_history_loader(get_futures_basis_history, ticker=ticker, start_date=start_date, end_date=end_date))
    except (ImportError, AttributeError):
        pass

    try:
        from data_access.trade_events import get_trade_events
        trade_rows = _generic_rows(_call_history_loader(get_trade_events, ticker=ticker, start_date=start_date, end_date=end_date))
    except (ImportError, AttributeError):
        pass

    try:
        from data_access.derivatives import get_unified_market_matrix
        microstructure_rows = _generic_rows(_call_history_loader(get_unified_market_matrix, ticker=ticker, start_date=start_date, end_date=end_date))
    except (ImportError, AttributeError):
        pass

    _progress(
        progress_logging,
        "Flow context loaded | institutional=%d options=%d basis=%d trade_events=%d microstructure=%d",
        len(institutional_rows), len(options_rows), len(basis_rows), len(trade_rows), len(microstructure_rows),
    )
    return institutional_rows, options_rows, basis_rows, trade_rows, microstructure_rows


def _build_flow_atoms(
    institutional_rows: list[dict[str, Any]],
    options_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    microstructure_rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
) -> tuple[dict[str, str], list[str]]:
    from analysis.flow_state import (
        build_derivatives_state_atoms,
        build_institutional_state_atoms,
        build_microstructure_state_atoms,
        build_trade_event_state_atoms,
    )
    states: dict[str, str] = {}
    limitations: list[str] = []
    if institutional_rows:
        states.update(build_institutional_state_atoms(institutional_rows, analysis_start, prediction_date))
    else:
        limitations.append("Institutional history unavailable through the cutoff.")
    if options_rows or basis_rows:
        states.update(build_derivatives_state_atoms(options_rows, basis_rows, analysis_start, prediction_date))
    else:
        limitations.append("Derivative options/futures history unavailable through the cutoff.")
    if trade_rows:
        states.update(build_trade_event_state_atoms(trade_rows, analysis_start, prediction_date))
    else:
        limitations.append("Trade-event history unavailable through the cutoff.")
    if microstructure_rows:
        states.update(build_microstructure_state_atoms(microstructure_rows, analysis_start, prediction_date))
    else:
        limitations.append("Unified market microstructure matrix unavailable through the cutoff.")
    return states, limitations


def _build_complete_state_atoms(
    *,
    ticker: str,
    benchmark: str,
    prediction_date: date,
    analysis_start: date,
    company_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    macro_histories: dict[str, list[dict[str, Any]]],
    global_histories: dict[str, list[dict[str, Any]]],
    industry: str | None,
    sector: str | None,
    industry_constituents: list[str],
    sector_constituents: list[str],
    institutional_rows: list[dict[str, Any]],
    options_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    microstructure_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str], list[str]]:
    """
    Build the real Phase 4 state surface from the existing Phase 3 engines.

    Historical financial states are intentionally NOT emitted into the
    relationship panel because the OLAP schema does not carry publication
    timestamps. Their absence is reported as a timing limitation rather than
    being treated as missing data or silently backdated.
    """
    from analysis.group_state import build_group_state

    states: dict[str, str] = {}
    limitations: list[str] = []

    selected_company = _subset_table(company_rows, analysis_start, prediction_date)
    historical_company = _subset_table(company_rows, company_rows[0]["report_date"], prediction_date)
    states.update(
        _build_company_market_atoms(
            ticker=ticker,
            selected_table=selected_company,
            historical_table=historical_company,
        )
    )

    benchmark_selected = [
        row for row in benchmark_rows
        if analysis_start <= row["report_date"] <= prediction_date
    ]
    benchmark_historical = [
        row for row in benchmark_rows
        if row["report_date"] <= prediction_date
    ]
    states.update(
        _build_generic_market_atoms(
            "benchmark",
            benchmark,
            benchmark_selected,
            benchmark_historical,
        )
    )

    states.update(
        _build_macro_atoms(
            macro_histories,
            analysis_start,
            prediction_date,
        )
    )

    states.update(
        _build_global_atoms(
            global_histories,
            analysis_start,
            prediction_date,
        )
    )

    flow_states, flow_limitations = _build_flow_atoms(
        institutional_rows, options_rows, basis_rows, trade_rows, microstructure_rows,
        analysis_start, prediction_date,
    )
    states.update(flow_states)
    limitations.extend(flow_limitations)

    benchmark_table = pa.Table.from_pylist(benchmark_historical) if benchmark_historical else None

    if industry and industry_constituents:
        industry_state = build_group_state(
            group_type="industry",
            group_name=industry,
            constituents=industry_constituents,
            analysis_start_date=analysis_start,
            analysis_end_date=prediction_date,
            benchmark_history=benchmark_table,
        )
        for metric, metric_state in industry_state.market_states.items():
            label = _state_label(metric_state)
            if label and not label.startswith("Unknown"):
                states[f"industry.market.{metric}"] = label
        limitations.extend(industry_state.limitations)

    if sector and sector_constituents:
        sector_state = build_group_state(
            group_type="sector",
            group_name=sector,
            constituents=sector_constituents,
            analysis_start_date=analysis_start,
            analysis_end_date=prediction_date,
            benchmark_history=benchmark_table,
        )
        for metric, metric_state in sector_state.market_states.items():
            label = _state_label(metric_state)
            if label and not label.startswith("Unknown"):
                states[f"sector.market.{metric}"] = label
        limitations.extend(sector_state.limitations)

    # Historical financial states are structurally available in Phase 3, but
    # they cannot be admitted here without a verified information-availability
    # timestamp. Record the limitation once per snapshot.
    limitations.append(
        "Historical company/industry/sector financial states are excluded from "
        "the relationship panel because OLAP ReportDate is accounting period end, "
        "not verified publication/availability date."
    )

    return states, limitations, list(FINANCIAL_TIMING_LIMITED_FAMILIES)


def build_real_olap_relationship_panel(
    config: RealOLAPValidationConfig,
) -> tuple[list[HistoricalRelationshipObservation], dict[str, Any], date | None]:
    """
    Build a real historical Phase 3-state / Phase 4-outcome panel.

    The adapter now exposes the complete *time-safe* real state surface:
        company.market, industry.market, sector.market, macro, global,
        benchmark, institutional, derivatives, microstructure, trade_events.

    Financial families are wired as a known timing limitation rather than
    backdated into the historical relationship panel.
    """
    from data_access.market import get_daily_history

    company_table = get_daily_history(config.ticker)
    company_rows = _table_rows(company_table)
    if not company_rows:
        raise RuntimeError(
            f"No real OLAP market history found for {config.ticker!r}."
        )

    latest_market_date = company_rows[-1]["report_date"]
    earliest_date = company_rows[0]["report_date"]

    benchmark_table = _resolve_benchmark_history(
        config.benchmark,
        earliest_date,
        latest_market_date,
    )
    benchmark_rows = _table_rows(benchmark_table)

    if not benchmark_rows:
        raise RuntimeError(
            f"No real OLAP benchmark history found for {config.benchmark!r}."
        )
        
    industry, sector, industry_constituents, sector_constituents = _current_membership(
    config.ticker
)

    # Keep the target's structural group labels stable; constituent lists are
    # refreshed point-in-time inside the historical snapshot loop.

    _configure_progress_logging(config.progress_logging)
    run_start = time.perf_counter()
    _progress(
        config.progress_logging,
        "Panel start | ticker=%s benchmark=%s timeframe=%s holding=%s | company_rows=%d",
        config.ticker, config.benchmark, config.analysis_timeframe,
        config.holding_period_months, len(company_rows),
    )
    _progress(
        config.progress_logging,
        "Membership | industry=%s (%d constituents) sector=%s (%d constituents)",
        industry or "<none>", len(industry_constituents),
        sector or "<none>", len(sector_constituents),
    )

    macro_histories = _load_all_macro_histories(
        earliest_date,
        latest_market_date,
        config.benchmark,
        progress_logging=config.progress_logging,
    )
    global_histories = _load_global_histories(
        earliest_date,
        latest_market_date,
        config.benchmark,
        config.ticker,
        progress_logging=config.progress_logging,
    )
    (
        institutional_rows,
        options_rows,
        basis_rows,
        trade_rows,
        microstructure_rows,
    ) = _load_flow_histories(
        earliest_date,
        latest_market_date,
        config.ticker,
        progress_logging=config.progress_logging,
    )

    _progress(
        config.progress_logging,
        "Context loaded | macro_series=%d global_series=%d institutional=%d options=%d basis=%d trade_events=%d microstructure=%d | %.2fs elapsed",
        len(macro_histories), len(global_histories), len(institutional_rows), len(options_rows), len(basis_rows), len(trade_rows), len(microstructure_rows),
        time.perf_counter() - run_start,
    )

    dates = _prediction_dates(
        company_rows,
        config.analysis_timeframe,
        config.holding_period_months,
        config.step_trading_days,
        config.max_folds,
    )

    snapshots: list[HistoricalStateSnapshot] = []
    outcomes: list[Any] = []
    skipped_no_state = 0
    skipped_no_outcome = 0
    state_future_violations = 0
    outcome_temporal_violations = 0
    financial_timing_limited_observations = 0
    candidate_observations = len(dates)

    timeframe_months = TIMEFRAME_MONTHS[config.analysis_timeframe]

    group_cache: DailyHistoryCache | None = None
    group_state_context = None
    if config.performance_cache:
        from data_access.market import get_daily_history as _raw_group_history_loader

        group_cache = DailyHistoryCache(_raw_group_history_loader)
        group_state_context = patch_group_state_history_loader(group_cache)
        group_state_context.__enter__()

    _progress(
        config.progress_logging,
        "Historical panel | candidate_dates=%d | step=%d trading days | cache=%s",
        len(dates), config.step_trading_days,
        "ON" if config.performance_cache else "OFF",
    )

    try:
        for date_index, prediction_date in enumerate(dates, start=1):
            def log_panel_progress() -> None:
                if (
                    date_index == 1
                    or date_index % max(1, config.progress_every) == 0
                    or date_index == len(dates)
                ):
                    cache_message = ""
                    if group_cache is not None:
                        cache_message = (
                            f" | group_cache hits={group_cache.stats.hits} "
                            f"misses={group_cache.stats.misses} "
                            f"unique={group_cache.unique_tickers}"
                        )
                    _progress(
                        config.progress_logging,
                        "Panel progress %d/%d (%.1f%%) | valid=%d no_state=%d no_outcome=%d | %.2fs elapsed%s",
                        date_index, len(dates), (date_index / len(dates)) * 100.0,
                        len(snapshots), skipped_no_state, skipped_no_outcome,
                        time.perf_counter() - run_start, cache_message,
                    )

            analysis_start = prediction_date + relativedelta(
                days=-max(1, round(timeframe_months * 30.4375))
            )

            _, _, industry_constituents_as_of, sector_constituents_as_of = _current_membership(
                config.ticker, prediction_date
            )
            states, limitations, timing_limited = _build_complete_state_atoms(
                ticker=config.ticker,
                benchmark=config.benchmark,
                prediction_date=prediction_date,
                analysis_start=analysis_start,
                company_rows=company_rows,
                benchmark_rows=benchmark_rows,
                macro_histories=macro_histories,
                global_histories=global_histories,
                industry=industry,
                sector=sector,
                industry_constituents=industry_constituents_as_of,
                sector_constituents=sector_constituents_as_of,
                institutional_rows=institutional_rows,
                options_rows=options_rows,
                basis_rows=basis_rows,
                trade_rows=trade_rows,
                microstructure_rows=microstructure_rows,
            )

            if states:
                state_dates = [
                    row["report_date"]
                    for row in company_rows
                    if row["report_date"] <= prediction_date
                ]
                if state_dates and max(state_dates) > prediction_date:
                    state_future_violations += 1

            if not states:
                skipped_no_state += 1
                log_panel_progress()
                continue

            outcome = _build_table_outcome(
                company_rows,
                benchmark_rows,
                config.ticker,
                prediction_date,
                config.holding_period_months,
                config.entry_mode,
                config.benchmark,
            )

            if (
                outcome.entry_date is not None
                and config.entry_mode == "next_trading_day"
                and outcome.entry_date <= prediction_date
            ):
                outcome_temporal_violations += 1
            if (
                outcome.entry_date is not None
                and outcome.exit_date is not None
                and outcome.exit_date < outcome.entry_date
            ):
                outcome_temporal_violations += 1

            if not outcome.valid or outcome.stock_return_pct is None:
                skipped_no_outcome += 1
                log_panel_progress()
                continue

            financial_timing_limited_observations += len(timing_limited)
            snapshots.append(
                HistoricalStateSnapshot(
                    as_of_date=prediction_date,
                    data_cutoff_date=prediction_date,
                    target=config.ticker,
                    scope="company",
                    states=states,
                    limitations=tuple(sorted(set(limitations))),
                )
            )
            outcomes.append(outcome)

            log_panel_progress()


    finally:
        if group_state_context is not None:
            group_state_context.__exit__(None, None, None)

    if not snapshots:
        raise RuntimeError(
            "Real OLAP validation produced no valid state/outcome pairs."
        )

    panel = build_historical_state_outcome_panel(
        snapshots=snapshots,
        outcomes=outcomes,
        cutoff_date=max(snapshot.as_of_date for snapshot in snapshots),
        strict=True,
    )

    stats: dict[str, Any] = {
        "valid_pairs": panel.stats.valid_pairs,
        "skipped_no_outcome": skipped_no_outcome,
        "skipped_no_state": skipped_no_state,
        "skipped_cutoff_violation": panel.stats.skipped_cutoff_violation,
        "skipped_identity_mismatch": panel.stats.skipped_identity_mismatch,
        "state_future_violations": state_future_violations,
        "outcome_temporal_violations": outcome_temporal_violations,
        "candidate_observations": candidate_observations,
        "financial_timing_limited_observations": financial_timing_limited_observations,
        "timing_limited_families": FINANCIAL_TIMING_LIMITED_FAMILIES,
        "industry": industry,
        "sector": sector,
        "industry_constituents": len(industry_constituents),
        "sector_constituents": len(sector_constituents),
        "macro_series": len(macro_histories),
        "global_series": len(global_histories),
        "institutional_rows": len(institutional_rows),
        "options_rows": len(options_rows),
        "basis_rows": len(basis_rows),
        "trade_event_rows": len(trade_rows),
        "microstructure_rows": len(microstructure_rows),
    }
    _progress(
        config.progress_logging,
        "Panel complete | valid_pairs=%d candidates=%d | %.2fs elapsed",
        panel.stats.valid_pairs, candidate_observations,
        time.perf_counter() - run_start,
    )
    if group_cache is not None:
        _progress(
            config.progress_logging,
            "History cache | requests=%d hits=%d misses=%d unique_tickers=%d",
            group_cache.stats.requests, group_cache.stats.hits,
            group_cache.stats.misses, group_cache.unique_tickers,
        )
    return list(panel.observations), stats, latest_market_date


def validate_real_olap_relationships(
    config: RealOLAPValidationConfig,
) -> RealOLAPValidationResult:
    validation_start = time.perf_counter()
    _configure_progress_logging(config.progress_logging)
    _progress(config.progress_logging, "Validation start | hardened=%s | observations will be built first", config.hardened_validation)
    observations, stats, latest_market_date = build_real_olap_relationship_panel(config)
    _progress(
        config.progress_logging,
        "Validation input ready | observations=%d | building walk-forward validation",
        len(observations),
    )

    from .hardening_4_10 import audit_state_surface

    combined_state_surface: dict[str, str] = {}
    if observations:
        combined_state_surface = observations[-1].states
    surface_audit = audit_state_surface(
        combined_state_surface,
        EXPECTED_PHASE4_STATE_FAMILIES,
    )
    time_safe_surface_audit = audit_state_surface(
        combined_state_surface,
        TIME_SAFE_PHASE4_STATE_FAMILIES,
    )

    engine = RelationshipDiscoveryEngine(
        max_order=3,
        min_observations=config.min_training_observations,
    )

    from .walk_forward_relationship import validate_walk_forward_relationships_hardened

    if config.hardened_validation:
        _progress(config.progress_logging, "Walk-forward hardening start | purge=True | max_order=3")
        result = validate_walk_forward_relationships_hardened(
            observations=observations,
            engine=engine,
            min_training_observations=config.min_training_observations,
            purge_overlapping_labels=True,
            parallel_workers=int(getattr(config, "phase5_parallel_workers", 1)),
        )
    else:
        _progress(config.progress_logging, "Walk-forward validation start | hardened=False")
        from .walk_forward_relationship import validate_walk_forward_relationships
        result = validate_walk_forward_relationships(
            observations=observations,
            engine=engine,
            min_training_observations=config.min_training_observations,
        )

    _progress(
        config.progress_logging,
        "Validation complete | folds=%d | leakage=%d | %.2fs elapsed",
        len(result.folds), result.leakage_violations,
        time.perf_counter() - validation_start,
    )

    first_prediction_date = min(
        (observation.as_of_date for observation in observations),
        default=None,
    )

    timing_limited = tuple(stats.get("timing_limited_families", ()))
    integrity_clean = (
        result.leakage_violations == 0
        and stats.get("state_future_violations", 0) == 0
        and stats.get("outcome_temporal_violations", 0) == 0
    )
    broad_validated = (
        not surface_audit.missing_families
        and not timing_limited
        and integrity_clean
    )
    time_safe_validated = (
        not time_safe_surface_audit.missing_families
        and integrity_clean
    )

    return RealOLAPValidationResult(
        ticker=config.ticker,
        benchmark=config.benchmark,
        analysis_timeframe=config.analysis_timeframe,
        holding_period_months=config.holding_period_months,
        entry_mode=config.entry_mode,
        panel_rows=len(observations),
        skipped_no_outcome=stats["skipped_no_outcome"],
        skipped_no_state=stats["skipped_no_state"],
        walk_forward_folds=len(result.folds),
        method_a_hit_rate_pct=result.method_a.directional_hit_rate_pct,
        method_b_hit_rate_pct=result.method_b.directional_hit_rate_pct,
        combined_hit_rate_pct=result.combined.directional_hit_rate_pct,
        leakage_violations=result.leakage_violations,
        state_future_violations=stats.get("state_future_violations", 0),
        outcome_temporal_violations=stats.get("outcome_temporal_violations", 0),
        latest_market_date=latest_market_date,
        first_prediction_date=first_prediction_date,
        state_surface_coverage_pct=surface_audit.coverage_pct,
        time_safe_state_surface_coverage_pct=time_safe_surface_audit.coverage_pct,
        missing_state_families=surface_audit.missing_families,
        timing_limited_state_families=timing_limited,
        broad_relationship_surface_validated=broad_validated,
        time_safe_relationship_surface_validated=time_safe_validated,
        purged_training_observations=getattr(result, "purged_training_observations", 0),
        unknown_overlap_observations=getattr(result, "unknown_overlap_observations", 0),
        selection_candidate_evaluations=getattr(result, "selection_candidate_evaluations", 0),
        selection_validated_predictions=getattr(result, "selection_validated_predictions", 0),
        multiple_testing_controlled_folds=getattr(result, "multiple_testing_controlled_folds", 0),
        method_a_directional_predictions=result.method_a.directional_predictions,
        method_b_directional_predictions=result.method_b.directional_predictions,
        combined_directional_predictions=result.combined.directional_predictions,
        candidate_observations=stats.get("candidate_observations", len(observations)),
        valid_state_observations=len(observations),
        financial_timing_limited_observations=stats.get("financial_timing_limited_observations", 0),
        industry_constituents=stats.get("industry_constituents", 0),
        sector_constituents=stats.get("sector_constituents", 0),
        macro_series=stats.get("macro_series", 0),
        global_series=stats.get("global_series", 0),
        institutional_rows=stats.get("institutional_rows", 0),
        options_rows=stats.get("options_rows", 0),
        basis_rows=stats.get("basis_rows", 0),
        trade_event_rows=stats.get("trade_event_rows", 0),
        microstructure_rows=stats.get("microstructure_rows", 0),
    )
