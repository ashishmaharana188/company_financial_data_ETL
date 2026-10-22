from __future__ import annotations

"""Phase 3 descriptive state adapters for institutional, derivatives and flow data.

This module deliberately performs no bullish/bearish interpretation. It converts
validated repository rows into Phase 3-style descriptive state atoms so Phase 4
can learn which states, if any, relate to later outcomes.
"""

from datetime import date, datetime
from math import isfinite
import re
from typing import Any, Callable, Iterable


StateBuilder = Callable[..., Any]


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            normalized[str(key).strip().lower()] = value
        raw_date = (
            normalized.get("report_date")
            or normalized.get("date")
            or normalized.get("event_date")
        )
        if raw_date is None:
            continue
        try:
            normalized["report_date"] = _as_date(raw_date)
        except (TypeError, ValueError):
            continue
        result.append(normalized)
    return sorted(result, key=lambda row: row["report_date"])


class _ValueColumn:
    def __init__(self, values: list[float]):
        self._values = values

    def to_pylist(self) -> list[float]:
        return list(self._values)


class _ValueTable:
    def __init__(self, values: list[float]):
        self._values = values
        self.num_rows = len(values)

    def column(self, name: str) -> _ValueColumn:
        if name != "value":
            raise KeyError(name)
        return _ValueColumn(self._values)


def _rows_to_state_tables(
    rows: list[dict[str, Any]],
    metric: str,
    analysis_start: date,
    prediction_date: date,
    *,
    use_pyarrow: bool,
):
    pa = None
    if use_pyarrow:
        import pyarrow as pa

    selected = [
        {"report_date": row["report_date"], "value": row[metric]}
        for row in rows
        if analysis_start <= row["report_date"] <= prediction_date
        and row.get(metric) is not None
    ]
    historical = [
        {"report_date": row["report_date"], "value": row[metric]}
        for row in rows
        if row["report_date"] <= prediction_date
        and row.get(metric) is not None
    ]

    if not use_pyarrow:
        def simple_table(data: list[dict[str, Any]]):
            return _ValueTable([float(row["value"]) for row in data])

        return simple_table(selected), simple_table(historical)

    empty = pa.table(
        {
            "report_date": pa.array([], type=pa.date32()),
            "value": pa.array([], type=pa.float64()),
        }
    )
    if not historical:
        return empty, empty

    def make_table(data: list[dict[str, Any]]):
        if not data:
            return empty
        return pa.Table.from_pydict(
            {
                "report_date": [row["report_date"] for row in data],
                "value": [float(row["value"]) for row in data],
            }
        )

    return make_table(selected), make_table(historical)


def _state_atoms_from_rows(
    family: str,
    rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
    metrics: list[str],
    state_builder: StateBuilder | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    use_pyarrow = state_builder is None
    if state_builder is None:
        from analysis.state import build_variable_state

        state_builder = build_variable_state

    atoms: dict[str, str] = {}
    normalized = _normalize_rows(rows)
    for metric in metrics:
        if not any(_numeric(row.get(metric)) is not None for row in normalized):
            continue
        usable_rows = [
            {**row, metric: _numeric(row.get(metric))}
            for row in normalized
            if _numeric(row.get(metric)) is not None
        ]
        selected_table, historical_table = _rows_to_state_tables(
            usable_rows,
            metric,
            analysis_start,
            prediction_date,
            use_pyarrow=use_pyarrow,
        )
        if historical_table.num_rows == 0:
            continue
        state = state_builder(
            variable=f"{family}.{prefix + '.' if prefix else ''}{metric}",
            selected_table=selected_table,
            historical_table=historical_table,
            value_column="value",
        )
        label = state.get("state") if isinstance(state, dict) else getattr(state, "state", None)
        if isinstance(label, str) and label.strip() and not label.startswith("Unknown"):
            key = f"{family}.{prefix + '.' if prefix else ''}{metric}"
            atoms[key] = label
    return atoms


def build_institutional_state_atoms(
    rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
    state_builder: StateBuilder | None = None,
) -> dict[str, str]:
    normalized = _normalize_rows(rows)
    if not normalized:
        return {}

    excluded = {"report_date", "date", "client_type"}
    metric_names = sorted({
        key
        for row in normalized
        for key, value in row.items()
        if key not in excluded and _numeric(value) is not None
    })

    result: dict[str, str] = {}
    groups = sorted({str(row.get("client_type", "ALL")) for row in normalized})
    for client_type in groups:
        group_rows = [
            row for row in normalized
            if str(row.get("client_type", "ALL")) == client_type
        ]
        result.update(
            _state_atoms_from_rows(
                "institutional",
                group_rows,
                analysis_start,
                prediction_date,
                metric_names,
                state_builder=state_builder,
                prefix=_slug(client_type),
            )
        )

        # Explicitly derive net contracts from the source long/short fields so
        # the positioning relationship is available without assigning a sign.
        derived: list[dict[str, Any]] = []
        for row in group_rows:
            item = dict(row)
            index_long = _numeric(row.get("future_index_long"))
            index_short = _numeric(row.get("future_index_short"))
            stock_long = _numeric(row.get("future_stock_long"))
            stock_short = _numeric(row.get("future_stock_short"))
            total_long = _numeric(row.get("total_long_contracts"))
            total_short = _numeric(row.get("total_short_contracts"))
            item["future_index_net"] = (
                index_long - index_short
                if index_long is not None and index_short is not None else None
            )
            item["future_stock_net"] = (
                stock_long - stock_short
                if stock_long is not None and stock_short is not None else None
            )
            item["total_net_contracts"] = (
                total_long - total_short
                if total_long is not None and total_short is not None else None
            )
            derived.append(item)
        result.update(
            _state_atoms_from_rows(
                "institutional",
                derived,
                analysis_start,
                prediction_date,
                ["future_index_net", "future_stock_net", "total_net_contracts"],
                state_builder=state_builder,
                prefix=_slug(client_type),
            )
        )
    return result


def _aggregate_options(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[date, dict[str, float]] = {}
    for row in _normalize_rows(rows):
        bucket = grouped.setdefault(
            row["report_date"],
            {
                "total_put_oi": 0.0,
                "total_call_oi": 0.0,
                "total_put_volume": 0.0,
                "total_call_volume": 0.0,
                "expiry_count": 0.0,
            },
        )
        for field in bucket:
            if field == "expiry_count":
                continue
            value = _numeric(row.get(field))
            if value is not None:
                bucket[field] += value
        bucket["expiry_count"] += 1.0

    result: list[dict[str, Any]] = []
    for report_date, bucket in sorted(grouped.items()):
        put_oi = bucket["total_put_oi"]
        call_oi = bucket["total_call_oi"]
        put_volume = bucket["total_put_volume"]
        call_volume = bucket["total_call_volume"]
        result.append({
            "report_date": report_date,
            **bucket,
            "oi_pcr": put_oi / call_oi if call_oi else None,
            "volume_pcr": put_volume / call_volume if call_volume else None,
        })
    return result


def build_derivatives_state_atoms(
    options_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
    state_builder: StateBuilder | None = None,
) -> dict[str, str]:
    result = _state_atoms_from_rows(
        "derivatives",
        _aggregate_options(options_rows),
        analysis_start,
        prediction_date,
        [
            "total_put_oi",
            "total_call_oi",
            "oi_pcr",
            "total_put_volume",
            "total_call_volume",
            "volume_pcr",
            "expiry_count",
        ],
        state_builder=state_builder,
        prefix="options",
    )

    # Match the upstream ETL's near-month basis selection: highest OI row per
    # ticker/date. This prevents multiple expiries from receiving equal weight.
    by_date: dict[date, dict[str, Any]] = {}
    for row in _normalize_rows(basis_rows):
        current = by_date.get(row["report_date"])
        oi = _numeric(row.get("open_interest"))
        if current is None or (oi is not None and _numeric(current.get("open_interest")) is None) or (
            oi is not None
            and _numeric(current.get("open_interest")) is not None
            and oi > _numeric(current.get("open_interest"))
        ):
            by_date[row["report_date"]] = row

    result.update(
        _state_atoms_from_rows(
            "derivatives",
            list(by_date.values()),
            analysis_start,
            prediction_date,
            ["spot_price", "futures_price", "open_interest", "absolute_basis", "basis_percentage"],
            state_builder=state_builder,
            prefix="futures",
        )
    )
    return result


def _aggregate_trade_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[date, dict[str, float]] = {}
    for row in _normalize_rows(rows):
        bucket = grouped.setdefault(
            row["report_date"],
            {
                "event_count": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "gross_event_volume": 0.0,
                "net_event_volume": 0.0,
                "trade_price_sum": 0.0,
                "trade_price_count": 0.0,
            },
        )
        quantity = _numeric(row.get("quantity"))
        price = _numeric(row.get("trade_price"))
        transaction_type = str(row.get("transaction_type", "")).upper()
        bucket["event_count"] += 1.0
        if quantity is not None:
            bucket["gross_event_volume"] += abs(quantity)
            if transaction_type == "BUY":
                bucket["buy_volume"] += abs(quantity)
            elif transaction_type == "SELL":
                bucket["sell_volume"] += abs(quantity)
            else:
                # Unknown transaction types are neither buy nor sell.
                # Do not silently convert malformed/novel values into sells.
                bucket["gross_event_volume"] -= abs(quantity)
        if price is not None:
            bucket["trade_price_sum"] += price
            bucket["trade_price_count"] += 1.0

    result: list[dict[str, Any]] = []
    for report_date, bucket in sorted(grouped.items()):
        result.append({
            "report_date": report_date,
            "event_count": bucket["event_count"],
            "buy_volume": bucket["buy_volume"],
            "sell_volume": bucket["sell_volume"],
            "gross_event_volume": bucket["gross_event_volume"],
            "net_event_volume": bucket["buy_volume"] - bucket["sell_volume"],
            "average_trade_price": (
                bucket["trade_price_sum"] / bucket["trade_price_count"]
                if bucket["trade_price_count"] else None
            ),
        })
    return result


def build_trade_event_state_atoms(
    rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
    state_builder: StateBuilder | None = None,
) -> dict[str, str]:
    return _state_atoms_from_rows(
        "trade_events",
        _aggregate_trade_events(rows),
        analysis_start,
        prediction_date,
        [
            "event_count",
            "buy_volume",
            "sell_volume",
            "gross_event_volume",
            "net_event_volume",
            "average_trade_price",
        ],
        state_builder=state_builder,
    )


def build_microstructure_state_atoms(
    rows: list[dict[str, Any]],
    analysis_start: date,
    prediction_date: date,
    state_builder: StateBuilder | None = None,
) -> dict[str, str]:
    normalized = _normalize_rows(rows)
    if not normalized:
        return {}

    metrics = [
        "delivery_percentage",
        "daily_hl_spread",
        "daily_vwap_dev",
        "short_volume",
        "short_percentage",
        "net_block_volume",
        "avg_block_premium",
        "oi_pcr",
        "delta_oi_pcr",
        "futures_basis",
    ]
    # Boolean eligibility/deal flags are retained in the state input as 0/1,
    # but no economic interpretation is attached to the resulting state.
    normalized_rows = []
    for row in normalized:
        item = dict(row)
        for field in ("has_block_deal", "is_fo_eligible"):
            if field in item:
                item[field] = 1.0 if bool(item[field]) else 0.0
        normalized_rows.append(item)
    metrics.extend(["has_block_deal", "is_fo_eligible"])
    return _state_atoms_from_rows(
        "microstructure",
        normalized_rows,
        analysis_start,
        prediction_date,
        metrics,
        state_builder=state_builder,
    )
