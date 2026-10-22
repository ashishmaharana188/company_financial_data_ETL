"""Instrument-type filters for raw derivative contracts.

Keep source cataloguing and history loading on the same SQL predicates. The
instrument codes below are the codes the upstream ETL explicitly handles.
"""
from __future__ import annotations

from typing import Literal


STOCK_INDEX_OPTION_TYPES = ("OPTSTK", "OPTIDX", "STO", "IDO")
COMMODITY_OPTION_TYPES = ("OPTFUT", "OPTCOM")
STOCK_INDEX_FUTURE_TYPES = ("FUTSTK", "FUTIDX", "STF", "IDF", "FUTIVX")
COMMODITY_FUTURE_TYPES = ("FUTCOM",)

ContractKind = Literal["stock_index_option", "commodity_option", "stock_index_future", "commodity_future", "other"]


def _in_sql(column: str, values: tuple[str, ...]) -> str:
    """Build a predicate from internal constant identifiers only."""
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"UPPER(TRIM(COALESCE({column}, ''))) IN ({quoted})"


def _option_type_sql(values: tuple[str, ...]) -> str:
    return _in_sql('"OptionType"', values)


def _instrument_type_sql(values: tuple[str, ...]) -> str:
    return _in_sql('"InstrumentType"', values)


OPTION_SIDE_SQL = _option_type_sql(("CE", "PE"))
FUTURES_SIDE_SQL = _option_type_sql(("XX", ""))

STOCK_INDEX_OPTIONS_SQL = f"({_instrument_type_sql(STOCK_INDEX_OPTION_TYPES)}) AND ({OPTION_SIDE_SQL})"
COMMODITY_OPTIONS_SQL = f"({_instrument_type_sql(COMMODITY_OPTION_TYPES)}) AND ({OPTION_SIDE_SQL})"
STOCK_INDEX_FUTURES_SQL = f"({_instrument_type_sql(STOCK_INDEX_FUTURE_TYPES)}) AND ({FUTURES_SIDE_SQL})"
COMMODITY_FUTURES_SQL = f"({_instrument_type_sql(COMMODITY_FUTURE_TYPES)}) AND ({FUTURES_SIDE_SQL})"


def classify_contract(instrument_type: object, option_type: object = "XX") -> ContractKind:
    """Pure-Python mirror of the SQL source classification, used by tests."""
    kind = str(instrument_type or "").strip().upper()
    side = str(option_type or "XX").strip().upper()
    if kind in COMMODITY_FUTURE_TYPES and side in {"XX", ""}:
        return "commodity_future"
    if kind in COMMODITY_OPTION_TYPES and side in {"CE", "PE"}:
        return "commodity_option"
    if kind in STOCK_INDEX_FUTURE_TYPES and side in {"XX", ""}:
        return "stock_index_future"
    if kind in STOCK_INDEX_OPTION_TYPES and side in {"CE", "PE"}:
        return "stock_index_option"
    return "other"
