from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_underlying_symbol(value: Any) -> str:
    """Normalize an instrument symbol for exact matching only.

    Exchange suffixes .NS/.BO are ignored; no semantic or economic aliases are
    guessed (for example BRENT_CRUDE is not silently treated as CRUDEOIL).
    """
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\.(NS|BO)$", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def find_exact_underlying_match(context_identifier: str | None, choices: list[dict[str, str]]) -> str | None:
    """Return only an exact ticker match; never fall back to the first choice."""
    target = normalize_underlying_symbol(context_identifier)
    if not target:
        return None
    matches = [str(item.get("value", "")) for item in choices
               if normalize_underlying_symbol(item.get("value", "")) == target]
    return matches[0] if len(matches) == 1 else None


def aggregate_option_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the selected underlying's option contracts per date.

    Sums available CE/PE open interest and volume across all strikes and expiry
    dates. The result is an underlying-level PCR series, not a single-strike view.
    """
    columns = ["report_date", "total_put_oi", "total_call_oi", "oi_pcr",
               "total_put_volume", "total_call_volume", "volume_pcr"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    df = frame.copy()
    lookup = {str(c).casefold(): c for c in df.columns}
    if "report_date" not in lookup or "option_type" not in lookup:
        return pd.DataFrame(columns=columns)
    date_col, type_col = lookup["report_date"], lookup["option_type"]
    oi_col, volume_col = lookup.get("open_interest"), lookup.get("volume")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[type_col] = df[type_col].astype(str).str.upper().str.strip()
    df = df[df[type_col].isin({"CE", "PE"}) & df[date_col].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    oi = pd.to_numeric(df[oi_col], errors="coerce") if oi_col else pd.Series(np.nan, index=df.index)
    vol = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else pd.Series(np.nan, index=df.index)
    df["_put_oi"] = np.where(df[type_col].eq("PE"), oi, 0.0)
    df["_call_oi"] = np.where(df[type_col].eq("CE"), oi, 0.0)
    df["_put_volume"] = np.where(df[type_col].eq("PE"), vol, 0.0)
    df["_call_volume"] = np.where(df[type_col].eq("CE"), vol, 0.0)
    result = df.groupby(date_col, as_index=False).agg(
        total_put_oi=("_put_oi", "sum"), total_call_oi=("_call_oi", "sum"),
        total_put_volume=("_put_volume", "sum"), total_call_volume=("_call_volume", "sum"),
    ).rename(columns={date_col: "report_date"})
    result["oi_pcr"] = result["total_put_oi"].div(result["total_call_oi"].where(result["total_call_oi"].ne(0)))
    result["volume_pcr"] = result["total_put_volume"].div(result["total_call_volume"].where(result["total_call_volume"].ne(0)))
    return result[columns].sort_values("report_date").reset_index(drop=True)


def select_representative_futures_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose one highest-OI futures contract per date; expiry breaks ties."""
    if frame is None:
        return pd.DataFrame()
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    lookup = {str(c).casefold(): c for c in df.columns}
    date_col = lookup.get("report_date")
    oi_col, expiry_col = lookup.get("open_interest"), lookup.get("expiry_date")
    if date_col is None:
        return df.iloc[0:0].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    sort_cols, ascending = [date_col], [True]
    if oi_col is not None:
        df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce")
        sort_cols.append(oi_col); ascending.append(False)
    if expiry_col is not None:
        df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")
        sort_cols.append(expiry_col); ascending.append(True)
    return (df.sort_values(sort_cols, ascending=ascending, na_position="last")
              .dropna(subset=[date_col]).drop_duplicates(date_col, keep="first")
              .sort_values(date_col).reset_index(drop=True))


def add_futures_basis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive basis only when the contract record includes an underlying price."""
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    if out.empty:
        return out
    lookup = {str(c).casefold(): c for c in out.columns}
    close_col, underlying_col = lookup.get("close"), lookup.get("underlying_price")
    if close_col is None or underlying_col is None:
        out["absolute_basis"] = np.nan
        out["basis_percentage"] = np.nan
        return out
    close = pd.to_numeric(out[close_col], errors="coerce")
    underlying = pd.to_numeric(out[underlying_col], errors="coerce")
    out["absolute_basis"] = close - underlying
    out["basis_percentage"] = ((close / underlying) - 1.0) * 100.0
    out.loc[underlying.eq(0) | underlying.isna() | close.isna(), "basis_percentage"] = np.nan
    out.loc[underlying.isna() | close.isna(), "absolute_basis"] = np.nan
    return out
