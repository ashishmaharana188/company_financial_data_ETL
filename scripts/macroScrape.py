"""Incremental macro/global-asset OHLC ingestion.

The live pipeline fetches and writes one symbol/timeframe at a time. This avoids
holding the entire macro and global-asset history in memory, which can restart
small hosted services before a database write begins.
"""
from __future__ import annotations

import time
from typing import Iterator

import pandas as pd
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval

from scripts.market_metadata import fetch_active_global_assets, upsert_market_metadata
from scripts.market_data_writer import MarketDataWriter, STANDARD_COLUMNS

YF_MACRO = {
    "US_10Y_Yield": "^TNX",
    "Brent_Crude": "BZ=F",
    "USD_INR": "INR=X",
    "US_Dollar_Index": "DX-Y.NYB",
    "Broad_Commodity": "DBC",
    "US_VIX": "^VIX",
    "Nifty_50": "^NSEI",
}
TV_MACRO = {
    "India_10Y_Yield": ("IN10Y", "TVC"),
    "India_CPI": ("INCPI", "ECONOMICS"),
    "India_VIX": ("INDIAVIX", "NSE"),
}
DEFAULT_INTERVALS = ["1d", "1h", "30m", "5m", "1m"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def register_discovered_tickers(tickers, data_source="auto"):
    """Upsert tickers into the same metadata target used by the downstream app."""
    records = []
    for value in tickers or []:
        ticker = str(value or "").strip().upper()
        if not ticker:
            continue
        if data_source.lower() in {"screener", "indianapi"} or ticker.endswith(".NS"):
            exchange = "NSE"
        elif ticker.endswith(".BO"):
            exchange = "BSE"
        else:
            try:
                raw = str(yf.Ticker(ticker).info.get("exchange", "NASDAQ")).upper()
            except Exception:
                raw = "NASDAQ"
            exchange = (
                "NSE" if raw in {"NSI", "NSE"}
                else "BSE" if raw in {"BSE", "BOM"}
                else "NYSE" if raw in {"NYQ", "NYSE"}
                else "NASDAQ"
            )
        records.append({
            "Ticker": ticker,
            "IndicatorName": ticker,
            "TargetTable": "global_assets",
            "AssetClass": "Equity",
            "Exchange": exchange,
            "IsActive": True,
            "Description": f"Auto-discovered via {data_source.upper()} ETL",
        })
    if not records:
        return 0
    count = upsert_market_metadata(records)
    print(f"SUCCESS: {count} ticker(s) synced to market_metadata.")
    return count


def get_active_global_assets() -> pd.DataFrame:
    assets = fetch_active_global_assets()
    if assets is None:
        raise RuntimeError("Active global assets query returned no result from market_metadata.")
    if not isinstance(assets, pd.DataFrame):
        assets = pd.DataFrame(assets)
    for column in ("Ticker", "AssetClass", "Exchange"):
        if column not in assets.columns:
            assets[column] = None
    print(f"Loaded {len(assets)} active global asset(s) from market_metadata.")
    return assets[["Ticker", "AssetClass", "Exchange"]]


def get_yf_period(interval: str) -> str:
    if interval == "1m":
        return "7d"
    if interval in {"5m", "15m", "30m"}:
        return "60d"
    if interval == "1h":
        return "730d"
    return "max"


def get_tv_interval(interval_str: str):
    mapping = {
        "1d": Interval.in_daily,
        "1h": Interval.in_1_hour,
        "30m": Interval.in_30_minute,
        "5m": Interval.in_5_minute,
        "1m": Interval.in_1_minute,
    }
    return mapping.get(interval_str, Interval.in_daily)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _clean_index(index) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index))
    if result.tz is not None:
        result = result.tz_localize(None)
    return result


def _normalise_ohlcv(raw: pd.DataFrame, entity: str, interval: str, category: str, asset_class=None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_frame()
    frame = raw.copy()
    rename = {}
    for source, target in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")):
        if source in frame.columns and target not in frame.columns:
            rename[source] = target
    frame.rename(columns=rename, inplace=True)
    frame.index = _clean_index(frame.index)
    frame.index.name = "ReportDate"
    frame = frame.reset_index()
    for column in OHLCV:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[["ReportDate"] + OHLCV].dropna(subset=["ReportDate", "Close"])
    frame["EntityName"] = entity
    frame["Timeframe"] = interval
    frame["Category"] = category
    frame["AssetClass"] = asset_class
    return frame[STANDARD_COLUMNS].reset_index(drop=True)


def _fetch_yf_batch(entity: str, ticker: str, interval: str, category: str, asset_class=None) -> pd.DataFrame:
    try:
        history = yf.Ticker(ticker).history(period=get_yf_period(interval), interval=interval)
        if history is None or history.empty:
            print(f" -> No YF data: {entity} ({ticker}) | {interval}.")
            return _empty_frame()
        frame = _normalise_ohlcv(history, entity, interval, category, asset_class)
        if not frame.empty:
            print(f" -> YF fetched {entity} | {interval}: {len(frame)} rows.")
        return frame
    except Exception as exc:
        print(f"    [ERROR] YF failed for {ticker} ({interval}): {exc}")
        return _empty_frame()


def _fetch_tv_batch(tv, entity: str, symbol: str, exchange: str, interval: str) -> pd.DataFrame:
    try:
        raw = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=get_tv_interval(interval),
            n_bars=1000,
        )
        if raw is None or raw.empty:
            print(f" -> No TradingView data: {entity} | {interval}.")
            return _empty_frame()
        frame = _normalise_ohlcv(raw, entity, interval, "MACRO", None)
        if not frame.empty:
            print(f" -> TV fetched {entity} | {interval}: {len(frame)} rows.")
        return frame
    except Exception as exc:
        print(f"    [ERROR] TV failed for {symbol} ({interval}): {exc}")
        return _empty_frame()


def _close_series(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    return frame.set_index("ReportDate")["Close"].sort_index()


def _yield_spread_frame(us_close: pd.Series | None, india_close: pd.Series | None, interval: str) -> pd.DataFrame:
    if us_close is None or india_close is None:
        return _empty_frame()
    aligned = pd.concat(
        [india_close.rename("India"), us_close.rename("US")], axis=1, join="outer"
    ).sort_index().ffill().dropna()
    if aligned.empty:
        return _empty_frame()
    frame = pd.DataFrame({
        "ReportDate": aligned.index,
        "Open": None,
        "High": None,
        "Low": None,
        "Close": aligned["India"] - aligned["US"],
        "Volume": None,
    })
    frame["EntityName"] = "Yield_Spread"
    frame["Timeframe"] = interval
    frame["Category"] = "MACRO"
    frame["AssetClass"] = None
    return frame[STANDARD_COLUMNS].dropna(subset=["Close"]).reset_index(drop=True)


def iter_macro_batches(intervals=None) -> Iterator[pd.DataFrame]:
    """Yield one small ticker/timeframe frame at a time for immediate persistence."""
    intervals = list(intervals or DEFAULT_INTERVALS)
    assets = get_active_global_assets()
    try:
        tv = TvDatafeed()
    except Exception as exc:
        print(f"[CRITICAL] TradingView connection failed entirely: {exc}")
        tv = None

    for interval in intervals:
        print(f"\n[MACRO] Fetching interval {interval}.")
        us_yield = None
        india_yield = None

        # Static macro instruments: fetch, capture only the small yield series,
        # yield to the writer, and allow the OHLC frame to be released.
        for name, ticker in YF_MACRO.items():
            frame = _fetch_yf_batch(name, ticker, interval, "MACRO", None)
            if name == "US_10Y_Yield":
                us_yield = _close_series(frame)
            if not frame.empty:
                yield frame
            time.sleep(0.2)

        # Global assets are intentionally fetched and persisted one ticker at a time.
        for row in assets.itertuples(index=False):
            raw_ticker = "" if pd.isna(row.Ticker) else str(row.Ticker).strip()
            if not raw_ticker or raw_ticker.lower() == "nan":
                continue
            exchange = "" if pd.isna(row.Exchange) else str(row.Exchange).upper()
            fetch_ticker = raw_ticker
            if exchange == "NSE" and not fetch_ticker.endswith(".NS"):
                fetch_ticker += ".NS"
            elif exchange == "BSE" and not fetch_ticker.endswith(".BO"):
                fetch_ticker += ".BO"
            frame = _fetch_yf_batch(
                raw_ticker, fetch_ticker, interval, "ASSET",
                ("Equity" if pd.isna(row.AssetClass) or not str(row.AssetClass).strip() else str(row.AssetClass).strip())
            )
            if not frame.empty:
                yield frame
            time.sleep(0.2)

        if tv is not None:
            for name, (symbol, exchange) in TV_MACRO.items():
                if name == "India_CPI" and interval != "1d":
                    continue
                frame = _fetch_tv_batch(tv, name, symbol, exchange, interval)
                if name == "India_10Y_Yield":
                    india_yield = _close_series(frame)
                if not frame.empty:
                    yield frame
                time.sleep(0.25)

        spread = _yield_spread_frame(us_yield, india_yield, interval)
        if not spread.empty:
            print(f" -> Calculated Yield_Spread | {interval}: {len(spread)} rows.")
            yield spread


def fetch_hybrid_macro_data(intervals=DEFAULT_INTERVALS, period_days=1000) -> pd.DataFrame:
    """Compatibility wrapper for callers expecting a single combined DataFrame.

    The live macro pipeline does not use this wrapper; it consumes the generator
    directly so the full history is never accumulated in memory.
    """
    batches = [batch for batch in iter_macro_batches(intervals) if not batch.empty]
    if not batches:
        return _empty_frame()
    result = pd.concat(batches, ignore_index=True)
    result = result.dropna(subset=["Close"])
    return result


def push_to_database(df: pd.DataFrame, writer: MarketDataWriter | None = None) -> int:
    """Compatibility entry point; writer commits groups incrementally."""
    own_writer = writer is None
    writer = writer or MarketDataWriter()
    try:
        count = 0
        if df is not None and not df.empty:
            count = writer.write_frame(df)
        print(f"[MARKET_DATA] Persisted {count} row(s) to {writer.destination}.")
        return count
    finally:
        if own_writer:
            writer.close()


def run_macro_pipeline(period_days=3000):
    intervals = DEFAULT_INTERVALS
    print("\nStarting incremental Macro & Global Asset Pipeline...")
    writer = MarketDataWriter()
    written_rows = 0
    batch_count = 0
    try:
        for batch in iter_macro_batches(intervals):
            if batch is None or batch.empty:
                continue
            count = writer.write_frame(batch)
            written_rows += count
            batch_count += 1
            print(
                f"[MACRO_PROGRESS] batch={batch_count} rows_written={count} "
                f"total_rows={written_rows} destination={writer.destination}."
            )
        if written_rows:
            print(
                f"[SUCCESS] Macro pipeline completed. batches={batch_count}, "
                f"rows_written={written_rows}, destination={writer.destination}."
            )
            return True, written_rows
        print("Pipeline aborted: no usable OHLC data was fetched; no data writes were made.")
        return False, 0
    finally:
        writer.close()


if __name__ == "__main__":
    success, rows = run_macro_pipeline()
    if not success:
        raise SystemExit(1)
