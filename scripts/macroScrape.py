import yfinance as yf
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from scripts.database import engine
from scripts.market_metadata import upsert_market_metadata, fetch_active_global_assets
import time
from datetime import datetime


def register_discovered_tickers(tickers, data_source="auto"):
    """Persist discovered tickers in the metadata store used by downstream jobs."""
    if not tickers:
        return 0

    print(f"\nRegistering discovered client tickers from {data_source.upper()}...")
    records = []
    for value in tickers:
        ticker = str(value or "").strip().upper()
        if not ticker:
            continue

        if data_source in ["screener", "indianapi"] or ticker.endswith(".NS"):
            exchange = "NSE"
        elif ticker.endswith(".BO"):
            exchange = "BSE"
        else:
            try:
                raw_exchange = str(yf.Ticker(ticker).info.get("exchange", "NASDAQ")).upper()
                exchange = (
                    "NSE" if raw_exchange in {"NSI", "NSE"}
                    else "BSE" if raw_exchange in {"BSE", "BOM"}
                    else "NYSE" if raw_exchange in {"NYQ", "NYSE"}
                    else "NASDAQ"
                )
            except Exception:
                exchange = "NASDAQ"

        records.append({
            "Ticker": ticker,
            "IndicatorName": ticker,
            "TargetTable": "global_assets",
            "Sector": None,
            "Industry": None,
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


def get_active_global_assets():
    """Read the registry from the same data store that receives metadata writes."""
    try:
        assets = fetch_active_global_assets()
    except Exception as exc:
        print(f"[ERROR] Failed to fetch active global assets from market_metadata: {exc}")
        raise
    print(f"Loaded {len(assets)} active global asset(s) from market_metadata.")
    return assets

def get_yf_period(interval):
    if interval == "1m":
        return "7d"
    elif interval in ["5m", "15m", "30m"]:
        return "60d"
    elif interval == "1h":
        return "730d"
    return "max"


def get_tv_interval(interval_str):
    mapping = {
        "1d": Interval.in_daily,
        "1h": Interval.in_1_hour,
        "30m": Interval.in_30_minute,
        "5m": Interval.in_5_minute,
        "1m": Interval.in_1_minute,
    }
    return mapping.get(interval_str, Interval.in_daily)


def fetch_hybrid_macro_data(
    intervals=["1d", "1h", "30m", "5m", "1m"], period_days=1000
):
    print("Initializing Hybrid Multi-Timeframe Spigots...")
    raw_data_frames = []

    req_cols = ["Open", "High", "Low", "Close", "Volume"]

    # --- 1. DEFINE TARGETS (Macro Indicators) ---
    yf_macro = {
        "US_10Y_Yield": "^TNX",
        "Brent_Crude": "BZ=F",
        "USD_INR": "INR=X",
        "US_Dollar_Index": "DX-Y.NYB",
        "Broad_Commodity": "DBC",
        "US_VIX": "^VIX",
        "Nifty_50": "^NSEI",
    }

    tv_macro = {
        "India_10Y_Yield": ("IN10Y", "TVC"),
        "India_CPI": ("INCPI", "ECONOMICS"),
        "India_VIX": ("INDIAVIX", "NSE"),
    }

    # --- 2. FETCH MACRO (YF) ---
    for name, ticker in yf_macro.items():
        for interval in intervals:
            try:
                print(f" -> YF Macro Fetch: {name} | {interval}...")
                tick = yf.Ticker(ticker)
                hist = tick.history(period=get_yf_period(interval), interval=interval)

                if not hist.empty:
                    df = hist.copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "ReportDate"
                    df.reset_index(inplace=True)

                    df["EntityName"] = name
                    df["Timeframe"] = interval
                    df["Category"] = "MACRO"
                    df["AssetClass"] = None

                    for col in req_cols:
                        if col not in df.columns:
                            df[col] = None

                    raw_data_frames.append(
                        df[
                            [
                                "EntityName",
                                "ReportDate",
                                "Timeframe",
                                "Category",
                                "AssetClass",
                            ]
                            + req_cols
                        ]
                    )
            except Exception as e:
                print(f"    [ERROR] YF failed for {ticker} ({interval}): {e}")
            time.sleep(0.5)

    # --- 3. FETCH GLOBAL ASSETS (YF) ---
    global_assets_df = get_active_global_assets()
    for _, row in global_assets_df.iterrows():
        raw_ticker = row["Ticker"]
        asset_class = row["AssetClass"]
        exchange = row["Exchange"]

        ticker = raw_ticker
        if exchange == "NSE" and not ticker.endswith(".NS"):
            ticker = f"{raw_ticker}.NS"
        elif exchange == "BSE" and not ticker.endswith(".BO"):
            ticker = f"{raw_ticker}.BO"

        for interval in intervals:
            try:
                print(f" -> YF Asset Fetch: {ticker} ({exchange}) | {interval}...")
                tick = yf.Ticker(ticker)
                hist = tick.history(period=get_yf_period(interval), interval=interval)

                if not hist.empty:
                    df = hist.copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "ReportDate"
                    df.reset_index(inplace=True)

                    df["EntityName"] = raw_ticker
                    df["Timeframe"] = interval
                    df["Category"] = "ASSET"
                    df["AssetClass"] = asset_class

                    for col in req_cols:
                        if col not in df.columns:
                            df[col] = None

                    raw_data_frames.append(
                        df[
                            [
                                "EntityName",
                                "ReportDate",
                                "Timeframe",
                                "Category",
                                "AssetClass",
                            ]
                            + req_cols
                        ]
                    )
            except Exception as e:
                print(f"    [ERROR] YF failed for {ticker} ({interval}): {e}")
            time.sleep(0.5)

    # --- 4. FETCH MACRO (TV) ---
    try:
        tv = TvDatafeed()
        for name, (symbol, exchange) in tv_macro.items():
            for interval in intervals:
                if "CPI" in name and interval != "1d":
                    continue

                try:
                    print(f" -> TV Macro Fetch: {name} | {interval}...")
                    tv_data = tv.get_hist(
                        symbol=symbol,
                        exchange=exchange,
                        interval=get_tv_interval(interval),
                        n_bars=1000,
                    )

                    if tv_data is not None and not tv_data.empty:
                        df = tv_data.copy()
                        df.index = pd.to_datetime(df.index)
                        df.index.name = "ReportDate"
                        df.reset_index(inplace=True)
                        df.rename(
                            columns={
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume",
                            },
                            inplace=True,
                        )

                        df["EntityName"] = name
                        df["Timeframe"] = interval
                        df["Category"] = "MACRO"
                        df["AssetClass"] = None

                        for col in req_cols:
                            if col not in df.columns:
                                df[col] = None

                        raw_data_frames.append(
                            df[
                                [
                                    "EntityName",
                                    "ReportDate",
                                    "Timeframe",
                                    "Category",
                                    "AssetClass",
                                ]
                                + req_cols
                            ]
                        )
                except Exception as e:
                    print(f"    [ERROR] TV failed for {symbol} ({interval}): {e}")
                time.sleep(1)
    except Exception as e:
        print(f"    [CRITICAL] TradingView connection failed entirely: {e}")

    # --- 5. SYNTHESIZE YIELD SPREAD ---
    if raw_data_frames:
        master_df = pd.concat(raw_data_frames, ignore_index=True)
        print("\\nCalculating Yield Spreads across timeframes...")
        spread_frames = []

        for interval in intervals:
            df_us = master_df[
                (master_df["EntityName"] == "US_10Y_Yield")
                & (master_df["Timeframe"] == interval)
            ].set_index("ReportDate")
            df_in = master_df[
                (master_df["EntityName"] == "India_10Y_Yield")
                & (master_df["Timeframe"] == interval)
            ].set_index("ReportDate")

            if not df_us.empty and not df_in.empty:
                aligned = df_in[["Close"]].join(
                    df_us[["Close"]], rsuffix="_us", how="outer"
                )
                aligned.ffill(inplace=True)
                aligned.dropna(inplace=True)

                spread = aligned["Close"] - aligned["Close_us"]
                spread_df = spread.to_frame("Close").reset_index()

                spread_df["EntityName"] = "Yield_Spread"
                spread_df["Timeframe"] = interval
                spread_df["Category"] = "MACRO"
                spread_df["AssetClass"] = None

                for col in ["Open", "High", "Low", "Volume"]:
                    spread_df[col] = None
                spread_frames.append(spread_df)

        if spread_frames:
            master_df = pd.concat([master_df] + spread_frames, ignore_index=True)

        return master_df.dropna(subset=["Close"])

    return pd.DataFrame()


def push_to_database(df):
    print("\\nRouting Data to strict Time-Series Architectures...")

    daily_df = df[df["Timeframe"] == "1d"].copy()
    intraday_df = df[df["Timeframe"] != "1d"].copy()

    # Create safe copies and format dates for DuckDB Casts
    daily_df["ReportDate"] = pd.to_datetime(daily_df["ReportDate"]).dt.date.astype(str)
    intraday_df["ReportDate"] = pd.to_datetime(intraday_df["ReportDate"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    daily_df = daily_df.where(pd.notnull(daily_df), None)
    intraday_df = intraday_df.where(pd.notnull(intraday_df), None)

    # 1. MACRO DAILY LEDGER
    macro_daily = daily_df[daily_df["Category"] == "MACRO"].copy()
    if not macro_daily.empty:
        macro_daily.rename(
            columns={"EntityName": "IndicatorName", "Close": "Close_Value"},
            inplace=True,
        )
        print(f" -> Pushing {len(macro_daily)} records to macro_daily_ledger...")

        # DUCKDB FIX: Native Upsert execution
        engine.register("temp_md", macro_daily)
        engine.execute(
            """
            INSERT INTO macro_daily_ledger ("IndicatorName", "ReportDate", "Open", "High", "Low", "Close_Value", "Volume")
            SELECT "IndicatorName", CAST("ReportDate" AS DATE), "Open", "High", "Low", "Close_Value", "Volume" FROM temp_md
            ON CONFLICT ("IndicatorName", "ReportDate") DO UPDATE SET 
                "Open"=EXCLUDED."Open", "High"=EXCLUDED."High", "Low"=EXCLUDED."Low", "Close_Value"=EXCLUDED."Close_Value", "Volume"=EXCLUDED."Volume";
        """
        )
        engine.unregister("temp_md")

    # 2. MACRO INTRADAY LEDGER
    macro_intra = intraday_df[intraday_df["Category"] == "MACRO"].copy()
    if not macro_intra.empty:
        macro_intra.rename(
            columns={"EntityName": "IndicatorName", "Close": "Close_Value"},
            inplace=True,
        )

        print(f" -> Pushing {len(macro_intra)} records to macro_intraday_ledger...")
        engine.register("temp_mi", macro_intra)

        # FIX: Removed split ReportTime column, using full TIMESTAMP cast for ReportDate
        engine.execute(
            """
            INSERT INTO macro_intraday_ledger ("IndicatorName", "ReportDate", "Timeframe", "Open", "High", "Low", "Close_Value", "Volume")
            SELECT "IndicatorName", CAST("ReportDate" AS TIMESTAMP), "Timeframe", "Open", "High", "Low", "Close_Value", "Volume" FROM temp_mi
            ON CONFLICT ("IndicatorName", "ReportDate", "Timeframe") DO UPDATE SET 
                "Open"=EXCLUDED."Open", "High"=EXCLUDED."High", "Low"=EXCLUDED."Low", "Close_Value"=EXCLUDED."Close_Value", "Volume"=EXCLUDED."Volume";
        """
        )
        engine.unregister("temp_mi")

    # 3. GLOBAL ASSETS DAILY
    asset_daily = daily_df[daily_df["Category"] == "ASSET"].copy()
    if not asset_daily.empty:
        asset_daily.rename(columns={"EntityName": "Ticker"}, inplace=True)
        print(f" -> Pushing {len(asset_daily)} records to global_assets_daily...")

        engine.register("temp_gd", asset_daily)
        engine.execute(
            """
            INSERT INTO global_assets_daily ("Ticker", "ReportDate", "AssetClass", "Open", "High", "Low", "Close", "Volume")
            SELECT "Ticker", CAST("ReportDate" AS DATE), "AssetClass", "Open", "High", "Low", "Close", "Volume" FROM temp_gd
            ON CONFLICT ("Ticker", "ReportDate") DO UPDATE SET 
                "Open"=EXCLUDED."Open", "High"=EXCLUDED."High", "Low"=EXCLUDED."Low", "Close"=EXCLUDED."Close", "Volume"=EXCLUDED."Volume";
        """
        )
        engine.unregister("temp_gd")

    # 4. GLOBAL ASSETS INTRADAY
    asset_intra = intraday_df[intraday_df["Category"] == "ASSET"].copy()
    if not asset_intra.empty:
        asset_intra.rename(columns={"EntityName": "Ticker"}, inplace=True)
        print(f" -> Pushing {len(asset_intra)} records to global_assets_intraday...")

        engine.register("temp_gi", asset_intra)
        engine.execute(
            """
            INSERT INTO global_assets_intraday ("Ticker", "ReportDate", "Timeframe", "Open", "High", "Low", "Close", "Volume")
            SELECT "Ticker", CAST("ReportDate" AS TIMESTAMP), "Timeframe", "Open", "High", "Low", "Close", "Volume" FROM temp_gi
            ON CONFLICT ("Ticker", "ReportDate", "Timeframe") DO UPDATE SET 
                "Open"=EXCLUDED."Open", "High"=EXCLUDED."High", "Low"=EXCLUDED."Low", "Close"=EXCLUDED."Close", "Volume"=EXCLUDED."Volume";
        """
        )
        engine.unregister("temp_gi")

    print("[SUCCESS] All Macro Pipeline Data Successfully Routed and Upserted.")


def run_macro_pipeline(period_days=3000):
    print(f"\\nStarting Macro Pipeline...")
    target_intervals = ["1d", "1h", "30m", "5m", "1m"]
    final_df = fetch_hybrid_macro_data(
        intervals=target_intervals, period_days=period_days
    )

    if not final_df.empty:
        push_to_database(final_df)
        return True, len(final_df)
    else:
        print("Pipeline aborted: No data extracted.")
        return False, 0


if __name__ == "__main__":
    success, rows = run_macro_pipeline()
