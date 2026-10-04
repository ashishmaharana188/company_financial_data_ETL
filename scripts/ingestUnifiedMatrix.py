import os
import glob
import json
import zipfile
import polars as pl
from datetime import datetime
from scripts.database import engine
import uuid
import logging
import re

CACHE_DIR = "offline_data_cache/master_archives"

# High-Speed Compact Logger
logging.basicConfig(
    filename="ingestion_audit.log",
    level=logging.INFO,
    format="%(asctime)s | FILE: %(file_name)-40s | RAW: %(raw)-8s | DEDUP: %(dedup)-8s | PUSH: %(push)-8s | STATUS: %(status)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def log_audit(file_name, raw, dedup, push, status):
    logging.info(
        "",
        extra={
            "file_name": os.path.basename(str(file_name)),
            "raw": raw,
            "dedup": dedup,
            "push": push,
            "status": status,
        },
    )


def _read_csv_safe(file_path_or_bytes):
    try:
        return pl.read_csv(
            file_path_or_bytes,
            encoding="utf8-lossy",
            ignore_errors=True,
            infer_schema_length=0,
        )
    except Exception as e:
        print(f"    [!] Format Error reading {file_path_or_bytes}: {e}")
        return pl.DataFrame()


def parse_cash_and_shorts(cash_file, preloaded_short_df=None):
    df_cash = _read_csv_safe(cash_file)
    if df_cash.is_empty():
        return pl.DataFrame()

    # Standardize columns
    df_cash = df_cash.rename({c: c.strip().upper() for c in df_cash.columns})

    # Ensure DATE1 exists
    if "DATE1" not in df_cash.columns:
        return pl.DataFrame()

    df_cash = df_cash.with_columns(
        pl.col("DATE1")
        .str.strip_chars()
        .str.strptime(pl.Date, format="%d-%b-%Y", strict=False)
        .alias("ReportDate")
    ).filter(pl.col("ReportDate").is_not_null())

    rename_map = {
        "SYMBOL": "Ticker",
        "SERIES": "Exchange_Series",
        "OPEN_PRICE": "Open",
        "HIGH_PRICE": "High",
        "LOW_PRICE": "Low",
        "CLOSE_PRICE": "Close",
        "TTL_TRD_QNTY": "Volume",
        "TURNOVER_LACS": "Turnover",
        "NO_OF_TRADES": "No_Of_Trades",
        "DELIV_QTY": "Delivery_Qty",
        "DELIV_PER": "Delivery_Percentage",
    }

    df_cash = df_cash.select(
        [
            pl.col(old).alias(new)
            for old, new in rename_map.items()
            if old in df_cash.columns
        ]
        + [pl.col("ReportDate")]
    )

    df_cash = df_cash.with_columns(
        [
            pl.lit("CASH").alias("InstrumentType"),
            pl.lit(datetime(2099, 12, 31).date()).alias("ExpiryDate"),
            pl.lit(0.0).alias("StrikePrice"),
        ]
    )

    if preloaded_short_df is not None and not preloaded_short_df.is_empty():
        df_cash = df_cash.join(
            preloaded_short_df.select(["ReportDate", "Ticker", "Short_Volume"]),
            on=["ReportDate", "Ticker"],
            how="left",
        )
    else:
        df_cash = df_cash.with_columns(
            pl.lit(None).cast(pl.Int64).alias("Short_Volume")
        )

    return df_cash


def parse_modern_fo_df(df):
    df = df.rename({c: c.strip() for c in df.columns})

    if "TradDt" not in df.columns or "XpryDt" not in df.columns:
        return pl.DataFrame()

    df = df.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("TradDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                    pl.col("TradDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d-%m-%Y", strict=False),
                    pl.col("TradDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.to_titlecase()
                    .str.strptime(pl.Date, format="%d-%b-%Y", strict=False),
                    pl.col("TradDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                    pl.col("TradDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y%m%d", strict=False),
                ]
            ).alias("ReportDate"),
            pl.coalesce(
                [
                    pl.col("XpryDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                    pl.col("XpryDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d-%m-%Y", strict=False),
                    pl.col("XpryDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.to_titlecase()
                    .str.strptime(pl.Date, format="%d-%b-%Y", strict=False),
                    pl.col("XpryDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                    pl.col("XpryDt")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y%m%d", strict=False),
                ]
            ).alias("ExpiryDate"),
        ]
    )

    col_mapping = {
        "TckrSymb": "Ticker",
        "FinInstrmTp": "InstrumentType",
        "StrkPric": "StrikePrice",
        "OptnTp": "OptionType",
        "OpnPric": "Open",
        "HghPric": "High",
        "LwPric": "Low",
        "ClsPric": "Close",
        "TtlTradgVol": "Volume",
        "TtlTrfVal": "Turnover",
        "TtlNbOfTxsExctd": "No_Of_Trades",
        "OpnIntrst": "Open_Interest",
        "ChngInOpnIntrst": "Change_In_OI",
        "SttlmPric": "Settlement_Price",
        "UndrlygPric": "Underlying_Price",
    }

    cols_to_select = [
        pl.col(k).alias(v) for k, v in col_mapping.items() if k in df.columns
    ] + [pl.col("ReportDate"), pl.col("ExpiryDate")]
    df = df.select(cols_to_select)

    if "InstrumentType" not in df.columns and "FinInstrmNm" in df.columns:
        df = df.with_columns(
            pl.col("FinInstrmNm").str.slice(0, 20).alias("InstrumentType")
        )

    return df


def parse_legacy_fo_df(df):
    df = df.rename({c: c.strip().upper() for c in df.columns})

    if "TIMESTAMP" not in df.columns or "EXPIRY_DT" not in df.columns:
        return pl.DataFrame()

    df = df.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("TIMESTAMP")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                    pl.col("TIMESTAMP")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d-%m-%Y", strict=False),
                    pl.col("TIMESTAMP")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.to_titlecase()
                    .str.strptime(pl.Date, format="%d-%b-%Y", strict=False),
                    pl.col("TIMESTAMP")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y%m%d", strict=False),
                ]
            ).alias("ReportDate"),
            pl.coalesce(
                [
                    pl.col("EXPIRY_DT")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                    pl.col("EXPIRY_DT")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d-%m-%Y", strict=False),
                    pl.col("EXPIRY_DT")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .str.to_titlecase()
                    .str.strptime(pl.Date, format="%d-%b-%Y", strict=False),
                    pl.col("EXPIRY_DT")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%Y%m%d", strict=False),
                ]
            ).alias("ExpiryDate"),
        ]
    )

    rename_map = {
        "SYMBOL": "Ticker",
        "INSTRUMENT": "InstrumentType",
        "STRIKE_PR": "StrikePrice",
        "OPTION_TYP": "OptionType",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "CONTRACTS": "Volume",
        "VAL_INLAKH": "Turnover",
        "OPEN_INT": "Open_Interest",
        "CHG_IN_OI": "Change_In_OI",
        "SETTLE_PR": "Settlement_Price",
    }

    cols_to_select = [
        pl.col(k).alias(v) for k, v in rename_map.items() if k in df.columns
    ] + [pl.col("ReportDate"), pl.col("ExpiryDate")]
    df = df.select(cols_to_select)

    if "OptionType" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("OptionType") == "XX")
            .then(None)
            .otherwise(pl.col("OptionType"))
            .alias("OptionType")
        )

    return df


def parse_mcx(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception:
        return pl.DataFrame()

    # 1. Aggressively unpack JSON to find the underlying array of records
    if isinstance(data, dict):
        if "d" in data and isinstance(data["d"], dict) and "Data" in data["d"]:
            data = data["d"]["Data"]
        elif "data" in data:
            data = data["data"]
        elif "Data" in data:
            data = data["Data"]
        else:
            # Auto-detect any top-level list if standard keys fail
            extracted_list = [v for k, v in data.items() if isinstance(v, list)]
            if extracted_list:
                data = extracted_list[0]
            else:
                data = [data]

    if not data:
        return pl.DataFrame()

    df = pl.DataFrame(data)
    if df.is_empty():
        return pl.DataFrame()

    df = df.rename({c: c.strip() for c in df.columns})

    # 2. Dynamic Column Detection
    # Protects against JSON key rotations
    date_candidates = [
        "Date",
        "date",
        "TradeDate",
        "trade_date",
        "Trade Date",
        "ReportDate",
    ]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if not date_col:
        # File is an API error payload or holiday response with no date keys
        return pl.DataFrame()

    if date_col != "Date":
        df = df.rename({date_col: "Date"})

    expiry_candidates = [
        "ExpiryDate",
        "expiryDate",
        "Expiry Date",
        "expiry_date",
        "Expiry",
    ]
    expiry_col = next((c for c in expiry_candidates if c in df.columns), None)

    if expiry_col and expiry_col != "ExpiryDate":
        df = df.rename({expiry_col: "ExpiryDate"})

    # 3. Universal Date Parser Block
    def parse_dates_safely(column):
        return pl.coalesce(
            [
                # Standard ISO / Timestamps (2026-05-25 or 2026-05-25T15:30)
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.slice(0, 10)
                .str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                # US Format (Unmasked by the May 29th payload: 05/29/2026)
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.strptime(pl.Date, format="%m/%d/%Y", strict=False),
                # Spaced or Dashed Indian Formats
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.strptime(pl.Date, format="%d-%m-%Y", strict=False),
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                # Textually named months (Requires titlecase normalization)
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .str.to_titlecase()
                .str.strptime(pl.Date, format="%d-%b-%Y", strict=False),
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .str.to_titlecase()
                .str.strptime(pl.Date, format="%d %b %Y", strict=False),
                # Condensed Formats (Highly common in MCX: 25May2026, 25052026)
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
                .str.to_titlecase()
                .str.strptime(pl.Date, format="%d%b%Y", strict=False),
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.strptime(pl.Date, format="%d%m%Y", strict=False),
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .str.strptime(pl.Date, format="%Y%m%d", strict=False),
            ]
        )

    # Apply Parser
    cols_to_assign = [parse_dates_safely("Date").alias("ReportDate")]
    if "ExpiryDate" in df.columns:
        cols_to_assign.append(parse_dates_safely("ExpiryDate").alias("ExpiryDate"))

    df = df.with_columns(cols_to_assign)

    # 4. Standardize Remaining Columns
    rename_map = {
        "Symbol": "Ticker",
        "InstrumentName": "InstrumentType",
        "StrikePrice": "StrikePrice",
        "OptionType": "OptionType",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
        "Value": "Turnover",
        "OpenInterest": "Open_Interest",
    }

    cols_to_select = [
        pl.col(k).alias(v) for k, v in rename_map.items() if k in df.columns
    ] + [pl.col("ReportDate")]
    if "ExpiryDate" in df.columns:
        cols_to_select.append(pl.col("ExpiryDate"))

    df = df.select(cols_to_select)

    if "StrikePrice" in df.columns:
        df = df.with_columns(pl.col("StrikePrice").fill_null("0.0"))

    return df


def push_chunk_to_db(df, file_name="Unknown_File"):
    if df.is_empty():
        log_audit(file_name, 0, 0, 0, "SKIPPED_EMPTY")
        return

    raw_count = df.height

    final_columns = [
        "Ticker",
        "ReportDate",
        "InstrumentType",
        "ExpiryDate",
        "StrikePrice",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Exchange_Series",
        "Turnover",
        "No_Of_Trades",
        "Delivery_Qty",
        "Delivery_Percentage",
        "Short_Volume",
        "OptionType",
        "Open_Interest",
        "Change_In_OI",
        "Settlement_Price",
        "Underlying_Price",
    ]

    # Add missing columns as null
    for col in final_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    df = df.select(final_columns)

    # Clean empty strings and dashes
    # Clean empty strings and dashes
    df = df.with_columns(
        [
            pl.when(pl.col(pl.String).str.strip_chars() == "-")
            .then(None)
            .otherwise(pl.col(pl.String))
            .name.keep()
        ]
    )

    debug_row = df.filter(
        (pl.col("Ticker") == "RELIANCE")
        & (pl.col("ReportDate") == datetime(2026, 9, 18).date())
    )

    if not debug_row.is_empty():
        row = debug_row.row(0, named=True)

        print("[DEBUG RAW VALUES]")
        print("Open:", repr(row["Open"]))
        print("High:", repr(row["High"]))
        print("Low:", repr(row["Low"]))
        print("Close:", repr(row["Close"]))
        print("Turnover:", repr(row["Turnover"]))

    # Cast Floats
    float_cols = [
        "StrikePrice",
        "Open",
        "High",
        "Low",
        "Close",
        "Turnover",
        "Delivery_Percentage",
        "Settlement_Price",
        "Underlying_Price",
    ]
    df = df.with_columns(
        [
            pl.col(c)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            for c in float_cols
            if c in df.columns
        ]
    )

    # Cast Ints
    int_cols = [
        "Volume",
        "No_Of_Trades",
        "Delivery_Qty",
        "Short_Volume",
        "Open_Interest",
        "Change_In_OI",
    ]
    df = df.with_columns(
        [
            pl.col(c)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(",", "")
            .str.replace_all(r"\.0$", "")
            .cast(pl.Int64, strict=False)
            for c in int_cols
            if c in df.columns
        ]
    )

    # Standardize Nulls for Composite Keys
    pk_cols = [
        "Ticker",
        "ReportDate",
        "InstrumentType",
        "ExpiryDate",
        "StrikePrice",
        "OptionType",
        "Exchange_Series",
    ]
    df = df.with_columns(
        [
            pl.col("OptionType").fill_null("XX"),
            pl.col("Exchange_Series").fill_null("XX"),
            pl.col("StrikePrice").fill_null(0.0),
        ]
    )

    # Needs Date objects not null for PK
    df = df.filter(pl.col("ReportDate").is_not_null())

    df = df.unique(subset=pk_cols, keep="last")
    dedup_count = df.height

    print(f"    -> [BULK PUSH] Zero-Copy Native Upsert for {dedup_count} rows...")

    temp_view = f"temp_master_{uuid.uuid4().hex[:8]}"
    try:
        debug_row = df.filter(
            (pl.col("Ticker") == "RELIANCE")
            & (pl.col("ReportDate") == datetime(2026, 9, 18).date())
        )

        if not debug_row.is_empty():
            row = debug_row.row(0, named=True)

            print(
                "[DEBUG]",
                row["Ticker"],
                row["ReportDate"],
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"],
                row["Turnover"],
                row["No_Of_Trades"],
                row["Delivery_Qty"],
                row["Delivery_Percentage"],
            )

        arrow_table = df.to_arrow()
        engine.register(temp_view, arrow_table)

        select_cols = []
        for col in final_columns:
            if col in ["Ticker", "InstrumentType", "OptionType", "Exchange_Series"]:
                select_cols.append(f'TRIM("{col}") AS "{col}"')
            else:
                select_cols.append(f'"{col}"')

        select_clause = ", ".join(select_cols)
        insert_cols = ", ".join([f'"{c}"' for c in final_columns])
        set_clause = ", ".join(
            [
                f'"{col}" = EXCLUDED."{col}"'
                for col in final_columns
                if col not in pk_cols
            ]
        )

        upsert_query = f"""
            INSERT INTO unified_market_master ({insert_cols})
            SELECT {select_clause} FROM {temp_view}
            ON CONFLICT ({", ".join([f'"{k}"' for k in pk_cols])})
            DO UPDATE SET {set_clause};
        """

        engine.execute(upsert_query)
        log_audit(file_name, raw_count, dedup_count, dedup_count, "SUCCESS")

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:80]
        log_audit(file_name, raw_count, dedup_count, 0, f"FAILED: {error_msg}")
        print(f"    [-] CRITICAL Bulk Push Error: {e}")

    finally:
        try:
            engine.unregister(temp_view)
        except:
            pass


def execute_pipeline(start_date_str="1900-01-01"):
    resume_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    cash_files = glob.glob(os.path.join(CACHE_DIR, "nse_cash_*.csv"))
    short_files = glob.glob(os.path.join(CACHE_DIR, "nse_short_selling_*.csv"))
    fo_zips = glob.glob(os.path.join(CACHE_DIR, "nse_fo_bhav_*.zip"))
    mcx_files = glob.glob(os.path.join(CACHE_DIR, "mcx_bhav_*.json"))

    total_cash = len(cash_files)
    total_fo = len(fo_zips)
    total_mcx = len(mcx_files)

    print(
        f"[*] Discovered Data: {total_cash} Cash CSVs | {total_fo} F&O Zips | {total_mcx} MCX JSONs."
    )
    print(f"[*] Resume Checkpoint: {resume_date}")

    def extract_date(filepath):
        match = re.search(r"(\d{8})", filepath)
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, "%d%m%Y").date()
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%Y%m%d").date()
                except ValueError:
                    pass
        return datetime.strptime("1900-01-01", "%Y-%m-%d").date()

    cash_files = sorted(cash_files, key=extract_date)
    fo_zips = sorted(fo_zips, key=extract_date)
    mcx_files = sorted(mcx_files, key=extract_date)

    global_short_df = None
    if short_files:
        print("[*] Pre-loading Master Short Selling Inventory into RAM...")
        df_short = _read_csv_safe(short_files[0])
        if not df_short.is_empty():
            df_short = df_short.rename({c: c.strip() for c in df_short.columns})
            if "Date" in df_short.columns:
                df_short = df_short.with_columns(
                    pl.col("Date")
                    .str.strip_chars()
                    .str.strptime(pl.Date, format="%d-%m-%Y", strict=False)
                    .alias("ReportDate")
                ).filter(pl.col("ReportDate").is_not_null())

                global_short_df = df_short.rename(
                    {"Symbol": "Ticker", "Quantity": "Short_Volume"}
                )

    # 1. PARSE CASH & SHORT SELLING
    for idx, cash_file in enumerate(cash_files, 1):
        if extract_date(cash_file) < resume_date:
            continue
        print(
            f"[*] Parsing Cash [ {idx} / {total_cash} ] : {os.path.basename(cash_file)}"
        )
        push_chunk_to_db(parse_cash_and_shorts(cash_file, global_short_df), cash_file)

    # 2. PARSE F&O
    for idx, z_path in enumerate(fo_zips, 1):
        if extract_date(z_path) < resume_date:
            continue
        print(
            f"[*] Parsing F&O Zip [ {idx} / {total_fo} ] : {os.path.basename(z_path)}"
        )
        try:
            with zipfile.ZipFile(z_path, "r") as z:
                for file_name in z.namelist():
                    if file_name.startswith("BhavCopy_NSE_FO") and file_name.endswith(
                        ".csv"
                    ):
                        with z.open(file_name) as f:
                            push_chunk_to_db(
                                parse_modern_fo_df(_read_csv_safe(f.read())), file_name
                            )
                    elif file_name.startswith("fo") and file_name.endswith("bhav.csv"):
                        with z.open(file_name) as f:
                            push_chunk_to_db(
                                parse_legacy_fo_df(_read_csv_safe(f.read())), file_name
                            )
        except zipfile.BadZipFile:
            print(f"    [-] Corrupted Zip File skipped: {z_path}")

    # 3. PARSE MCX
    for idx, mcx_file in enumerate(mcx_files, 1):
        if extract_date(mcx_file) < resume_date:
            continue
        print(f"[*] Parsing MCX [ {idx} / {total_mcx} ] : {os.path.basename(mcx_file)}")
        push_chunk_to_db(parse_mcx(mcx_file), mcx_file)

    print("[SUCCESS] Unified Matrix Update Complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="1900-01-01")
    args = parser.parse_args()
    execute_pipeline(args.start)
