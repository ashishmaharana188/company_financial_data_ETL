import requests
import pandas as pd
from sqlalchemy import text
from scripts.database import engine
import time
from functools import lru_cache


@lru_cache(maxsize=1)
def _fetch_sec_tickers():
    """Fetches and caches the SEC ticker list so it only downloads once per run."""
    headers = {"User-Agent": "SwarmAgent Admin@yourdomain.com"}
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching SEC tickers: {e}")
    return {}


def get_cik(ticker: str) -> str:
    # Strip exchange suffixes to ensure clean matching
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    sec_data = _fetch_sec_tickers()

    for item in sec_data.values():
        if item["ticker"].upper() == clean_ticker:
            return str(item["cik_str"]).zfill(10)
    return None


def get_structural_break_date(ticker: str):
    cik = get_cik(ticker)
    if not cik:
        return None

    headers = {"User-Agent": "SwarmAgent Admin@yourdomain.com"}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return None

    filings = response.json().get("filings", {}).get("recent", {})
    for form, date in zip(filings.get("form", []), filings.get("filingDate", [])):
        if form in ["10-12B", "15-12G", "15-12B"]:
            return date

    return None


def backfill_structural_breaks(target_tickers=None):
    print("Starting Corporate Action Audit...")

    with engine.begin() as conn:
        # 1. Use passed tickers if available, otherwise fetch all
        if target_tickers:
            tickers = target_tickers
        else:
            result = conn.execute(text('SELECT "Ticker" FROM market_metadata;'))
            tickers = [row[0] for row in result]

        for ticker in tickers:
            print(f"[{ticker}] Checking EDGAR for structural breaks...")

            # 2. Ping EDGAR
            break_date = get_structural_break_date(ticker)

            if break_date:
                print(
                    f"  -> WARNING: Break detected on {break_date}. Updating database."
                )
                # 3. Save the date to the database
                update_query = text(
                    """
                    UPDATE market_metadata
                    SET valid_data_since = :break_date 
                    WHERE "Ticker" = :ticker
                """
                )
                conn.execute(update_query, {"break_date": break_date, "ticker": ticker})
            else:
                print(f"  -> Clean history. No action required.")

            # Respect SEC rate limits
            time.sleep(1)

    print("Audit Complete.")
