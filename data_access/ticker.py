from __future__ import annotations


# ============================================================
# CANONICAL TICKER NORMALIZATION
# ============================================================

KNOWN_EXCHANGE_SUFFIXES = (
    ".NS",
    ".BO",
)

def normalize_ticker(ticker: str) -> list[str]:
    """
    Return ticker lookup candidates.

    OLAP/Screener generally use the canonical ticker (e.g. RELIANCE),
    while Yahoo Finance may use the NSE suffix (e.g. RELIANCE.NS).

    The canonical ticker is always tried first, with .NS as fallback.
    """
    ticker = ticker.strip().upper()

    if ticker.endswith(".NS"):
        base_ticker = ticker[:-3]
        return [base_ticker, ticker]

    return [ticker, f"{ticker}.NS"]