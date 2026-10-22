from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from data_access.db import get_connection
from data_access.ticker import normalize_ticker


def load_freshest_market_history(
    ticker: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Load daily prices from the freshest available ticker alias for the UI.

    The legacy data_access.market.get_daily_history function returns the first
    alias with any rows. If the canonical ticker is stale while its .NS alias is
    current, that behaviour can hide recent data. This app-scoped loader picks
    the alias with the latest non-null close date, breaking ties by row count.
    It does not alter the data-access behaviour used by existing backtests.
    """
    candidates = normalize_ticker(ticker)
    coverage_sql = '''
        SELECT
            MAX(CASE WHEN "Close" IS NOT NULL THEN "ReportDate" END) AS latest_date,
            COUNT("Close") AS rows_with_close
        FROM global_assets_daily
        WHERE "Ticker" = ?
    '''
    with get_connection() as conn:
        coverage: list[tuple[Any, int, int, str]] = []
        for priority, candidate in enumerate(candidates):
            row = conn.execute(coverage_sql, [candidate]).fetchone()
            if row and row[0] is not None:
                coverage.append((row[0], int(row[1] or 0), -priority, candidate))
        chosen = max(coverage)[-1] if coverage else candidates[0]

        sql = '''
            SELECT
                "Ticker" AS ticker,
                "ReportDate" AS report_date,
                "AssetClass" AS asset_class,
                "Open" AS open,
                "High" AS high,
                "Low" AS low,
                "Close" AS close,
                "Volume" AS volume
            FROM global_assets_daily
            WHERE "Ticker" = ?
        '''
        params: list[Any] = [chosen]
        if start_date is not None:
            sql += ' AND "ReportDate" >= ?'
            params.append(start_date.date() if isinstance(start_date, datetime) else start_date)
        if end_date is not None:
            sql += ' AND "ReportDate" <= ?'
            params.append(end_date.date() if isinstance(end_date, datetime) else end_date)
        sql += ' ORDER BY "ReportDate"'
        result = conn.execute(sql, params).fetch_df()
    return result
