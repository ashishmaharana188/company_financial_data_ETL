from datetime import date, datetime
from typing import Any
from .db import get_connection
from .ticker import normalize_ticker

def get_active_companies() -> list[dict[str, Any]]:
    """
    Return all active companies from market_metadata.
    """
    sql = """
        SELECT
            Ticker,
            IndicatorName,
            TargetTable,
            Sector,
            Industry,
            AssetClass,
            Exchange,
            IsActive,
            valid_data_since,
            Description
        FROM market_metadata
        WHERE IsActive = TRUE
        ORDER BY Ticker
    """

    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()

    columns = [
        "Ticker",
        "IndicatorName",
        "TargetTable",
        "Sector",
        "Industry",
        "AssetClass",
        "Exchange",
        "IsActive",
        "valid_data_since",
        "Description",
    ]

    return [dict(zip(columns, row)) for row in rows]


def get_company(ticker: str):
    candidates = normalize_ticker(ticker)

    sql = """
        SELECT
            Ticker,
            IndicatorName,
            TargetTable,
            Sector,
            Industry,
            AssetClass,
            Exchange,
            IsActive,
            valid_data_since,
            Description
        FROM market_metadata
        WHERE Ticker = ?
        LIMIT 1
    """

    with get_connection() as conn:
        for candidate in candidates:
            row = conn.execute(sql, [candidate]).fetchone()

            if row is not None:
                break
        else:
            return None

    columns = [
        "Ticker",
        "IndicatorName",
        "TargetTable",
        "Sector",
        "Industry",
        "AssetClass",
        "Exchange",
        "IsActive",
        "valid_data_since",
        "Description",
    ]

    return dict(zip(columns, row))


def get_companies_by_sector(
    sector: str,
    active_only: bool = True,
    as_of_date: str | date | datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Return companies belonging to a sector.
    """
    sql = """
        SELECT
            Ticker,
            IndicatorName,
            TargetTable,
            Sector,
            Industry,
            AssetClass,
            Exchange,
            IsActive,
            valid_data_since,
            Description
        FROM market_metadata
        WHERE Sector = ?
    """

    params = [sector]

    if active_only:
        sql += " AND IsActive = TRUE"

    if as_of_date is not None:
        sql += " AND (valid_data_since IS NULL OR valid_data_since <= ?)"
        params.append(as_of_date)

    sql += " ORDER BY Ticker"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    columns = [
        "Ticker",
        "IndicatorName",
        "TargetTable",
        "Sector",
        "Industry",
        "AssetClass",
        "Exchange",
        "IsActive",
        "valid_data_since",
        "Description",
    ]

    return [dict(zip(columns, row)) for row in rows]


def get_companies_by_industry(
    industry: str,
    active_only: bool = True,
    as_of_date: str | date | datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Return companies belonging to an industry.
    """
    sql = """
        SELECT
            Ticker,
            IndicatorName,
            TargetTable,
            Sector,
            Industry,
            AssetClass,
            Exchange,
            IsActive,
            valid_data_since,
            Description
        FROM market_metadata
        WHERE Industry = ?
    """

    params = [industry]

    if active_only:
        sql += "AND IsActive = TRUE"

    if as_of_date is not None:
        sql += " AND (valid_data_since IS NULL OR valid_data_since <= ?)"
        params.append(as_of_date)

    sql += " ORDER BY Ticker"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    columns = [
        "Ticker",
        "IndicatorName",
        "TargetTable",
        "Sector",
        "Industry",
        "AssetClass",
        "Exchange",
        "IsActive",
        "valid_data_since",
        "Description",
    ]

    return [dict(zip(columns, row)) for row in rows]


def get_sectors(active_only: bool = True) -> list[str]:
    """
    Return distinct sectors.
    """
    sql = """
        SELECT DISTINCT Sector
        FROM market_metadata
        WHERE Sector IS NOT NULL
    """

    if active_only:
        sql += " AND IsActive = TRUE"

    sql += " ORDER BY Sector"

    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()

    return [row[0] for row in rows]


def get_industries(
    sector: str | None = None,
    active_only: bool = True,
) -> list[str]:
    """
    Return distinct industries, optionally restricted to a sector.
    """
    sql = """
        SELECT DISTINCT Industry
        FROM market_metadata
        WHERE Industry IS NOT NULL
    """

    params: list[Any] = []

    if sector is not None:
        sql += " AND Sector = ?"
        params.append(sector)

    if active_only:
        sql += " AND IsActive = TRUE"

    sql += " ORDER BY Industry"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [row[0] for row in rows]