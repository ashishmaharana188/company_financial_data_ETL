"""DuckDB query helpers compatible with current DuckDB Python APIs.

Keep Arrow conversion on DuckDB relations returned by ``connection.sql``.
DuckDB 2.x removed ``fetch_arrow_table`` and ``fetch_record_batch`` from
``connection.execute`` results, which are connection objects rather than
separate result objects. The relational API also avoids an unnecessary full
result copy for these Arrow reads.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from .db import get_connection


def _relation(sql: str, params: list[Any] | tuple[Any, ...] | None):
    """Build a parameterized DuckDB relation for a read query."""
    con = get_connection()
    try:
        # The relation owns the query plan, not the connection; materialize the
        # Arrow result in the caller while the connection remains open.
        return con, con.sql(sql, params=params or [])
    except Exception:
        con.close()
        raise


def execute_arrow(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
) -> pa.Table:
    """Execute a read query and return its complete result as an Arrow table.

    Intended for metadata and moderate-sized histories. Use ``stream_arrow``
    for large result sets that should be consumed in bounded record batches.
    """
    con, relation = _relation(sql, params)
    try:
        return relation.to_arrow_table()
    finally:
        con.close()


def execute_rows(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
) -> list[tuple]:
    """Execute a small read query and return Python rows."""
    con = get_connection()
    try:
        # fetchall() remains supported by DuckDB's DB-API connection.
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


def stream_arrow(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
    batch_size: int = 100_000,
) -> Iterator[pa.RecordBatch]:
    """Yield Arrow record batches from a read query.

    Use the relation API's ``to_arrow_reader`` rather than the removed
    ``fetch_record_batch`` method on ``connection.execute`` results.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    con, relation = _relation(sql, params)
    try:
        reader = relation.to_arrow_reader(batch_size)
        for batch in reader:
            yield batch
    finally:
        con.close()
