# data_access/query.py

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pyarrow as pa

from .db import get_connection


def execute_arrow(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
) -> pa.Table:
    """
    Execute a query and return an Arrow table.

    Suitable for metadata and moderately sized results.
    """
    con = get_connection()

    try:
        result = con.execute(sql, params or [])
        return result.fetch_arrow_table()
    finally:
        con.close()


def execute_rows(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
) -> list[tuple]:
    """
    Execute a small query and return Python rows.
    """
    con = get_connection()

    try:
        return con.execute(
            sql,
            params or [],
        ).fetchall()
    finally:
        con.close()


def stream_arrow(
    sql: str,
    params: list[Any] | tuple[Any, ...] | None = None,
    batch_size: int = 100_000,
) -> Iterator[pa.RecordBatch]:
    """
    Stream large OLAP queries without materializing the
    entire result in Python memory.
    """
    con = get_connection()

    try:
        reader = con.execute(
            sql,
            params or [],
        ).fetch_record_batch(batch_size)

        for batch in reader:
            yield batch

    finally:
        con.close()