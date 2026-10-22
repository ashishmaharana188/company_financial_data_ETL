# data_access/catalog.py

from __future__ import annotations

from dataclasses import dataclass

from .query import execute_arrow


@dataclass(frozen=True)
class ColumnInfo:
    schema: str
    table: str
    column: str
    data_type: str
    ordinal_position: int


def list_tables():
    sql = """
        SELECT
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        ORDER BY table_schema, table_name
    """

    return execute_arrow(sql)


def list_columns(table_name: str):
    sql = """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type,
            ordinal_position
        FROM information_schema.columns
        WHERE table_name = ?
        ORDER BY ordinal_position
    """

    return execute_arrow(sql, [table_name])