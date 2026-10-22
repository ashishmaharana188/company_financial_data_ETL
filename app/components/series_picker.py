from __future__ import annotations

import streamlit as st

from app.services.catalog import SOURCES, SeriesRequest, list_identifiers, metric_options


def series_picker(prefix: str, title: str, catalog_items: dict[str, list[dict[str, str]]] | None = None) -> SeriesRequest | None:
    """Render a source -> identifier -> metric selector for one plotted series."""
    st.markdown(f"**{title}**")
    source = st.selectbox("Data domain", SOURCES, key=f"{prefix}_source")
    try:
        identifiers = (catalog_items or {}).get(source)
        if identifiers is None:
            identifiers = list_identifiers(source)
    except Exception as exc:
        st.warning(f"Could not load the {source} catalogue: {exc}")
        return None
    if not identifiers:
        st.warning(f"No selectable series were found for {source}.")
        return None
    id_map = {item["value"]: item["label"] for item in identifiers}
    identifier = st.selectbox(
        "Variable / instrument / company",
        options=list(id_map),
        format_func=lambda value: id_map.get(value, value),
        key=f"{prefix}_identifier",
    )
    metrics = metric_options(source)
    if not metrics:
        st.warning(f"No numeric metrics are configured for {source}.")
        return None
    metric_map = {item.key: item.label for item in metrics}
    metric = st.selectbox(
        "Measure",
        options=list(metric_map),
        format_func=lambda value: metric_map.get(value, value),
        key=f"{prefix}_metric",
    )
    return SeriesRequest(source=source, identifier=identifier, metric=metric)
