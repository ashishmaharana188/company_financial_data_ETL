from __future__ import annotations

from datetime import date
from math import ceil
from pathlib import Path
from typing import Mapping

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# High-contrast colors chosen explicitly so lines remain visible on dark themes.
TERMINAL_COLORS = [
    "#58A6FF",  # blue
    "#3DDC97",  # green
    "#F6BD4F",  # amber
    "#D2A8FF",  # purple
    "#FF7B72",  # coral
    "#79C0FF",  # light blue
    "#FFA657",  # orange
    "#A5D6FF",  # pale blue
    "#56D4DD",  # cyan
    "#F778BA",  # pink
]


def _layout_specs(series_count: int, columns_per_row: int):
    """Build a subplot grid whose incomplete final row expands to fill its width.

    The internal grid uses two slots per visible chart. A final row with one or
    two charts expands those charts across the available row instead of leaving
    a large empty panel to the right.
    """
    effective_columns = min(columns_per_row, series_count)
    grid_columns = effective_columns * 2
    rows = ceil(series_count / effective_columns)
    specs: list[list[dict | None]] = []
    titles: list[str] = []
    cursor = 0

    for row_index in range(rows):
        remaining = series_count - cursor
        in_row = min(effective_columns, remaining)
        span = grid_columns // in_row
        row_specs: list[dict | None] = []
        row_titles: list[str] = []
        for col_start in range(0, grid_columns, span):
            if len(row_specs) >= grid_columns:
                break
            if len(row_titles) == col_start:
                row_titles.append("")
            # Each chart starts at this column and consumes `span` grid slots.
            row_specs.append({"colspan": span})
            row_titles.append("")
            row_specs.extend([None] * (span - 1))
            row_titles.extend([""] * (span - 1))
            cursor += 1
            if cursor >= series_count:
                # Remaining cells in this row are deliberately left empty only
                # if the row layout needs them; colspan placeholders hide them.
                pass
        # Make arrays exactly grid_columns wide. Titles only mark chart starts.
        row_specs = (row_specs + [None] * grid_columns)[:grid_columns]
        row_titles = (row_titles + [""] * grid_columns)[:grid_columns]
        specs.append(row_specs)
        titles.extend(row_titles)

    return rows, grid_columns, specs, titles


def synchronized_series_chart(
    series: Mapping[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    columns_per_row: int = 2,
    sync_crosshair: bool = True,
    indicator_overlays: Mapping[str, Mapping[str, pd.DataFrame]] | Mapping[str, pd.DataFrame] | None = None,
    indicator_target_label: str | None = None,
):
    """Build a responsive, high-contrast terminal-style chart grid.

    Charts use distinct value axes and synchronized date axes. The last
    incomplete row expands horizontally so a lone chart never occupies only a
    fraction of the available width. At most three visible charts appear per row.
    """
    if columns_per_row not in (2, 3):
        raise ValueError("columns_per_row must be 2 or 3")

    nonempty: list[tuple[str, pd.DataFrame]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for label, frame in series.items():
        if frame is None or frame.empty or "date" not in frame or "value" not in frame:
            continue
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"])
        out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
        out = out.sort_values("date")
        if not out.empty:
            nonempty.append((label, out))

    if not nonempty:
        return None

    rows, grid_columns, specs, _ = _layout_specs(len(nonempty), columns_per_row)

    # Do not use Plotly's implicit subplot_titles here. With colspan placeholders
    # and a partially filled last row they can be assigned to the wrong cell or
    # omitted. We add one explicit paper-coordinate annotation per real panel
    # after the subplot domains have been calculated.
    fig = make_subplots(
        rows=rows,
        cols=grid_columns,
        specs=specs,
        horizontal_spacing=0.018,
        vertical_spacing=min(0.14, 0.10 + 0.01 * rows),
    )

    for index, (label, frame) in enumerate(nonempty):
        row = index // min(columns_per_row, len(nonempty)) + 1
        within_row = index % min(columns_per_row, len(nonempty))
        row_start = (row - 1) * min(columns_per_row, len(nonempty))
        in_row = min(min(columns_per_row, len(nonempty)), len(nonempty) - row_start)
        span = grid_columns // in_row
        col = within_row * span + 1
        color = TERMINAL_COLORS[index % len(TERMINAL_COLORS)]
        candle_fields = ["open", "high", "low", "close", "volume"]
        has_ohlc = all(field in frame.columns and frame[field].notna().any()
                       for field in ("open", "high", "low", "close"))
        customdata = [
            [
                (float(point[field]) if field in frame.columns and pd.notna(point.get(field)) else None)
                for field in candle_fields
            ]
            for _, point in frame.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["value"],
                customdata=customdata,
                meta={"has_ohlc": bool(has_ohlc), "series_label": label,
                      "panel_index": index, "is_main_series": True},
                name=label,
                mode="lines",
                line={"width": 2.0, "color": color},
                connectgaps=False,
                # Keep Plotly's floating tooltip out of the way. A fixed OHLC/date
                # display in the terminal header is populated by the crosshair JS.
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            row=row,
            col=col,
            fixedrange=False,
            automargin=True,
            showgrid=True,
            gridcolor="rgba(154, 167, 186, 0.16)",
            zeroline=False,
            tickfont={"color": "#B7C0CD", "size": 10},
            title_font={"color": "#D9E2F0", "size": 11},
        )

    # Indicators are calculated independently for each selected series and are
    # rendered on that series' own panel. A flat mapping of indicator name to
    # frame remains supported as a backward-compatible main-panel overlay.
    if indicator_overlays:
        values_are_frames = all(isinstance(value, pd.DataFrame) for value in indicator_overlays.values())
        if values_are_frames:
            legacy_target = indicator_target_label or nonempty[0][0]
            overlay_groups = {legacy_target: indicator_overlays}
        else:
            overlay_groups = {
                str(label): overlay_map
                for label, overlay_map in indicator_overlays.items()
                if isinstance(overlay_map, Mapping)
            }

        overlay_colors = {
            "EMA 9": "#F6BD4F", "EMA 20": "#FF7B72", "EMA 50": "#D2A8FF",
            "SMA 20": "#3DDC97", "SMA 50": "#79C0FF",
            "Bollinger middle": "#A5D6FF", "Bollinger upper": "#8B949E",
            "Bollinger lower": "#8B949E",
        }
        for target_index, (target_label, _target_frame) in enumerate(nonempty):
            overlay_group = overlay_groups.get(target_label, {})
            if not overlay_group:
                continue
            target_row = target_index // min(columns_per_row, len(nonempty)) + 1
            target_row_start = (target_row - 1) * min(columns_per_row, len(nonempty))
            target_in_row = min(min(columns_per_row, len(nonempty)), len(nonempty) - target_row_start)
            target_span = grid_columns // target_in_row
            target_col = (target_index % min(columns_per_row, len(nonempty))) * target_span + 1
            for indicator_index, (indicator_name, indicator_frame) in enumerate(overlay_group.items()):
                if indicator_frame is None or indicator_frame.empty:
                    continue
                overlay = indicator_frame.copy()
                overlay["date"] = pd.to_datetime(overlay["date"], errors="coerce")
                overlay["value"] = pd.to_numeric(overlay["value"], errors="coerce")
                overlay = overlay.dropna(subset=["date", "value"])
                overlay = overlay[(overlay["date"] >= start_ts) & (overlay["date"] <= end_ts)]
                if overlay.empty:
                    continue
                color = next((c for prefix, c in overlay_colors.items() if indicator_name.startswith(prefix)), TERMINAL_COLORS[(len(nonempty) + indicator_index) % len(TERMINAL_COLORS)])
                dashed = "Bollinger upper" in indicator_name or "Bollinger lower" in indicator_name
                fig.add_trace(
                    go.Scatter(
                        x=overlay["date"], y=overlay["value"],
                        name=f"{target_label} · {indicator_name}",
                        mode="lines", connectgaps=False,
                        line={"width": 1.35, "color": color, "dash": "dot" if dashed else "solid"},
                        hoverinfo="skip",
                        meta={"panel_index": target_index, "is_indicator": True, "is_main_series": False,
                              "series_label": target_label, "indicator_name": indicator_name},
                    ),
                    row=target_row, col=target_col,
                )

    # Axis numbering follows the non-empty subplot creation order. All panels
    # share the first x-axis range while retaining independent y scales.
    for axis_number in range(1, len(nonempty) + 1):
        xaxis_name = "xaxis" if axis_number == 1 else f"xaxis{axis_number}"
        axis = fig.layout[xaxis_name]
        axis.matches = None if axis_number == 1 else "x"
        axis.range = [start_ts, end_ts]
        axis.showgrid = True
        axis.gridcolor = "rgba(154, 167, 186, 0.16)"
        axis.zeroline = False
        axis.tickfont = {"color": "#B7C0CD", "size": 10}
        axis.tickformat = "%d %b\n%Y"
        axis.automargin = True

    figure_height_per_row = 345 if min(columns_per_row, len(nonempty)) == 2 else 310
    if len(nonempty) == 1:
        figure_height_per_row = 440
    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        height=max(420, rows * figure_height_per_row + 68),
        margin={"l": 58, "r": 22, "t": 64, "b": 44},
        hovermode="closest",
        showlegend=False,
        dragmode="pan",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"family": "Arial, sans-serif", "color": "#D9E2F0", "size": 11},
    )

    # Explicit title per real panel. Split long labels into two lines so titles
    # remain readable in narrow two/three-column layouts.
    panel_titles = []
    for index, (label, _frame) in enumerate(nonempty):
        axis_number = index + 1
        xaxis_name = "xaxis" if axis_number == 1 else f"xaxis{axis_number}"
        yaxis_name = "yaxis" if axis_number == 1 else f"yaxis{axis_number}"
        x_domain = list(fig.layout[xaxis_name].domain or (0, 1))
        y_domain = list(fig.layout[yaxis_name].domain or (0, 1))
        parts = [part.strip() for part in str(label).split(" · ")]
        title_text = " · ".join(parts) if len(parts) <= 2 else " · ".join(parts[:2]) + "<br>" + " · ".join(parts[2:])
        panel_titles.append({
            "xref": "paper", "yref": "paper",
            "x": (x_domain[0] + x_domain[1]) / 2,
            "y": min(0.985, y_domain[1] + 0.028),
            "text": title_text,
            "showarrow": False,
            "xanchor": "center", "yanchor": "bottom",
            "align": "center",
            "font": {"color": "#E6EDF3", "size": 12},
            "bgcolor": "rgba(14,17,23,0.86)",
            "borderpad": 2,
        })
    fig.update_layout(annotations=panel_titles)

    # A vertical date cursor and horizontal value guide are created for every
    # visible subplot. Browser-side JS moves them all on each hover event.
    if sync_crosshair:
        shapes = []
        for index in range(len(nonempty)):
            number = index + 1
            xref = "x" if number == 1 else f"x{number}"
            yref = "y" if number == 1 else f"y{number}"
            x_layout_name = "xaxis" if number == 1 else f"xaxis{number}"
            y_layout_name = "yaxis" if number == 1 else f"yaxis{number}"
            x_domain = list(fig.layout[x_layout_name].domain or (0, 1))
            y_domain = list(fig.layout[y_layout_name].domain or (0, 1))
            shapes.extend([
                {
                    "type": "line", "xref": xref, "yref": "paper",
                    "x0": start_ts, "x1": start_ts,
                    "y0": y_domain[0], "y1": y_domain[1],
                    "line": {"color": "#F6BD4F", "width": 1.25, "dash": "dot"},
                    "visible": False, "layer": "above",
                },
                {
                    "type": "line", "xref": "paper", "yref": yref,
                    "x0": x_domain[0], "x1": x_domain[1],
                    "y0": 0, "y1": 0,
                    "line": {"color": "#F6BD4F", "width": 1, "dash": "dot"},
                    "visible": False, "layer": "above",
                },
            ])
        fig.update_layout(shapes=shapes)
    return fig


def terminal_chart_html(fig, sync_crosshair: bool = True) -> str:
    """Render a chart grid with trading-terminal navigation and linked crosshair.

    Interaction model:
    - Drag inside a plot: pan the chart (date range horizontally, values vertically).
    - Mouse wheel inside a plot: zoom the shared date range around the cursor.
    - Shift + mouse wheel inside a plot: zoom that panel's value range.
    - Mouse wheel over an x-axis: zoom the shared date range.
    - Mouse wheel over a y-axis: zoom only that panel's value range.
    - Drag an x-axis: horizontal drag shifts time; vertical drag zooms time.
    - Drag a y-axis: vertical drag zooms values; horizontal drag shifts values.
    - Axis drags and axis-wheel gestures never change the other axis.

    Browser-side interaction is implemented inside the Plotly iframe so it does
    not depend on Streamlit reruns or browser/network access to external assets.
    Hover shows an inline fixed OHLC/date readout. Click locks a plus crosshair;
    drag it to move date and value guides, press Escape or use the toggle to unlock.
    """
    shape_count = len(fig.layout.shapes or [])
    trace_count = len(fig.data)
    sync_flag = "true" if sync_crosshair and shape_count else "false"
    post_script = r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd || !window.Plotly) return;
  const seriesCount = __TRACE_COUNT__;
  const shapeCount = __SHAPE_COUNT__;
  const syncCrosshair = __SYNC_CROSSHAIR__;
  let gesture = null;
  let rangeUpdateQueued = false;
  let latestRangePatch = null;
  let crosshairPatchQueued = false;
  let latestCrosshairPatch = null;
  let crosshairLocked = false;
  let relayoutPending = false;
  let lastCursorKey = null;
  let lastCursorState = null;

  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const axisLayoutName = (axisRef, fallback) => {
    const name = axisRef || fallback;
    return name.startsWith('x') ? 'xaxis' + name.slice(1) : 'yaxis' + name.slice(1);
  };
  const byId = id => document.getElementById(id);
  function setLockUi() {
    const button = byId('ta-crosshair-toggle');
    const status = byId('ta-crosshair-status');
    if (button) {
      button.textContent = crosshairLocked ? 'Unlock crosshair' : 'Lock crosshair';
      button.setAttribute('aria-pressed', crosshairLocked ? 'true' : 'false');
    }
    if (status) status.textContent = crosshairLocked
      ? 'CROSSHAIR LOCKED · drag the plus to reposition · Esc to unlock'
      : 'HOVER TO INSPECT · click to lock the crosshair';
  }
  function plotMetrics() {
    const rect = gd.getBoundingClientRect();
    const size = gd._fullLayout && gd._fullLayout._size;
    if (!size || !size.w || !size.h) return null;
    return {rect, left: rect.left + size.l, top: rect.top + size.t, width: size.w, height: size.h};
  }
  function panelMetadata(index, m) {
    const trace = gd.data[index];
    const xName = axisLayoutName(trace.xaxis, 'xaxis');
    const yName = axisLayoutName(trace.yaxis, 'yaxis');
    const xAxis = gd._fullLayout[xName];
    const yAxis = gd._fullLayout[yName];
    if (!xAxis || !yAxis) return null;
    const xd = xAxis.domain || [0, 1];
    const yd = yAxis.domain || [0, 1];
    const x0 = m.left + xd[0] * m.width;
    const x1 = m.left + xd[1] * m.width;
    const y0 = m.top + (1 - yd[1]) * m.height;
    const y1 = m.top + (1 - yd[0]) * m.height;
    return {index, trace, xName, yName, xAxis, yAxis, x0, x1, y0, y1,
      xAxisLineY: y1, yAxisLineX: x0};
  }
  function getPanels(m) {
    const panels = [];
    if (!m) return panels;
    const seen = new Set();
    for (let i = 0; i < seriesCount; i++) {
      const trace = gd.data[i];
      if (!trace) continue;
      const xName = axisLayoutName(trace.xaxis, 'xaxis');
      const yName = axisLayoutName(trace.yaxis, 'yaxis');
      const key = xName + '|' + yName;
      if (seen.has(key)) continue;
      seen.add(key);
      const panel = panelMetadata(i, m);
      if (panel) { panel.index = panels.length; panels.push(panel); }
    }
    return panels;
  }
  function hitTest(clientX, clientY) {
    const m = plotMetrics();
    if (!m) return null;
    const panels = getPanels(m);
    for (const p of panels) {
      if (clientX >= p.x0 - 4 && clientX <= p.x1 + 4 &&
          clientY >= p.xAxisLineY - 4 && clientY <= p.xAxisLineY + 38) {
        return {...p, kind: 'x-axis', metrics: m};
      }
    }
    for (const p of panels) {
      if (clientY >= p.y0 - 3 && clientY <= p.y1 + 3 &&
          clientX >= p.yAxisLineX - 62 && clientX <= p.yAxisLineX + 7) {
        return {...p, kind: 'y-axis', metrics: m};
      }
    }
    for (const p of panels) {
      if (clientX >= p.x0 && clientX <= p.x1 && clientY >= p.y0 && clientY <= p.y1) {
        return {...p, kind: 'plot', metrics: m};
      }
    }
    return null;
  }
  function readRange(panel, axisType) {
    const axis = axisType === 'x' ? panel.xAxis : panel.yAxis;
    const name = axisType === 'x' ? panel.xName : panel.yName;
    let raw = axis.range;
    if ((!raw || raw.length !== 2) && gd.layout[name]) raw = gd.layout[name].range;
    if (!raw || raw.length !== 2) return null;
    if (axisType === 'x') {
      const a = raw[0] instanceof Date ? raw[0].getTime() : new Date(raw[0]).getTime();
      const b = raw[1] instanceof Date ? raw[1].getTime() : new Date(raw[1]).getTime();
      return Number.isFinite(a) && Number.isFinite(b) && a !== b ? [a, b] : null;
    }
    const a = Number(raw[0]), b = Number(raw[1]);
    return Number.isFinite(a) && Number.isFinite(b) && a !== b ? [a, b] : null;
  }
  function rangePatch(panel, axisType, range) {
    const value = axisType === 'x' ? range.map(v => new Date(v).toISOString()) : range;
    if (axisType === 'x') {
      const patch = {};
      for (const p of getPanels(plotMetrics())) patch[p.xName + '.range'] = value;
      return patch;
    }
    return {[panel.yName + '.range']: value};
  }
  function applyPatch(patch) {
    if (!patch || !Object.keys(patch).length) return;
    latestRangePatch = {...(latestRangePatch || {}), ...patch};
    if (rangeUpdateQueued) return;
    rangeUpdateQueued = true;
    requestAnimationFrame(() => {
      const next = latestRangePatch;
      latestRangePatch = null;
      rangeUpdateQueued = false;
      if (next) Plotly.relayout(gd, next);
    });
  }
  function applyCrosshairPatch(patch) {
    if (!patch || !Object.keys(patch).length) return;
    latestCrosshairPatch = {...(latestCrosshairPatch || {}), ...patch};
    if (crosshairPatchQueued) return;
    crosshairPatchQueued = true;
    requestAnimationFrame(() => {
      const next = latestCrosshairPatch;
      latestCrosshairPatch = null;
      crosshairPatchQueued = false;
      if (next) Plotly.relayout(gd, next);
    });
  }
  function panRange(range, delta) { return [range[0] + delta, range[1] + delta]; }
  function zoomRange(range, factor, anchor) {
    return [anchor + (range[0] - anchor) * factor, anchor + (range[1] - anchor) * factor];
  }
  function xAtPointer(panel, clientX) {
    const range = readRange(panel, 'x');
    if (!range) return null;
    const fraction = clamp((clientX - panel.x0) / Math.max(1, panel.x1 - panel.x0), 0, 1);
    return range[0] + fraction * (range[1] - range[0]);
  }
  function yAtPointer(panel, clientY) {
    const range = readRange(panel, 'y');
    if (!range) return null;
    const fraction = clamp((clientY - panel.y0) / Math.max(1, panel.y1 - panel.y0), 0, 1);
    return range[1] - fraction * (range[1] - range[0]);
  }
  function startGesture(event, hit) {
    if (event.button !== 0) return false;
    const xRange = readRange(hit, 'x');
    const yRange = readRange(hit, 'y');
    if (!xRange || !yRange) return false;
    gesture = {
      kind: hit.kind, panel: hit, pointerId: event.pointerId,
      startX: event.clientX, startY: event.clientY,
      lastX: event.clientX, lastY: event.clientY,
      xRange, yRange,
      anchorX: xAtPointer(hit, event.clientX),
      anchorY: yAtPointer(hit, event.clientY),
      width: Math.max(1, hit.x1 - hit.x0), height: Math.max(1, hit.y1 - hit.y0),
      changed: false
    };
    return true;
  }
  function startCursorGesture(event, hit) {
    gesture = {kind: 'cursor', panel: hit, pointerId: event.pointerId,
      startX: event.clientX, startY: event.clientY, changed: false};
    return true;
  }
  function updateGesture(event) {
    if (!gesture) return;
    const g = gesture;
    const dx = event.clientX - g.startX;
    const dy = event.clientY - g.startY;
    if (g.kind === 'cursor') {
      g.changed = g.changed || Math.abs(dx) + Math.abs(dy) > 1;
      moveCrosshair(g.panel, event.clientX, event.clientY, true);
      event.preventDefault();
      return;
    }
    // A stationary press is a click-to-lock crosshair, not a chart pan.
    if (g.kind === 'plot' && Math.hypot(dx, dy) < 5) return;
    const xSpan = g.xRange[1] - g.xRange[0];
    const ySpan = g.yRange[1] - g.yRange[0];
    let patch = {};
    if (g.kind === 'plot') {
      // The plot itself pans the graph: horizontal movement shifts the shared
      // timeline, while vertical movement shifts only the active panel's values.
      // Axis gestures are handled in their own hit zones below and never alter
      // the orthogonal axis.
      patch = rangePatch(g.panel, 'x', panRange(g.xRange, -dx / g.width * xSpan));
      patch[g.panel.yName + '.range'] = panRange(g.yRange, dy / g.height * ySpan);
    } else if (g.kind === 'x-axis') {
      if (Math.abs(dx) > Math.abs(dy)) {
        patch = rangePatch(g.panel, 'x', panRange(g.xRange, -dx / g.width * xSpan));
      } else {
        const factor = Math.exp(clamp(dy * 0.008, -2.3, 2.3));
        patch = rangePatch(g.panel, 'x', zoomRange(g.xRange, factor, g.anchorX));
      }
    } else if (g.kind === 'y-axis') {
      if (Math.abs(dy) >= Math.abs(dx)) {
        const factor = Math.exp(clamp(dy * 0.008, -2.3, 2.3));
        patch[g.panel.yName + '.range'] = zoomRange(g.yRange, factor, g.anchorY);
      } else {
        patch[g.panel.yName + '.range'] = panRange(g.yRange, dx / g.height * ySpan);
      }
    }
    if (Math.hypot(dx, dy) >= 5) g.changed = true;
    applyPatch(patch);
    event.preventDefault();
  }
  function endGesture(event) {
    if (!gesture) return;
    const g = gesture;
    const id = g.pointerId;
    if (g.kind === 'plot' && !g.changed && !crosshairLocked && syncCrosshair) {
      crosshairLocked = true;
      setLockUi();
      moveCrosshair(g.panel, event ? event.clientX : g.startX, event ? event.clientY : g.startY, true);
    } else if (g.kind === 'cursor') {
      crosshairLocked = true;
      setLockUi();
    }
    gesture = null;
    try { if (gd.hasPointerCapture(id)) gd.releasePointerCapture(id); } catch (_) {}
    if (event) event.preventDefault();
  }

  function toMillis(value) {
    const parsed = value instanceof Date ? value.getTime() : new Date(value).getTime();
    return Number.isFinite(parsed) ? parsed : NaN;
  }
  function traceYValues(trace) {
    // Plotly may represent large numeric arrays as typed-array descriptors.
    // Read the decoded input array when present, otherwise use the normal array.
    const values = trace && trace.y ? trace.y : [];
    return values && values._inputArray ? values._inputArray : values;
  }
  function nearestIndex(trace, targetMillis) {
    const xs = trace.x || [], ys = traceYValues(trace);
    let bestIndex = -1, bestDistance = Infinity;
    for (let i = 0; i < xs.length && i < ys.length; i++) {
      const t = toMillis(xs[i]);
      if (!Number.isFinite(t) || ys[i] === null || ys[i] === undefined) continue;
      const distance = Math.abs(t - targetMillis);
      if (distance < bestDistance) { bestDistance = distance; bestIndex = i; }
    }
    return bestIndex;
  }
  function nearestValue(trace, targetMillis) {
    const index = nearestIndex(trace, targetMillis);
    if (index < 0) return null;
    const value = Number(traceYValues(trace)[index]);
    return Number.isFinite(value) ? value : null;
  }
  function formatValue(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return Number(value).toLocaleString(undefined, {maximumFractionDigits: 6});
  }
  function updateReadout(panel, targetMillis, cursorY) {
    const trace = panel.trace;
    const idx = nearestIndex(trace, targetMillis);
    if (idx < 0) return;
    const actualDate = toMillis((trace.x || [])[idx]);
    const dateText = Number.isFinite(actualDate) ? new Date(actualDate).toISOString().slice(0, 10) : '—';
    const dateNode = byId('ta-hud-date');
    const titleNode = byId('ta-hud-series');
    const valueNode = byId('ta-hud-value');
    const ohlcNode = byId('ta-hud-ohlc');
    const cursorNode = byId('ta-hud-cursor-y');
    if (dateNode) dateNode.textContent = dateText;
    if (titleNode) titleNode.textContent = trace.name || 'Selected series';
    const candle = ((trace.customdata || [])[idx]) || [];
    const hasOhlc = Boolean(trace.meta && trace.meta.has_ohlc);
    if (ohlcNode) {
      ohlcNode.style.display = hasOhlc ? 'flex' : 'none';
      if (hasOhlc) {
        ['open', 'high', 'low', 'close'].forEach((field, i) => {
          const node = byId('ta-hud-' + field);
          if (node) node.textContent = formatValue(candle[i]);
        });
      }
    }
    if (valueNode) {
      valueNode.style.display = hasOhlc ? 'none' : 'inline-flex';
      valueNode.textContent = 'Value ' + formatValue(traceYValues(trace)[idx]);
    }
    if (cursorNode) cursorNode.textContent = 'Cursor Y ' + formatValue(cursorY);
  }
  function moveCrosshair(panel, clientX, clientY, lockedAction) {
    const targetMillis = xAtPointer(panel, clientX);
    if (!Number.isFinite(targetMillis)) return;
    const activeY = yAtPointer(panel, clientY);
    const key = Math.round(targetMillis) + ':' + panel.index + ':' + Math.round(clientY) + ':' + (lockedAction ? 'L' : 'H');
    lastCursorState = {panelIndex: panel.index, clientX, clientY, targetMillis};
    if (key === lastCursorKey) return;
    lastCursorKey = key;
    if (syncCrosshair && shapeCount) {
      const panels = getPanels(plotMetrics());
      const patch = {};
      const iso = new Date(targetMillis).toISOString();
      for (const p of panels) {
        const v = p.index * 2;
        const h = v + 1;
        patch[`shapes[${v}].x0`] = iso;
        patch[`shapes[${v}].x1`] = iso;
        patch[`shapes[${v}].visible`] = true;
        const y = p.index === panel.index && activeY !== null ? activeY : nearestValue(p.trace, targetMillis);
        if (y === null) {
          patch[`shapes[${h}].visible`] = false;
        } else {
          patch[`shapes[${h}].y0`] = y;
          patch[`shapes[${h}].y1`] = y;
          patch[`shapes[${h}].visible`] = true;
        }
      }
      applyCrosshairPatch(patch);
    }
    updateReadout(panel, targetMillis, activeY);
  }
  function hideCrosshair() {
    if (!syncCrosshair || !shapeCount || crosshairLocked) return;
    const patch = {};
    for (let i = 0; i < shapeCount; i++) patch[`shapes[${i}].visible`] = false;
    applyCrosshairPatch(patch);
    lastCursorKey = null;
  }
  function unlockCrosshair() {
    crosshairLocked = false;
    setLockUi();
    hideCrosshair();
  }

  gd.addEventListener('pointerdown', function(event) {
    const hit = hitTest(event.clientX, event.clientY);
    if (!hit || !['plot', 'x-axis', 'y-axis'].includes(hit.kind)) return;
    if (hit.kind === 'plot' && crosshairLocked) {
      startCursorGesture(event, hit);
      moveCrosshair(hit, event.clientX, event.clientY, true);
    } else if (!startGesture(event, hit)) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    try { gd.setPointerCapture(event.pointerId); } catch (_) {}
  }, true);
  window.addEventListener('pointermove', function(event) {
    if (gesture && event.pointerId === gesture.pointerId) {
      event.preventDefault();
      updateGesture(event);
      return;
    }
    if (!crosshairLocked) {
      const hit = hitTest(event.clientX, event.clientY);
      if (hit && hit.kind === 'plot') moveCrosshair(hit, event.clientX, event.clientY, false);
    }
  }, {capture: true, passive: false});
  window.addEventListener('pointerup', function(event) {
    if (gesture && event.pointerId === gesture.pointerId) endGesture(event);
  }, true);
  window.addEventListener('pointercancel', function(event) {
    if (gesture && event.pointerId === gesture.pointerId) endGesture(event);
  }, true);
  gd.addEventListener('mouseleave', hideCrosshair);
  const lockButton = byId('ta-crosshair-toggle');
  if (lockButton) {
    lockButton.disabled = !syncCrosshair;
    lockButton.title = syncCrosshair ? 'Lock or unlock the synchronized crosshair' : 'Enable Sync crosshair to lock the crosshair';
  }
  if (lockButton) lockButton.addEventListener('click', function(event) {
    event.preventDefault();
    if (crosshairLocked) unlockCrosshair();
    else {
      // Lock the most recently inspected point; subsequent drag moves the plus cursor.
      crosshairLocked = true;
      setLockUi();
      if (lastCursorState) {
        const panel = getPanels(plotMetrics()).find(p => p.index === lastCursorState.panelIndex);
        if (panel) moveCrosshair(panel, lastCursorState.clientX, lastCursorState.clientY, true);
      }
    }
  });
  window.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && panelFullscreen) { panelFullscreen.remove(); panelFullscreen = null; return; }
    if (event.key === 'Escape' && crosshairLocked) unlockCrosshair();
  });
  setLockUi();

  // Terminal shell scrolls internally so the OHLC/date readout remains visible.
  const terminalShell = byId('ta-terminal-shell');
  const chartStage = byId('ta-chart-stage');
  const graphScroll = byId('ta-graph-scroll');
  let panelLayer = null;
  let panelFullscreen = null;
  function requestFullscreenSafe(element) {
    if (element && element.requestFullscreen) element.requestFullscreen().catch(() => {});
  }
  function makePanelFullscreen(panelIndex) {
    const panel = getPanels(plotMetrics()).find(p => p.index === panelIndex);
    if (!panel) return;
    if (panelFullscreen) panelFullscreen.remove();
    const overlay = document.createElement('div');
    overlay.id = 'ta-single-panel-fullscreen';
    overlay.innerHTML = '<div class="ta-fs-header"><strong></strong><span>Hover for date/value · Esc to close</span><button type="button">Close chart</button></div><div class="ta-fs-plot"></div>';
    overlay.querySelector('strong').textContent = panel.trace.name || 'Chart';
    const close = () => {
      try { if (document.fullscreenElement === overlay) document.exitFullscreen().catch(() => {}); } catch (_) {}
      if (panelFullscreen) panelFullscreen.remove();
      panelFullscreen = null;
    };
    overlay.querySelector('button').addEventListener('click', close);
    document.body.appendChild(overlay);
    panelFullscreen = overlay;
    const related = gd.data.filter(t => axisLayoutName(t.xaxis, 'xaxis') === panel.xName && axisLayoutName(t.yaxis, 'yaxis') === panel.yName);
    const traces = related.map(t => ({
      type: 'scatter', mode: t.mode || 'lines', name: t.name,
      x: Array.from(t.x || []), y: Array.from(traceYValues(t) || []),
      customdata: t.customdata ? Array.from(t.customdata) : undefined,
      connectgaps: t.connectgaps,
      line: t.line ? {...t.line} : {width: 2, color: '#58A6FF'},
      hoverinfo: 'skip'
    }));
    const fsLayout = {
      autosize: true, height: Math.max(500, window.innerHeight - 64),
      margin: {l: 72, r: 28, t: 26, b: 48},
      paper_bgcolor: '#0E1117', plot_bgcolor: '#0E1117',
      font: {family: 'Arial, sans-serif', color: '#D9E2F0'},
      showlegend: true, legend: {orientation: 'h', y: 1.06}, hovermode: 'x',
      xaxis: {type: 'date', showgrid: true, gridcolor: 'rgba(154,167,186,0.16)',
        showspikes: true, spikemode: 'across', spikesnap: 'cursor', spikecolor: '#F6BD4F', spikethickness: 1},
      yaxis: {showgrid: true, gridcolor: 'rgba(154,167,186,0.16)',
        showspikes: true, spikemode: 'across', spikesnap: 'cursor', spikecolor: '#79C0FF', spikethickness: 1}
    };
    Plotly.newPlot(overlay.querySelector('.ta-fs-plot'), traces, fsLayout,
      {responsive: true, displaylogo: false, scrollZoom: true});
    requestFullscreenSafe(overlay);
  }
  function installPanelFullscreenButtons() {
    if (!chartStage) return;
    if (!panelLayer) {
      panelLayer = document.createElement('div');
      panelLayer.id = 'ta-panel-fullscreen-layer';
      chartStage.appendChild(panelLayer);
    }
    const m = plotMetrics();
    if (!m) return;
    const panels = getPanels(m);
    let buttons = Array.from(panelLayer.querySelectorAll('.ta-panel-fullscreen-button'));
    // Reposition existing buttons rather than recreating them after every relayout;
    // repeated DOM replacement can interrupt pointer clicks during crosshair motion.
    if (buttons.length !== panels.length) {
      panelLayer.innerHTML = '';
      buttons = panels.map(panel => {
        const button = document.createElement('button');
        button.className = 'ta-panel-fullscreen-button';
        button.type = 'button'; button.textContent = 'Expand';
        button.title = 'Open this chart full screen';
        button.addEventListener('click', event => { event.preventDefault(); event.stopPropagation(); makePanelFullscreen(Number(button.dataset.panelIndex)); });
        button.dataset.panelIndex = String(panel.index);
        panelLayer.appendChild(button);
        return button;
      });
    }
    panels.forEach((panel, index) => {
      const button = buttons[index];
      button.dataset.panelIndex = String(panel.index);
      button.style.left = Math.max(4, panel.x1 - m.rect.left - 70) + 'px';
      button.style.top = Math.max(3, panel.y0 - m.rect.top - 24) + 'px';
    });
  }
  const allFullscreenButton = byId('ta-fullscreen-terminal');
  if (allFullscreenButton && terminalShell) {
    allFullscreenButton.addEventListener('click', () => {
      if (document.fullscreenElement === terminalShell) document.exitFullscreen().catch(() => {});
      else requestFullscreenSafe(terminalShell);
    });
    document.addEventListener('fullscreenchange', () => {
      allFullscreenButton.textContent = document.fullscreenElement === terminalShell ? 'Exit full screen' : 'Full-screen terminal';
      if (panelFullscreen && !document.fullscreenElement) { panelFullscreen.remove(); panelFullscreen = null; }
      setTimeout(installPanelFullscreenButtons, 80);
    });
  }
  window.addEventListener('resize', () => setTimeout(installPanelFullscreenButtons, 100));
  if (graphScroll) graphScroll.addEventListener('scroll', () => installPanelFullscreenButtons(), {passive: true});
  gd.on('plotly_afterplot', () => setTimeout(installPanelFullscreenButtons, 60));
  setTimeout(installPanelFullscreenButtons, 350);

  gd.addEventListener('wheel', function(event) {
    const hit = hitTest(event.clientX, event.clientY);
    if (!hit) return;
    const amount = clamp(event.deltaY, -240, 240);
    if (Math.abs(amount) < 1) return;
    event.preventDefault();
    event.stopPropagation();
    // Axis zones take priority over modifier keys: the wheel always acts on
    // the axis directly under the pointer, like a charting terminal.
    if (hit.kind === 'x-axis') {
      const range = readRange(hit, 'x');
      if (!range) return;
      const anchor = xAtPointer(hit, event.clientX) ?? (range[0] + range[1]) / 2;
      const factor = Math.exp(clamp(amount * 0.0018, -1.2, 1.2));
      applyPatch(rangePatch(hit, 'x', zoomRange(range, factor, anchor)));
    } else if (hit.kind === 'y-axis' || (hit.kind === 'plot' && event.shiftKey)) {
      const range = readRange(hit, 'y');
      if (!range) return;
      const anchor = yAtPointer(hit, event.clientY) ?? (range[0] + range[1]) / 2;
      const factor = Math.exp(clamp(amount * 0.0018, -1.2, 1.2));
      applyPatch({[hit.yName + '.range']: zoomRange(range, factor, anchor)});
    } else if (hit.kind === 'plot') {
      // Plain wheel over the chart zooms time, anchored at the pointer date.
      // Shift+wheel is the explicit shortcut for zooming the active value axis.
      const range = readRange(hit, 'x');
      if (!range) return;
      const anchor = xAtPointer(hit, event.clientX) ?? (range[0] + range[1]) / 2;
      const factor = Math.exp(clamp(amount * 0.0018, -1.2, 1.2));
      applyPatch(rangePatch(hit, 'x', zoomRange(range, factor, anchor)));
    }
  }, {capture: true, passive: false});
})();
""".replace("__TRACE_COUNT__", str(trace_count)) \
   .replace("__SHAPE_COUNT__", str(shape_count)) \
   .replace("__SYNC_CROSSHAIR__", sync_flag)

    html = pio.to_html(
        fig,
        config={
            "displaylogo": False,
            # Wheel behavior is handled explicitly below: plot and x-axis wheel
            # zoom time; y-axis wheel zooms only the active panel's value range.
            "scrollZoom": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "pan2d", "zoom2d"],
            "doubleClick": "reset",
        },
        include_plotlyjs=False,
        full_html=True,
        default_width="100%",
        default_height=f"{int(fig.layout.height or 700)}px",
        post_script=[post_script],
        div_id="trendanalysis-terminal-chart",
    )
    # Fixed terminal readout lives in the chart header, not in a floating tooltip.
    header = """<div id="ta-terminal-topbar">
      <div id="ta-crosshair-status">HOVER TO INSPECT · click to lock the crosshair</div>
      <div id="ta-terminal-readout" aria-live="polite">
        <span id="ta-hud-series">Hover a chart</span>
        <span class="ta-separator">|</span><span id="ta-hud-date">—</span>
        <div id="ta-hud-ohlc" class="ta-readout-values">
          <span>O <b id="ta-hud-open">—</b></span><span>H <b id="ta-hud-high">—</b></span>
          <span>L <b id="ta-hud-low">—</b></span><span>C <b id="ta-hud-close">—</b></span>
        </div>
        <span id="ta-hud-value" style="display:none">Value —</span>
        <span id="ta-hud-cursor-y">Cursor Y —</span>
        <button id="ta-crosshair-toggle" type="button" aria-pressed="false">Lock crosshair</button>
        <button id="ta-fullscreen-terminal" type="button">Full-screen terminal</button>
      </div>
    </div>"""
    shell_open = '<div id="ta-terminal-shell">' + header + '<div id="ta-graph-scroll"><div id="ta-chart-stage">'
    html = html.replace("<body>", "<body>" + shell_open, 1)
    html = html.replace("</body>", "</div></div></div></body>", 1)
    bundle_path = Path(__file__).resolve().parents[1] / "assets" / "plotly.min.js"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Bundled Plotly.js not found: {bundle_path}")
    plotly_js = bundle_path.read_text(encoding="utf-8")
    bundled_script = '<script type="text/javascript">\n' + plotly_js + "\n</script>"
    html = html.replace("</head>", bundled_script + "</head>", 1)
    html = html.replace(
        "</head>",
        "<style>html,body{margin:0;padding:0;width:100%;height:100%;background:#0E1117;overflow:hidden;color:#D9E2F0;font-family:Arial,sans-serif;} "
        "#ta-terminal-shell{height:100vh;width:100%;display:flex;flex-direction:column;background:#0E1117;overflow:hidden;} "
        "#ta-terminal-topbar{flex:0 0 auto;min-height:62px;box-sizing:border-box;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:7px 14px;background:#0E1117;border-bottom:1px solid #303844;position:relative;z-index:100;} "
        "#ta-graph-scroll{flex:1 1 auto;min-height:0;overflow:auto;overscroll-behavior:contain;scrollbar-color:#485361 #0E1117;} "
        "#ta-chart-stage{position:relative;min-height:100%;width:100%;} "
        "#ta-crosshair-status{font-size:11px;color:#8B949E;white-space:nowrap;} "
        "#ta-terminal-readout{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap;font-size:11px;font-variant-numeric:tabular-nums;} "
        "#ta-hud-series{font-weight:600;color:#D9E2F0;max-width:230px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;} "
        ".ta-separator{color:#484F58;} .ta-readout-values{display:flex;gap:10px;color:#8B949E;} .ta-readout-values b{color:#D9E2F0;font-weight:600;} "
        "#ta-hud-date{color:#F6BD4F;font-weight:700;} #ta-hud-cursor-y{color:#79C0FF;} "
        "#ta-crosshair-toggle,#ta-fullscreen-terminal{background:#21262D;color:#C9D1D9;border:1px solid #484F58;border-radius:4px;padding:5px 8px;font-size:11px;cursor:pointer;white-space:nowrap;} "
        "#ta-crosshair-toggle[aria-pressed=true]{background:#3B2D10;border-color:#F6BD4F;color:#F6BD4F;} "
        "#trendanalysis-terminal-chart{width:100% !important;} .plot-container.plotly{width:100% !important;} "
        "#trendanalysis-terminal-chart .main-svg{touch-action:none;cursor:crosshair;} "
        "#ta-panel-fullscreen-layer{position:absolute;inset:0;pointer-events:none;z-index:30;} .ta-panel-fullscreen-button{position:absolute;pointer-events:auto;background:rgba(33,38,45,.94);color:#C9D1D9;border:1px solid #57616F;border-radius:4px;padding:3px 7px;font-size:10px;cursor:pointer;} "
        "#ta-terminal-shell:fullscreen{width:100vw;height:100vh;} #ta-terminal-shell:fullscreen #ta-graph-scroll{height:calc(100vh - 62px);} "
        "#ta-single-panel-fullscreen{position:fixed;inset:0;background:#0E1117;color:#D9E2F0;z-index:999999;display:flex;flex-direction:column;padding:0;box-sizing:border-box;} "
        "#ta-single-panel-fullscreen:fullscreen{width:100vw;height:100vh;} .ta-fs-header{height:48px;flex:0 0 48px;display:flex;align-items:center;gap:16px;padding:0 16px;border-bottom:1px solid #303844;} .ta-fs-header strong{margin-right:auto;} .ta-fs-header span{font-size:12px;color:#8B949E;} .ta-fs-header button{background:#21262D;color:#D9E2F0;border:1px solid #484F58;border-radius:4px;padding:6px 10px;cursor:pointer;} .ta-fs-plot{flex:1;min-height:0;width:100%;} "
        "</style></head>",
        1,
    )
    return html
