from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from app.components.charts import synchronized_series_chart, terminal_chart_html


class SynchronizedChartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = date(2026, 1, 1)
        self.end = date(2026, 1, 5)
        self.series = {
            f"Series {name}": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
                    "value": [index, index + 1, index + 3],
                }
            )
            for index, name in enumerate(["A", "B", "C", "D"], start=1)
        }

    def test_series_are_arranged_side_by_side_two_per_row(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        self.assertEqual(len(fig.data), 4)
        # First and second panel share the same vertical domain (top row).
        self.assertEqual(fig.layout.yaxis.domain, fig.layout.yaxis2.domain)
        # Third and fourth panel share the next vertical domain (second row).
        self.assertEqual(fig.layout.yaxis3.domain, fig.layout.yaxis4.domain)
        self.assertNotEqual(fig.layout.yaxis.domain, fig.layout.yaxis3.domain)

    def test_hover_is_linked_across_matching_time_axes(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        self.assertEqual(fig.layout.hovermode, "closest")
        self.assertIsNone(fig.layout.xaxis.matches)
        self.assertEqual(fig.layout.xaxis2.matches, "x")
        self.assertEqual(fig.layout.xaxis3.matches, "x")
        self.assertEqual(fig.layout.xaxis4.matches, "x")
        # Each panel has one vertical and one horizontal crosshair shape.
        self.assertEqual(len(fig.layout.shapes), len(self.series) * 2)
        self.assertEqual(fig.layout.shapes[0].xref, "x")
        self.assertEqual(fig.layout.shapes[0].yref, "paper")
        self.assertEqual(fig.layout.shapes[1].xref, "paper")
        self.assertEqual(fig.layout.shapes[1].yref, "y")
        self.assertFalse(fig.layout.shapes[0].visible)

    def test_three_per_row_is_supported(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=3)
        self.assertEqual(len(fig.data), 4)
        # First three panels share the top-row domain.
        self.assertEqual(fig.layout.yaxis.domain, fig.layout.yaxis2.domain)
        self.assertEqual(fig.layout.yaxis2.domain, fig.layout.yaxis3.domain)
        self.assertNotEqual(fig.layout.yaxis.domain, fig.layout.yaxis4.domain)

    def test_rejects_more_than_three_panels_per_row(self) -> None:
        with self.assertRaises(ValueError):
            synchronized_series_chart(self.series, self.start, self.end, columns_per_row=4)

    def test_one_chart_uses_full_width_instead_of_reserving_empty_column(self) -> None:
        single = {"Only series": self.series["Series A"]}
        fig = synchronized_series_chart(single, self.start, self.end, columns_per_row=2)
        self.assertEqual(len(fig.data), 1)
        self.assertAlmostEqual(fig.layout.xaxis.domain[0], 0.0, places=3)
        self.assertAlmostEqual(fig.layout.xaxis.domain[1], 1.0, places=3)
        self.assertEqual(fig.data[0].line.color, "#58A6FF")
        self.assertEqual(fig.layout.paper_bgcolor, "#0E1117")

    def test_incomplete_final_row_expands_to_available_width(self) -> None:
        three = {key: self.series[key] for key in ("Series A", "Series B", "Series C")}
        fig = synchronized_series_chart(three, self.start, self.end, columns_per_row=2)
        self.assertEqual(len(fig.data), 3)
        # Third panel should span the full final row.
        self.assertAlmostEqual(fig.layout.xaxis3.domain[0], 0.0, places=3)
        self.assertAlmostEqual(fig.layout.xaxis3.domain[1], 1.0, places=3)

    def test_crosshair_can_be_disabled(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2, sync_crosshair=False)
        self.assertEqual(len(fig.layout.shapes or []), 0)

    def test_html_has_terminal_readout_and_locked_plus_crosshair(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        html = terminal_chart_html(fig, sync_crosshair=True)
        self.assertIn("ta-terminal-topbar", html)
        self.assertLess(html.index("ta-terminal-topbar"), html.index("trendanalysis-terminal-chart"))
        self.assertIn("ta-hud-date", html)
        self.assertIn("ta-hud-open", html)
        self.assertIn("ta-hud-high", html)
        self.assertIn("ta-hud-low", html)
        self.assertIn("ta-hud-close", html)
        self.assertIn("ta-crosshair-toggle", html)
        self.assertIn("crosshairLocked", html)
        self.assertIn("moveCrosshair(g.panel, event.clientX, event.clientY, true)", html)
        self.assertIn("event.key === 'Escape'", html)
        self.assertIn("applyCrosshairPatch(patch)", html)
        self.assertIn("hoverinfo\":\"skip\"", html)
        self.assertNotIn("<script src=\"plotly.min.js", html)

    def test_price_series_customdata_preserves_ohlc_for_readout(self) -> None:
        data = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
            "value": [10.5, 11.0, 12.0],
            "open": [10.0, 10.5, 11.2],
            "high": [11.0, 11.4, 12.4],
            "low": [9.8, 10.1, 11.0],
            "close": [10.5, 11.0, 12.0],
            "volume": [100, 120, 150],
        })
        fig = synchronized_series_chart({"Price": data}, self.start, self.end)
        self.assertTrue(fig.data[0].meta["has_ohlc"])
        self.assertEqual(list(fig.data[0].customdata[0]), [10.0, 11.0, 9.8, 10.5, 100.0])

    def test_non_ohlc_series_keeps_value_readout_mode(self) -> None:
        fig = synchronized_series_chart({"PCR": self.series["Series A"]}, self.start, self.end)
        self.assertFalse(fig.data[0].meta["has_ohlc"])
        self.assertTrue(all(len(point) == 5 for point in fig.data[0].customdata))


    def test_terminal_navigation_supports_pan_and_axis_specific_zoom(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        html = terminal_chart_html(fig, sync_crosshair=True)
        # Dragging on a plot pans the visible date and active panel's value range.
        self.assertIn("pointerdown", html)
        self.assertIn("pointermove", html)
        self.assertIn("rangePatch(g.panel, 'x', panRange(g.xRange", html)
        self.assertIn("g.panel.yName + '.range'", html)
        # Plot drag pans the graph; plain plot wheel zooms time instead of panning.
        self.assertIn("hit.kind === 'plot'", html)
        self.assertIn("hit.kind === 'x-axis'", html)
        self.assertIn("hit.kind === 'y-axis'", html)
        self.assertIn("Plain wheel over the chart zooms time, anchored at the pointer date.", html)
        self.assertIn("applyPatch(rangePatch(hit, 'x', zoomRange(range, factor, anchor)))", html)
        self.assertIn("Axis zones take priority over modifier keys", html)
        self.assertIn("scrollZoom\": false", html)
        self.assertIn("event.deltaY", html)

    def test_axis_wheel_scales_only_axis_under_pointer(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        html = terminal_chart_html(fig, sync_crosshair=True)
        x_wheel = html.index("if (hit.kind === 'x-axis')")
        y_wheel = html.index("} else if (hit.kind === 'y-axis' || (hit.kind === 'plot' && event.shiftKey))")
        plot_wheel = html.index("} else if (hit.kind === 'plot')", y_wheel)
        self.assertLess(x_wheel, y_wheel)
        self.assertLess(y_wheel, plot_wheel)
        # The x-axis branch updates all linked x ranges; the y-axis branch only its own y range.
        self.assertIn("applyPatch(rangePatch(hit, 'x', zoomRange(range, factor, anchor)))", html[x_wheel:y_wheel])
        self.assertIn("applyPatch({[hit.yName + '.range']: zoomRange(range, factor, anchor)})", html[y_wheel:plot_wheel])

    def test_dragging_value_axis_does_not_patch_time_axis(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        html = terminal_chart_html(fig, sync_crosshair=True)
        start = html.index("} else if (g.kind === 'y-axis')")
        end = html.index("if (Math.hypot(dx, dy) >= 5)", start)
        block = html[start:end]
        self.assertIn("g.panel.yName + '.range'", block)
        self.assertNotIn("rangePatch(g.panel, 'x'", block)
        self.assertNotIn("xaxis", block)

    def test_navigation_preserves_shared_date_ranges_for_all_x_axes(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        html = terminal_chart_html(fig, sync_crosshair=True)
        self.assertIn("for (const p of getPanels(plotMetrics())) patch[p.xName + '.range'] = value", html)
        self.assertIn("x-axis'", html)
        self.assertIn("y-axis'", html)


    def test_indicators_overlay_main_panel_without_creating_new_panels(self) -> None:
        from app.services.indicators import calculate_indicators
        overlays = calculate_indicators(self.series["Series A"], ema_periods=(2,), sma_periods=(2,))
        fig = synchronized_series_chart(
            self.series, self.start, self.end,
            indicator_overlays=overlays, indicator_target_label="Series A",
        )
        self.assertEqual(len(fig.data), len(self.series) + 2)
        self.assertEqual(fig.data[-2].meta["panel_index"], 0)
        self.assertTrue(fig.data[-2].meta["is_indicator"])
        html = terminal_chart_html(fig, sync_crosshair=True)
        self.assertIn("function getPanels(m)", html)
        self.assertIn("const key = xName + '|' + yName", html)


    def test_explicit_title_is_added_for_every_real_panel(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end, columns_per_row=2)
        titles = [a.text for a in (fig.layout.annotations or [])]
        self.assertEqual(len(titles), len(self.series))
        for label in self.series:
            self.assertTrue(any(label in title.replace("<br>", " ") for title in titles), label)
        # Titles are anchored per subplot domain, not generated for empty grid cells.
        self.assertTrue(all(a.xref == "paper" and a.yref == "paper" for a in fig.layout.annotations))

    def test_indicators_can_overlay_each_selected_series(self) -> None:
        from app.services.indicators import calculate_indicators
        grouped = {
            label: calculate_indicators(frame, ema_periods=(2,), sma_periods=(2,))
            for label, frame in self.series.items()
        }
        fig = synchronized_series_chart(
            self.series, self.start, self.end, columns_per_row=2,
            indicator_overlays=grouped,
        )
        self.assertEqual(len(fig.data), len(self.series) * 3)
        for panel_index, label in enumerate(self.series):
            overlays = [trace for trace in fig.data if trace.meta and trace.meta.get("is_indicator") and trace.meta.get("panel_index") == panel_index]
            self.assertEqual(len(overlays), 2, label)
            self.assertTrue(all(trace.meta.get("series_label") == label for trace in overlays))

    def test_terminal_uses_internal_scroll_area_and_fullscreen_controls(self) -> None:
        fig = synchronized_series_chart(self.series, self.start, self.end)
        html = terminal_chart_html(fig)
        self.assertIn("ta-terminal-shell", html)
        self.assertIn("ta-graph-scroll", html)
        self.assertIn("position:sticky", html) if "position:sticky" in html else self.assertIn("overflow:auto", html)
        self.assertIn("ta-fullscreen-terminal", html)
        self.assertIn("ta-panel-fullscreen-button", html)
        self.assertIn("makePanelFullscreen", html)

    def test_returns_none_when_no_series_has_data_in_range(self) -> None:
        series = {"Outside": pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "value": [1]})}
        self.assertIsNone(synchronized_series_chart(series, self.start, self.end))


if __name__ == "__main__":
    unittest.main()
