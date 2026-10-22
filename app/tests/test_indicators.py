from __future__ import annotations

import unittest

import pandas as pd

from app.services.indicators import calculate_indicators
from app.services.statistics_service import distribution_frame, distribution_summary, variable_summary


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=60, freq="D"),
            "value": [float(i + 1) for i in range(60)],
        })

    def test_ema_sma_and_bollinger_overlay_series(self):
        result = calculate_indicators(
            self.frame, ema_periods=(9, 20), sma_periods=(20,), bollinger_period=20
        )
        self.assertIn("EMA 9", result)
        self.assertIn("EMA 20", result)
        self.assertIn("SMA 20", result)
        self.assertIn("Bollinger upper 20 (+2σ)", result)
        self.assertIn("Bollinger lower 20 (-2σ)", result)
        # 20-observation averages begin only after sufficient observations.
        self.assertTrue(pd.isna(result["SMA 20"]["value"].iloc[18]))
        self.assertAlmostEqual(result["SMA 20"]["value"].iloc[19], 10.5)

    def test_distribution_summary(self):
        stats = distribution_summary(self.frame, "2026-01-01", "2026-01-10")
        self.assertEqual(stats["observations"], 10)
        self.assertAlmostEqual(stats["mean"], 5.5)
        self.assertAlmostEqual(stats["median"], 5.5)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 10.0)


    def test_distribution_frame_supports_levels_and_observation_changes(self):
        levels = distribution_frame(self.frame, "levels", "2026-01-03", "2026-01-05")
        self.assertEqual(levels["value"].tolist(), [3.0, 4.0, 5.0])

        differences = distribution_frame(self.frame, "absolute_change", "2026-01-03", "2026-01-05")
        self.assertEqual(differences["value"].tolist(), [1.0, 1.0, 1.0])

        returns = distribution_frame(self.frame, "percentage_change", "2026-01-03", "2026-01-05")
        self.assertAlmostEqual(returns["value"].iloc[0], (3 / 2 - 1) * 100)

    def test_percentage_distribution_omits_nonpositive_prior_values(self):
        frame = self.frame.iloc[:4].copy()
        frame["value"] = [-2.0, -1.0, 0.0, 2.0]
        result = distribution_frame(frame, "percentage_change")
        self.assertEqual(len(result.dropna(subset=["value"])), 0)

    def test_change_is_endpoint_difference_not_sum_of_daily_changes(self):
        stats = variable_summary(self.frame, "2026-01-11", "2026-01-20")
        self.assertEqual(stats["first_date"], "2026-01-11")
        self.assertEqual(stats["latest_date"], "2026-01-20")
        self.assertEqual(stats["change"], 9.0)
        self.assertAlmostEqual(stats["percent_change"], (20 / 11 - 1) * 100)

    def test_percentage_change_not_reported_for_zero_or_negative_start(self):
        for values in ([0.0, 2.0], [-5.0, -2.0]):
            frame = self.frame.iloc[:2].copy()
            frame["value"] = values
            stats = variable_summary(frame)
            self.assertIsNone(stats["percent_change"])
            self.assertFalse(stats["percent_change_available"])


if __name__ == "__main__":
    unittest.main()
