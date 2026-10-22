from __future__ import annotations

import unittest

import pandas as pd

from app.services.catalog import (
    PRICE_METRICS,
    RAW_DERIVATIVE_FILTERS,
    SOURCE_COMMODITY_PRICES,
    metric_options,
)
from app.services.derivative_filters import COMMODITY_FUTURES_SQL
from app.services.series_service import _series_frame


class CommodityMarketPriceSourceTests(unittest.TestCase):
    def test_market_price_source_exposes_daily_ohlcv_measures(self):
        keys = [metric.key for metric in metric_options(SOURCE_COMMODITY_PRICES)]
        self.assertEqual(keys, ["close", "open", "high", "low", "volume"])
        self.assertEqual(PRICE_METRICS, metric_options(SOURCE_COMMODITY_PRICES))

    def test_market_price_source_uses_commodity_futures_filter_only(self):
        self.assertEqual(RAW_DERIVATIVE_FILTERS[SOURCE_COMMODITY_PRICES], COMMODITY_FUTURES_SQL)
        predicate = RAW_DERIVATIVE_FILTERS[SOURCE_COMMODITY_PRICES]
        self.assertIn("FUTCOM", predicate)
        self.assertNotIn("OPTFUT", predicate)
        self.assertNotIn("OPTCOM", predicate)

    def test_series_frame_keeps_raw_ohlc_fields_for_terminal_readout(self):
        raw = pd.DataFrame({
            "ReportDate": pd.to_datetime(["2026-10-15", "2026-10-16"]),
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [103.0, 104.0],
            "Volume": [200, 250],
        })
        result = _series_frame(raw, "ReportDate", "close", "MCX commodity close")
        self.assertEqual(result["value"].tolist(), [103.0, 104.0])
        for field in ("open", "high", "low", "close", "volume"):
            self.assertIn(field, result.columns)
        self.assertEqual(result.iloc[0]["open"], 100.0)
        self.assertEqual(result.iloc[1]["high"], 106.0)


if __name__ == "__main__":
    unittest.main()
