from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.services.derivative_helpers import (
    add_futures_basis_columns,
    aggregate_option_contracts,
    find_exact_underlying_match,
    normalize_underlying_symbol,
    select_representative_futures_contract,
)


class DerivativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.choices = [
            {"value": "ABC", "label": "ABC · OPTSTK"},
            {"value": "CRUDEOIL", "label": "CRUDEOIL · FUTCOM"},
            {"value": "RELIANCE", "label": "RELIANCE · OPTSTK"},
        ]

    def test_nse_suffix_is_normalized_for_exact_company_link(self):
        self.assertEqual(normalize_underlying_symbol("RELIANCE.NS"), "RELIANCE")
        self.assertEqual(find_exact_underlying_match("RELIANCE.NS", self.choices), "RELIANCE")

    def test_no_exact_match_does_not_choose_first_unrelated_company(self):
        self.assertIsNone(find_exact_underlying_match("BRENT_CRUDE", self.choices))
        self.assertIsNone(find_exact_underlying_match(None, self.choices))

    def test_option_aggregation_sums_strikes_and_expiries_for_selected_underlying_rows(self):
        frame = pd.DataFrame({
            "report_date": pd.to_datetime(["2026-01-02"] * 4 + ["2026-01-05"] * 2),
            "expiry_date": pd.to_datetime([
                "2026-01-29", "2026-01-29", "2026-02-26", "2026-02-26",
                "2026-02-26", "2026-02-26",
            ]),
            "strike_price": [100, 100, 110, 110, 100, 100],
            "option_type": ["CE", "PE", "CE", "PE", "CE", "PE"],
            "open_interest": [100, 100, 50, 50, 0, 80],
            "volume": [10, 20, 5, 10, 2, 4],
        })
        out = aggregate_option_contracts(frame)
        first = out.iloc[0]
        self.assertEqual(first["total_put_oi"], 150)
        self.assertEqual(first["total_call_oi"], 150)
        self.assertAlmostEqual(first["oi_pcr"], 1.0)
        self.assertEqual(first["total_put_volume"], 30)
        self.assertEqual(first["total_call_volume"], 15)
        self.assertAlmostEqual(first["volume_pcr"], 2.0)
        self.assertTrue(np.isnan(out.iloc[1]["oi_pcr"]))

    def test_futures_picker_chooses_highest_open_interest_and_near_expiry_breaks_ties(self):
        frame = pd.DataFrame({
            "report_date": pd.to_datetime(["2026-01-02"] * 3),
            "expiry_date": pd.to_datetime(["2026-01-29", "2026-02-26", "2026-03-26"]),
            "open_interest": [100, 200, 200],
            "close": [101, 102, 103],
        })
        selected = select_representative_futures_contract(frame)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["close"], 102)
        self.assertEqual(selected.iloc[0]["expiry_date"], pd.Timestamp("2026-02-26"))

    def test_basis_is_unavailable_when_record_has_no_underlying_price(self):
        frame = pd.DataFrame({"close": [101.0, 99.0], "underlying_price": [100.0, np.nan]})
        out = add_futures_basis_columns(frame)
        self.assertEqual(out.iloc[0]["absolute_basis"], 1.0)
        self.assertAlmostEqual(out.iloc[0]["basis_percentage"], 1.0)
        self.assertTrue(np.isnan(out.iloc[1]["absolute_basis"]))
        self.assertTrue(np.isnan(out.iloc[1]["basis_percentage"]))


if __name__ == "__main__":
    unittest.main()
