from __future__ import annotations

import unittest
from datetime import date

from app.services.time_ranges import normalize_date_range, subtract_period


class TimeRangeTests(unittest.TestCase):
    def test_days_and_weeks(self):
        end = date(2026, 10, 16)
        self.assertEqual(subtract_period(end, 10, "days"), date(2026, 10, 6))
        self.assertEqual(subtract_period(end, 2, "weeks"), date(2026, 10, 2))

    def test_months_use_calendar_arithmetic(self):
        self.assertEqual(subtract_period(date(2026, 10, 31), 1, "months"), date(2026, 9, 30))

    def test_years_handle_leap_dates(self):
        self.assertEqual(subtract_period(date(2024, 2, 29), 1, "years"), date(2023, 2, 28))

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            subtract_period(date(2026, 1, 1), 0, "days")
        with self.assertRaises(ValueError):
            subtract_period(date(2026, 1, 1), 1, "fortnights")

    def test_normalizes_custom_date_input(self):
        start, end = normalize_date_range((date(2026, 1, 1), date(2026, 3, 1)), date(2025, 1, 1), date(2026, 1, 1))
        self.assertEqual((start, end), (date(2026, 1, 1), date(2026, 3, 1)))


if __name__ == "__main__":
    unittest.main()
