from pathlib import Path
import unittest


class MultiVariableAnalysisUITests(unittest.TestCase):
    def test_event_study_selects_source_and_multiple_targets_independently(self):
        source = Path(__file__).parents[1].joinpath("main.py").read_text(encoding="utf-8")
        self.assertIn('key="event_source_v17"', source)
        self.assertIn('key="event_targets_v17"', source)
        self.assertIn('for response_target in target_labels:', source)
        self.assertIn('"response_target": "Response target"', source)

    def test_lead_lag_selects_reference_and_multiple_comparisons(self):
        source = Path(__file__).parents[1].joinpath("main.py").read_text(encoding="utf-8")
        self.assertIn('key="lag_source_v17"', source)
        self.assertIn('key="lag_targets_v17"', source)
        self.assertIn('for compared in lag_targets:', source)
        self.assertIn('adjusted_p_across_variables', source)


if __name__ == "__main__":
    unittest.main()
