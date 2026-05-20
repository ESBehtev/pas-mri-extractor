import unittest
from pathlib import Path

from pas_mri_extractor.rules import rule_extract_features


class RulePreviousCsCountTest(unittest.TestCase):
    def test_rubec_after_numeric_cs_count(self) -> None:
        result = rule_extract_features(
            "Беременность 34 недели, рубец на матке после 2 КС."
        )

        self.assertEqual(result.case_info.gestational_week, 34)
        self.assertEqual(result.case_info.previous_cs_count, 2)

    def test_textual_cs_count(self) -> None:
        result = rule_extract_features("После двух кесаревых сечений.")

        self.assertEqual(result.case_info.previous_cs_count, 2)

    def test_gestational_week_is_not_previous_cs_count(self) -> None:
        result = rule_extract_features("Беременность 34 недели.")

        self.assertEqual(result.case_info.gestational_week, 34)
        self.assertIsNone(result.case_info.previous_cs_count)

    def test_sample_text_extracts_previous_cs_count(self) -> None:
        text = Path("examples/sample_mri.txt").read_text(encoding="utf-8")
        result = rule_extract_features(text)

        self.assertEqual(result.case_info.previous_cs_count, 2)


if __name__ == "__main__":
    unittest.main()
