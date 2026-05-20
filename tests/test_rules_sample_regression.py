import unittest
from pathlib import Path

from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


class RulesSampleRegressionTest(unittest.TestCase):
    def test_sample_mri_rules_pipeline_does_not_regress_to_percreta(self) -> None:
        text = Path("examples/sample_mri.txt").read_text(encoding="utf-8")

        extraction = rule_extract_features(text)
        result = normalize_mri_result(extraction)

        self.assertEqual(result.schema_version, "1.0")
        self.assertEqual(result.case_info.previous_cs_count, 2)
        self.assertEqual(result.extracted_features.invasion.type, "increta")
        self.assertNotEqual(result.extracted_features.invasion.type, "percreta")
        self.assertEqual(
            result.extracted_features.anatomy.bladder_involvement,
            "possible",
        )
        self.assertEqual(result.score.red_flag, 0)
        self.assertEqual(result.score.clinical_score, 11)
        self.assertEqual(result.score.risk_group, "high")
        self.assertEqual(
            result.predicted_risks.estimated_blood_loss_ml_range,
            "1500–2500 мл",
        )
        self.assertEqual(result.recommendation.readiness_level, "3")


if __name__ == "__main__":
    unittest.main()
