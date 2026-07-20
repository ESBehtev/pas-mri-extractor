import unittest
from pas_mri_extractor.core.rules import rule_extract_features
from pas_mri_extractor.core.scoring import normalize_mri_result


class RulesSampleRegressionTest(unittest.TestCase):
    def test_sample_mri_rules_pipeline_does_not_regress_to_percreta(self) -> None:
        text = "Беременность 34 недели. Рубец на матке после 2 КС. Признаки placenta increta."

        extraction = rule_extract_features(text)
        result = normalize_mri_result(extraction)

        self.assertEqual(result.schema_version, "1.0")
        self.assertEqual(result.case_info.previous_cs_count, 2)
        self.assertEqual(result.extracted_features.invasion.type, "increta")
        self.assertNotEqual(result.extracted_features.invasion.type, "percreta")
        self.assertEqual(result.score.risk_group, "moderate")


if __name__ == "__main__":
    unittest.main()
