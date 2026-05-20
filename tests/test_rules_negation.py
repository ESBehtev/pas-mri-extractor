import unittest

from pas_mri_extractor.rules import rule_extract_features


class RuleNegationTest(unittest.TestCase):
    def test_bladder_involvement_absent_when_negated(self) -> None:
        result = rule_extract_features(
            "Признаков инвазии мочевого пузыря нет."
        )

        self.assertEqual(
            result.extracted_features.anatomy.bladder_involvement,
            "absent",
        )
        self.assertEqual(result.extracted_features.invasion.type, "none")
        self.assertTrue(result.evidence.negative_findings)

    def test_lacunae_absent_when_negated(self) -> None:
        result = rule_extract_features(
            "Плацентарные лакуны не определяются."
        )

        self.assertEqual(result.extracted_features.mri_signs.lacunae, "absent")
        self.assertTrue(result.evidence.negative_findings)

    def test_invasion_absent_when_negated(self) -> None:
        result = rule_extract_features(
            "Данных за врастание плаценты нет."
        )

        self.assertEqual(result.extracted_features.invasion.type, "none")
        self.assertEqual(result.extracted_features.invasion.confidence, "absent")
        self.assertTrue(result.evidence.negative_findings)

    def test_uncertain_bladder_involvement_is_preserved(self) -> None:
        result = rule_extract_features(
            "Нельзя исключить вовлечение мочевого пузыря."
        )

        self.assertEqual(
            result.extracted_features.anatomy.bladder_involvement,
            "possible",
        )
        self.assertIn(
            "возможное вовлечение мочевого пузыря",
            result.evidence.uncertain_findings,
        )


if __name__ == "__main__":
    unittest.main()
