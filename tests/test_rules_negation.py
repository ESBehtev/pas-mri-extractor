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

    def test_placenta_increta_is_detected_as_increta(self) -> None:
        result = rule_extract_features(
            "МР-картина соответствует placenta increta в области рубца."
        )

        self.assertEqual(result.extracted_features.invasion.type, "increta")

    def test_possible_serosal_involvement_does_not_force_percreta(self) -> None:
        result = rule_extract_features(
            "Нельзя исключить начальное вовлечение серозной оболочки."
        )

        self.assertNotEqual(result.extracted_features.invasion.type, "percreta")

    def test_definite_bladder_wall_invasion_can_be_percreta(self) -> None:
        result = rule_extract_features(
            "Определяется достоверная инвазия стенки мочевого пузыря."
        )

        self.assertEqual(result.extracted_features.invasion.type, "percreta")

    def test_sample_like_increta_with_possible_bladder_is_not_percreta(self) -> None:
        result = rule_extract_features(
            "Маточно-пузырное пространство частично не дифференцируется, "
            "однако достоверных признаков инвазии стенки мочевого пузыря "
            "не получено. Заключение: МР-картина соответствует placenta "
            "increta. Признаки возможного начального вовлечения "
            "маточно-пузырного пространства без убедительных данных за "
            "placenta percreta."
        )

        self.assertEqual(result.extracted_features.invasion.type, "increta")
        self.assertNotEqual(
            result.extracted_features.anatomy.bladder_involvement,
            "present",
        )


if __name__ == "__main__":
    unittest.main()
