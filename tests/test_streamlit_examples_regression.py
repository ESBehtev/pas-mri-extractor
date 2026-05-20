import unittest

from app.examples import STREAMLIT_EXAMPLES
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


EXPECTED_EXAMPLES = {
    "Без PAS — простой случай": {
        "invasion_type": "none",
        "invasion_confidence": "absent",
        "bladder_involvement": "absent",
        "placenta_previa": "absent",
        "anterior_placenta": "absent",
        "clinical_score": 0,
        "risk_group": "low",
        "readiness_level": "1",
        "evidence": ["плацентарные лакуны не определяются"],
    },
    "Без PAS — сложная отрицательная формулировка": {
        "invasion_type": "none",
        "invasion_confidence": "absent",
        "bladder_involvement": "absent",
        "placenta_previa": "absent",
        "anterior_placenta": "present",
        "clinical_score": 1,
        "risk_group": "low",
        "readiness_level": "1",
        "evidence": ["данных за врастание плаценты нет"],
    },
    "Increta — простой случай": {
        "invasion_type": "increta",
        "invasion_confidence": "probable",
        "bladder_involvement": "absent",
        "placenta_previa": "present",
        "anterior_placenta": "present",
        "clinical_score": 10,
        "risk_group": "moderate",
        "readiness_level": "2",
        "evidence": [
            "множественные лакуны",
            "ретроплацентарные сосуды расширены",
            "истончение миометрия",
        ],
    },
    "Increta — сложный случай с возможным вовлечением": {
        "invasion_type": "increta",
        "invasion_confidence": "possible",
        "bladder_involvement": "possible",
        "placenta_previa": "absent",
        "anterior_placenta": "present",
        "clinical_score": 11,
        "risk_group": "high",
        "readiness_level": "3",
        "evidence": [
            "множественные плацентарные лакуны",
            "ретроплацентарные сосуды",
            "маточно-пузырное пространство",
        ],
    },
    "Percreta — простой случай": {
        "invasion_type": "percreta",
        "invasion_confidence": "probable",
        "bladder_involvement": "present",
        "placenta_previa": "present",
        "anterior_placenta": "present",
        "clinical_score": 15,
        "risk_group": "high",
        "readiness_level": "3",
        "evidence": ["достоверная инвазия стенки мочевого пузыря"],
    },
    "Percreta — сложный случай со смешанными формулировками": {
        "invasion_type": "percreta",
        "invasion_confidence": "probable",
        "bladder_involvement": "present",
        "placenta_previa": "absent",
        "anterior_placenta": "present",
        "clinical_score": 13,
        "risk_group": "high",
        "readiness_level": "3",
        "evidence": ["достоверной инвазии стенки мочевого пузыря"],
    },
}


def all_evidence_text(result) -> str:
    evidence = result.evidence
    findings = (
        evidence.positive_findings
        + evidence.uncertain_findings
        + evidence.negative_findings
    )
    return "\n".join(findings).lower()


class StreamlitExamplesRegressionTest(unittest.TestCase):
    def test_all_expected_examples_are_present(self) -> None:
        names = {example["name"] for example in STREAMLIT_EXAMPLES}

        self.assertEqual(names, set(EXPECTED_EXAMPLES))

    def test_streamlit_examples_rules_scoring_contract(self) -> None:
        examples_by_name = {example["name"]: example for example in STREAMLIT_EXAMPLES}

        for name, expected in EXPECTED_EXAMPLES.items():
            with self.subTest(example=name):
                extraction = rule_extract_features(examples_by_name[name]["report_text"])
                result = normalize_mri_result(extraction)

                self.assertEqual(
                    result.extracted_features.invasion.type,
                    expected["invasion_type"],
                )
                self.assertEqual(
                    result.extracted_features.invasion.confidence,
                    expected["invasion_confidence"],
                )
                self.assertEqual(
                    result.extracted_features.anatomy.bladder_involvement,
                    expected["bladder_involvement"],
                )
                self.assertEqual(
                    result.extracted_features.placenta_location.placenta_previa,
                    expected["placenta_previa"],
                )
                self.assertEqual(
                    result.extracted_features.placenta_location.anterior_placenta,
                    expected["anterior_placenta"],
                )
                self.assertEqual(
                    result.score.clinical_score,
                    expected["clinical_score"],
                )
                self.assertEqual(result.score.risk_group, expected["risk_group"])
                self.assertEqual(
                    result.recommendation.readiness_level,
                    expected["readiness_level"],
                )

                evidence_text = all_evidence_text(result)
                for phrase in expected["evidence"]:
                    self.assertIn(phrase, evidence_text)


if __name__ == "__main__":
    unittest.main()
