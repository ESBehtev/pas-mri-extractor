import unittest

from pydantic import ValidationError

from pas_mri_extractor.schemas import FullMRIResult, MRIExtractionResult


CANONICAL_PAYLOAD = {
    "schema_version": "1.0",
    "case_info": {
        "gestational_week": 34,
        "previous_cs_count": 2,
    },
    "extracted_features": {
        "invasion": {
            "type": "increta",
            "confidence": "probable",
        },
        "anatomy": {
            "bladder_involvement": "possible",
            "parametrium_involvement": "absent",
            "posterior_wall_involvement": "absent",
        },
        "placenta_location": {
            "placenta_previa": "present",
            "anterior_placenta": "present",
        },
        "mri_signs": {
            "retroplacental_vessels": "present",
            "lacunae": "present",
            "uterine_wall_thinning": "present",
            "uterine_hernia_or_bulging": "absent",
        },
        "clinical_context": {
            "preoperative_bleeding": "absent",
        },
    },
    "evidence": {
        "positive_findings": [],
        "uncertain_findings": [],
        "negative_findings": [],
    },
}

FULL_PAYLOAD = {
    **CANONICAL_PAYLOAD,
    "score": {
        "clinical_score": 9,
        "risk_group": "moderate",
        "red_flag": 0,
        "score_reasons": "increta: +3",
    },
    "predicted_risks": {
        "massive_blood_loss_over_1500_ml_percent": 35,
        "estimated_blood_loss_ml_range": "1000-1500 ml",
        "vascular_intervention_percent": 25,
        "bladder_involvement_percent": 15,
        "risk_summary_text": "summary",
    },
    "recommendation": {
        "readiness_level": "2",
        "readiness_text": "moderate risk",
    },
    "computed_rationale": "computed rationale",
}


class SchemaContractTest(unittest.TestCase):
    def test_canonical_nested_payload_is_valid(self) -> None:
        result = MRIExtractionResult.model_validate(CANONICAL_PAYLOAD)

        self.assertEqual(result.schema_version, "1.0")
        self.assertEqual(result.extracted_features.invasion.type, "increta")

    def test_full_output_payload_requires_scoring_blocks(self) -> None:
        result = FullMRIResult.model_validate(FULL_PAYLOAD)

        self.assertEqual(result.score.risk_group, "moderate")
        self.assertEqual(result.recommendation.readiness_level, "2")

    def test_legacy_flat_features_payload_is_rejected(self) -> None:
        legacy_payload = {
            "features": {
                "invasion_type": "increta",
                "placenta_previa": "present",
            }
        }

        with self.assertRaises(ValidationError) as context:
            MRIExtractionResult.model_validate(legacy_payload)

        self.assertIn("Legacy flat 'features' format", str(context.exception))


if __name__ == "__main__":
    unittest.main()
