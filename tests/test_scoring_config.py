import copy
import unittest
from unittest.mock import patch

from pas_mri_extractor import scoring


SCORING_PAYLOAD = {
    "schema_version": "1.0",
    "case_info": {
        "gestational_week": 34,
        "previous_cs_count": None,
    },
    "extracted_features": {
        "invasion": {
            "type": "none",
            "confidence": "absent",
        },
        "anatomy": {
            "bladder_involvement": "absent",
            "parametrium_involvement": "absent",
            "posterior_wall_involvement": "absent",
        },
        "placenta_location": {
            "placenta_previa": "present",
            "anterior_placenta": "absent",
        },
        "mri_signs": {
            "retroplacental_vessels": "absent",
            "lacunae": "absent",
            "uterine_wall_thinning": "absent",
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


class ScoringConfigTest(unittest.TestCase):
    def test_scoring_uses_yaml_weights_thresholds_and_readiness(self) -> None:
        test_cfg = copy.deepcopy(scoring.score_cfg)
        test_cfg["scoring"]["features"]["placenta_previa"]["present"] = 5
        test_cfg["risk_groups"] = {
            "low": {"min": 0, "max": 2},
            "moderate": {"min": 3, "max": 4},
            "high": {"min": 5, "max": 100},
        }
        test_cfg["risk_predictions"]["high"]["blood_loss_range"] = "yaml high"
        test_cfg["readiness_levels"]["high"]["level"] = "Y"
        test_cfg["readiness_levels"]["high"]["text"] = "YAML readiness"

        with patch.object(scoring, "score_cfg", test_cfg):
            result = scoring.normalize_mri_result(SCORING_PAYLOAD)

        self.assertEqual(result.score.clinical_score, 5)
        self.assertEqual(result.score.risk_group, "high")
        self.assertEqual(
            result.predicted_risks.estimated_blood_loss_ml_range,
            "yaml high",
        )
        self.assertEqual(result.recommendation.readiness_level, "Y")
        self.assertEqual(result.recommendation.readiness_text, "YAML readiness")


if __name__ == "__main__":
    unittest.main()
