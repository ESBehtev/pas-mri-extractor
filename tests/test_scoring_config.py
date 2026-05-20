import copy
import unittest
from pathlib import Path

import yaml

from pas_mri_extractor.config import clear_config_cache
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
    local_path = Path("runtime_configs/risk_score.local.yaml")

    def setUp(self) -> None:
        self.original_local_content = None
        if self.local_path.exists():
            self.original_local_content = self.local_path.read_text(encoding="utf-8")
            self.local_path.unlink()

        clear_config_cache()
        scoring.clear_score_config_cache()

    def tearDown(self) -> None:
        if self.original_local_content is None:
            if self.local_path.exists():
                self.local_path.unlink()
        else:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            self.local_path.write_text(self.original_local_content, encoding="utf-8")

        clear_config_cache()
        scoring.clear_score_config_cache()

    def test_scoring_uses_yaml_weights_thresholds_and_readiness(self) -> None:
        test_cfg = copy.deepcopy(scoring.get_score_config())
        test_cfg["scoring"]["features"]["placenta_previa"]["present"] = 5
        test_cfg["risk_groups"] = {
            "low": {"min": 0, "max": 2},
            "moderate": {"min": 3, "max": 4},
            "high": {"min": 5, "max": 100},
        }
        test_cfg["risk_predictions"]["high"]["blood_loss_range"] = "yaml high"
        test_cfg["readiness_levels"]["high"]["level"] = "Y"
        test_cfg["readiness_levels"]["high"]["text"] = "YAML readiness"

        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(
            yaml.safe_dump(test_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        clear_config_cache()
        scoring.clear_score_config_cache()

        result = scoring.normalize_mri_result(SCORING_PAYLOAD)

        self.assertEqual(result.score.clinical_score, 5)
        self.assertEqual(result.score.risk_group, "high")
        self.assertEqual(
            result.predicted_risks.estimated_blood_loss_ml_range,
            "yaml high",
        )
        self.assertEqual(result.recommendation.readiness_level, "Y")
        self.assertEqual(result.recommendation.readiness_text, "YAML readiness")

    def test_scoring_uses_base_config_when_local_override_absent(self) -> None:
        result = scoring.normalize_mri_result(SCORING_PAYLOAD)

        self.assertEqual(result.score.clinical_score, 1)
        self.assertEqual(result.score.risk_group, "low")
        self.assertEqual(result.recommendation.readiness_level, "1")

    def test_invalid_local_override_does_not_silently_fallback(self) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text("scoring: [", encoding="utf-8")
        clear_config_cache()
        scoring.clear_score_config_cache()

        with self.assertRaises(yaml.YAMLError):
            scoring.normalize_mri_result(SCORING_PAYLOAD)


if __name__ == "__main__":
    unittest.main()
