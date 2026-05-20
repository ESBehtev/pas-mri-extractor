import copy
import unittest
from pathlib import Path

import yaml

from pas_mri_extractor.config import clear_config_cache, config_overrides, load_config
from pas_mri_extractor import scoring
from pas_mri_extractor.rules import rule_extract_features


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

    def write_stale_runtime_override(self, config: dict) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def test_scoring_uses_explicit_in_memory_override(self) -> None:
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

        with config_overrides({"risk_score.yaml": test_cfg}):
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

    def test_load_config_ignores_stale_runtime_file_on_disk(self) -> None:
        stale_cfg = copy.deepcopy(scoring.get_score_config())
        stale_cfg["scoring"]["features"]["placenta_previa"]["present"] = 99
        self.write_stale_runtime_override(stale_cfg)
        clear_config_cache()
        scoring.clear_score_config_cache()

        config = load_config("risk_score.yaml")

        self.assertEqual(
            config["scoring"]["features"]["placenta_previa"]["present"],
            1,
        )

    def test_sample_regression_ignores_stale_runtime_file_on_disk(self) -> None:
        stale_cfg = copy.deepcopy(scoring.get_score_config())
        stale_cfg["scoring"]["invasion_type"]["increta"] = 10
        self.write_stale_runtime_override(stale_cfg)
        clear_config_cache()
        scoring.clear_score_config_cache()

        text = Path("examples/sample_mri.txt").read_text(encoding="utf-8")
        extraction = rule_extract_features(text)
        result = scoring.normalize_mri_result(extraction)

        self.assertEqual(result.score.clinical_score, 11)
        self.assertEqual(result.score.risk_group, "high")


if __name__ == "__main__":
    unittest.main()
