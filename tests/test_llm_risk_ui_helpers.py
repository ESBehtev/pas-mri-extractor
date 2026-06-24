import unittest

from app.llm_risk_helpers import (
    build_extracted_result_for_llm_risk,
    stage_result_to_llm_risk_ui,
)
from pas_mri_extractor.stages import StageResult, StageStatus


class LLMRiskUIHelpersTest(unittest.TestCase):
    def test_build_extracted_result_for_llm_risk_removes_scoring_blocks(self) -> None:
        result = {
            "schema_version": "1.0",
            "case_info": {"gestational_week": 34},
            "extracted_features": {"invasion": {"type": "increta"}},
            "suspicion": {"percreta_suspicion": "absent"},
            "evidence": {"positive_findings": ["lacunae"]},
            "score": {"clinical_score": 10},
            "predicted_risks": {"estimated_blood_loss_ml_range": "1000–1500 мл"},
            "recommendation": {"readiness_level": "2"},
        }

        extracted = build_extracted_result_for_llm_risk(result)

        self.assertEqual(
            set(extracted),
            {
                "schema_version",
                "case_info",
                "extracted_features",
                "suspicion",
                "evidence",
            },
        )
        self.assertNotIn("score", extracted)
        self.assertNotIn("predicted_risks", extracted)
        self.assertNotIn("recommendation", extracted)

    def test_stage_result_to_llm_risk_ui_omits_debug_artifacts(self) -> None:
        stage_result = StageResult(
            stage_name="LLMRiskPredictionStage",
            status=StageStatus.SUCCESS,
            output={
                "risk_assessment": {
                    "estimated_blood_loss_ml": 1200,
                }
            },
            warnings=["mock warning"],
            metadata={
                "debug_artifacts": {
                    "prompt": "hidden",
                    "raw_output": "hidden",
                }
            },
        )

        payload = stage_result_to_llm_risk_ui(stage_result)

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["warnings"], ["mock warning"])
        self.assertEqual(
            payload["llm_risk"],
            {"risk_assessment": {"estimated_blood_loss_ml": 1200}},
        )
        self.assertNotIn("metadata", payload)


if __name__ == "__main__":
    unittest.main()
