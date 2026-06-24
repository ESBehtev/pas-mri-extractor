import unittest

from app.llm_risk_helpers import (
    build_extracted_result_for_llm_risk,
    format_ml,
    format_percent,
    normalize_llm_risk,
    normalize_rule_based_risk,
    risk_level_from_percent,
    reset_llm_risk_state_values,
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

    def test_format_percent_and_ml_handle_missing_values(self) -> None:
        self.assertEqual(format_percent(None), "—")
        self.assertEqual(format_percent(25), "25%")
        self.assertEqual(format_percent(12.5), "12.5%")
        self.assertEqual(format_ml(None), "—")
        self.assertEqual(format_ml(1200), "1200 мл")

    def test_risk_level_from_percent(self) -> None:
        self.assertEqual(risk_level_from_percent(None), "unknown")
        self.assertEqual(risk_level_from_percent(10), "low")
        self.assertEqual(risk_level_from_percent(25), "moderate")
        self.assertEqual(risk_level_from_percent(50), "high")
        self.assertEqual(risk_level_from_percent(75), "very_high")

    def test_normalize_rule_based_risk(self) -> None:
        result = {
            "schema_version": "1.0",
            "case_info": {"gestational_week": 34},
            "extracted_features": {"invasion": {"type": "increta"}},
            "evidence": {"positive_findings": ["lacunae"]},
            "predicted_risks": {
                "massive_blood_loss_over_1500_ml_percent": 70,
                "estimated_blood_loss_ml_range": "1500–3000 мл",
                "vascular_intervention_percent": 50,
                "bladder_involvement_percent": 30,
                "risk_summary_text": "summary",
            },
            "recommendation": {
                "readiness_level": "3",
                "readiness_text": "high readiness",
            },
            "score": {
                "score_reasons": "increta: +3",
            },
        }

        normalized = normalize_rule_based_risk(result)

        self.assertTrue(normalized["available"])
        self.assertEqual(normalized["massive_blood_loss_risk"], 70)
        self.assertEqual(normalized["estimated_blood_loss"], "1500–3000 мл")
        self.assertIsNone(normalized["hysterectomy_risk"])
        self.assertEqual(normalized["readiness_level"], "3")
        self.assertEqual(
            set(normalized["raw"]),
            {"score", "predicted_risks", "recommendation", "computed_rationale"},
        )
        self.assertNotIn("schema_version", normalized["raw"])
        self.assertNotIn("extracted_features", normalized["raw"])
        self.assertNotIn("evidence", normalized["raw"])

    def test_normalize_llm_risk_success_and_skipped(self) -> None:
        payload = {
            "status": "success",
            "llm_risk": {
                "risk_assessment": {
                    "massive_blood_loss_risk_percent": 65,
                    "estimated_blood_loss_ml": 1800,
                    "estimated_blood_loss_range": "1500–2500 мл",
                    "vascular_intervention_risk_percent": 55,
                    "bladder_involvement_risk_percent": 30,
                    "hysterectomy_risk_percent": 25,
                    "transfusion_risk_percent": 70,
                },
                "readiness": {
                    "level": "3",
                    "rationale": "rationale",
                },
                "operative_risk_summary": {
                    "text": "operative",
                },
                "clinical_summary": {
                    "text": "clinical",
                },
                "confidence": "medium",
            },
        }

        normalized = normalize_llm_risk(payload)
        skipped = normalize_llm_risk(None)

        self.assertTrue(normalized["available"])
        self.assertEqual(normalized["estimated_blood_loss"], 1800)
        self.assertEqual(normalized["hysterectomy_risk"], 25)
        self.assertEqual(normalized["confidence"], "medium")
        self.assertFalse(skipped["available"])
        self.assertEqual(skipped["status"], "skipped")

    def test_normalize_llm_risk_running_and_failed(self) -> None:
        running = normalize_llm_risk({"status": "running"})
        failed = normalize_llm_risk(
            {
                "status": "failed",
                "errors": ["boom"],
                "warnings": ["check input"],
            }
        )

        self.assertFalse(running["available"])
        self.assertEqual(running["status"], "running")
        self.assertIn("Выполняется", running["message"])
        self.assertFalse(failed["available"])
        self.assertEqual(failed["errors"], ["boom"])
        self.assertEqual(failed["warnings"], ["check input"])

    def test_reset_llm_risk_state_clears_previous_result(self) -> None:
        state = {
            "last_llm_risk_result": {"status": "success"},
            "llm_risk_result": {"risk_assessment": {}},
            "llm_risk_status": "success",
            "llm_risk_errors": ["old"],
            "llm_risk_warnings": ["old"],
        }

        reset_llm_risk_state_values(state)

        self.assertIsNone(state["last_llm_risk_result"])
        self.assertIsNone(state["llm_risk_result"])
        self.assertEqual(state["llm_risk_status"], "skipped")
        self.assertEqual(state["llm_risk_errors"], [])
        self.assertEqual(state["llm_risk_warnings"], [])


if __name__ == "__main__":
    unittest.main()
