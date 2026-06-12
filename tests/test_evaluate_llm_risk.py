import unittest

from pas_mri_extractor.stages import StageResult, StageStatus
from scripts.evaluate_llm_risk import (
    calculate_summary,
    extract_case_fields,
    join_gold_with_text_records,
    process_record,
)


LLM_RISK = {
    "risk_assessment": {
        "massive_blood_loss_risk_percent": 60,
        "estimated_blood_loss_ml": 1400,
        "estimated_blood_loss_range": "1000–1500 мл",
        "vascular_intervention_risk_percent": 55,
        "bladder_involvement_risk_percent": 40,
        "hysterectomy_risk_percent": 70,
        "transfusion_risk_percent": 80,
    },
    "readiness": {
        "level": "3",
        "rationale": "mock rationale",
    },
    "operative_risk_summary": {
        "text": "mock operative risk",
    },
    "clinical_summary": {
        "text": "mock summary",
    },
    "confidence": "medium",
}


class EvaluateLLMRiskTest(unittest.TestCase):
    def test_extract_case_fields_supports_aliases_and_nested_actual(self) -> None:
        record = {
            "id": "case-1",
            "MRI_description": "MRI report",
            "actual_outcome": {
                "blood_loss_ml": "1500 мл",
                "transfusion": "yes",
                "hysterectomy": "no",
                "vascular_stage": "1",
                "bladder_injury": "нет",
                "pas_type": "increta",
                "readiness_level": "3",
            },
        }

        case_id, text, actual, warnings = extract_case_fields(record, 0)

        self.assertEqual(case_id, "case-1")
        self.assertEqual(text, "MRI report")
        self.assertEqual(actual["blood_loss_ml"], 1500)
        self.assertIs(actual["transfusion"], True)
        self.assertIs(actual["hysterectomy"], False)
        self.assertIs(actual["vascular_intervention"], True)
        self.assertIs(actual["bladder_involvement"], False)
        self.assertEqual(actual["final_pas"], "increta")
        self.assertEqual(actual["readiness_level"], "3")
        self.assertNotIn("actual field not found: blood_loss_ml", warnings)

    def test_calculate_summary_uses_threshold_binary_metrics(self) -> None:
        records = [
            {
                "status": "success",
                "actual": {
                    "blood_loss_ml": 1500,
                    "readiness_level": "3",
                    "transfusion": True,
                    "hysterectomy": True,
                    "vascular_intervention": True,
                    "bladder_involvement": False,
                },
                "llm_risk": LLM_RISK,
            },
            {
                "status": "failed",
                "actual": {},
                "llm_risk": None,
            },
        ]

        summary = calculate_summary(records)

        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_success"], 1)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["blood_loss_mae_ml"], 100)
        self.assertEqual(summary["readiness_exact_match"], 1.0)
        self.assertEqual(summary["binary_metrics"]["transfusion"]["tp"], 1)
        self.assertEqual(summary["binary_metrics"]["hysterectomy"]["tp"], 1)
        self.assertEqual(summary["binary_metrics"]["vascular_intervention"]["tp"], 1)
        self.assertEqual(summary["binary_metrics"]["bladder_involvement"]["tn"], 1)

    def test_process_record_uses_mocked_stages_without_model(self) -> None:
        record = {
            "case_id": "case-1",
            "report_text": "MRI report",
            "gold_blood_loss_ml": 1500,
        }

        def mock_run_case_pipeline(text, model_id):
            return [
                StageResult(
                    stage_name="ExtractorStage",
                    status=StageStatus.SUCCESS,
                    output={
                        "extracted_result": {"evidence": {}},
                        "evidence": {},
                        "schema_version": "1.0",
                    },
                ),
                StageResult(
                    stage_name="RiskPredictionStage",
                    status=StageStatus.SUCCESS,
                    output={
                        "predicted_risks": {
                            "estimated_blood_loss_ml_range": "1000–1500 мл",
                        }
                    },
                ),
            ]

        def mock_run_risk_prediction(**kwargs):
            return StageResult(
                stage_name="LLMRiskPredictionStage",
                status=StageStatus.SUCCESS,
                output=LLM_RISK,
                metadata={
                    "debug_artifacts": {
                        "prompt": "should be stripped",
                    }
                },
            )

        result = process_record(
            record=record,
            index=0,
            model_id="mock-model",
            text_field="auto",
            dry_run=False,
            shared_model=object(),
            run_case_pipeline_fn=mock_run_case_pipeline,
            run_risk_prediction_fn=mock_run_risk_prediction,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["case_id"], "case-1")
        self.assertEqual(result["llm_risk"], LLM_RISK)
        self.assertNotIn("debug_artifacts", result["llm_risk"])

    def test_join_gold_with_text_records_builds_mri_text_from_text_input(self) -> None:
        gold_records = [
            {
                "case_id": "case-1",
                "gold_blood_loss_ml": 1500,
                "gold_massive_blood_loss": "yes",
                "gold_bladder_involvement": "no",
                "gold_vascular_intervention": "yes",
                "gold_pas_type": "increta",
                "gold_readiness_level": "3",
            }
        ]
        text_records = [
            {
                "case_id": "case-1",
                "МРТ_Описание": "Описание МРТ case-1",
                "МРТ_Заключение": "Заключение МРТ case-1",
            }
        ]

        joined = join_gold_with_text_records(gold_records, text_records)
        case_id, text, actual, warnings = extract_case_fields(joined[0], 0)

        self.assertEqual(case_id, "case-1")
        self.assertEqual(
            text,
            "Описание:\nОписание МРТ case-1\n\n"
            "Заключение:\nЗаключение МРТ case-1",
        )
        self.assertEqual(actual["blood_loss_ml"], 1500)
        self.assertIs(actual["massive_blood_loss"], True)
        self.assertIs(actual["bladder_involvement"], False)
        self.assertIs(actual["vascular_intervention"], True)
        self.assertEqual(actual["final_pas"], "increta")
        self.assertEqual(actual["readiness_level"], "3")
        self.assertFalse(any("text not found" in warning for warning in warnings))

    def test_process_record_fails_when_joined_text_is_missing(self) -> None:
        gold_records = [{"case_id": "case-2", "gold_blood_loss_ml": 1200}]
        joined = join_gold_with_text_records(gold_records, [], join_key="case_id")

        result = process_record(
            record=joined[0],
            index=0,
            model_id="mock-model",
            text_field="auto",
            dry_run=True,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing MRI text", result["errors"])
        self.assertTrue(
            any("text not found for case_id=case-2" in warning for warning in result["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
