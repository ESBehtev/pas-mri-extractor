"""Тесты batch-прогона полного dataset_sheet3 pipeline.
Проверяют:
- создание case folder и JSON artifacts;
- сохранение failed case;
- расчёт blood loss metrics;
- skip-existing без запуска pipeline.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pas_mri_extractor.stages import StageResult, StageStatus
from scripts.run_full_dataset_pipeline import (
    calculate_blood_loss_metrics,
    calculate_summary,
    extract_actual_blood_loss,
    process_case,
)


LLM_RISK = {
    "risk_assessment": {
        "estimated_blood_loss_ml": 1400,
        "massive_blood_loss_risk_percent": 60,
        "estimated_blood_loss_range": "1000–1500 мл",
        "vascular_intervention_risk_percent": 50,
        "bladder_involvement_risk_percent": 20,
        "hysterectomy_risk_percent": 30,
        "transfusion_risk_percent": 70,
    },
    "readiness": {"level": "2", "rationale": "mock"},
    "operative_risk_summary": {"text": "operative"},
    "clinical_summary": {"text": "clinical"},
    "confidence": "medium",
}


def mock_extract_success(**kwargs):
    return {
        "result": {
            "schema_version": "1.0",
            "case_info": {"gestational_week": 34},
            "extracted_features": {"invasion": {"type": "increta"}},
            "suspicion": {"highest_suspected_extent": "increta"},
            "evidence": {"positive_findings": ["lacunae"]},
            "score": {"clinical_score": 7},
            "predicted_risks": {"vascular_intervention_percent": 50},
            "recommendation": {"readiness_level": "2"},
            "computed_rationale": "rule rationale",
            "debug_artifacts": {"prompt": "hidden"},
        }
    }


def mock_risk_success(**kwargs):
    return StageResult(
        stage_name="LLMRiskPredictionStage",
        status=StageStatus.SUCCESS,
        output=LLM_RISK,
    )


class FullDatasetPipelineTest(unittest.TestCase):
    def test_process_case_creates_folder_and_saves_original_and_result(self) -> None:
        record = {
            "case_id": "case_000001",
            "МРТ_Описание": "Описание",
            "МРТ_Заключение": "Заключение",
            "КровопотеряРоды": "1000.000000",
            "КровопотеряОперация": "400.000000",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            progress = process_case(
                record=record,
                index=0,
                output_dir=output_dir,
                model_id="mock-model",
                loaded_model=object(),
                extract_fn=mock_extract_success,
                risk_fn=mock_risk_success,
            )

            case_dir = output_dir / "case_000001"
            original = json.loads((case_dir / "original.json").read_text("utf-8"))
            result = json.loads((case_dir / "result.json").read_text("utf-8"))

        self.assertEqual(progress["status"], "success")
        self.assertEqual(original, record)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["llm_risk"], LLM_RISK)
        self.assertEqual(
            result["actual_blood_loss"]["actual_blood_loss_for_metrics_ml"],
            400,
        )
        self.assertEqual(
            result["actual_blood_loss"]["actual_blood_loss_source"],
            "operation_blood_loss",
        )
        self.assertIn("rule_based_risk", result)
        self.assertNotIn("debug_artifacts", result)

    def test_process_case_failed_still_saves_result(self) -> None:
        record = {
            "case_id": "case_000002",
            "МРТ_Описание": "Описание",
            "МРТ_Заключение": "Заключение",
        }

        def mock_extract_failed(**kwargs):
            raise RuntimeError("extract failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            progress = process_case(
                record=record,
                index=1,
                output_dir=output_dir,
                model_id="mock-model",
                loaded_model=object(),
                extract_fn=mock_extract_failed,
                risk_fn=mock_risk_success,
            )
            result = json.loads(
                (output_dir / "case_000002" / "result.json").read_text("utf-8")
            )

        self.assertEqual(progress["status"], "failed")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"], ["extract failed"])

    def test_extract_actual_blood_loss_prefers_text_then_operation_then_birth(self) -> None:
        actual = extract_actual_blood_loss(
            {
                "КровопотеряРоды": "1000.000000",
                "КровопотеряОперация": "600.000000",
                "Ход Вмешательства": "Общая кровопотеря 1600 мл.",
            }
        )

        self.assertEqual(actual["birth_blood_loss_ml"], 1000)
        self.assertEqual(actual["operation_blood_loss_ml"], 600)
        self.assertEqual(actual["total_blood_loss_from_text_ml"], 1600)
        self.assertEqual(actual["actual_blood_loss_for_metrics_ml"], 1600)
        self.assertEqual(actual["actual_blood_loss_source"], "total_blood_loss_from_text")

        operation_only = extract_actual_blood_loss(
            {
                "КровопотеряРоды": "1000.000000",
                "КровопотеряОперация": "600.000000",
            }
        )
        birth_only = extract_actual_blood_loss({"КровопотеряРоды": "1000.000000"})
        unavailable = extract_actual_blood_loss({})

        self.assertEqual(operation_only["actual_blood_loss_for_metrics_ml"], 600)
        self.assertEqual(operation_only["actual_blood_loss_source"], "operation_blood_loss")
        self.assertEqual(birth_only["actual_blood_loss_for_metrics_ml"], 1000)
        self.assertEqual(birth_only["actual_blood_loss_source"], "birth_blood_loss")
        self.assertIsNone(unavailable["actual_blood_loss_for_metrics_ml"])
        self.assertEqual(unavailable["actual_blood_loss_source"], "unavailable")

    def test_blood_loss_metrics(self) -> None:
        results = [
            {
                "status": "success",
                "actual_blood_loss": {"actual_blood_loss_for_metrics_ml": 1000},
                "llm_risk": {"risk_assessment": {"estimated_blood_loss_ml": 1200}},
            },
            {
                "status": "success",
                "actual_blood_loss": {"actual_blood_loss_for_metrics_ml": 2000},
                "llm_risk": {"risk_assessment": {"estimated_blood_loss_ml": 2600}},
            },
            {
                "status": "failed",
                "actual_blood_loss": {"actual_blood_loss_for_metrics_ml": 1000},
                "llm_risk": {"risk_assessment": {"estimated_blood_loss_ml": 2000}},
            },
        ]

        metrics = calculate_blood_loss_metrics(results)

        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae_ml"], 400)
        self.assertAlmostEqual(metrics["mape_percent"], 25.0)
        self.assertAlmostEqual(metrics["rmse_ml"], (200000) ** 0.5)
        self.assertEqual(metrics["within_250_ml"], 0.5)
        self.assertEqual(metrics["within_500_ml"], 0.5)
        self.assertEqual(metrics["within_1000_ml"], 1.0)

    def test_skip_existing_does_not_call_pipeline(self) -> None:
        record = {"case_id": "case_000003", "МРТ_Описание": "Описание"}

        def fail_if_called(**kwargs):
            raise AssertionError("pipeline should not run")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            case_dir = output_dir / "case_000003"
            case_dir.mkdir(parents=True)
            (case_dir / "result.json").write_text(
                json.dumps({"status": "success"}, ensure_ascii=False),
                encoding="utf-8",
            )
            progress = process_case(
                record=record,
                index=2,
                output_dir=output_dir,
                model_id="mock-model",
                loaded_model=object(),
                skip_existing=True,
                extract_fn=fail_if_called,
                risk_fn=mock_risk_success,
            )

        self.assertEqual(progress["status"], "skipped")

    def test_calculate_summary_counts_failed_cases(self) -> None:
        summary = calculate_summary(
            [
                {
                    "case_id": "case_1",
                    "status": "success",
                    "actual_blood_loss": {"actual_blood_loss_for_metrics_ml": 1000},
                    "llm_risk": {
                        "risk_assessment": {"estimated_blood_loss_ml": 1200}
                    },
                },
                {
                    "case_id": "case_2",
                    "status": "failed",
                    "errors": ["boom"],
                    "warnings": [],
                },
            ]
        )

        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_success"], 1)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["failed_cases"][0]["case_id"], "case_2")
        self.assertEqual(summary["blood_loss_metrics"]["mae_ml"], 200)


if __name__ == "__main__":
    unittest.main()
