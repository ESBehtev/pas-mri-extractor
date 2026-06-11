import unittest

from pydantic import ValidationError

from pas_mri_extractor.orchestrator import run_risk_prediction_experiment
from pas_mri_extractor.prompt_registry import load_stage_prompt
from pas_mri_extractor.stages import (
    LLMRiskPredictionOutput,
    LLMRiskPredictionStage,
    PipelineContext,
    StageStatus,
)


EXTRACTED_RESULT = {
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
        "positive_findings": ["lacunae"],
        "uncertain_findings": [],
        "negative_findings": [],
    },
}


LLM_OUTPUT = {
    "risk_assessment": {
        "blood_loss_risk_percent": 45,
        "blood_loss_range": "1000-1500 ml",
        "vascular_intervention_risk_percent": 30,
        "bladder_involvement_risk_percent": 20,
    },
    "readiness": {
        "level": "2",
        "rationale": "probable increta with possible bladder involvement",
    },
    "clinical_summary": {
        "text": "Research-only PAS risk summary.",
    },
    "confidence": "medium",
}


class LLMRiskPredictionStageTest(unittest.TestCase):
    def test_llm_risk_prediction_schema_accepts_expected_output(self) -> None:
        result = LLMRiskPredictionOutput.model_validate(LLM_OUTPUT)

        self.assertEqual(result.risk_assessment.blood_loss_risk_percent, 45)
        self.assertEqual(result.confidence, "medium")

    def test_llm_risk_prediction_schema_rejects_unknown_confidence(self) -> None:
        payload = {
            **LLM_OUTPUT,
            "confidence": "uncertain",
        }

        with self.assertRaises(ValidationError):
            LLMRiskPredictionOutput.model_validate(payload)

    def test_prompt_registry_loads_experimental_risk_prediction_prompt(self) -> None:
        prompt_config = load_stage_prompt("risk_prediction")

        self.assertEqual(prompt_config["stage"], "risk_prediction")
        self.assertEqual(prompt_config["status"], "experimental")
        self.assertFalse(prompt_config["runtime_active"])

    def test_llm_risk_prediction_stage_wires_context_into_prompt(self) -> None:
        captured = {}

        def mock_runner(prompt, model_id):
            captured["prompt"] = prompt
            captured["model_id"] = model_id
            return LLM_OUTPUT

        context = PipelineContext(
            source_text="MRI source text",
            conclusion_text="MRI conclusion text",
            extracted_result=EXTRACTED_RESULT,
            evidence=EXTRACTED_RESULT["evidence"],
            metadata={"case_id": "case-1"},
        )

        result = LLMRiskPredictionStage(
            model_id="mock-model",
            runner=mock_runner,
        ).run(context)

        self.assertEqual(result.status, StageStatus.SUCCESS)
        self.assertEqual(result.output, LLM_OUTPUT)
        self.assertEqual(context.predicted_risks, LLM_OUTPUT["risk_assessment"])
        self.assertEqual(captured["model_id"], "mock-model")
        self.assertIn("MRI source text", captured["prompt"])
        self.assertIn("MRI conclusion text", captured["prompt"])
        self.assertIn("lacunae", captured["prompt"])
        self.assertIn("debug_artifacts", result.metadata)

    def test_risk_prediction_experiment_runs_without_main_orchestrator_change(self) -> None:
        def mock_runner(prompt, model_id):
            return LLM_OUTPUT

        result = run_risk_prediction_experiment(
            text="MRI source text",
            extracted_result=EXTRACTED_RESULT,
            model_id="mock-model",
            runner=mock_runner,
        )

        self.assertEqual(result.stage_name, "LLMRiskPredictionStage")
        self.assertEqual(result.status, StageStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
