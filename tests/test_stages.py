import json
import unittest

from pas_mri_extractor.orchestrator import Orchestrator
from pas_mri_extractor.schemas import MRIExtractionResult
from pas_mri_extractor.scoring import normalize_mri_result
from pas_mri_extractor.stages import (
    ExtractorStage,
    PipelineContext,
    RiskPredictionStage,
    StageResult,
    StageStatus,
)


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
        "positive_findings": ["lacunae"],
        "uncertain_findings": [],
        "negative_findings": [],
    },
}


class StageContractTest(unittest.TestCase):
    def test_stage_result_to_dict_is_json_serializable(self) -> None:
        validated = MRIExtractionResult.model_validate(CANONICAL_PAYLOAD)
        result = StageResult(
            stage_name="test",
            status=StageStatus.SUCCESS,
            output={"validated": validated},
            metadata={"model_id": "mock"},
        )

        payload = result.to_dict()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(result["stage_name"], "test")
        json.dumps(payload, ensure_ascii=False)

    def test_risk_prediction_stage_uses_existing_scoring(self) -> None:
        context = PipelineContext(
            source_text="MRI report text",
            conclusion_text=None,
            extracted_result=CANONICAL_PAYLOAD,
            evidence=CANONICAL_PAYLOAD["evidence"],
            metadata={"model_id": "mock"},
        )

        stage_result = RiskPredictionStage().run(context)
        expected = normalize_mri_result(CANONICAL_PAYLOAD).model_dump()

        self.assertEqual(stage_result.status, StageStatus.SUCCESS)
        self.assertEqual(
            stage_result.output["predicted_risks"],
            expected["predicted_risks"],
        )
        self.assertEqual(
            stage_result.output["recommendation"],
            expected["recommendation"],
        )
        self.assertEqual(context.predicted_risks, expected["predicted_risks"])
        self.assertEqual(
            stage_result.output["case_context"]["source_text"],
            "MRI report text",
        )

    def test_orchestrator_runs_extractor_then_risk_prediction(self) -> None:
        validated = MRIExtractionResult.model_validate(CANONICAL_PAYLOAD)
        full_result = normalize_mri_result(validated).model_dump()

        def mock_runner(**kwargs):
            self.assertEqual(kwargs["text"], "MRI report text")
            self.assertEqual(kwargs["model_name"], "mock-model")
            return {
                "prompt": "mock prompt",
                "raw_output": json.dumps(CANONICAL_PAYLOAD),
                "parsed": CANONICAL_PAYLOAD,
                "validated": validated,
                "result": full_result,
            }

        context = PipelineContext(source_text="MRI report text")
        orchestrator = Orchestrator(
            stages=[
                ExtractorStage(model_id="mock-model", runner=mock_runner),
                RiskPredictionStage(),
            ]
        )

        results = orchestrator.run(context)

        self.assertEqual(
            [result.stage_name for result in results],
            ["ExtractorStage", "RiskPredictionStage"],
        )
        self.assertEqual([result.status for result in results], [StageStatus.SUCCESS] * 2)
        self.assertIsNotNone(context.extracted_result)
        self.assertIsNotNone(context.predicted_risks)

        extractor_result = results[0]
        self.assertEqual(
            set(extractor_result.output),
            {"extracted_result", "evidence", "schema_version"},
        )
        self.assertEqual(extractor_result.output["schema_version"], "1.0")
        self.assertEqual(
            extractor_result.output["evidence"],
            CANONICAL_PAYLOAD["evidence"],
        )
        self.assertNotIn("prompt", extractor_result.output)
        self.assertNotIn("raw_output", extractor_result.output)
        self.assertNotIn("parsed", extractor_result.output)
        self.assertEqual(
            extractor_result.metadata["debug_artifacts"]["prompt"],
            "mock prompt",
        )
        self.assertIn("raw_output", extractor_result.metadata["debug_artifacts"])
        self.assertIn("parsed", extractor_result.metadata["debug_artifacts"])

    def test_extractor_stage_returns_failed_result_on_runner_error(self) -> None:
        def failing_runner(**kwargs):
            raise ValueError("mock extraction failed")

        context = PipelineContext(source_text="MRI report text")

        result = ExtractorStage(runner=failing_runner).run(context)

        self.assertEqual(result.status, StageStatus.FAILED)
        self.assertIn("mock extraction failed", result.error)


if __name__ == "__main__":
    unittest.main()
