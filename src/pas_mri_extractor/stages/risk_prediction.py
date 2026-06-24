from __future__ import annotations

from typing import Any

from pas_mri_extractor.scoring import normalize_mri_result

from .base import PipelineContext, StageResult, StageStatus, to_serializable


class RiskPredictionStage:
    name = "RiskPredictionStage"

    def run(self, context: PipelineContext) -> StageResult:
        if context.extracted_result is None:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
                warnings=["No extracted_result is available for risk prediction."],
            )

        try:
            full_result = normalize_mri_result(context.extracted_result).model_dump()
        except Exception as error:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                metadata=dict(context.metadata),
                error=str(error),
            )

        output: dict[str, Any] = {
            "score": full_result["score"],
            "predicted_risks": full_result["predicted_risks"],
            "recommendation": full_result["recommendation"],
            "computed_rationale": full_result["computed_rationale"],
            "result": full_result,
            "case_context": {
                "source_text": context.source_text,
                "conclusion_text": context.conclusion_text,
                "extracted_result": to_serializable(context.extracted_result),
                "evidence": context.evidence,
                "metadata": dict(context.metadata),
            },
        }

        context.predicted_risks = output["predicted_risks"]
        context.metadata["schema_version"] = full_result.get("schema_version")

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            output=output,
            metadata=dict(context.metadata),
        )
