"""Stage-обёртка над текущим extraction pipeline.
Зачем нужен:
- не менять существующую extractor logic и prompt;
- упаковать результат извлечения в StageResult;
- передать extracted_result/evidence дальше по PipelineContext.
"""

from __future__ import annotations

from typing import Any, Callable

from pas_mri_extractor.pipeline import extract_features_with_artifacts

from .base import PipelineContext, StageResult, StageStatus, to_serializable


ExtractorRunner = Callable[..., dict[str, Any]]


def _get_evidence(extracted_result: Any) -> dict[str, Any] | None:
    if extracted_result is None:
        return None

    if hasattr(extracted_result, "evidence"):
        return to_serializable(extracted_result.evidence)

    if isinstance(extracted_result, dict):
        evidence = extracted_result.get("evidence")
        return evidence if isinstance(evidence, dict) else None

    return None


def _get_schema_version(extracted_result: Any, result: Any) -> str | None:
    if hasattr(extracted_result, "schema_version"):
        return str(extracted_result.schema_version)

    if isinstance(extracted_result, dict):
        schema_version = extracted_result.get("schema_version")
        return str(schema_version) if schema_version is not None else None

    if isinstance(result, dict):
        schema_version = result.get("schema_version")
        return str(schema_version) if schema_version is not None else None

    return None


class ExtractorStage:
    name = "ExtractorStage"

    def __init__(
        self,
        model_id: str | None = None,
        runner: ExtractorRunner = extract_features_with_artifacts,
        print_raw_output: bool = False,
    ) -> None:
        self.model_id = model_id
        self.runner = runner
        self.print_raw_output = print_raw_output

    def run(self, context: PipelineContext) -> StageResult:
        try:
            artifacts = self.runner(
                text=context.source_text,
                model_name=self.model_id,
                print_raw_output=self.print_raw_output,
            )
        except Exception as error:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                metadata={"model_id": self.model_id},
                error=str(error),
            )

        extracted_result = artifacts.get("validated") or artifacts.get("result")
        context.extracted_result = extracted_result
        context.evidence = _get_evidence(extracted_result)
        context.metadata["model_id"] = self.model_id

        result = artifacts.get("result")
        schema_version = _get_schema_version(extracted_result, result)
        context.metadata["schema_version"] = schema_version

        output = {
            "extracted_result": to_serializable(extracted_result),
            "evidence": context.evidence,
            "schema_version": schema_version,
        }

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            output=output,
            metadata={
                "model_id": self.model_id,
                "schema_version": schema_version,
                "debug_artifacts": artifacts,
            },
        )
