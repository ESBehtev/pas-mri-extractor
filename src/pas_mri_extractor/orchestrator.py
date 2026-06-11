from __future__ import annotations

from pas_mri_extractor.stages import (
    ExtractorStage,
    PipelineContext,
    RiskPredictionStage,
    StageResult,
    StageStatus,
)


class Orchestrator:
    def __init__(self, stages: list[object]) -> None:
        self.stages = stages

    def run(self, context: PipelineContext) -> list[StageResult]:
        results: list[StageResult] = []

        for stage in self.stages:
            result = stage.run(context)
            results.append(result)

            if result.status == StageStatus.FAILED:
                break

        return results


def run_case_pipeline(
    text: str,
    model_id: str | None = None,
) -> list[StageResult]:
    context = PipelineContext(
        source_text=text,
        metadata={"model_id": model_id},
    )
    orchestrator = Orchestrator(
        stages=[
            ExtractorStage(model_id=model_id),
            RiskPredictionStage(),
        ],
    )

    return orchestrator.run(context)
