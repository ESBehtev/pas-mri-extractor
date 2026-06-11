from __future__ import annotations

import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from pas_mri_extractor.prompt_registry import load_stage_prompt

from .base import PipelineContext, StageResult, StageStatus, to_serializable


ConfidenceLevel = Literal["low", "medium", "high"]
LLMRiskPredictionRunner = Callable[[str, str | None], str | dict[str, Any]]


class LLMRiskAssessment(BaseModel):
    blood_loss_risk_percent: int = Field(ge=0, le=100)
    blood_loss_range: str
    vascular_intervention_risk_percent: int = Field(ge=0, le=100)
    bladder_involvement_risk_percent: int = Field(ge=0, le=100)


class LLMReadiness(BaseModel):
    level: str
    rationale: str


class LLMClinicalSummary(BaseModel):
    text: str


class LLMRiskPredictionOutput(BaseModel):
    risk_assessment: LLMRiskAssessment
    readiness: LLMReadiness
    clinical_summary: LLMClinicalSummary
    confidence: ConfidenceLevel


def _extract_evidence(extracted_result: Any) -> dict[str, Any] | None:
    if extracted_result is None:
        return None

    if hasattr(extracted_result, "evidence"):
        return to_serializable(extracted_result.evidence)

    if isinstance(extracted_result, dict):
        evidence = extracted_result.get("evidence")
        return evidence if isinstance(evidence, dict) else None

    return None


def build_risk_prediction_prompt(context: PipelineContext) -> str:
    prompt_config = load_stage_prompt("risk_prediction")
    template = prompt_config.get("template")
    if not template:
        raise ValueError("risk_prediction prompt config must contain 'template'")

    placeholder = prompt_config.get(
        "case_context_placeholder",
        "__CASE_CONTEXT_JSON__",
    )
    case_context = {
        "source_text": context.source_text,
        "conclusion_text": context.conclusion_text,
        "extracted_result": to_serializable(context.extracted_result),
        "evidence": context.evidence or _extract_evidence(context.extracted_result),
        "metadata": dict(context.metadata),
    }
    case_context_json = json.dumps(
        case_context,
        ensure_ascii=False,
        indent=2,
    )

    return str(template).replace(str(placeholder), case_context_json)


def parse_llm_risk_prediction_output(raw_output: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return raw_output

    text = raw_output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise

        return json.loads(text[start : end + 1])


def default_llm_risk_prediction_runner(
    prompt: str,
    model_id: str | None,
) -> str:
    from pas_mri_extractor.models import generate_text, load_llm

    loaded_model = load_llm(model_id)
    return generate_text(loaded_model, prompt)


class LLMRiskPredictionStage:
    name = "LLMRiskPredictionStage"

    def __init__(
        self,
        model_id: str | None = None,
        runner: LLMRiskPredictionRunner | None = None,
    ) -> None:
        self.model_id = model_id
        self.runner = runner or default_llm_risk_prediction_runner

    def run(self, context: PipelineContext) -> StageResult:
        if context.extracted_result is None:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
                warnings=["No extracted_result is available for LLM risk prediction."],
                metadata={"model_id": self.model_id},
            )

        try:
            prompt = build_risk_prediction_prompt(context)
            raw_output = self.runner(prompt, self.model_id)
            parsed = parse_llm_risk_prediction_output(raw_output)
            validated = LLMRiskPredictionOutput.model_validate(parsed)
        except Exception as error:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                metadata={"model_id": self.model_id},
                error=str(error),
            )

        output = validated.model_dump()
        context.predicted_risks = output["risk_assessment"]

        return StageResult(
            stage_name=self.name,
            status=StageStatus.SUCCESS,
            output=output,
            metadata={
                "model_id": self.model_id,
                "prompt_stage": "risk_prediction",
                "debug_artifacts": {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed": parsed,
                },
            },
        )
