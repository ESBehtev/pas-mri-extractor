from __future__ import annotations

import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from pas_mri_extractor.prompt_registry import load_stage_prompt

from .base import PipelineContext, StageResult, StageStatus, to_serializable


ConfidenceLevel = Literal["low", "medium", "high"]
EstimatedBloodLossRange = Literal[
    "<1000 мл",
    "1000–1500 мл",
    "1500–2500 мл",
    ">2500 мл",
]
ReadinessLevel = Literal["1", "2", "3", "4"]
RiskPredictionMode = Literal["direct_json", "reason_then_json"]
LLMRiskPredictionRunner = Callable[[str, str | None], str | dict[str, Any]]


class LLMRiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    massive_blood_loss_risk_percent: int = Field(ge=0, le=100)
    estimated_blood_loss_ml: int = Field(ge=0)
    estimated_blood_loss_range: EstimatedBloodLossRange
    vascular_intervention_risk_percent: int = Field(ge=0, le=100)
    bladder_involvement_risk_percent: int = Field(ge=0, le=100)
    hysterectomy_risk_percent: int = Field(ge=0, le=100)
    transfusion_risk_percent: int = Field(ge=0, le=100)


class LLMReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ReadinessLevel
    rationale: str


class LLMClinicalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class LLMOperativeRiskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class LLMRiskPredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_assessment: LLMRiskAssessment
    readiness: LLMReadiness
    operative_risk_summary: LLMOperativeRiskSummary
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


def build_case_context_json(context: PipelineContext) -> str:
    case_context = {
        "source_text": context.source_text,
        "conclusion_text": context.conclusion_text,
        "extracted_result": to_serializable(context.extracted_result),
        "evidence": context.evidence or _extract_evidence(context.extracted_result),
        "metadata": dict(context.metadata),
    }
    return json.dumps(
        case_context,
        ensure_ascii=False,
        indent=2,
    )


def build_prompt_from_config(
    context: PipelineContext,
    prompt_config_name: str,
    reasoning_text: str | None = None,
) -> str:
    prompt_config = load_stage_prompt(prompt_config_name)
    template = prompt_config.get("template")
    if not template:
        raise ValueError(f"{prompt_config_name} prompt config must contain 'template'")

    placeholder = prompt_config.get(
        "case_context_placeholder",
        "__CASE_CONTEXT_JSON__",
    )
    prompt = str(template).replace(str(placeholder), build_case_context_json(context))

    if reasoning_text is not None:
        reasoning_placeholder = prompt_config.get(
            "reasoning_placeholder",
            "__REASONING_TEXT__",
        )
        prompt = prompt.replace(str(reasoning_placeholder), reasoning_text)

    return prompt


def build_risk_prediction_prompt(context: PipelineContext) -> str:
    return build_prompt_from_config(context, "risk_prediction")


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
    from pas_mri_extractor.models import generate_text
    from pas_mri_extractor.pipeline import get_cached_model

    loaded_model = get_cached_model(model_id)
    return generate_text(loaded_model, prompt)


class LLMRiskPredictionStage:
    name = "LLMRiskPredictionStage"

    def __init__(
        self,
        model_id: str | None = None,
        runner: LLMRiskPredictionRunner | None = None,
        loaded_model: Any | None = None,
        mode: RiskPredictionMode = "direct_json",
    ) -> None:
        self.model_id = model_id
        self.loaded_model = loaded_model
        self.mode = mode
        self.runner = runner or self._default_runner

    def _default_runner(self, prompt: str, model_id: str | None) -> str:
        if self.loaded_model is not None:
            from pas_mri_extractor.models import generate_text

            return generate_text(self.loaded_model, prompt)

        return default_llm_risk_prediction_runner(prompt, model_id)

    def run(self, context: PipelineContext) -> StageResult:
        if context.extracted_result is None:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
                warnings=["No extracted_result is available for LLM risk prediction."],
                metadata={"model_id": self.model_id},
            )

        try:
            if self.mode == "direct_json":
                prompt = build_risk_prediction_prompt(context)
                raw_output = self.runner(prompt, self.model_id)
                parsed = parse_llm_risk_prediction_output(raw_output)
                debug_artifacts = {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed": parsed,
                }
            elif self.mode == "reason_then_json":
                reasoning_prompt = build_prompt_from_config(
                    context,
                    "risk_prediction_reasoning",
                )
                reasoning_text = str(self.runner(reasoning_prompt, self.model_id))
                finalizer_prompt = build_prompt_from_config(
                    context,
                    "risk_prediction_finalizer",
                    reasoning_text=reasoning_text,
                )
                raw_output = self.runner(finalizer_prompt, self.model_id)
                parsed = parse_llm_risk_prediction_output(raw_output)
                debug_artifacts = {
                    "reasoning_prompt": reasoning_prompt,
                    "reasoning_text": reasoning_text,
                    "finalizer_prompt": finalizer_prompt,
                    "raw_output": raw_output,
                    "parsed": parsed,
                }
            else:
                raise ValueError(f"Unsupported risk prediction mode: {self.mode}")

            validated = LLMRiskPredictionOutput.model_validate(parsed)
        except Exception as error:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                metadata={"model_id": self.model_id, "risk_mode": self.mode},
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
                "risk_mode": self.mode,
                "prompt_stage": "risk_prediction",
                "debug_artifacts": debug_artifacts,
            },
        )
