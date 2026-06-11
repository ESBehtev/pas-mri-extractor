from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import load_config


class PromptRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class PromptSpec:
    stage_name: str
    config_name: str
    active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    def __init__(self, specs: dict[str, PromptSpec] | None = None) -> None:
        self.specs = specs or DEFAULT_PROMPT_SPECS

    def get_spec(self, stage_name: str) -> PromptSpec:
        normalized_stage = stage_name.strip().lower()
        spec = self.specs.get(normalized_stage)
        if spec is None:
            available = ", ".join(sorted(self.specs))
            raise PromptRegistryError(
                f"Prompt config for stage '{stage_name}' was not found. "
                f"Available stages: {available}"
            )

        return spec

    def load(self, stage_name: str) -> dict[str, Any]:
        spec = self.get_spec(stage_name)
        try:
            config = load_config(spec.config_name)
        except FileNotFoundError as error:
            raise PromptRegistryError(
                f"Prompt config for stage '{stage_name}' was not found: "
                f"{spec.config_name}"
            ) from error

        config.setdefault("stage", spec.stage_name)
        config.setdefault("prompt_config_name", spec.config_name)
        if spec.metadata:
            config.setdefault("registry_metadata", dict(spec.metadata))

        return config


DEFAULT_PROMPT_SPECS: dict[str, PromptSpec] = {
    "extractor": PromptSpec(
        stage_name="extractor",
        config_name="prompt.yaml",
        active=True,
        metadata={
            "alias_config_name": "prompts/extractor.yaml",
            "source_of_truth": "configs/prompt.yaml",
        },
    ),
    "risk_prediction": PromptSpec(
        stage_name="risk_prediction",
        config_name="prompts/risk_prediction.example.yaml",
        active=False,
    ),
    "clinical_summary": PromptSpec(
        stage_name="clinical_summary",
        config_name="prompts/clinical_summary.example.yaml",
        active=False,
    ),
    "case_chat": PromptSpec(
        stage_name="case_chat",
        config_name="prompts/case_chat.example.yaml",
        active=False,
    ),
}


PromptManager = PromptRegistry


def load_stage_prompt(stage_name: str) -> dict[str, Any]:
    return PromptRegistry().load(stage_name)
