from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_serializable(model_dump())

    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]

    return value


@dataclass
class StageResult:
    stage_name: str
    status: StageStatus
    output: Any = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "output": to_serializable(self.output),
            "warnings": to_serializable(self.warnings),
            "metadata": to_serializable(self.metadata),
            "error": self.error,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass
class PipelineContext:
    source_text: str
    conclusion_text: str | None = None
    extracted_result: Any | None = None
    predicted_risks: Any | None = None
    evidence: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
