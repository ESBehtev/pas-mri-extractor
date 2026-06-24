from typing import Any


def build_extracted_result_for_llm_risk(result: dict | None) -> dict | None:
    if not result:
        return None

    extracted_result = {
        key: result[key]
        for key in [
            "schema_version",
            "case_info",
            "extracted_features",
            "suspicion",
            "evidence",
        ]
        if key in result
    }

    return extracted_result or None


def stage_result_to_llm_risk_ui(stage_result: Any) -> dict | None:
    if stage_result is None:
        return None

    status = getattr(stage_result, "status", None)
    status_value = getattr(status, "value", status)
    error = getattr(stage_result, "error", None)
    warnings = getattr(stage_result, "warnings", []) or []

    return {
        "stage_name": getattr(stage_result, "stage_name", "LLMRiskPredictionStage"),
        "status": status_value,
        "llm_risk": getattr(stage_result, "output", None),
        "errors": [error] if error else [],
        "warnings": list(warnings),
    }
