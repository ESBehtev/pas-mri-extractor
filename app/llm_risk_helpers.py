from typing import Any


MISSING_DISPLAY = "—"


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


def reset_llm_risk_state_values(state: Any) -> None:
    state["last_llm_risk_result"] = None
    state["llm_risk_result"] = None
    state["llm_risk_status"] = "skipped"
    state["llm_risk_errors"] = []
    state["llm_risk_warnings"] = []


def risk_level_from_percent(value: int | float | str | None) -> str:
    numeric_value = to_number(value)
    if numeric_value is None:
        return "unknown"
    if numeric_value >= 75:
        return "very_high"
    if numeric_value >= 50:
        return "high"
    if numeric_value >= 25:
        return "moderate"
    return "low"


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def format_percent(value: Any) -> str:
    numeric_value = to_number(value)
    if numeric_value is None:
        return MISSING_DISPLAY
    if numeric_value.is_integer():
        return f"{int(numeric_value)}%"
    return f"{numeric_value:.1f}%"


def format_ml(value: Any) -> str:
    numeric_value = to_number(value)
    if numeric_value is None:
        return MISSING_DISPLAY
    if numeric_value.is_integer():
        return f"{int(numeric_value)} мл"
    return f"{numeric_value:.0f} мл"


def normalize_rule_based_risk(rule_based_result: dict | None) -> dict[str, Any]:
    if not rule_based_result:
        return {
            "available": False,
            "raw": None,
        }

    predicted_risks = rule_based_result.get("predicted_risks") or {}
    recommendation = rule_based_result.get("recommendation") or {}
    score = rule_based_result.get("score") or {}
    raw_rule_based = {
        "score": score,
        "predicted_risks": predicted_risks,
        "recommendation": recommendation,
        "computed_rationale": rule_based_result.get("computed_rationale"),
    }

    return {
        "available": bool(predicted_risks or recommendation or score),
        "massive_blood_loss_risk": predicted_risks.get(
            "massive_blood_loss_over_1500_ml_percent"
        ),
        "estimated_blood_loss": predicted_risks.get("estimated_blood_loss_ml_range"),
        "estimated_blood_loss_range": predicted_risks.get(
            "estimated_blood_loss_ml_range"
        ),
        "vascular_intervention_risk": predicted_risks.get(
            "vascular_intervention_percent"
        ),
        "bladder_involvement_risk": predicted_risks.get(
            "bladder_involvement_percent"
        ),
        "hysterectomy_risk": None,
        "transfusion_risk": None,
        "readiness_level": recommendation.get("readiness_level"),
        "readiness_rationale": recommendation.get("readiness_text"),
        "score_reasons": score.get("score_reasons"),
        "risk_summary_text": predicted_risks.get("risk_summary_text"),
        "computed_rationale": rule_based_result.get("computed_rationale"),
        "raw": raw_rule_based,
    }


def normalize_llm_risk(stage_result: dict | None) -> dict[str, Any]:
    if not stage_result:
        return {
            "available": False,
            "status": "skipped",
            "message": "LLM-прогноз рисков отключён",
            "errors": [],
            "warnings": [],
            "raw": None,
        }

    status = stage_result.get("status")
    if status != "success":
        if status == "skipped":
            message = "LLM-прогноз рисков отключён"
        elif status == "running":
            message = "Выполняется LLM-прогноз хирургических рисков..."
        else:
            message = "LLM-прогноз рисков не выполнен"

        return {
            "available": False,
            "status": status,
            "message": message,
            "errors": stage_result.get("errors") or [],
            "warnings": stage_result.get("warnings") or [],
            "raw": stage_result.get("llm_risk"),
        }

    llm_risk = stage_result.get("llm_risk") or {}
    risk_assessment = llm_risk.get("risk_assessment") or {}
    readiness = llm_risk.get("readiness") or {}
    operative_summary = llm_risk.get("operative_risk_summary") or {}
    clinical_summary = llm_risk.get("clinical_summary") or {}

    return {
        "available": True,
        "status": status,
        "massive_blood_loss_risk": risk_assessment.get(
            "massive_blood_loss_risk_percent"
        ),
        "estimated_blood_loss": risk_assessment.get("estimated_blood_loss_ml"),
        "estimated_blood_loss_range": risk_assessment.get(
            "estimated_blood_loss_range"
        ),
        "vascular_intervention_risk": risk_assessment.get(
            "vascular_intervention_risk_percent"
        ),
        "bladder_involvement_risk": risk_assessment.get(
            "bladder_involvement_risk_percent"
        ),
        "hysterectomy_risk": risk_assessment.get("hysterectomy_risk_percent"),
        "transfusion_risk": risk_assessment.get("transfusion_risk_percent"),
        "readiness_level": readiness.get("level"),
        "readiness_rationale": readiness.get("rationale"),
        "confidence": llm_risk.get("confidence"),
        "operative_risk_summary": operative_summary.get("text"),
        "clinical_summary": clinical_summary.get("text"),
        "errors": stage_result.get("errors") or [],
        "warnings": stage_result.get("warnings") or [],
        "raw": llm_risk,
    }
