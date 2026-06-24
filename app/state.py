import streamlit as st

try:
    from llm_risk_helpers import (
        build_combined_result_json,
        build_rule_based_risk_json,
        reset_llm_risk_state_values,
    )
except ModuleNotFoundError:
    from app.llm_risk_helpers import (
        build_combined_result_json,
        build_rule_based_risk_json,
        reset_llm_risk_state_values,
    )


SESSION_DEFAULTS = {
    "is_running": False,
    "model_loaded": False,
    "last_result": None,
    "last_dual_result": None,
    "last_sections": None,
    "last_llm_risk_result": None,
    "extraction_result": None,
    "rule_based_result": None,
    "llm_risk_result": None,
    "llm_risk_status": "skipped",
    "llm_risk_errors": [],
    "llm_risk_warnings": [],
    "combined_result_json": None,
    "analysis_run_id": 0,
    "last_model_name": None,
    "last_diagnostic_mode": False,
    "report_text": "",
    "extract_requested": False,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_running(is_running: bool) -> None:
    st.session_state["is_running"] = is_running


def request_extraction() -> None:
    st.session_state["extract_requested"] = True
    st.session_state["analysis_run_id"] = st.session_state.get("analysis_run_id", 0) + 1


def clear_extraction_request() -> None:
    st.session_state["extract_requested"] = False


def build_rule_based_result(result: dict | None) -> dict | None:
    if not result:
        return None

    return build_rule_based_risk_json(result)


def reset_llm_risk_state(session_state: object | None = None) -> None:
    state = session_state if session_state is not None else st.session_state
    reset_llm_risk_state_values(state)


def save_extraction_result(
    *,
    result: dict,
    dual_result: dict | None,
    sections: object,
    model_name: str,
    diagnostic_mode: bool,
    llm_risk_result: dict | None = None,
) -> None:
    rule_based_result = build_rule_based_result(result)
    llm_status = llm_risk_result.get("status") if llm_risk_result else "skipped"
    if llm_status == "skipped":
        combined_llm_status = "disabled"
    else:
        combined_llm_status = llm_status
    llm_result = llm_risk_result.get("llm_risk") if llm_risk_result else None
    llm_errors = llm_risk_result.get("errors") if llm_risk_result else []
    llm_warnings = llm_risk_result.get("warnings") if llm_risk_result else []

    st.session_state["model_loaded"] = True
    st.session_state["last_result"] = result
    st.session_state["last_dual_result"] = dual_result
    st.session_state["last_llm_risk_result"] = llm_risk_result
    st.session_state["extraction_result"] = result
    st.session_state["rule_based_result"] = rule_based_result
    st.session_state["llm_risk_result"] = llm_result
    st.session_state["llm_risk_status"] = llm_status
    st.session_state["llm_risk_errors"] = llm_errors
    st.session_state["llm_risk_warnings"] = llm_warnings
    st.session_state["combined_result_json"] = build_combined_result_json(
        extraction_result=result,
        rule_based_risk=rule_based_result,
        llm_risk=llm_result,
        llm_risk_status=combined_llm_status,
        llm_risk_errors=llm_errors,
        llm_risk_warnings=llm_warnings,
    )
    st.session_state["last_sections"] = sections
    st.session_state["last_model_name"] = model_name
    st.session_state["last_diagnostic_mode"] = diagnostic_mode


def get_last_outputs() -> tuple[dict | None, dict | None, object | None, bool]:
    return (
        st.session_state.get("last_result"),
        st.session_state.get("last_dual_result"),
        st.session_state.get("last_sections"),
        st.session_state.get("last_diagnostic_mode", False),
    )


def get_last_llm_risk_output() -> dict | None:
    return st.session_state.get("last_llm_risk_result")
