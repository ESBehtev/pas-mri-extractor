"""Presentation helpers for the production Streamlit review screen."""

import json
from typing import Any

import streamlit as st


def _status(value: Any) -> str:
    return {
        "present": "выявлено",
        "probable": "вероятно",
        "possible": "возможно",
        "absent": "не выявлено",
    }.get(str(value), str(value or "—"))


def render_clinical_result(result: dict[str, Any]) -> None:
    features = result.get("extracted_features", {})
    invasion = features.get("invasion", {})
    score = result.get("score", {})
    recommendation = result.get("recommendation", {})

    st.subheader("Структурированный результат")
    columns = st.columns(3)
    columns[0].metric("Инвазия", invasion.get("type", "none"))
    columns[1].metric("Группа риска", score.get("risk_group", "—"))
    columns[2].metric("Готовность", recommendation.get("readiness_level", "—"))

    st.caption(f"Уверенность: {invasion.get('confidence', 'unclear')}")
    st.json(result, expanded=False)

    evidence = result.get("evidence", {})
    if any(evidence.get(key) for key in ("positive_findings", "uncertain_findings", "negative_findings")):
        st.subheader("Источник признаков")
        for title, key in (
            ("Положительные", "positive_findings"),
            ("Неопределённые", "uncertain_findings"),
            ("Отрицательные", "negative_findings"),
        ):
            findings = evidence.get(key) or []
            if findings:
                st.write(f"{title}: " + "; ".join(map(str, findings)))


def render_json_download(result: dict[str, Any]) -> None:
    st.download_button(
        "Скачать JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="pas_mri_result.json",
        mime="application/json",
    )
