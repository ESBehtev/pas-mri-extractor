import html
import json
from typing import Any

import streamlit as st


VALUE_TRANSLATIONS = {
    None: "нет данных",
    "unknown": "нет данных",
    "none": "нет признаков",
    "absent": "отсутствует",
    "present": "выявлено",
    "possible": "возможно",
    "probable": "вероятно",
    "definite": "достоверно",
    "unclear": "неясно",
    "low": "низкий",
    "moderate": "умеренный",
    "medium": "умеренный",
    "high": "высокий",
    "accreta": "accreta",
    "increta": "increta",
    "percreta": "percreta",
}


def ru(value: Any) -> str:
    if value is None:
        return "нет данных"

    if isinstance(value, str):
        return VALUE_TRANSLATIONS.get(value.lower(), value)

    return str(value)


def ru_upper(value: Any) -> str:
    return ru(value).upper()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, str):
        if ";" in value:
            return [part.strip() for part in value.split(";") if part.strip()]
        return [value.strip()] if value.strip() else []

    return [str(value)]


def colorize_risk(value: Any) -> str:
    value = str(value).lower()

    if value in ["low", "низкий"]:
        return "#16a34a"

    if value in ["moderate", "medium", "умеренный"]:
        return "#f59e0b"

    if value in ["high", "высокий"]:
        return "#dc2626"

    return "#9ca3af"


def colorize_invasion(value: Any) -> str:
    value = str(value).lower()

    mapping = {
        "none": "#16a34a",
        "accreta": "#f59e0b",
        "increta": "#ea580c",
        "percreta": "#dc2626",
    }

    return mapping.get(value, "#9ca3af")


def colorize_confidence(value: Any) -> str:
    value = str(value).lower()

    mapping = {
        "absent": "#16a34a",
        "possible": "#eab308",
        "probable": "#f97316",
        "definite": "#dc2626",
    }

    return mapping.get(value, "#9ca3af")


def colorize_score(value: Any) -> str:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return "#9ca3af"

    if score <= 3:
        return "#16a34a"

    if score <= 8:
        return "#f59e0b"

    return "#dc2626"


def colorize_readiness_level(value: Any) -> str:
    value = str(value)

    if value == "1":
        return "#16a34a"

    if value == "2":
        return "#f59e0b"

    if value == "3":
        return "#dc2626"

    return "#9ca3af"


def badge(label: str, value: Any, color: str) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(ru(value))

    st.markdown(
        f"""
        <div style="
            padding: 16px 18px;
            border-radius: 14px;
            background-color: {color}22;
            border: 1px solid {color};
            margin-bottom: 12px;
            min-height: 120px;
        ">
            <div style="
                font-size: 13px;
                color: #cbd5e1;
                margin-bottom: 10px;
                font-weight: 600;
            ">
                {safe_label}
            </div>
            <div style="
                font-size: 30px;
                line-height: 1.15;
                font-weight: 800;
                color: {color};
                word-break: break-word;
            ">
                {safe_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def finding_box(text: str, color: str) -> None:
    safe_text = html.escape(str(text))

    st.markdown(
        f"""
        <div style="
            padding: 12px 14px;
            border-radius: 10px;
            background-color: {color}22;
            border-left: 4px solid {color};
            margin-bottom: 10px;
            font-size: 14px;
            line-height: 1.45;
        ">
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_findings(items: list[str], kind: str) -> None:
    if not items:
        st.write("Нет")
        return

    for item in items:
        if kind == "positive":
            finding_box(item, "#dc2626")
        elif kind == "uncertain":
            finding_box(item, "#f59e0b")
        else:
            finding_box(item, "#16a34a")


def readiness_box(level: Any, text: str | None) -> None:
    color = colorize_readiness_level(level)
    safe_level = html.escape(ru(level))
    safe_text = html.escape(text or "Нет текстовой рекомендации")

    st.markdown(
        f"""
        <div style="
            padding: 16px 18px;
            border-radius: 14px;
            background-color: {color}22;
            border: 1px solid {color};
            margin-bottom: 12px;
        ">
            <div style="
                font-size: 15px;
                font-weight: 800;
                color: {color};
                margin-bottom: 10px;
            ">
                Уровень готовности: {safe_level}
            </div>
            <div style="
                font-size: 15px;
                line-height: 1.5;
            ">
                {safe_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_short_result(title: str, item: dict | None) -> None:
    st.markdown(f"### {title}")

    if not item:
        st.write("Нет данных")
        return

    features = item.get("extracted_features", {})
    invasion = features.get("invasion", {})
    score = item.get("score", {})

    st.write(f"**Тип врастания:** {ru(invasion.get('type'))}")
    st.write(f"**Уверенность:** {ru(invasion.get('confidence'))}")
    st.write(f"**Группа риска:** {ru(score.get('risk_group'))}")
    st.write(f"**Score:** {ru(score.get('clinical_score'))}")


def render_report_sections(sections: object | None) -> None:
    if sections and sections.has_conclusion:
        st.success("Заключение найдено")
    else:
        st.warning("Заключение не найдено")

    st.markdown("**Описательная часть:**")
    st.text((sections.body or "Нет данных") if sections else "Нет данных")

    st.markdown("**Заключение:**")
    st.text((sections.conclusion or "Нет данных") if sections else "Нет данных")


def render_dual_comparison(dual_result: dict | None) -> None:
    if not dual_result:
        st.write("Нет данных diagnostic extraction")
        return

    full_result = dual_result.get("full")
    body_result = dual_result.get("body")
    conclusion_result = dual_result.get("conclusion")

    col_full, col_body, col_conclusion = st.columns(3)

    with col_full:
        render_short_result("Полный отчёт", full_result)

    with col_body:
        render_short_result("Описание", body_result)

    with col_conclusion:
        render_short_result("Заключение", conclusion_result)


def render_summary_cards(result: dict) -> None:
    features = result.get("extracted_features", {})
    invasion = features.get("invasion", {})
    score = result.get("score", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        badge(
            "Группа риска",
            ru_upper(score.get("risk_group")),
            colorize_risk(score.get("risk_group")),
        )

    with col2:
        badge(
            "Клинический score",
            score.get("clinical_score"),
            colorize_score(score.get("clinical_score")),
        )

    with col3:
        badge(
            "Тип врастания",
            invasion.get("type"),
            colorize_invasion(invasion.get("type")),
        )

    with col4:
        badge(
            "Уверенность",
            invasion.get("confidence"),
            colorize_confidence(invasion.get("confidence")),
        )


def render_clinical_result(result: dict | None) -> None:
    if not result:
        st.info("Вставьте отчёт и нажмите Extract на вкладке Input.")
        return

    case_info = result.get("case_info", {})
    features = result.get("extracted_features", {})
    anatomy = features.get("anatomy", {})
    placenta_location = features.get("placenta_location", {})
    mri_signs = features.get("mri_signs", {})
    clinical_context = features.get("clinical_context", {})
    evidence = result.get("evidence", {})
    score = result.get("score", {})
    predicted_risks = result.get("predicted_risks", {})
    recommendation = result.get("recommendation", {})

    st.subheader("Краткое заключение")
    render_summary_cards(result)

    st.subheader("Клинические признаки")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Клинические данные")
        st.write(
            f"**Срок беременности:** "
            f"{ru(case_info.get('gestational_week'))} недель"
        )
        st.write(
            f"**Кесаревых сечений в анамнезе:** "
            f"{ru(case_info.get('previous_cs_count'))}"
        )

        st.markdown("#### Анатомия")
        st.write(f"**Мочевой пузырь:** {ru(anatomy.get('bladder_involvement'))}")
        st.write(f"**Параметрий:** {ru(anatomy.get('parametrium_involvement'))}")
        st.write(f"**Задняя стенка:** {ru(anatomy.get('posterior_wall_involvement'))}")

        st.markdown("#### Локализация плаценты")
        st.write(
            f"**Предлежание плаценты:** "
            f"{ru(placenta_location.get('placenta_previa'))}"
        )
        st.write(
            f"**Плацента по передней стенке:** "
            f"{ru(placenta_location.get('anterior_placenta'))}"
        )

    with col_right:
        st.markdown("#### MRI-признаки")
        sign_labels = {
            "retroplacental_vessels": "Ретроплацентарные сосуды",
            "lacunae": "Лакуны",
            "uterine_wall_thinning": "Истончение миометрия/рубца",
            "uterine_hernia_or_bulging": "Выбухание/грыжевидная деформация",
        }

        for key, label in sign_labels.items():
            st.write(f"**{label}:** {ru(mri_signs.get(key))}")

        st.markdown("#### Клинический контекст")
        st.write(
            f"**Кровотечение до операции:** "
            f"{ru(clinical_context.get('preoperative_bleeding'))}"
        )

    st.subheader("Evidence")
    positive_findings = as_list(evidence.get("positive_findings"))
    uncertain_findings = as_list(evidence.get("uncertain_findings"))
    negative_findings = as_list(evidence.get("negative_findings"))

    col_pos, col_unc, col_neg = st.columns(3)

    with col_pos:
        st.markdown("#### Признаки PAS")
        show_findings(positive_findings, "positive")

    with col_unc:
        st.markdown("#### Неопределённые находки")
        show_findings(uncertain_findings, "uncertain")

    with col_neg:
        st.markdown("#### Признаки против PAS")
        show_findings(negative_findings, "negative")

    st.subheader("Прогнозируемые риски")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        st.metric(
            "Кровопотеря >1500 мл",
            f"{ru(predicted_risks.get('massive_blood_loss_over_1500_ml_percent'))}%",
        )

    with risk_col2:
        st.metric(
            "Оценочная кровопотеря",
            ru(predicted_risks.get("estimated_blood_loss_ml_range")),
        )

    with risk_col3:
        st.metric(
            "Вероятность сосудистого вмешательства",
            f"{ru(predicted_risks.get('vascular_intervention_percent'))}%",
        )

    with risk_col4:
        st.metric(
            "Риск вовлечения мочевого пузыря",
            f"{ru(predicted_risks.get('bladder_involvement_percent'))}%",
        )

    score_reasons = as_list(score.get("score_reasons"))
    if score_reasons:
        st.subheader("Причины score")
        for reason in score_reasons[:3]:
            finding_box(reason, "#64748b")

        hidden_reasons = score_reasons[3:]
        if hidden_reasons:
            with st.expander(
                f"Показать ещё ({len(hidden_reasons)})",
                expanded=False,
            ):
                for reason in hidden_reasons:
                    finding_box(reason, "#64748b")

    st.subheader("Рекомендация по готовности")
    readiness_level = recommendation.get("readiness_level")
    readiness_text = recommendation.get("readiness_text")

    if readiness_level or readiness_text:
        readiness_box(readiness_level, readiness_text)
    else:
        st.write("Нет данных")

    risk_summary_text = predicted_risks.get("risk_summary_text")
    if risk_summary_text:
        st.subheader("Сводка рисков")
        finding_box(risk_summary_text, colorize_risk(score.get("risk_group")))

    computed_rationale = result.get("computed_rationale")
    if computed_rationale:
        st.subheader("Расчётное обоснование")
        st.write(computed_rationale)


def render_json_export(result: dict | None) -> None:
    if not result:
        st.info("Нет результата для отображения.")
        return

    st.json(result)
    st.download_button(
        label="Скачать JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="pas_mri_extraction.json",
        mime="application/json",
    )
