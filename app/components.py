import html
import json
from textwrap import dedent
from typing import Any

import streamlit as st

try:
    from llm_risk_helpers import (
        build_extracted_result_for_llm_risk,
        stage_result_to_llm_risk_ui,
    )
except ModuleNotFoundError:
    from app.llm_risk_helpers import (
        build_extracted_result_for_llm_risk,
        stage_result_to_llm_risk_ui,
    )

try:
    from provenance import build_report_highlighting
except ModuleNotFoundError:
    from app.provenance import build_report_highlighting


VALUE_TRANSLATIONS = {
    None: "нет данных",
    "unknown": "нет данных",
    "none": "нет признаков",
    "absent": "отсутствует",
    "present": "выявлено",
    "possible": "возможно",
    "probable": "вероятно",
    "definite": "достоверно",
    "confirmed": "подтверждено",
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


def percent_value(value: Any) -> str:
    if value is None:
        return "нет данных"
    return f"{value}%"


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


def colorize_status(value: Any) -> str:
    value = str(value).lower()

    if "отсутств" in value:
        return "#16a34a"

    mapping = {
        "none": "#16a34a",
        "absent": "#16a34a",
        "negative": "#16a34a",
        "отсутствует": "#16a34a",
        "present": "#dc2626",
        "выявлено": "#dc2626",
        "possible": "#f97316",
        "возможно": "#f97316",
        "probable": "#ea580c",
        "вероятно": "#ea580c",
        "definite": "#dc2626",
        "достоверно": "#dc2626",
    }

    return mapping.get(value, "#9ca3af")


def colorize_confidence(value: Any) -> str:
    return colorize_status(value)


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


def render_html(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def badge(label: str, value: Any, color: str) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(ru(value))

    render_html(
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
    )


def finding_box(text: str, color: str) -> None:
    safe_text = html.escape(str(text))

    render_html(
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
    )


def feature_card(
    title: str,
    rows: list[tuple[str, Any, str | None]],
) -> None:
    safe_title = html.escape(title)
    row_html = []

    for label, value, color in rows:
        row_color = color or colorize_status(value)
        safe_label = html.escape(str(label))
        safe_value = html.escape(ru(value))
        row_html.append(
            dedent(
                f"""
            <div style="
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: center;
                border-top: 1px solid #334155;
                padding: 9px 0;
            ">
                <span style="color: #cbd5e1;">{safe_label}</span>
                <span style="
                    color: {row_color};
                    font-weight: 800;
                    text-align: right;
                    white-space: nowrap;
                ">{safe_value}</span>
            </div>
            """
            ).strip()
        )

    render_html(
        f"""
        <div style="
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
            margin-bottom: 14px;
        ">
            <div style="
                font-size: 15px;
                font-weight: 800;
                margin-bottom: 4px;
            ">{safe_title}</div>
            {''.join(row_html)}
        </div>
        """,
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

    render_html(
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
    st.write(f"**Клиническая оценка:** {ru(score.get('clinical_score'))}")


DIAGNOSTIC_COMPARISON_FIELDS = [
    ("Тип врастания", ("extracted_features", "invasion", "type")),
    ("Уверенность", ("extracted_features", "invasion", "confidence")),
    ("Группа риска", ("score", "risk_group")),
    ("Клиническая оценка", ("score", "clinical_score")),
]


def get_nested_value(item: dict | None, path: tuple[str, ...]) -> Any:
    current: Any = item or {}

    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def diagnostic_value_color(value: Any, is_consistent: bool, is_different: bool) -> str:
    if is_consistent:
        return "#16a34a"

    if not is_different:
        return "#64748b"

    value_text = str(value).lower()
    if value_text in ["percreta", "high"] or value_text.isdigit():
        return "#dc2626"

    return "#f59e0b"


def render_diagnostic_column(
    title: str,
    item: dict | None,
    field_stats: dict[str, dict[str, Any]],
) -> None:
    safe_title = html.escape(title)
    rows = []

    for label, path in DIAGNOSTIC_COMPARISON_FIELDS:
        value = get_nested_value(item, path)
        value_key = str(value)
        stats = field_stats[label]
        is_consistent = len(stats["counts"]) == 1
        is_different = stats["counts"].get(value_key, 0) == 1
        color = diagnostic_value_color(value, is_consistent, is_different)

        rows.append(
            dedent(
                f"""
            <div style="
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: center;
                border-top: 1px solid #334155;
                padding: 9px 0;
            ">
                <span style="color: #cbd5e1;">{html.escape(label)}</span>
                <span style="
                    color: {color};
                    font-weight: 800;
                    text-align: right;
                    white-space: nowrap;
                ">{html.escape(ru(value))}</span>
            </div>
            """
            ).strip()
        )

    render_html(
        f"""
        <div style="
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
            margin-bottom: 14px;
        ">
            <div style="
                font-size: 15px;
                font-weight: 800;
                margin-bottom: 4px;
            ">{safe_title}</div>
            {''.join(rows)}
        </div>
        """
    )


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
        st.write("Нет данных диагностического извлечения")
        return

    items = {
        "Полный отчёт": dual_result.get("full"),
        "Описание": dual_result.get("body"),
        "Заключение": dual_result.get("conclusion"),
    }

    field_stats = {}
    for label, path in DIAGNOSTIC_COMPARISON_FIELDS:
        values = [str(get_nested_value(item, path)) for item in items.values()]
        counts = {value: values.count(value) for value in set(values)}
        field_stats[label] = {"counts": counts}

    col_full, col_body, col_conclusion = st.columns(3)

    with col_full:
        render_diagnostic_column("Полный отчёт", items["Полный отчёт"], field_stats)

    with col_body:
        render_diagnostic_column("Описание", items["Описание"], field_stats)

    with col_conclusion:
        render_diagnostic_column("Заключение", items["Заключение"], field_stats)


def render_evidence_highlighting(result: dict, report_text: str | None) -> None:
    if not report_text:
        return

    evidence = result.get("evidence", {})
    positive_findings = as_list(evidence.get("positive_findings"))
    uncertain_findings = as_list(evidence.get("uncertain_findings"))
    negative_findings = as_list(evidence.get("negative_findings"))

    provenance = build_report_highlighting(
        report_text,
        result,
    )
    highlighted_text = provenance["html"]
    unmatched_evidence = provenance["unmatched_evidence"]

    with st.expander("Подсветка отчёта", expanded=False):
        if not positive_findings and not uncertain_findings and not negative_findings:
            st.write("Фраз для подсветки нет.")

        render_html(
            f"""
            <div style="
                white-space: pre-wrap;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 14px 16px;
                line-height: 1.55;
                margin-top: 12px;
            ">{highlighted_text}</div>
            """
        )

        if unmatched_evidence:
            st.markdown("**Не найдено в тексте:**")
            for phrase in unmatched_evidence:
                st.write(f"- {phrase}")


def render_summary_cards(result: dict) -> None:
    features = result.get("extracted_features", {})
    invasion = features.get("invasion", {})
    score = result.get("score", {})
    invasion_type = invasion.get("type")
    has_no_pas = invasion_type in [None, "none", "absent"]

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
            invasion_type,
            colorize_invasion(invasion_type),
        )

    with col4:
        if has_no_pas:
            badge("PAS", "признаков нет", "#16a34a")
        else:
            badge(
                "Уверенность",
                invasion.get("confidence"),
                colorize_confidence(invasion.get("confidence")),
            )


def render_suspicion_block(suspicion: dict) -> None:
    st.subheader("Клиническое подозрение")
    feature_card(
        "Клиническое подозрение",
        [
            (
                "Наиболее тяжёлый предполагаемый вариант PAS",
                suspicion.get("highest_suspected_extent", "none"),
                colorize_invasion(suspicion.get("highest_suspected_extent")),
            ),
            (
                "Подозрение на percreta",
                suspicion.get("percreta_suspicion", "absent"),
                None,
            ),
            (
                "Подозрение вовлечения серозы/мочевого пузыря",
                suspicion.get("bladder_serosa_suspicion", "absent"),
                None,
            ),
        ],
    )

    rationale = as_list(suspicion.get("rationale"))
    if rationale:
        with st.expander("Фразы, подтверждающие подозрение", expanded=False):
            for phrase in rationale:
                finding_box(phrase, "#f59e0b")


def render_clinical_result(result: dict | None, report_text: str | None = None) -> None:
    if not result:
        st.info("Вставьте отчёт и нажмите «Извлечь признаки».")
        return

    case_info = result.get("case_info", {})
    features = result.get("extracted_features", {})
    invasion = features.get("invasion", {})
    anatomy = features.get("anatomy", {})
    placenta_location = features.get("placenta_location", {})
    mri_signs = features.get("mri_signs", {})
    clinical_context = features.get("clinical_context", {})
    suspicion = result.get("suspicion") or {}
    evidence = result.get("evidence", {})
    score = result.get("score", {})
    predicted_risks = result.get("predicted_risks", {})
    recommendation = result.get("recommendation", {})

    st.subheader("Краткое заключение")
    render_summary_cards(result)

    st.subheader("Клинические признаки")
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        feature_card(
            "Клинические данные",
            [
                ("Срок беременности", case_info.get("gestational_week"), "#cbd5e1"),
                ("КС в анамнезе", case_info.get("previous_cs_count"), "#cbd5e1"),
            ],
        )
        feature_card(
            "Локализация плаценты",
            [
                ("Предлежание", placenta_location.get("placenta_previa"), None),
                (
                    "Передняя стенка",
                    placenta_location.get("anterior_placenta"),
                    None,
                ),
            ],
        )

    with col_mid:
        feature_card(
            "Врастание",
            [
                ("Тип", invasion.get("type"), colorize_invasion(invasion.get("type"))),
                (
                    "Уверенность",
                    invasion.get("confidence"),
                    colorize_confidence(invasion.get("confidence")),
                ),
            ],
        )
        feature_card(
            "Анатомическое вовлечение",
            [
                (
                    "Вовлечение мочевого пузыря",
                    anatomy.get("bladder_involvement"),
                    None,
                ),
                (
                    "Вовлечение параметрия",
                    anatomy.get("parametrium_involvement"),
                    None,
                ),
                (
                    "Вовлечение задней стенки матки",
                    anatomy.get("posterior_wall_involvement"),
                    None,
                ),
            ],
        )

    with col_right:
        feature_card(
            "МР-признаки",
            [
                (
                    "Ретроплацентарные сосуды",
                    mri_signs.get("retroplacental_vessels"),
                    None,
                ),
                ("Лакуны", mri_signs.get("lacunae"), None),
                (
                    "Истончение миометрия/рубца",
                    mri_signs.get("uterine_wall_thinning"),
                    None,
                ),
                (
                    "Выбухание/грыжевидная деформация",
                    mri_signs.get("uterine_hernia_or_bulging"),
                    None,
                ),
            ],
        )
        feature_card(
            "Клинический контекст",
            [
                (
                    "Кровотечение до операции",
                    clinical_context.get("preoperative_bleeding"),
                    None,
                ),
            ],
        )

    render_suspicion_block(suspicion)

    positive_findings = as_list(evidence.get("positive_findings"))
    uncertain_findings = as_list(evidence.get("uncertain_findings"))
    negative_findings = as_list(evidence.get("negative_findings"))

    with st.expander("Подтверждающие фразы", expanded=False):
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

    render_evidence_highlighting(result, report_text)

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
        st.subheader("Причины оценки")
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


def render_llm_risk_prediction(stage_result: dict | None) -> None:
    if not stage_result:
        return

    st.subheader("LLM-прогноз хирургических рисков")

    status = stage_result.get("status")
    warnings = stage_result.get("warnings") or []
    errors = stage_result.get("errors") or []

    if status == "success":
        st.success("LLM-прогноз рисков выполнен")
    elif status == "failed":
        st.error("LLM-прогноз рисков не выполнен")
    else:
        st.warning(f"Статус LLM-прогноза: {ru(status)}")

    if warnings or errors:
        with st.expander("Ошибки и предупреждения LLM-прогноза", expanded=False):
            for error in errors:
                st.error(error)
            for warning in warnings:
                st.warning(warning)

    if status != "success":
        return

    llm_risk = stage_result.get("llm_risk") or {}
    risk_assessment = llm_risk.get("risk_assessment") or {}
    readiness = llm_risk.get("readiness") or {}
    operative_summary = llm_risk.get("operative_risk_summary") or {}
    clinical_summary = llm_risk.get("clinical_summary") or {}

    st.markdown("#### Основные числовые риски")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Кровопотеря >1500 мл",
            percent_value(risk_assessment.get("massive_blood_loss_risk_percent")),
        )
    with col2:
        st.metric(
            "Оценка кровопотери",
            ru(risk_assessment.get("estimated_blood_loss_ml")),
        )
    with col3:
        st.metric(
            "Диапазон кровопотери",
            ru(risk_assessment.get("estimated_blood_loss_range")),
        )
    with col4:
        st.metric(
            "Сосудистое вмешательство",
            percent_value(risk_assessment.get("vascular_intervention_risk_percent")),
        )

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric(
            "Вовлечение мочевого пузыря",
            percent_value(risk_assessment.get("bladder_involvement_risk_percent")),
        )
    with col6:
        st.metric(
            "Гистерэктомия",
            percent_value(risk_assessment.get("hysterectomy_risk_percent")),
        )
    with col7:
        st.metric(
            "Трансфузия",
            percent_value(risk_assessment.get("transfusion_risk_percent")),
        )

    st.markdown("#### Готовность")
    readiness_box(readiness.get("level"), readiness.get("rationale"))

    st.markdown("#### Резюме")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Операционный риск**")
        st.write(operative_summary.get("text") or "Нет данных")
    with summary_col2:
        st.markdown("**Клиническое резюме**")
        st.write(clinical_summary.get("text") or "Нет данных")

    st.markdown("#### Уверенность")
    st.write(ru(llm_risk.get("confidence")))

    with st.expander("LLM risk JSON", expanded=False):
        st.json(llm_risk)


def render_json_export(result: dict | None) -> None:
    if not result:
        st.info("Нет результата для отображения.")
        return

    with st.expander("Структурированный JSON", expanded=False):
        st.json(result)
        st.download_button(
            label="Скачать JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="pas_mri_extraction.json",
            mime="application/json",
        )
