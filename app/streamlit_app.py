from pathlib import Path

import streamlit as st

from components import (
    render_clinical_result,
    render_dual_comparison,
    render_json_export,
    render_report_sections,
)
from state import (
    get_last_outputs,
    init_session_state,
    save_extraction_result,
    set_running,
)

from pas_mri_extractor.pipeline import extract_features, extract_features_dual
from pas_mri_extractor.report_sections import split_report_sections


st.set_page_config(
    page_title="PAS MRI Extractor",
    page_icon="🧠",
    layout="wide",
)


init_session_state()

st.title("PAS MRI Extractor")
st.caption("Структурированное извлечение признаков из MRI-отчётов")


with st.sidebar:
    st.header("Статус")

    if st.session_state.get("model_loaded"):
        st.success("Модель загружена в память")
    else:
        st.info("Модель будет загружена при первом извлечении")

    st.markdown("---")
    st.warning(
        "Исследовательский прототип. "
        "Не предназначен для самостоятельного клинического применения."
    )


input_tab, clinical_tab, json_tab, diagnostics_tab = st.tabs(
    ["Input", "Clinical Result", "Structured JSON", "Diagnostics"]
)

result, dual_result, sections, last_diagnostic_mode = get_last_outputs()


with input_tab:
    st.subheader("Input")

    model_name = st.selectbox(
        "Модель",
        ["qwen_7b"],
        index=0,
        key="model_name",
    )

    diagnostic_mode = st.checkbox(
        "Diagnostic extraction comparison",
        value=False,
        help="Run full/body/conclusion extraction for comparison.",
    )

    example_path = Path("examples/sample_mri.txt")

    if st.button("Загрузить пример"):
        if example_path.exists():
            st.session_state["report_text"] = example_path.read_text(
                encoding="utf-8"
            )
        else:
            st.error("Файл examples/sample_mri.txt не найден")

    text = st.text_area(
        "Текст MRI-отчёта",
        key="report_text",
        height=280,
        placeholder="Вставьте MRI-отчёт...",
    )

    run = st.button(
        "Извлечь признаки",
        type="primary",
        disabled=st.session_state["is_running"],
    )

    if run:
        if not text.strip():
            st.error("Вставьте текст MRI-отчёта")
            st.stop()

        current_sections = split_report_sections(text)
        set_running(True)

        try:
            with st.spinner("Выполняется извлечение признаков..."):
                if diagnostic_mode:
                    current_dual_result = extract_features_dual(
                        text=text,
                        model_name=model_name,
                    )
                    current_result = current_dual_result["full"]
                else:
                    current_dual_result = None
                    current_result = extract_features(
                        text=text,
                        model_name=model_name,
                    )

            save_extraction_result(
                result=current_result,
                dual_result=current_dual_result,
                sections=current_sections,
                model_name=model_name,
                diagnostic_mode=diagnostic_mode,
            )

            result, dual_result, sections, last_diagnostic_mode = get_last_outputs()

        except Exception as error:
            st.error(f"Ошибка: {error}")
            st.stop()

        finally:
            set_running(False)


with clinical_tab:
    render_clinical_result(result)


with json_tab:
    render_json_export(result)


with diagnostics_tab:
    if not last_diagnostic_mode:
        st.info("Diagnostic extraction comparison выключен.")
    else:
        st.subheader("Разбор структуры отчёта")
        render_report_sections(sections)

        st.subheader("Сравнение: полный отчёт / описание / заключение")
        render_dual_comparison(dual_result)
