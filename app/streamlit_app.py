import streamlit as st

from components import (
    render_clinical_result,
    render_dual_comparison,
    render_json_export,
    render_report_sections,
)
from config_studio import render_config_studio
from examples import get_example_by_name, get_example_names
from state import (
    clear_extraction_request,
    get_last_outputs,
    init_session_state,
    request_extraction,
    save_extraction_result,
    set_running,
)

from pas_mri_extractor.pipeline import extract_features, extract_features_dual
from pas_mri_extractor.pipeline import get_cached_model
from pas_mri_extractor.report_sections import split_report_sections


@st.cache_resource(show_spinner=False)
def preload_model(model_name: str):
    return get_cached_model(model_name)


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
    model_status = st.empty()

    st.markdown("---")
    st.warning(
        "Исследовательский прототип. "
        "Не предназначен для самостоятельного клинического применения."
    )


result, dual_result, sections, last_diagnostic_mode = get_last_outputs()

extract_tab, config_tab = st.tabs(["Извлечение", "Конфигурация"])

with extract_tab:
    st.subheader("Input")

    control_col, example_col = st.columns([1, 2])

    with control_col:
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

    model_ready = False

    model_status.info("Модель загружается...")
    try:
        with st.spinner("Модель загружается..."):
            preload_model(model_name)
    except Exception as error:
        st.session_state["model_loaded"] = False
        model_status.error("Ошибка загрузки модели")
        st.error(f"Не удалось загрузить модель: {error}")
    else:
        st.session_state["model_loaded"] = True
        st.session_state["last_model_name"] = model_name
        model_status.success("Модель загружена в память")
        model_ready = True

    with example_col:
        example_name = st.selectbox(
            "Пример отчёта",
            get_example_names(),
            index=0,
            key="example_name",
        )
        selected_example = get_example_by_name(example_name)

        st.caption(
            f"{selected_example['category']} | "
            f"{selected_example['difficulty']} | "
            f"{selected_example['description']}"
        )

        if st.button("Загрузить пример"):
            st.session_state["report_text"] = selected_example["report_text"]

    text = st.text_area(
        "Текст MRI-отчёта",
        key="report_text",
        height=280,
        placeholder="Вставьте MRI-отчёт...",
    )

    st.button(
        "Извлечь признаки",
        type="primary",
        disabled=(
            not model_ready
            or st.session_state.get("is_running", False)
            or st.session_state.get("extract_requested", False)
        ),
        on_click=request_extraction,
    )

    if st.session_state.get("extract_requested"):
        if st.session_state.get("is_running"):
            clear_extraction_request()
            st.stop()

        if not model_ready:
            st.error("Модель не загружена")
            clear_extraction_request()
            st.stop()

        if not text.strip():
            st.error("Вставьте текст MRI-отчёта")
            clear_extraction_request()
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
            clear_extraction_request()

    if last_diagnostic_mode and dual_result:
        st.markdown("---")
        st.subheader("Сравнение: полный отчёт / описание / заключение")
        render_dual_comparison(dual_result)

        with st.expander(
            "Разбор структуры отчёта",
            expanded=False,
        ):
            render_report_sections(sections)

    if result:
        st.markdown("---")
        render_clinical_result(result)

        st.markdown("---")
        render_json_export(result)
    else:
        st.markdown("---")
        render_clinical_result(result)

with config_tab:
    render_config_studio()
