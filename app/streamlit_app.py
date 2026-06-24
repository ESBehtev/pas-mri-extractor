import streamlit as st

from components import (
    build_extracted_result_for_llm_risk,
    render_clinical_result,
    render_dual_comparison,
    render_json_export,
    render_report_sections,
    render_risk_prediction_comparison,
    stage_result_to_llm_risk_ui,
)
from config_studio import render_config_studio
from examples import get_example_by_name, get_example_names
from state import (
    clear_extraction_request,
    get_last_llm_risk_output,
    get_last_outputs,
    init_session_state,
    request_extraction,
    reset_llm_risk_state,
    save_extraction_result,
    set_running,
)

from pas_mri_extractor.pipeline import extract_features, extract_features_dual
from pas_mri_extractor.pipeline import get_cached_model, unload_current_model
from pas_mri_extractor.orchestrator import run_risk_prediction_experiment
from pas_mri_extractor.models import get_available_models, get_default_model_name
from pas_mri_extractor.config import config_overrides
from pas_mri_extractor.scoring import clear_score_config_cache
from pas_mri_extractor.report_sections import split_report_sections


def preload_model(model_name: str):
    return get_cached_model(model_name)


st.set_page_config(
    page_title="PAS MRI Extractor",
    page_icon="🧠",
    layout="wide",
)


init_session_state()

st.title("Извлечение признаков PAS по МРТ")
st.caption("Структурированное извлечение признаков из МРТ-отчётов")


with st.sidebar:
    st.header("Статус")
    model_status = st.empty()

    st.markdown("---")
    st.warning(
        "Исследовательский прототип. "
        "Не предназначен для самостоятельного клинического применения."
    )


result, dual_result, sections, last_diagnostic_mode = get_last_outputs()
llm_risk_result = get_last_llm_risk_output()

extract_tab, config_tab = st.tabs(["Извлечение", "Конфигурация"])

with extract_tab:
    st.subheader("Input")

    control_col, example_col = st.columns([1, 2])

    with control_col:
        available_models = get_available_models()
        model_options = list(available_models.keys())
        default_model = get_default_model_name()
        default_model_index = (
            model_options.index(default_model)
            if default_model in model_options
            else 0
        )

        model_name = st.selectbox(
            "Модель",
            model_options,
            index=default_model_index,
            format_func=lambda name: available_models[name].get(
                "display_name",
                name,
            ),
            key="model_name",
        )

        diagnostic_mode = st.checkbox(
            "Диагностическое сравнение извлечения",
            value=False,
            help="Сравнить извлечение по полному отчёту, описанию и заключению.",
        )

        run_llm_risk_prediction = st.checkbox(
            "Выполнять LLM-прогноз рисков",
            value=True,
            help=(
                "Выполняет второй LLM-вызов для прогноза хирургических рисков "
                "на основе исходного текста и извлечённого JSON."
            ),
        )

    model_ready = False
    previous_model_name = st.session_state.get("last_model_name")
    if previous_model_name and previous_model_name != model_name:
        unload_current_model()
        st.cache_resource.clear()

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
        "Текст МРТ-отчёта",
        key="report_text",
        height=280,
        placeholder="Вставьте МРТ-отчёт...",
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

    rendered_current_request = False
    diagnostic_placeholder = st.empty()
    result_placeholder = st.empty()
    comparison_placeholder = st.empty()
    export_placeholder = st.empty()

    if st.session_state.get("extract_requested"):
        if st.session_state.get("is_running"):
            clear_extraction_request()
            st.stop()

        if not model_ready:
            st.error("Модель не загружена")
            clear_extraction_request()
            st.stop()

        if not text.strip():
            st.error("Вставьте текст МРТ-отчёта")
            clear_extraction_request()
            st.stop()

        current_sections = split_report_sections(text)
        set_running(True)
        reset_llm_risk_state()

        try:
            with st.spinner("Выполняется извлечение признаков..."):
                session_overrides = st.session_state.get("config_overrides", {})
                with config_overrides(session_overrides):
                    clear_score_config_cache()
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

            current_llm_risk_result = {
                "stage_name": "LLMRiskPredictionStage",
                "status": "running" if run_llm_risk_prediction else "skipped",
                "llm_risk": None,
                "errors": [],
                "warnings": [],
            }
            save_extraction_result(
                result=current_result,
                dual_result=current_dual_result,
                sections=current_sections,
                model_name=model_name,
                diagnostic_mode=diagnostic_mode,
                llm_risk_result=current_llm_risk_result,
            )

            if diagnostic_mode and current_dual_result:
                with diagnostic_placeholder.container():
                    st.markdown("---")
                    st.subheader("Сравнение: полный отчёт / описание / заключение")
                    render_dual_comparison(current_dual_result)

                    with st.expander(
                        "Разбор структуры отчёта",
                        expanded=False,
                    ):
                        render_report_sections(current_sections)

            with result_placeholder.container():
                st.markdown("---")
                render_clinical_result(current_result, text)

            with comparison_placeholder.container():
                st.markdown("---")
                render_risk_prediction_comparison(
                    current_result,
                    current_llm_risk_result,
                )

            with export_placeholder.container():
                st.markdown("---")
                render_json_export(
                    st.session_state.get("combined_result_json") or current_result,
                    extractor_only_result=current_result,
                )

            rendered_current_request = True

            if run_llm_risk_prediction:
                with st.spinner("Выполняется LLM-прогноз хирургических рисков..."):
                    try:
                        loaded_model = get_cached_model(model_name)
                        extracted_result = build_extracted_result_for_llm_risk(
                            current_result,
                        )
                        risk_stage_result = run_risk_prediction_experiment(
                            text=text,
                            extracted_result=extracted_result,
                            model_id=model_name,
                            loaded_model=loaded_model,
                        )
                        current_llm_risk_result = stage_result_to_llm_risk_ui(
                            risk_stage_result,
                        )
                    except Exception as risk_error:
                        current_llm_risk_result = {
                            "stage_name": "LLMRiskPredictionStage",
                            "status": "failed",
                            "llm_risk": None,
                            "errors": [str(risk_error)],
                            "warnings": [],
                        }

                save_extraction_result(
                    result=current_result,
                    dual_result=current_dual_result,
                    sections=current_sections,
                    model_name=model_name,
                    diagnostic_mode=diagnostic_mode,
                    llm_risk_result=current_llm_risk_result,
                )
                with comparison_placeholder.container():
                    st.markdown("---")
                    render_risk_prediction_comparison(
                        current_result,
                        current_llm_risk_result,
                    )
                with export_placeholder.container():
                    st.markdown("---")
                    render_json_export(
                        st.session_state.get("combined_result_json") or current_result,
                        extractor_only_result=current_result,
                    )

            result, dual_result, sections, last_diagnostic_mode = get_last_outputs()
            llm_risk_result = get_last_llm_risk_output()

        except Exception as error:
            st.error(f"Ошибка: {error}")
            st.stop()

        finally:
            clear_score_config_cache()
            set_running(False)
            clear_extraction_request()

    if not rendered_current_request and last_diagnostic_mode and dual_result:
        st.markdown("---")
        st.subheader("Сравнение: полный отчёт / описание / заключение")
        render_dual_comparison(dual_result)

        with st.expander(
            "Разбор структуры отчёта",
            expanded=False,
        ):
            render_report_sections(sections)

    if not rendered_current_request and result:
        st.markdown("---")
        render_clinical_result(result, st.session_state.get("report_text"))

        st.markdown("---")
        render_risk_prediction_comparison(result, llm_risk_result)

        st.markdown("---")
        render_json_export(
            st.session_state.get("combined_result_json") or result,
            extractor_only_result=result,
        )
    elif not rendered_current_request:
        st.markdown("---")
        render_clinical_result(result, st.session_state.get("report_text"))

with config_tab:
    render_config_studio()
