"""Streamlit entry point for PAS MRI extraction."""

import streamlit as st

from app.ui.components import render_clinical_result, render_json_download
from app.ui.examples import get_example_by_name, get_example_names
from app.ui.state import init_session_state, save_result
from pas_mri_extractor.llm.client import get_available_models, get_default_model_name
from pas_mri_extractor.services.pipeline import extract_features, unload_current_client


st.set_page_config(page_title="PAS MRI Extractor", page_icon="🧠", layout="wide")
init_session_state()

st.title("PAS MRI Extractor")
st.caption("Research/prototype tool. The result is not a clinical decision.")

model_name = st.sidebar.selectbox(
    "API profile",
    options=list(get_available_models()),
    index=list(get_available_models()).index(get_default_model_name()),
)
example_name = st.sidebar.selectbox("Пример", ["—"] + get_example_names())
if example_name != "—" and st.sidebar.button("Загрузить пример"):
    st.session_state["report_text"] = get_example_by_name(example_name)["report_text"]

text = st.text_area("Текст MRI-отчёта", key="report_text", height=260)
if st.button("Извлечь признаки", type="primary", disabled=not text.strip()):
    try:
        st.session_state["is_running"] = True
        result = extract_features(text, model_name=model_name)
        save_result(result)
    except Exception as error:
        save_result(None, str(error))
    finally:
        st.session_state["is_running"] = False

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
if st.session_state["last_result"]:
    render_clinical_result(st.session_state["last_result"])
    render_json_download(st.session_state["last_result"])

if st.sidebar.button("Закрыть API-клиент"):
    unload_current_client()
    st.sidebar.success("API-клиент закрыт")
