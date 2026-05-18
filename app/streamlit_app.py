import json
from pathlib import Path

import streamlit as st

from pas_mri_extractor.pipeline import extract_features


st.set_page_config(
    page_title="PAS MRI Extractor",
    page_icon="🧠",
    layout="wide",
)

st.title("PAS MRI Extractor")
st.caption("Структурированное извлечение признаков из MRI-отчётов")

with st.sidebar:
    st.header("Настройки")

    model_name = st.selectbox(
        "Модель",
        ["qwen_7b"],
        index=0,
    )

    st.markdown("---")

    st.warning(
        "Исследовательский прототип. "
        "Не предназначен для самостоятельного клинического применения."
    )

example_path = Path("examples/sample_mri.txt")

if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""

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
    height=300,
    placeholder="Вставьте MRI-отчёт...",
)

run = st.button("Извлечь признаки", type="primary")

if run:
    if not text.strip():
        st.error("Вставьте текст MRI-отчёта")
        st.stop()

    try:
        with st.spinner("Выполняется извлечение признаков..."):
            result = extract_features(
                text=text,
                model_name=model_name,
            )

    except Exception as error:
        st.error(f"Ошибка: {error}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Структурированный JSON")
        st.json(result)

    with col2:
        st.subheader("Ключевые признаки")

        extracted_features = result.get("extracted_features", {})

        invasion = extracted_features.get("invasion", {})
        anatomy = extracted_features.get("anatomy", {})

        st.metric(
            "Тип инвазии",
            invasion.get("type", "unknown"),
        )

        st.metric(
            "Уверенность",
            invasion.get("confidence", "unknown"),
        )

        st.write("### Анатомия")
        st.write(anatomy)

    st.download_button(
        label="Скачать JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="pas_mri_extraction.json",
        mime="application/json",
    )