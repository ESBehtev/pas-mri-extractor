from pathlib import Path

import streamlit as st
import yaml

from pas_mri_extractor.config import clear_config_cache
from pas_mri_extractor.scoring import clear_score_config_cache


CONFIG_DIR = Path("configs")

CONFIG_FILES = [
    "prompt.yaml",
    "risk_score.yaml",
    "rules.yaml",
]


def get_source_path(config_name: str) -> Path:
    return CONFIG_DIR / config_name


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate_yaml(content: str) -> dict:
    return yaml.safe_load(content) or {}


def init_config_override_state() -> None:
    st.session_state.setdefault("config_overrides", {})
    st.session_state.setdefault("config_override_texts", {})


def get_editor_text(config_name: str) -> str:
    override_texts = st.session_state["config_override_texts"]
    if config_name in override_texts:
        return override_texts[config_name]

    return read_text(get_source_path(config_name))


def clear_runtime_config_caches() -> None:
    clear_config_cache()
    clear_score_config_cache()


def save_session_override(config_name: str, content: str) -> None:
    parsed = validate_yaml(content)
    st.session_state["config_overrides"][config_name] = parsed
    st.session_state["config_override_texts"][config_name] = content
    clear_runtime_config_caches()


def reset_session_override(config_name: str) -> None:
    st.session_state["config_overrides"].pop(config_name, None)
    st.session_state["config_override_texts"].pop(config_name, None)
    clear_runtime_config_caches()


def render_config_studio() -> None:
    init_config_override_state()

    st.subheader("Конфигурация")
    st.warning(
        "Переопределение действует только в текущей сессии приложения и не "
        "записывается в файлы проекта."
    )
    st.caption("Старые локальные YAML-файлы конфигурации игнорируются приложением.")

    config_name = st.selectbox(
        "Файл конфигурации",
        CONFIG_FILES,
        key="config_studio_file",
    )

    source_path = get_source_path(config_name)
    editor_key = f"config_editor_{config_name}"
    show_source_key = f"config_show_source_{config_name}"

    source_text = read_text(source_path)

    if editor_key not in st.session_state:
        st.session_state[editor_key] = get_editor_text(config_name)

    st.caption(f"Источник: `{source_path}`")

    if config_name in st.session_state["config_overrides"]:
        st.caption("Переопределение сессии: активно")
    else:
        st.caption("Переопределение сессии: нет")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("Показать исходную конфигурацию"):
            st.session_state[show_source_key] = not st.session_state.get(
                show_source_key,
                False,
            )

    with action_col2:
        if st.button("Сбросить переопределение сессии"):
            reset_session_override(config_name)
            st.session_state[editor_key] = source_text
            st.success("Переопределение сессии сброшено.")

    if st.session_state.get(show_source_key, False):
        with st.expander("Исходная конфигурация", expanded=True):
            st.code(source_text, language="yaml")

    content = st.text_area(
        "Содержимое YAML",
        key=editor_key,
        height=520,
    )

    if st.button("Сохранить переопределение на время сессии", type="primary"):
        try:
            save_session_override(config_name, content)
        except yaml.YAMLError as error:
            st.error(f"Некорректный YAML: {error}")
        else:
            st.success("Переопределение сессии сохранено.")
            st.warning("Переопределение будет применено при следующем извлечении.")
