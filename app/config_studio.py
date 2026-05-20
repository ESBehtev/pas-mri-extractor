from pathlib import Path

import streamlit as st
import yaml


CONFIG_DIR = Path("configs")
RUNTIME_CONFIG_DIR = Path("runtime_configs")

CONFIG_FILES = {
    "prompt.yaml": "prompt.local.yaml",
    "risk_score.yaml": "risk_score.local.yaml",
    "rules.yaml": "rules.local.yaml",
}


def get_source_path(config_name: str) -> Path:
    return CONFIG_DIR / config_name


def get_local_path(config_name: str) -> Path:
    return RUNTIME_CONFIG_DIR / CONFIG_FILES[config_name]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_active_config_text(config_name: str) -> str:
    local_path = get_local_path(config_name)
    if local_path.exists():
        return read_text(local_path)

    return read_text(get_source_path(config_name))


def validate_yaml(content: str) -> None:
    yaml.safe_load(content)


def save_local_override(config_name: str, content: str) -> Path:
    validate_yaml(content)
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    local_path = get_local_path(config_name)
    local_path.write_text(content, encoding="utf-8")

    return local_path


def reset_local_override(config_name: str) -> None:
    local_path = get_local_path(config_name)
    if local_path.exists():
        local_path.unlink()


def render_config_studio() -> None:
    st.subheader("Конфигурация")
    st.warning("Local override сохранён, но пока не применяется к pipeline.")

    config_name = st.selectbox(
        "Config file",
        list(CONFIG_FILES),
        key="config_studio_file",
    )

    source_path = get_source_path(config_name)
    local_path = get_local_path(config_name)
    editor_key = f"config_editor_{config_name}"
    show_source_key = f"config_show_source_{config_name}"

    source_text = read_text(source_path)

    if editor_key not in st.session_state:
        st.session_state[editor_key] = read_active_config_text(config_name)

    st.caption(f"Source: `{source_path}`")
    if local_path.exists():
        st.caption(f"Local override: `{local_path}`")
    else:
        st.caption("Local override: none")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("Показать исходный config"):
            st.session_state[show_source_key] = not st.session_state.get(
                show_source_key,
                False,
            )

    with action_col2:
        if st.button("Сбросить local override"):
            reset_local_override(config_name)
            st.session_state[editor_key] = source_text
            st.success("Local override сброшен. В редактор загружен исходный config.")

    if st.session_state.get(show_source_key, False):
        with st.expander("Исходный config", expanded=True):
            st.code(source_text, language="yaml")

    content = st.text_area(
        "YAML content",
        key=editor_key,
        height=520,
    )

    if st.button("Сохранить как local override", type="primary"):
        try:
            saved_path = save_local_override(config_name, content)
        except yaml.YAMLError as error:
            st.error(f"YAML invalid: {error}")
        else:
            st.success(f"Local override сохранён: {saved_path}")
            st.warning("Local override сохранён, но пока не применяется к pipeline.")
