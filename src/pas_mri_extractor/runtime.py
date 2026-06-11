"""Runtime environment settings for deployment wrappers.

This module is intentionally passive for now. It centralizes environment
defaults for Docker and future services without changing the current extractor
pipeline, schemas, prompts, scoring, or rules.
"""

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    app_env: str = "local"
    app_mode: str = "streamlit"
    active_stage: str = "extractor"
    model_config: str = "configs/models.yaml"
    prompt_config: str = "configs/prompt.yaml"
    output_dir: str = "outputs"
    data_dir: str = "data"
    models_dir: str = "models"
    runtime_config_dir: str = "runtime_configs"
    log_level: str = "INFO"
    enable_audit_log: bool = False
    enable_parallel_review: bool = False


def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        app_env=os.getenv("PAS_APP_ENV", RuntimeSettings.app_env),
        app_mode=os.getenv("PAS_APP_MODE", RuntimeSettings.app_mode),
        active_stage=os.getenv("PAS_ACTIVE_STAGE", RuntimeSettings.active_stage),
        model_config=os.getenv("PAS_MODEL_CONFIG", RuntimeSettings.model_config),
        prompt_config=os.getenv("PAS_PROMPT_CONFIG", RuntimeSettings.prompt_config),
        output_dir=os.getenv("PAS_OUTPUT_DIR", RuntimeSettings.output_dir),
        data_dir=os.getenv("PAS_DATA_DIR", RuntimeSettings.data_dir),
        models_dir=os.getenv("PAS_MODELS_DIR", RuntimeSettings.models_dir),
        runtime_config_dir=os.getenv(
            "PAS_RUNTIME_CONFIG_DIR",
            RuntimeSettings.runtime_config_dir,
        ),
        log_level=os.getenv("PAS_LOG_LEVEL", RuntimeSettings.log_level),
        enable_audit_log=_env_bool(
            "PAS_ENABLE_AUDIT_LOG",
            RuntimeSettings.enable_audit_log,
        ),
        enable_parallel_review=_env_bool(
            "PAS_ENABLE_PARALLEL_REVIEW",
            RuntimeSettings.enable_parallel_review,
        ),
    )
