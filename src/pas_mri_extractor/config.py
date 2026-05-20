"""
Загрузка YAML-конфигов проекта.

Используется для моделей, промптов, scoring-конфигов и regex-правил.
"""

from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNTIME_CONFIG_DIR = PROJECT_ROOT / "runtime_configs"

LOCAL_CONFIG_OVERRIDES = {
    "prompt.yaml": "prompt.local.yaml",
    "risk_score.yaml": "risk_score.local.yaml",
    "rules.yaml": "rules.local.yaml",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


@lru_cache(maxsize=16)
def load_config(name: str) -> dict[str, Any]:
    local_name = LOCAL_CONFIG_OVERRIDES.get(name)
    if local_name:
        local_path = RUNTIME_CONFIG_DIR / local_name
        if local_path.exists():
            return load_yaml(local_path)

    return load_yaml(CONFIG_DIR / name)


def clear_config_cache() -> None:
    load_config.cache_clear()
