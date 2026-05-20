"""
Загрузка YAML-конфигов проекта.

Используется для моделей, промптов, scoring-конфигов и regex-правил.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
_CONFIG_OVERRIDES: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "CONFIG_OVERRIDES",
    default=None,
)


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
def load_base_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def load_config(name: str) -> dict[str, Any]:
    overrides = _CONFIG_OVERRIDES.get()
    if overrides and name in overrides:
        return deepcopy(overrides[name])

    return deepcopy(load_base_config(name))


@contextmanager
def config_overrides(overrides: dict[str, dict[str, Any]] | None):
    token = _CONFIG_OVERRIDES.set(overrides or None)
    try:
        yield
    finally:
        _CONFIG_OVERRIDES.reset(token)


def clear_config_cache() -> None:
    load_base_config.cache_clear()
