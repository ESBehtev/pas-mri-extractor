"""Production extraction pipeline and process-level API-client lifecycle."""

from __future__ import annotations

from threading import RLock
from typing import Any

from ..core.scoring import normalize_mri_result
from ..llm.client import LLMClient, create_llm_client, resolve_model_name
from .extractor import extract_mri_features_with_raw


_CURRENT_CLIENT: LLMClient | None = None
_CURRENT_PROFILE_NAME: str | None = None
_CLIENT_LOCK = RLock()


def unload_current_client() -> None:
    """Close and discard the process-level API client."""
    global _CURRENT_CLIENT, _CURRENT_PROFILE_NAME
    with _CLIENT_LOCK:
        client, _CURRENT_CLIENT = _CURRENT_CLIENT, None
        _CURRENT_PROFILE_NAME = None
        if client is not None:
            client.close()


def get_cached_client(model_name: str | None = None) -> LLMClient:
    """Return one API client for the selected configured profile."""
    global _CURRENT_CLIENT, _CURRENT_PROFILE_NAME
    profile_name = resolve_model_name(model_name)
    with _CLIENT_LOCK:
        if _CURRENT_CLIENT is not None and _CURRENT_PROFILE_NAME == profile_name:
            return _CURRENT_CLIENT
        unload_current_client()
        _CURRENT_CLIENT = create_llm_client(profile_name)
        _CURRENT_PROFILE_NAME = profile_name
        return _CURRENT_CLIENT


def extract_features(text: str, model_name: str | None = None) -> dict[str, Any]:
    """Extract validated PAS features and deterministic research risk outputs."""
    return extract_features_with_artifacts(text=text, model_name=model_name)["result"]


def extract_features_with_artifacts(
    text: str,
    model_name: str | None = None,
    client: LLMClient | None = None,
    print_raw_output: bool = False,
) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Пустой текст MRI-отчёта")
    client = client or get_cached_client(model_name)
    artifacts = extract_mri_features_with_raw(
        mri_text=text,
        client=client,
        print_raw_output=print_raw_output,
    )
    return {**artifacts, "result": normalize_mri_result(artifacts["validated"]).model_dump()}
