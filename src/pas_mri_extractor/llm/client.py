"""OpenAI-compatible LLM client and model configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        OpenAI,
    )
except ImportError:  # pragma: no cover - exercised only in incomplete local environments
    OpenAI = None  # type: ignore[assignment,misc]

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        status_code = 0

    class APITimeoutError(Exception):
        pass

    class BadRequestError(Exception):
        pass

from ..config import load_config


load_dotenv()
logger = logging.getLogger(__name__)

STRICT_JSON_SYSTEM_PROMPT = (
    "Return ONLY one valid JSON object. Do not use markdown. "
    "Do not include explanations. The first character of the response must be {. "
    "The last character of the response must be }."
)
RETRY_JSON_SYSTEM_PROMPT = (
    "Your previous answer was invalid. Output ONLY valid JSON matching the schema. "
    "Do not use markdown. Do not include explanations. "
    "The first character must be { and the last character must be }."
)


class ModelConfigError(RuntimeError):
    """Raised when an OpenAI-compatible API profile is incomplete or invalid."""


class LLMRequestError(RuntimeError):
    """Raised when an OpenAI-compatible server cannot complete a request."""


@dataclass(frozen=True)
class LLMConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float
    temperature: float
    max_tokens: int
    response_format: dict[str, Any] | None
    fallback_on_unsupported_response_format: bool


class LLMClient(Protocol):
    config: LLMConfig

    def generate(
        self,
        prompt: str,
        *,
        generation_overrides: dict[str, Any] | None = None,
        retry_json: bool = False,
    ) -> str: ...

    def close(self) -> None: ...


def get_models_config() -> dict[str, Any]:
    return load_config("models.yaml")


def get_available_models() -> dict[str, dict[str, Any]]:
    return get_models_config().get("models", {})


def resolve_model_name(model_name: str | None = None) -> str:
    if model_name:
        return model_name
    return os.getenv("PAS_API_PROFILE") or get_models_config()["default_model"]


def get_default_model_name() -> str:
    return resolve_model_name()


def get_model_config(model_name: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved_name = resolve_model_name(model_name)
    models = get_available_models()
    if resolved_name not in models:
        available = ", ".join(sorted(models)) or "none"
        raise ModelConfigError(
            f"Unknown API profile '{resolved_name}'. Available profiles: {available}"
        )
    return resolved_name, models[resolved_name]


def _env_or_config(env_name: str, config_value: Any) -> Any:
    value = os.getenv(env_name)
    return value if value not in (None, "") else config_value


def _as_positive_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelConfigError(f"{field_name} must be a positive number.") from error
    if result <= 0:
        raise ModelConfigError(f"{field_name} must be a positive number.")
    return result


def _as_nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ModelConfigError(f"{field_name} must be a non-negative integer.") from error
    if result < 0:
        raise ModelConfigError(f"{field_name} must be a non-negative integer.")
    return result


def resolve_llm_config(model_name: str | None = None) -> LLMConfig:
    name, profile = get_model_config(model_name)
    generation = profile.get("generation") or {}
    output = profile.get("output") or {}
    api_key_env = str(profile.get("api_key_env", "PAS_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    base_url = str(_env_or_config("PAS_API_BASE_URL", profile.get("base_url", ""))).rstrip("/")
    model = str(_env_or_config("PAS_MODEL", profile.get("model", "")))

    if not base_url:
        raise ModelConfigError("API base URL is missing. Set PAS_API_BASE_URL or models.yaml.")
    if not model:
        raise ModelConfigError("API model is missing. Set PAS_MODEL or models.yaml.")
    if not api_key:
        raise ModelConfigError(f"API key is missing. Set {api_key_env}.")

    timeout = _as_positive_float(_env_or_config("PAS_TIMEOUT", profile.get("timeout", 60)), "timeout")
    temperature = float(_env_or_config("PAS_TEMPERATURE", generation.get("temperature", 0.0)))
    max_tokens = _as_nonnegative_int(
        _env_or_config("PAS_MAX_TOKENS", generation.get("max_tokens", 2048)),
        "max_tokens",
    )
    response_format = output.get("response_format") if output.get("enforce_json") else None
    if response_format is not None and not isinstance(response_format, dict):
        raise ModelConfigError("output.response_format must be an object.")

    return LLMConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        fallback_on_unsupported_response_format=bool(
            output.get("fallback_on_unsupported_response_format", False)
        ),
    )


def dry_run_model_config(model_name: str | None = None) -> dict[str, Any]:
    config = resolve_llm_config(model_name)
    return {
        "profile": config.name,
        "provider": "openai_compatible",
        "base_url": config.base_url,
        "model": config.model,
        "timeout": config.timeout,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": config.response_format,
        "fallback_on_unsupported_response_format": config.fallback_on_unsupported_response_format,
        "api_key_configured": True,
    }


def _response_format_is_unsupported(error: BadRequestError) -> bool:
    message = str(error).lower()
    return "response_format" in message and any(
        marker in message
        for marker in ("unsupported", "not supported", "unknown", "unrecognized")
    )


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, sdk_client: Any | None = None) -> None:
        self.config = config
        if sdk_client is not None:
            self._client = sdk_client
        else:
            if OpenAI is None:
                raise ModelConfigError(
                    "The OpenAI Python SDK is not installed. Run `pip install -r requirements.txt`."
                )
            self._client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
            )

    def generate(
        self,
        prompt: str,
        *,
        generation_overrides: dict[str, Any] | None = None,
        retry_json: bool = False,
    ) -> str:
        overrides = generation_overrides or {}
        max_tokens = _as_nonnegative_int(
            overrides.get("max_tokens", self.config.max_tokens), "max_tokens"
        )
        temperature = float(overrides.get("temperature", self.config.temperature))
        system_prompt = RETRY_JSON_SYSTEM_PROMPT if retry_json else STRICT_JSON_SYSTEM_PROMPT
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.response_format is not None:
            kwargs["response_format"] = self.config.response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
        except BadRequestError as error:
            if not (
                self.config.response_format is not None
                and self.config.fallback_on_unsupported_response_format
                and _response_format_is_unsupported(error)
            ):
                raise LLMRequestError(f"API request was rejected: {error}") from error
            logger.warning("API profile %s rejected response_format; retrying without it", self.config.name)
            kwargs.pop("response_format")
            try:
                response = self._client.chat.completions.create(**kwargs)
            except BadRequestError as retry_error:
                raise LLMRequestError(
                    f"API request was rejected after response-format fallback: {retry_error}"
                ) from retry_error
            except (APIConnectionError, APITimeoutError, APIStatusError) as retry_error:
                raise LLMRequestError(f"API request failed after response-format fallback: {retry_error}") from retry_error
        except APITimeoutError as error:
            raise LLMRequestError(f"API request timed out after {self.config.timeout:g} seconds.") from error
        except APIConnectionError as error:
            raise LLMRequestError(f"Could not connect to API server at {self.config.base_url}.") from error
        except APIStatusError as error:
            raise LLMRequestError(f"API server returned status {error.status_code}.") from error

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMRequestError("API response contains no message content.")
        return content.strip()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class MockLLMClient:
    """In-memory deterministic client for unit tests; it never uses a network."""

    def __init__(self, responses: list[str], config: LLMConfig | None = None) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.config = config or LLMConfig(
            name="mock",
            base_url="http://mock.invalid/v1",
            api_key="not-used",
            model="mock-model",
            timeout=1.0,
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
            fallback_on_unsupported_response_format=False,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise LLMRequestError("MockLLMClient has no queued response.")
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def create_llm_client(model_name: str | None = None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(resolve_llm_config(model_name))
