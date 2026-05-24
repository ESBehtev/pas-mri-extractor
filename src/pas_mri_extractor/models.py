"""
Загрузка LLM и генерация ответа.

Модель и параметры берутся из configs/models.yaml.
"""

import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_2_5_FALLBACK_COMMAND = (
    "python run_single.py --model qwen_2_5_7b --text-file examples/sample_mri.txt"
)

STRICT_JSON_SYSTEM_PROMPT = (
    "Return ONLY one valid JSON object. Do not use markdown. "
    "Do not include explanations. The first character of the response must be {. "
    "The last character of the response must be }."
)

STRICT_JSON_TEXT_PREFIX = (
    "Return ONLY one valid JSON object.\n"
    "Do not use markdown.\n"
    "Do not include explanations.\n"
    "The first character of the response must be {.\n"
    "The last character of the response must be }.\n\n"
)

RETRY_JSON_TEXT_PREFIX = (
    "Your previous answer was invalid. "
    "Output ONLY valid JSON matching the schema.\n"
    "Do not use markdown. Do not include explanations.\n"
    "The first character must be { and the last character must be }.\n\n"
)


@dataclass
class LoadedModel:
    name: str
    model_id: str
    backend: str
    tokenizer: Any
    model: Any
    generation_config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    output_config: dict[str, Any] = field(default_factory=dict)


class ModelConfigError(RuntimeError):
    pass


def resolve_model_name(model_name: str | None = None) -> str:
    if model_name:
        return model_name

    env_model = os.getenv("PAS_MODEL")
    if env_model:
        return env_model

    config = load_config("models.yaml")
    return config["default_model"]


def get_models_config() -> dict[str, Any]:
    return load_config("models.yaml")


def get_available_models() -> dict[str, dict[str, Any]]:
    return get_models_config().get("models", {})


def get_default_model_name() -> str:
    return resolve_model_name(None)


def get_model_config(model_name: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved_name = resolve_model_name(model_name)
    config = get_models_config()
    models = config.get("models", {})

    if resolved_name not in models:
        available = ", ".join(sorted(models)) or "none"
        raise ModelConfigError(
            f"Unknown model '{resolved_name}'. Available models: {available}"
        )

    return resolved_name, models[resolved_name]


def resolve_model_path(path_or_id: str) -> Path | None:
    path = Path(path_or_id).expanduser()

    if path.is_absolute():
        return path.resolve()

    if "/" in path_or_id and not path_or_id.startswith("models/"):
        return None

    return (PROJECT_ROOT / path).resolve()


def get_model_id_or_path(model_cfg: dict[str, Any]) -> str:
    model_id_or_path = (
        model_cfg.get("model_path")
        or model_cfg.get("model_id_or_path")
        or model_cfg.get("model_id")
    )
    if not model_id_or_path:
        raise ModelConfigError(
            "Model config must define model_path, model_id_or_path, or model_id."
        )

    return str(model_id_or_path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_quantization_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    quantization_config = model_cfg.get("quantization_config")

    if isinstance(quantization_config, dict):
        return quantization_config

    legacy_config = model_cfg.get("quantization")
    if isinstance(legacy_config, dict):
        return legacy_config

    if model_cfg.get("load_in_4bit"):
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": model_cfg.get("torch_dtype", "float16"),
            "bnb_4bit_use_double_quant": model_cfg.get(
                "bnb_4bit_use_double_quant",
                True,
            ),
            "bnb_4bit_quant_type": model_cfg.get("bnb_4bit_quant_type", "nf4"),
        }

    return {}


def assert_model_available(model_name: str, model_cfg: dict[str, Any]) -> None:
    model_id_or_path = get_model_id_or_path(model_cfg)
    local_path = resolve_model_path(model_id_or_path)

    if local_path is None or local_path.exists():
        return

    if model_cfg.get("backend") == "llama_cpp":
        hf_repo = model_cfg.get("hf_repo", model_id_or_path)
        hf_filename = model_cfg.get("hf_filename")
        download_command = f"hf download {hf_repo}"
        if hf_filename:
            download_command = f"{download_command} {hf_filename}"
        download_target = local_path.parent if hf_filename else local_path
        download_command = f"{download_command} --local-dir {download_target}"

        raise ModelConfigError(
            "GGUF model file not found:\n"
            f"  {local_path}\n\n"
            "Please download the model to this location.\n\n"
            "Download on server:\n"
            f"{download_command}"
        )

    hf_repo = model_cfg.get("hf_repo", model_id_or_path)
    hf_filename = model_cfg.get("hf_filename")
    download_command = f"hf download {hf_repo}"
    if hf_filename:
        download_command = f"{download_command} {hf_filename}"
    download_target = local_path.parent if hf_filename else local_path
    download_command = (
        f"{download_command} --local-dir {display_path(download_target)}"
    )

    raise ModelConfigError(
        f"Model {model_name} not found.\n"
        f"Expected path:\n{display_path(local_path)}\n\n"
        f"Download on server:\n"
        f"{download_command}\n\n"
        f"Fallback:\n{QWEN_2_5_FALLBACK_COMMAND}"
    )


def dry_run_model_config(model_name: str | None = None) -> dict[str, Any]:
    resolved_name, model_cfg = get_model_config(model_name)
    model_id_or_path = get_model_id_or_path(model_cfg)
    local_path = resolve_model_path(model_id_or_path)
    exists = None if local_path is None else local_path.exists()
    hf_filename = model_cfg.get("hf_filename")
    output_config = model_cfg.get("output") or {}
    response_format = output_config.get("response_format")
    response_format_type = (
        response_format.get("type") if isinstance(response_format, dict) else None
    )
    download_command = None
    if local_path is not None and model_cfg.get("hf_repo"):
        download_command = f"hf download {model_cfg.get('hf_repo')}"
        if hf_filename:
            download_command = f"{download_command} {hf_filename}"
        download_target = local_path.parent if hf_filename else local_path
        download_command = f"{download_command} --local-dir {download_target}"

    return {
        "model": resolved_name,
        "display_name": model_cfg.get("display_name", resolved_name),
        "backend": model_cfg.get("backend", "transformers"),
        "model_id_or_path": model_id_or_path,
        "expected_path": str(local_path) if local_path is not None else None,
        "path_exists": exists,
        "hf_repo": model_cfg.get("hf_repo"),
        "hf_filename": hf_filename,
        "download_command": download_command,
        "fallback_command": QWEN_2_5_FALLBACK_COMMAND,
        "output.enforce_json": bool(output_config.get("enforce_json", False)),
        "output.response_format.type": response_format_type,
        "output": {
            "enforce_json": bool(output_config.get("enforce_json", False)),
            "response_format": response_format,
        },
    }


def get_torch_device(device: str = "auto") -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_dtype(dtype: str = "auto") -> Any:
    if dtype != "auto":
        return getattr(torch, dtype)

    if torch.cuda.is_available():
        return torch.float16

    if torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


def build_quantization_config(quantization_config: dict[str, Any]) -> Any | None:
    if not quantization_config.get("load_in_4bit", False):
        return None

    from transformers import BitsAndBytesConfig

    compute_dtype = get_torch_dtype(
        quantization_config.get("bnb_4bit_compute_dtype", "float16")
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization_config.get(
            "bnb_4bit_use_double_quant",
            True,
        ),
        bnb_4bit_quant_type=quantization_config.get(
            "bnb_4bit_quant_type",
            "nf4",
        ),
    )


def load_llama_cpp_model(
    model_name: str,
    model_cfg: dict[str, Any],
) -> LoadedModel:
    assert_model_available(model_name, model_cfg)

    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise ModelConfigError(
            "llama-cpp-python is required for backend 'llama_cpp'.\n"
            "Install on server:\n"
            "pip install llama-cpp-python"
        ) from error

    model_path = resolve_model_path(get_model_id_or_path(model_cfg))
    if model_path is None:
        raise ModelConfigError("llama_cpp backend requires a local model path.")

    runtime_config = model_cfg.get("runtime", {})
    generation_config = model_cfg.get("generation", {})
    tokenizer_config = model_cfg.get("tokenizer", {})
    output_config = model_cfg.get("output", {})

    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=int(runtime_config.get("n_gpu_layers", -1)),
        n_ctx=int(runtime_config.get("n_ctx", 4096)),
        n_batch=int(runtime_config.get("n_batch", 512)),
        verbose=bool(runtime_config.get("verbose", False)),
    )

    return LoadedModel(
        name=model_name,
        model_id=str(model_path),
        backend="llama_cpp",
        tokenizer=None,
        model=model,
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
        output_config=output_config if isinstance(output_config, dict) else {},
    )


def load_transformers_model(
    model_name: str,
    model_cfg: dict[str, Any],
) -> LoadedModel:
    assert_model_available(model_name, model_cfg)

    model_id = get_model_id_or_path(model_cfg)
    generation_config = model_cfg.get("generation", {})
    tokenizer_config = model_cfg.get("tokenizer", {})
    output_config = model_cfg.get("output", {})
    runtime_config = model_cfg.get("runtime", {})
    quantization_config = get_quantization_config(model_cfg)

    device = get_torch_device(
        runtime_config.get("device", "auto")
    )

    dtype = get_torch_dtype(
        runtime_config.get("dtype", "auto")
    )

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = build_quantization_config(quantization_config)

    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = runtime_config.get("device_map", "auto")
    else:
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    if bnb_config is None:
        model.to(device)

    model.eval()

    return LoadedModel(
        name=model_name,
        model_id=model_id,
        backend="transformers",
        tokenizer=tokenizer,
        model=model,
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
        output_config=output_config if isinstance(output_config, dict) else {},
    )


def load_llm(model_name: str | None = None) -> LoadedModel:
    model_name, model_cfg = get_model_config(model_name)
    backend = model_cfg.get("backend", "transformers")

    if backend == "transformers":
        return load_transformers_model(model_name, model_cfg)

    if backend == "llama_cpp":
        return load_llama_cpp_model(model_name, model_cfg)

    raise ModelConfigError(
        f"Model {model_name} uses unsupported backend '{backend}'. "
        "Supported backends: transformers, llama_cpp."
    )


def format_prompt(loaded_model: LoadedModel, prompt: str) -> str:
    if loaded_model.backend == "llama_cpp":
        return prompt

    tokenizer = loaded_model.tokenizer
    tokenizer_config = loaded_model.tokenizer_config

    use_chat_template = tokenizer_config.get("use_chat_template", True)

    if not use_chat_template:
        return prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_llama_cpp_prompt(prompt: str, retry_json: bool = False) -> str:
    prefix = RETRY_JSON_TEXT_PREFIX if retry_json else STRICT_JSON_TEXT_PREFIX
    return f"{prefix}{prompt}"


def merge_generation_config(
    loaded_model: LoadedModel,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_config = dict(loaded_model.generation_config)
    if overrides:
        generation_config.update(overrides)

    return generation_config


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }


def supports_kwarg(callable_obj: Any, kwarg: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return True

    return kwarg in signature.parameters


def get_llama_cpp_generation_kwargs(
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    max_tokens = generation_config.get(
        "max_tokens",
        generation_config.get("max_new_tokens", 3000),
    )

    kwargs = {
        "max_tokens": int(max_tokens),
        "temperature": float(generation_config.get("temperature", 0.0)),
        "top_p": float(generation_config.get("top_p", 1.0)),
    }

    repeat_penalty = generation_config.get(
        "repeat_penalty",
        generation_config.get("repetition_penalty"),
    )
    if repeat_penalty is not None:
        kwargs["repeat_penalty"] = float(repeat_penalty)

    seed = generation_config.get("seed")
    if seed is not None:
        kwargs["seed"] = int(seed)

    return kwargs


def get_llama_cpp_response_format(
    loaded_model: LoadedModel,
) -> dict[str, Any] | None:
    output_config = loaded_model.output_config
    if not output_config.get("enforce_json", False):
        return None

    response_format = output_config.get("response_format")
    if not isinstance(response_format, dict):
        return None

    return response_format


def create_llama_cpp_chat_completion(
    loaded_model: LoadedModel,
    messages: list[dict[str, str]],
    generation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    create_chat_completion = loaded_model.model.create_chat_completion
    call_kwargs = filter_supported_kwargs(
        create_chat_completion,
        generation_kwargs,
    )
    response_format = get_llama_cpp_response_format(loaded_model)

    if response_format is None:
        return create_chat_completion(
            messages=messages,
            **call_kwargs,
        )

    if not supports_kwarg(create_chat_completion, "response_format"):
        logger.warning(
            "response_format not supported by current llama_cpp backend, "
            "falling back to prompt+postprocessing JSON extraction"
        )
        return create_chat_completion(
            messages=messages,
            **call_kwargs,
        )

    logger.info(
        "Using llama_cpp JSON response_format for model: %s",
        loaded_model.name,
    )

    try:
        return create_chat_completion(
            messages=messages,
            response_format=response_format,
            **call_kwargs,
        )
    except TypeError as error:
        if "response_format" not in str(error):
            raise

        logger.warning(
            "response_format not supported by current llama_cpp backend, "
            "falling back to prompt+postprocessing JSON extraction"
        )
        return create_chat_completion(
            messages=messages,
            **call_kwargs,
        )


def generate_text(
    loaded_model: LoadedModel,
    prompt: str,
    generation_overrides: dict[str, Any] | None = None,
    retry_json: bool = False,
) -> str:
    generation_config = merge_generation_config(loaded_model, generation_overrides)

    if loaded_model.backend == "llama_cpp":
        tokenizer_config = loaded_model.tokenizer_config
        generation_kwargs = get_llama_cpp_generation_kwargs(generation_config)

        if tokenizer_config.get("use_chat_template", False) and hasattr(
            loaded_model.model,
            "create_chat_completion",
        ):
            system_prompt = (
                RETRY_JSON_TEXT_PREFIX.strip()
                if retry_json
                else STRICT_JSON_SYSTEM_PROMPT
            )
            output = create_llama_cpp_chat_completion(
                loaded_model=loaded_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                generation_kwargs=generation_kwargs,
            )
            return output["choices"][0]["message"]["content"].strip()

        output = loaded_model.model(
            build_llama_cpp_prompt(prompt, retry_json=retry_json),
            **filter_supported_kwargs(loaded_model.model, generation_kwargs),
        )
        return output["choices"][0]["text"].strip()

    tokenizer = loaded_model.tokenizer
    model = loaded_model.model
    tokenizer_config = loaded_model.tokenizer_config

    device = next(model.parameters()).device

    formatted_prompt = format_prompt(
        loaded_model,
        prompt,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=tokenizer_config.get("truncation", True),
        max_length=tokenizer_config.get("max_input_tokens", None),
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=tokenizer_config.get("skip_special_tokens", True),
    ).strip()
