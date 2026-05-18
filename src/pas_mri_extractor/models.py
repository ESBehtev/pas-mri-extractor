"""
Загрузка LLM и генерация ответа.

Модель и параметры берутся из configs/models.yaml.
"""

import os
from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


load_dotenv()


@dataclass
class LoadedModel:
    name: str
    model_id: str
    tokenizer: Any
    model: Any
    generation_config: dict[str, Any]
    tokenizer_config: dict[str, Any]


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


def load_llm(model_name: str | None = None) -> LoadedModel:
    config = load_config("models.yaml")

    if model_name is None:
        model_name = config["default_model"]

    model_cfg = config["models"][model_name]

    model_id = model_cfg["model_id"]
    generation_config = model_cfg.get("generation", {})
    tokenizer_config = model_cfg.get("tokenizer", {})
    runtime_config = model_cfg.get("runtime", {})
    quantization_config = model_cfg.get("quantization", {})

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
        tokenizer=tokenizer,
        model=model,
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
    )


def format_prompt(loaded_model: LoadedModel, prompt: str) -> str:
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


def generate_text(loaded_model: LoadedModel, prompt: str) -> str:
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
            **loaded_model.generation_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=tokenizer_config.get("skip_special_tokens", True),
    ).strip()