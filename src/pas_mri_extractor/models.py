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


def load_llm(model_name: str | None = None) -> LoadedModel:
    config = load_config("models.yaml")

    if model_name is None:
        model_name = config["default_model"]

    model_cfg = config["models"][model_name]

    model_id = model_cfg["model_id"]
    generation_config = model_cfg.get("generation", {})
    runtime_config = model_cfg.get("runtime", {})

    device = get_torch_device(runtime_config.get("device", "auto"))
    torch_dtype = get_torch_dtype(runtime_config.get("torch_dtype", "auto"))

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return LoadedModel(
        name=model_name,
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        generation_config=generation_config,
    )


def generate_text(loaded_model: LoadedModel, prompt: str) -> str:
    tokenizer = loaded_model.tokenizer
    model = loaded_model.model

    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **loaded_model.generation_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()