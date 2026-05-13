"""
Главная логика извлечения признаков из MRI-текста.

Собирает prompt, вызывает модель, парсит JSON и валидирует результат через Pydantic.
"""

from .json_utils import extract_json_object
from .models import LoadedModel, generate_text, load_llm
from .prompts import build_prompt
from .schemas import MRIExtractionResult


def extract_mri_features(
    mri_text: str,
    model_name: str | None = None,
    loaded_model: LoadedModel | None = None,
) -> MRIExtractionResult:
    if not mri_text or not mri_text.strip():
        raise ValueError("MRI text is empty")

    if loaded_model is None:
        loaded_model = load_llm(model_name)

    prompt = build_prompt(mri_text)
    raw_output = generate_text(loaded_model, prompt)
    parsed = extract_json_object(raw_output)

    return MRIExtractionResult.model_validate(parsed)


def extract_mri_features_with_raw(
    mri_text: str,
    model_name: str | None = None,
    loaded_model: LoadedModel | None = None,
) -> dict:
    if not mri_text or not mri_text.strip():
        raise ValueError("MRI text is empty")

    if loaded_model is None:
        loaded_model = load_llm(model_name)

    prompt = build_prompt(mri_text)
    raw_output = generate_text(loaded_model, prompt)
    parsed = extract_json_object(raw_output)
    validated = MRIExtractionResult.model_validate(parsed)

    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed": parsed,
        "validated": validated.model_dump(),
    }