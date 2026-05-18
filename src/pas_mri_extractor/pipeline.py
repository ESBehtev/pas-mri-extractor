from typing import Any

from pas_mri_extractor.extractor import extract_mri_features
from pas_mri_extractor.scoring import normalize_mri_result


def extract_features(
    text: str,
    model_name: str = "qwen_7b",
) -> dict[str, Any]:
    """
    Главная точка входа для Streamlit, CLI и будущего API.
    Выполняет extraction через LLM и возвращает dict.
    """

    text = text.strip()

    if not text:
        raise ValueError("Пустой текст MRI-отчёта")

    extracted = extract_mri_features(
        mri_text=text,
        model_name=model_name,
    )

    result = normalize_mri_result(extracted)

    return result.model_dump()