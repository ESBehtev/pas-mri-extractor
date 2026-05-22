from functools import lru_cache
from typing import Any

from pas_mri_extractor.extractor import extract_mri_features
from pas_mri_extractor.models import LoadedModel, load_llm
from pas_mri_extractor.report_sections import split_report_sections
from pas_mri_extractor.scoring import normalize_mri_result


@lru_cache(maxsize=2)
def get_cached_model(model_name: str | None) -> LoadedModel:
    """
    Загружает LLM один раз на процесс Python и переиспользует её
    для следующих извлечений.
    """
    return load_llm(model_name)


def extract_features(
    text: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Основная точка входа для Streamlit / CLI / будущего API.
    Извлекает признаки из полного текста MRI-отчёта.
    """

    text = text.strip()

    if not text:
        raise ValueError("Пустой текст MRI-отчёта")

    loaded_model = get_cached_model(model_name)

    extracted = extract_mri_features(
        mri_text=text,
        loaded_model=loaded_model,
    )

    result = normalize_mri_result(extracted)

    return result.model_dump()


def extract_features_dual(
    text: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Раздельно анализирует:
    1. полный отчёт,
    2. описательную часть,
    3. заключение.

    Пока это диагностический режим: нужен, чтобы понять,
    что сильнее влияет на результат — описание или заключение.
    """

    text = text.strip()

    if not text:
        raise ValueError("Пустой текст MRI-отчёта")

    sections = split_report_sections(text)
    loaded_model = get_cached_model(model_name)

    full_extracted = extract_mri_features(
        mri_text=text,
        loaded_model=loaded_model,
    )
    full_result = normalize_mri_result(full_extracted).model_dump()

    body_result = None
    if sections.body and sections.body.strip():
        body_extracted = extract_mri_features(
            mri_text=sections.body,
            loaded_model=loaded_model,
        )
        body_result = normalize_mri_result(body_extracted).model_dump()

    conclusion_result = None
    if sections.conclusion and sections.conclusion.strip():
        conclusion_extracted = extract_mri_features(
            mri_text=sections.conclusion,
            loaded_model=loaded_model,
        )
        conclusion_result = normalize_mri_result(conclusion_extracted).model_dump()

    return {
        "sections": {
            "body": sections.body,
            "conclusion": sections.conclusion,
            "has_conclusion": sections.has_conclusion,
        },
        "full": full_result,
        "body": body_result,
        "conclusion": conclusion_result,
    }
