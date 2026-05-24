import gc
import logging
from typing import Any

from pas_mri_extractor.extractor import (
    extract_mri_features,
    extract_mri_features_with_raw,
)
from pas_mri_extractor.models import LoadedModel, load_llm
from pas_mri_extractor.report_sections import split_report_sections
from pas_mri_extractor.scoring import normalize_mri_result


logger = logging.getLogger(__name__)
_CURRENT_MODEL: LoadedModel | None = None
_CURRENT_MODEL_NAME: str | None = None


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("cuda cache cleared")


def close_model_object(model: Any) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


def unload_current_model() -> None:
    """
    Выгружает текущую модель из process-level singleton перед загрузкой другой.
    """

    global _CURRENT_MODEL, _CURRENT_MODEL_NAME

    if _CURRENT_MODEL is None:
        clear_cuda_cache()
        gc.collect()
        return

    loaded_model = _CURRENT_MODEL
    model_name = _CURRENT_MODEL_NAME or loaded_model.name

    try:
        close_model_object(loaded_model.model)
    except Exception as error:
        logger.warning("Failed to close model %s: %s", model_name, error)

    loaded_model.model = None
    loaded_model.tokenizer = None
    _CURRENT_MODEL = None
    _CURRENT_MODEL_NAME = None

    del loaded_model
    gc.collect()
    logger.debug("model unloaded: %s", model_name)
    clear_cuda_cache()


def get_cached_model(model_name: str | None) -> LoadedModel:
    """
    Держит в памяти только одну LLM на процесс Python.
    При смене model_name старая модель выгружается перед загрузкой новой.
    """

    global _CURRENT_MODEL, _CURRENT_MODEL_NAME

    if _CURRENT_MODEL is not None and _CURRENT_MODEL_NAME == model_name:
        return _CURRENT_MODEL

    unload_current_model()
    _CURRENT_MODEL = load_llm(model_name)
    _CURRENT_MODEL_NAME = model_name
    return _CURRENT_MODEL


def extract_features(
    text: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Основная точка входа для Streamlit / CLI / будущего API.
    Извлекает признаки из полного текста MRI-отчёта.
    """

    return extract_features_with_artifacts(
        text=text,
        model_name=model_name,
    )["result"]


def extract_features_with_artifacts(
    text: str,
    model_name: str | None = None,
    loaded_model: LoadedModel | None = None,
    print_raw_output: bool = False,
) -> dict[str, Any]:
    """
    Единая inference-точка для CLI/eval, когда нужны диагностические артефакты.
    Streamlit продолжает использовать extract_features(), который возвращает
    только итоговый клинический JSON.
    """

    text = text.strip()

    if not text:
        raise ValueError("Пустой текст MRI-отчёта")

    if loaded_model is None:
        loaded_model = get_cached_model(model_name)

    artifacts = extract_mri_features_with_raw(
        mri_text=text,
        loaded_model=loaded_model,
        print_raw_output=print_raw_output,
    )
    result = normalize_mri_result(artifacts["validated"])

    return {
        **artifacts,
        "result": result.model_dump(),
    }


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
