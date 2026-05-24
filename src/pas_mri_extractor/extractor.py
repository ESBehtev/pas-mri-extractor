"""
Главная логика извлечения признаков из MRI-текста.

Собирает prompt, вызывает модель, парсит JSON и валидирует результат через Pydantic.
"""

import os
import sys

from .json_utils import extract_json_object
from .models import LoadedModel, generate_text, load_llm
from .prompts import build_prompt
from .schemas import MRIExtractionResult


VALID_INVASION_TYPES = {
    "none",
    "accreta",
    "increta",
    "percreta",
}

CONFIDENCE_VALUES = {
    "possible",
    "probable",
    "definite",
    "unclear",
    "absent",
}

STATUS_VALUES = {
    "absent",
    "possible",
    "probable",
    "present",
}

JSON_PARSE_FAILURE_MARKERS = (
    "No JSON object found in model output",
    "Incomplete JSON object in model output",
    "No valid JSON object found in model output",
)


def should_print_raw_output(print_raw_output: bool = False) -> bool:
    return print_raw_output or os.getenv("PAS_PRINT_RAW_OUTPUT") == "1"


def print_raw_model_output(raw_output: str) -> None:
    print("----- RAW MODEL OUTPUT START -----", file=sys.stderr)
    print(raw_output, file=sys.stderr)
    print("----- RAW MODEL OUTPUT END -----", file=sys.stderr)


def is_json_parse_failure(error: Exception) -> bool:
    message = str(error)
    return any(marker in message for marker in JSON_PARSE_FAILURE_MARKERS)


def build_raw_output_debug_message(raw_output: str) -> str:
    return (
        "Model output could not be parsed as JSON.\n"
        f"raw_output_length={len(raw_output)}\n"
        "raw_output_first_1500:\n"
        f"{raw_output[:1500]}\n"
        "raw_output_last_1500:\n"
        f"{raw_output[-1500:]}"
    )


def parse_json_with_retry(
    loaded_model: LoadedModel,
    prompt: str,
    raw_output: str,
    print_raw_output: bool = False,
) -> tuple[dict, str]:
    if should_print_raw_output(print_raw_output):
        print_raw_model_output(raw_output)

    try:
        return extract_json_object(raw_output), raw_output
    except ValueError as first_error:
        if not is_json_parse_failure(first_error):
            raise

    retry_raw_output = generate_text(
        loaded_model,
        prompt,
        generation_overrides={
            "max_new_tokens": 3500,
            "temperature": 0.0,
        },
        retry_json=True,
    )

    if should_print_raw_output(print_raw_output):
        print_raw_model_output(retry_raw_output)

    try:
        return extract_json_object(retry_raw_output), retry_raw_output
    except ValueError as retry_error:
        if is_json_parse_failure(retry_error):
            debug_message = build_raw_output_debug_message(retry_raw_output)
            raise ValueError(debug_message) from retry_error
        raise


def normalize_invasion_type(value: object) -> str:
    if value is None:
        return "none"

    text = str(value).strip().lower()

    if text in VALID_INVASION_TYPES:
        return text

    if "percreta" in text:
        return "percreta"

    if "increta" in text:
        return "increta"

    if "accreta" in text:
        return "accreta"

    return "none"


def normalize_confidence(value: object) -> str:
    if value is None:
        return "unclear"

    text = str(value).strip().lower()

    if text in CONFIDENCE_VALUES:
        return text

    if any(marker in text for marker in ["definite", "confirmed", "convincing", "достовер"]):
        return "definite"

    if any(marker in text for marker in ["probable", "likely", "вероят"]):
        return "probable"

    if any(
        marker in text
        for marker in [
            "possible",
            "cannot exclude",
            "can't exclude",
            "not excluded",
            "не исключ",
            "возмож",
        ]
    ):
        return "possible"

    if any(marker in text for marker in ["absent", "no ", "нет", "не выяв"]):
        return "absent"

    return "unclear"


def normalize_status(value: object) -> str:
    if value is None:
        return "absent"

    text = str(value).strip().lower()

    if text in STATUS_VALUES:
        return text

    if any(
        marker in text
        for marker in [
            "cannot exclude",
            "can't exclude",
            "not excluded",
            "не исключ",
            "possible",
            "possibly",
            "возмож",
            "под вопросом",
        ]
    ):
        return "possible"

    if any(marker in text for marker in ["probable", "likely", "вероят"]):
        return "probable"

    if any(
        marker in text
        for marker in [
            "present",
            "detected",
            "visualized",
            "identified",
            "выяв",
            "определ",
            "визуализ",
            "имеется",
            "есть",
        ]
    ):
        return "present"

    if any(
        marker in text
        for marker in [
            "absent",
            "not present",
            "no convincing",
            "no definite",
            "no signs",
            "не выяв",
            "нет",
            "без убедительных",
            "без достоверных",
        ]
    ):
        return "absent"

    return "absent"


def repair_common_model_errors(parsed: dict) -> dict:
    """
    Чинит частые ошибки LLM до Pydantic-валидации.
    """

    extracted_features = parsed.get("extracted_features", {})

    invasion = extracted_features.get("invasion", {})
    if isinstance(invasion, dict):
        invasion_type = invasion.get("type")

        if str(invasion_type).strip().lower() in CONFIDENCE_VALUES:
            invasion["confidence"] = normalize_confidence(invasion_type)
            invasion["type"] = "none"
        else:
            invasion["type"] = normalize_invasion_type(invasion_type)

        invasion["confidence"] = normalize_confidence(
            invasion.get("confidence")
        )

        extracted_features["invasion"] = invasion

    anatomy = extracted_features.get("anatomy", {})
    if isinstance(anatomy, dict):
        for field in [
            "bladder_involvement",
            "parametrium_involvement",
            "posterior_wall_involvement",
        ]:
            if field in anatomy:
                anatomy[field] = normalize_status(anatomy.get(field))
        extracted_features["anatomy"] = anatomy

    placenta_location = extracted_features.get("placenta_location", {})
    if isinstance(placenta_location, dict):
        for field in [
            "placenta_previa",
            "anterior_placenta",
        ]:
            if field in placenta_location:
                placenta_location[field] = normalize_status(
                    placenta_location.get(field)
                )
        extracted_features["placenta_location"] = placenta_location

    mri_signs = extracted_features.get("mri_signs", {})
    if isinstance(mri_signs, dict):
        for field in [
            "retroplacental_vessels",
            "lacunae",
            "uterine_wall_thinning",
            "uterine_hernia_or_bulging",
        ]:
            if field in mri_signs:
                mri_signs[field] = normalize_status(mri_signs.get(field))
        extracted_features["mri_signs"] = mri_signs

    clinical_context = extracted_features.get("clinical_context", {})
    if isinstance(clinical_context, dict):
        if "preoperative_bleeding" in clinical_context:
            clinical_context["preoperative_bleeding"] = normalize_status(
                clinical_context.get("preoperative_bleeding")
            )
        extracted_features["clinical_context"] = clinical_context

    parsed["extracted_features"] = extracted_features

    return parsed


def postprocess_extraction(parsed: dict) -> dict:
    if "features" in parsed:
        raise ValueError(
            "Model output uses legacy flat 'features' format. "
            "Expected canonical nested 'extracted_features'."
        )

    if "extracted_features" not in parsed:
        raise ValueError(
            "Model output is missing required canonical 'extracted_features' field."
        )

    parsed = repair_common_model_errors(parsed)

    return parsed


def extract_mri_features(
    mri_text: str,
    model_name: str | None = None,
    loaded_model: LoadedModel | None = None,
    print_raw_output: bool = False,
) -> MRIExtractionResult:
    if not mri_text or not mri_text.strip():
        raise ValueError("MRI text is empty")

    if loaded_model is None:
        loaded_model = load_llm(model_name)

    prompt = build_prompt(mri_text)
    raw_output = generate_text(loaded_model, prompt)

    parsed, _ = parse_json_with_retry(
        loaded_model,
        prompt,
        raw_output,
        print_raw_output=print_raw_output,
    )
    parsed = postprocess_extraction(parsed)

    return MRIExtractionResult.model_validate(parsed)


def extract_mri_features_with_raw(
    mri_text: str,
    model_name: str | None = None,
    loaded_model: LoadedModel | None = None,
    print_raw_output: bool = False,
) -> dict:
    if not mri_text or not mri_text.strip():
        raise ValueError("MRI text is empty")

    if loaded_model is None:
        loaded_model = load_llm(model_name)

    prompt = build_prompt(mri_text)
    raw_output = generate_text(loaded_model, prompt)

    parsed, final_raw_output = parse_json_with_retry(
        loaded_model,
        prompt,
        raw_output,
        print_raw_output=print_raw_output,
    )
    parsed = postprocess_extraction(parsed)

    validated = MRIExtractionResult.model_validate(parsed)

    return {
        "prompt": prompt,
        "raw_output": final_raw_output,
        "parsed": parsed,
        "validated": validated.model_dump(),
    }
