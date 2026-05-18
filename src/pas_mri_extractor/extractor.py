"""
Главная логика извлечения признаков из MRI-текста.

Собирает prompt, вызывает модель, парсит JSON и валидирует результат через Pydantic.
"""

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


def repair_common_model_errors(parsed: dict) -> dict:
    """
    Чинит частые ошибки LLM до Pydantic-валидации.

    Поддерживает оба возможных формата:
    1. старый плоский parsed["features"]
    2. новый вложенный parsed["extracted_features"]["invasion"]
    """

    features = parsed.get("features", {})

    invasion_type = features.get("invasion_type")

    if invasion_type in CONFIDENCE_VALUES:
        text_blob = " ".join(
            [
                str(features),
                str(parsed.get("clinical_summary", "")),
                str(parsed.get("clinical_rationale", "")),
            ]
        ).lower()

        features["invasion_confidence"] = invasion_type

        if "percreta" in text_blob:
            features["invasion_type"] = "percreta"
        elif "increta" in text_blob:
            features["invasion_type"] = "increta"
        elif "accreta" in text_blob:
            features["invasion_type"] = "accreta"
        else:
            features["invasion_type"] = "none"

    if "invasion_type" in features:
        features["invasion_type"] = normalize_invasion_type(
            features.get("invasion_type")
        )

    parsed["features"] = features

    extracted_features = parsed.get("extracted_features", {})
    invasion = extracted_features.get("invasion", {})

    if isinstance(invasion, dict):
        invasion_type = invasion.get("type")
        confidence = invasion.get("confidence")

        if str(invasion_type).lower() in CONFIDENCE_VALUES:
            invasion["confidence"] = str(invasion_type).lower()
            invasion["type"] = "none"
        else:
            invasion["type"] = normalize_invasion_type(invasion_type)

        if confidence is not None:
            confidence_text = str(confidence).strip().lower()
            if confidence_text not in CONFIDENCE_VALUES:
                invasion["confidence"] = "unclear"

        extracted_features["invasion"] = invasion
        parsed["extracted_features"] = extracted_features

    return parsed


def trim_text_fields(parsed: dict, max_chars: int = 350) -> dict:
    for field in [
        "clinical_summary",
        "clinical_rationale",
    ]:
        value = parsed.get(field)

        if isinstance(value, str) and len(value) > max_chars:
            parsed[field] = value[:max_chars].rstrip() + "..."

    features = parsed.get("features", {})
    explanation = features.get("short_explanation")

    if isinstance(explanation, str) and len(explanation) > max_chars:
        features["short_explanation"] = explanation[:max_chars].rstrip() + "..."

    parsed["features"] = features

    return parsed


def rebuild_short_explanation(parsed: dict) -> dict:
    features = parsed.get("features", {})

    found = []

    invasion_type = features.get("invasion_type")
    if invasion_type and invasion_type != "none":
        found.append(f"тип врастания: {invasion_type}")

    if features.get("invasion_confidence") not in [None, "absent"]:
        found.append(f"уверенность: {features['invasion_confidence']}")

    labels = {
        "bladder_involvement": "возможное вовлечение мочевого пузыря",
        "parametrium_involvement": "вовлечение параметрия",
        "posterior_wall_involvement": "вовлечение задней стенки",
        "placenta_previa": "предлежание плаценты",
        "anterior_placenta": "плацента по передней стенке",
        "retroplacental_vessels": "расширенные/ретроплацентарные сосуды",
        "lacunae": "плацентарные лакуны",
        "uterine_wall_thinning": "истончение миометрия/рубца",
        "uterine_hernia_or_bulging": "выбухание/грыжевидная деформация",
        "preoperative_bleeding": "кровотечение",
    }

    for key, label in labels.items():
        value = features.get(key)

        if value == "present":
            found.append(label)
        elif value in ["possible", "probable"]:
            found.append(f"{label}: {value}")

    if features.get("previous_cs_count") is not None:
        found.append(f"КС: {features['previous_cs_count']}")

    if features.get("gestational_week") is not None:
        found.append(f"{features['gestational_week']} недель")

    features["short_explanation"] = (
        "; ".join(found)
        if found
        else "Значимых признаков врастания по тексту не выделено."
    )

    parsed["features"] = features

    return parsed


def postprocess_extraction(parsed: dict) -> dict:
    parsed = repair_common_model_errors(parsed)
    parsed = rebuild_short_explanation(parsed)
    parsed = trim_text_fields(parsed)

    return parsed


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
    parsed = postprocess_extraction(parsed)

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
    parsed = postprocess_extraction(parsed)

    validated = MRIExtractionResult.model_validate(parsed)

    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed": parsed,
        "validated": validated.model_dump(),
    }