"""
Расчёт клинического score, риск-группы и рекомендаций.

На вход получает валидированный результат экстракции.
Основная логика пока перенесена из исходного ноутбука.
"""

from copy import deepcopy
from functools import lru_cache

from .schemas import (
    FullMRIResult,
    MRIExtractionResult,
    reject_legacy_features_payload,
)

from .config import load_config


@lru_cache(maxsize=1)
def get_score_config() -> dict:
    return load_config("risk_score.yaml")


def clear_score_config_cache() -> None:
    get_score_config.cache_clear()


def cfg_get(path: tuple[str, ...], default=None):
    current = get_score_config()

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return default if current is None else current


def cfg_weight(path: tuple[str, ...], default: int = 0) -> int:
    return int(cfg_get(path, default))


def resolve_risk_group(clinical_score: int) -> str:
    risk_groups = cfg_get(("risk_groups",), {})

    for group_name in ["low", "moderate", "high"]:
        group_cfg = risk_groups.get(group_name, {})
        min_score = group_cfg.get("min")
        max_score = group_cfg.get("max")

        if min_score is None or max_score is None:
            continue

        if int(min_score) <= clinical_score <= int(max_score):
            return group_name

    if clinical_score <= 3:
        return "low"

    if clinical_score <= 9:
        return "moderate"

    return "high"


def get_base_risk_prediction(risk_group: str) -> tuple[int, int, int, str]:
    prediction = cfg_get(("risk_predictions", risk_group), {})

    return (
        int(prediction.get("blood_loss_percent", 0)),
        int(prediction.get("vascular_percent", 0)),
        int(prediction.get("bladder_percent", 0)),
        str(prediction.get("blood_loss_range", "нет данных")),
    )


def get_readiness(risk_group: str) -> tuple[str, str]:
    readiness = cfg_get(("readiness_levels", risk_group), {})

    return (
        str(readiness.get("level", "")),
        str(readiness.get("text", "")),
    )


def normalize_mri_result(result: MRIExtractionResult | dict) -> FullMRIResult:
    if isinstance(result, MRIExtractionResult):
        result_dict = result.model_dump()
    else:
        result_dict = reject_legacy_features_payload(deepcopy(result))

    if "extracted_features" not in result_dict:
        raise ValueError(
            "Result is missing required canonical 'extracted_features' field."
        )

    case_info = result_dict.get("case_info", {})
    extracted = result_dict.get("extracted_features", {})

    invasion_block = extracted.get("invasion", {})
    anatomy = extracted.get("anatomy", {})
    placenta_location = extracted.get("placenta_location", {})
    mri_signs = extracted.get("mri_signs", {})
    clinical_context = extracted.get("clinical_context", {})

    invasion = invasion_block.get("type", "none")
    confidence = invasion_block.get("confidence", "absent")

    bladder = anatomy.get("bladder_involvement", "absent")
    parametrium = anatomy.get("parametrium_involvement", "absent")

    vessels = mri_signs.get("retroplacental_vessels", "absent")
    lacunae = mri_signs.get("lacunae", "absent")
    thinning = mri_signs.get("uterine_wall_thinning", "absent")
    hernia = mri_signs.get("uterine_hernia_or_bulging", "absent")

    previa = placenta_location.get("placenta_previa", "absent")
    anterior = placenta_location.get("anterior_placenta", "absent")

    bleeding = clinical_context.get("preoperative_bleeding", "absent")
    prev_cs = case_info.get("previous_cs_count", None)

    clinical_score = 0
    reasons = []

    invasion_score = cfg_weight(("scoring", "invasion_type", invasion), 0)
    if invasion_score > 0:
        clinical_score += invasion_score
        reasons.append(f"{invasion}: +{invasion_score}")

    if confidence == "definite" and invasion != "none":
        confidence_score = cfg_weight(
            ("scoring", "invasion_confidence", confidence),
            0,
        )
        clinical_score += confidence_score
        reasons.append(f"явное врастание: +{confidence_score}")
    elif confidence in ["probable", "possible"] and invasion != "none":
        confidence_score = cfg_weight(
            ("scoring", "invasion_confidence", confidence),
            0,
        )
        clinical_score += confidence_score
        reasons.append(f"вероятное/возможное врастание: +{confidence_score}")

    if bladder == "present":
        bladder_score = cfg_weight(
            ("scoring", "features", "bladder_involvement", bladder),
            0,
        )
        clinical_score += bladder_score
        reasons.append(f"вовлечение мочевого пузыря: +{bladder_score}")
    elif bladder in ["possible", "probable"]:
        bladder_score = cfg_weight(
            ("scoring", "features", "bladder_involvement", bladder),
            0,
        )
        clinical_score += bladder_score
        reasons.append(f"возможное вовлечение мочевого пузыря: +{bladder_score}")

    if parametrium == "present":
        parametrium_score = cfg_weight(
            ("scoring", "features", "parametrium_involvement", parametrium),
            0,
        )
        clinical_score += parametrium_score
        reasons.append(f"вовлечение параметрия: +{parametrium_score}")

    if thinning == "present":
        thinning_score = cfg_weight(
            ("scoring", "features", "uterine_wall_thinning", thinning),
            0,
        )
        clinical_score += thinning_score
        reasons.append(f"истончение миометрия/рубца: +{thinning_score}")

    if hernia == "present":
        hernia_score = cfg_weight(
            ("scoring", "features", "uterine_hernia_or_bulging", hernia),
            0,
        )
        clinical_score += hernia_score
        reasons.append(f"выбухание/грыжа: +{hernia_score}")

    if vessels == "present":
        vessels_score = cfg_weight(
            ("scoring", "features", "retroplacental_vessels", vessels),
            0,
        )
        clinical_score += vessels_score
        reasons.append(f"расширенные сосуды: +{vessels_score}")

    if lacunae == "present":
        lacunae_score = cfg_weight(
            ("scoring", "features", "lacunae", lacunae),
            0,
        )
        clinical_score += lacunae_score
        reasons.append(f"лакуны: +{lacunae_score}")

    if previa == "present":
        previa_score = cfg_weight(
            ("scoring", "features", "placenta_previa", previa),
            0,
        )
        clinical_score += previa_score
        reasons.append(f"предлежание плаценты: +{previa_score}")

    if anterior == "present":
        anterior_score = cfg_weight(
            ("scoring", "features", "anterior_placenta", anterior),
            0,
        )
        clinical_score += anterior_score
        reasons.append(f"передняя плацента: +{anterior_score}")

    if bleeding == "present":
        bleeding_score = cfg_weight(
            ("scoring", "features", "preoperative_bleeding", bleeding),
            0,
        )
        clinical_score += bleeding_score
        reasons.append(f"кровотечение: +{bleeding_score}")

    previous_cs_min_count = cfg_get(("scoring", "previous_cs_count", "min_count"), 2)
    if isinstance(prev_cs, int) and prev_cs >= int(previous_cs_min_count):
        previous_cs_score = cfg_weight(("scoring", "previous_cs_count", "score"), 0)
        clinical_score += previous_cs_score
        reasons.append(f"≥{previous_cs_min_count} КС: +{previous_cs_score}")

    red_flag = 0

    if bladder in ["present", "probable"]:
        red_flag = 1

    if invasion == "percreta" and bladder in ["possible", "probable", "present"]:
        red_flag = 1

    if invasion == "percreta":
        clinical_score = max(clinical_score, 8)

    if invasion == "percreta" and bladder in ["possible", "probable", "present"]:
        clinical_score = max(clinical_score, 9)

    risk_group = resolve_risk_group(clinical_score)

    if invasion == "percreta":
        risk_group = "high"

    if invasion == "increta" and bladder == "absent":
        risk_group = "moderate"

    if invasion == "increta" and risk_group == "low":
        risk_group = "moderate"

    if bladder == "present" and invasion == "percreta":
        risk_group = "high"

    blood_risk, vascular_risk, bladder_risk, blood_loss_range = (
        get_base_risk_prediction(risk_group)
    )

    if invasion == "percreta":
        blood_risk = max(blood_risk, 70)
        vascular_risk = max(vascular_risk, 50)
        blood_loss_range = "1500–3000 мл"

    if invasion == "increta" and bladder == "absent":
        blood_risk = min(max(blood_risk, 40), 50)
        vascular_risk = min(max(vascular_risk, 25), 35)
        bladder_risk = min(bladder_risk, 15)
        blood_loss_range = "1000–1500 мл"

    if bladder == "present":
        bladder_risk = max(bladder_risk, 40)
        blood_loss_range = "1500–3000 мл"
    elif bladder in ["possible", "probable"]:
        bladder_risk = max(bladder_risk, 30)

    if vessels == "present":
        blood_risk += 5
        vascular_risk += 5

    if lacunae == "present":
        blood_risk += 5

    if previa == "present":
        blood_risk += 5

    if anterior == "present":
        vascular_risk += 5

    if thinning == "present":
        blood_risk += 5

    if invasion == "increta" and bladder == "absent":
        blood_risk = min(blood_risk, 50)
        vascular_risk = min(vascular_risk, 40)
        bladder_risk = min(bladder_risk, 15)
        blood_loss_range = "1000–1500 мл"

    blood_risk = int(min(blood_risk, 85))
    vascular_risk = int(min(vascular_risk, 75))
    bladder_risk = int(min(bladder_risk, 65))

    readiness_level, readiness_text = get_readiness(risk_group)

    score_reasons = "; ".join(reasons) if reasons else "значимых признаков высокого риска не выявлено"

    result_dict["score"] = {
        "clinical_score": clinical_score,
        "risk_group": risk_group,
        "red_flag": red_flag,
        "score_reasons": score_reasons,
    }

    result_dict["predicted_risks"] = {
        "massive_blood_loss_over_1500_ml_percent": blood_risk,
        "estimated_blood_loss_ml_range": blood_loss_range,
        "vascular_intervention_percent": vascular_risk,
        "bladder_involvement_percent": bladder_risk,
        "risk_summary_text": (
            f"Риск массивной кровопотери >1500 мл: {blood_risk}%; "
            f"прогнозируемый объём кровопотери: {blood_loss_range}; "
            f"риск необходимости сосудистого вмешательства: {vascular_risk}%; "
            f"риск вовлечения мочевого пузыря: {bladder_risk}%"
        ),
    }

    result_dict["recommendation"] = {
        "readiness_level": readiness_level,
        "readiness_text": readiness_text,
    }

    result_dict["computed_rationale"] = (
        f"Уровень готовности выбран на основании признаков: {score_reasons}. "
        f"Оценочные риски: кровопотеря >1500 мл — {blood_risk}%, "
        f"прогнозируемый объём кровопотери — {blood_loss_range}, "
        f"сосудистое вмешательство — {vascular_risk}%, "
        f"вовлечение мочевого пузыря — {bladder_risk}%."
    )

    return FullMRIResult.model_validate(result_dict)
