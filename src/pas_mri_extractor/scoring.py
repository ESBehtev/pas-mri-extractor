"""
Расчёт клинического score, риск-группы и рекомендаций.

На вход получает валидированный результат экстракции.
Основная логика пока перенесена из исходного ноутбука.
"""

from copy import deepcopy

from .schemas import (
    FullMRIResult,
    MRIExtractionResult,
    reject_legacy_features_payload,
)

from .config import load_config
score_cfg = load_config("risk_score.yaml")

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

    if invasion == "percreta":
        clinical_score += 5
        reasons.append("percreta: +5")
    elif invasion == "increta":
        clinical_score += 3
        reasons.append("increta: +3")
    elif invasion == "accreta":
        clinical_score += 1
        reasons.append("accreta: +1")

    if confidence == "definite" and invasion != "none":
        clinical_score += 2
        reasons.append("явное врастание: +2")
    elif confidence in ["probable", "possible"] and invasion != "none":
        clinical_score += 1
        reasons.append("вероятное/возможное врастание: +1")

    if bladder == "present":
        clinical_score += 3
        reasons.append("вовлечение мочевого пузыря: +3")
    elif bladder in ["possible", "probable"]:
        clinical_score += 2
        reasons.append("возможное вовлечение мочевого пузыря: +2")

    if parametrium == "present":
        clinical_score += 3
        reasons.append("вовлечение параметрия: +3")

    if thinning == "present":
        clinical_score += 1
        reasons.append("истончение миометрия/рубца: +1")

    if hernia == "present":
        clinical_score += 2
        reasons.append("выбухание/грыжа: +2")

    if vessels == "present":
        clinical_score += 1
        reasons.append("расширенные сосуды: +1")

    if lacunae == "present":
        clinical_score += 1
        reasons.append("лакуны: +1")

    if previa == "present":
        clinical_score += 1
        reasons.append("предлежание плаценты: +1")

    if anterior == "present":
        clinical_score += 1
        reasons.append("передняя плацента: +1")

    if bleeding == "present":
        clinical_score += 2
        reasons.append("кровотечение: +2")

    if isinstance(prev_cs, int) and prev_cs >= 2:
        clinical_score += 1
        reasons.append("≥2 КС: +1")

    red_flag = 0

    if bladder in ["present", "probable"]:
        red_flag = 1

    if invasion == "percreta" and bladder in ["possible", "probable", "present"]:
        red_flag = 1

    if invasion == "percreta":
        clinical_score = max(clinical_score, 8)

    if invasion == "percreta" and bladder in ["possible", "probable", "present"]:
        clinical_score = max(clinical_score, 9)

    if clinical_score <= 3:
        risk_group = "low"
    elif clinical_score <= 9:
        risk_group = "moderate"
    else:
        risk_group = "high"

    if invasion == "percreta":
        risk_group = "high"

    if invasion == "increta" and bladder == "absent":
        risk_group = "moderate"

    if invasion == "increta" and risk_group == "low":
        risk_group = "moderate"

    if bladder == "present" and invasion == "percreta":
        risk_group = "high"

    if risk_group == "low":
        blood_risk, vascular_risk, bladder_risk = 10, 10, 5
        blood_loss_range = "500–1000 мл"
    elif risk_group == "moderate":
        blood_risk, vascular_risk, bladder_risk = 35, 25, 15
        blood_loss_range = "1000–1500 мл"
    else:
        blood_risk, vascular_risk, bladder_risk = 70, 50, 30
        blood_loss_range = "1500–2500 мл"

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

    if risk_group == "low":
        readiness_level = "1"
        readiness_text = "Уровень 1: низкий риск, стандартная бригада, обычная подготовка."
    elif risk_group == "moderate":
        readiness_level = "2"
        readiness_text = "Уровень 2: умеренный риск, усиленная подготовка, запас компонентов крови, сосудистый хирург по вызову."
    else:
        readiness_level = "3"
        readiness_text = "Уровень 3: высокий риск, мультидисциплинарная команда, сосудистый хирург/уролог заранее, готовность к расширенной операции."

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
