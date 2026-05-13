"""
Rule-based fallback для извлечения признаков.

Используется как простой baseline или запасной вариант,
если LLM не вернула валидный JSON.
"""

import re
from typing import Any

from .config import load_config
from .schemas import MRIExtractionResult


def normalize_text(text: str, config: dict[str, Any]) -> str:
    text = str(text)

    normalization = config.get("normalization", {})

    if normalization.get("lowercase", True):
        text = text.lower()

    for old, new in normalization.get("replace", {}).items():
        text = text.replace(old, new)

    return text


def extract_numeric_features(text: str, config: dict[str, Any], features: dict[str, Any]) -> None:
    for feature_name, rule in config.get("regex_numeric", {}).items():
        match = re.search(rule["pattern"], text)

        if not match:
            continue

        value = match.group(rule.get("group", 1))

        if rule.get("type") == "int":
            value = int(value)

        features[feature_name] = value


def extract_regex_features(text: str, config: dict[str, Any], features: dict[str, Any]) -> None:
    for feature_name, status_rules in config.get("regex_features", {}).items():
        for status, pattern in status_rules.items():
            if re.search(pattern, text):
                features[feature_name] = status
                break


def extract_invasion_type(text: str, config: dict[str, Any], features: dict[str, Any]) -> None:
    matched_type = "none"
    matched_priority = 0

    for invasion_type, rule in config.get("invasion_type_rules", {}).items():
        priority = rule.get("priority", 0)

        for pattern in rule.get("patterns", []):
            if re.search(pattern, text) and priority > matched_priority:
                matched_type = invasion_type
                matched_priority = priority

    features["invasion_type"] = matched_type


def extract_invasion_confidence(text: str, config: dict[str, Any], features: dict[str, Any]) -> None:
    confidence_rules = config.get("invasion_confidence_rules", {})

    if features.get("invasion_type") == "none":
        features["invasion_confidence"] = confidence_rules.get("default_if_no_invasion", "absent")
        return

    for confidence in ["possible", "probable"]:
        for pattern in confidence_rules.get(confidence, []):
            if re.search(pattern, text):
                features["invasion_confidence"] = confidence
                return

    features["invasion_confidence"] = confidence_rules.get("default_if_invasion_present", "definite")


def build_short_explanation(config: dict[str, Any], features: dict[str, Any]) -> str:
    templates = config.get("short_explanation_templates", {})
    found = []

    if features.get("invasion_type") != "none":
        template = templates.get("invasion_type")
        if template:
            found.append(template.format(value=features["invasion_type"]))

    for feature_name in [
        "placenta_previa",
        "anterior_placenta",
        "uterine_wall_thinning",
        "lacunae",
        "retroplacental_vessels",
    ]:
        if features.get(feature_name) == "present" and feature_name in templates:
            found.append(templates[feature_name])

    if features.get("bladder_involvement") in ["possible", "present"]:
        template = templates.get("bladder_involvement")
        if template:
            found.append(template.format(value=features["bladder_involvement"]))

    return "; ".join(found) if found else config.get(
        "short_explanation_default",
        "Значимых признаков врастания по тексту не выделено.",
    )


def rule_extract_features(mri_text: str, rules_config_name: str = "rules.yaml") -> MRIExtractionResult:
    config = load_config(rules_config_name)
    text = normalize_text(mri_text, config)

    features = dict(config.get("default_features", {}))

    extract_regex_features(text, config, features)
    extract_numeric_features(text, config, features)
    extract_invasion_type(text, config, features)
    extract_invasion_confidence(text, config, features)

    features["short_explanation"] = build_short_explanation(config, features)

    result = {
        "features": features,
        "clinical_summary": features["short_explanation"],
        "clinical_rationale": "Результат получен rule-based методом по regex-правилам.",
    }

    return MRIExtractionResult.model_validate(result)