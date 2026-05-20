"""
Rule-based fallback для извлечения признаков.

Используется как простой baseline или источник простых фактических признаков
для hybrid merge с LLM-результатом.
"""

import re
from typing import Any

from .config import load_config
from .schemas import MRIExtractionResult


NEGATION_WINDOW_CHARS = 60

NEGATION_PATTERNS = [
    r"\bнет\b",
    r"\bне\s+выяв\w*",
    r"\bне\s+определя\w*",
    r"\bне\s+получен\w*",
    r"\bбез\s+признак\w*",
    r"\bбез\s+убедительн\w*",
    r"\bбез\s+достоверн\w*",
    r"\bданн\w*\s+за\b.{0,80}\bнет\b",
    r"\bпризнак\w*\b.{0,80}\bнет\b",
    r"(?<!не\s)\bисключается\b",
    r"(?<!не\s)\bисключен[аыо]?\b",
]

UNCERTAIN_PATTERNS = [
    r"\bнельзя\s+исключ",
    r"\bне\s+исключ",
    r"\bневозможно\s+исключ",
    r"\bсомнитель",
    r"\bподозр",
    r"\bвозможн",
    r"\bвероят",
]

PREVIOUS_CS_COUNT_VALUES = {
    "1": 1,
    "один": 1,
    "одна": 1,
    "одного": 1,
    "одной": 1,
    "2": 2,
    "две": 2,
    "два": 2,
    "двух": 2,
    "3": 3,
    "три": 3,
    "трех": 3,
}

PREVIOUS_CS_COUNT_TOKEN = (
    r"(?P<count>1|2|3|один|одна|одного|одной|две|два|двух|три|трех)"
)

PREVIOUS_CS_COUNT_PATTERNS = [
    rf"\bпосле\s+{PREVIOUS_CS_COUNT_TOKEN}\s*(?:кс|кесарев\w*\s+сечен\w*)",
    rf"\bанамнезе\s+{PREVIOUS_CS_COUNT_TOKEN}\s*(?:кс|кесарев\w*\s+сечен\w*)",
    rf"\bрубец\w*\s+на\s+матк\w*.{{0,40}}\bпосле\s+{PREVIOUS_CS_COUNT_TOKEN}\s*(?:кс|кесарев\w*\s+сечен\w*)",
    rf"\b{PREVIOUS_CS_COUNT_TOKEN}\s*(?:кс|кесарев\w*\s+сечен\w*)",
]

NEGATIVE_EVIDENCE_LABELS = {
    "bladder_involvement": "отрицательный контекст: вовлечение мочевого пузыря",
    "parametrium_involvement": "отрицательный контекст: вовлечение параметрия",
    "posterior_wall_involvement": "отрицательный контекст: вовлечение задней стенки",
    "placenta_previa": "отрицательный контекст: предлежание плаценты",
    "retroplacental_vessels": "отрицательный контекст: ретроплацентарные сосуды",
    "lacunae": "отрицательный контекст: плацентарные лакуны",
    "uterine_wall_thinning": "отрицательный контекст: истончение миометрия/рубца",
    "uterine_hernia_or_bulging": "отрицательный контекст: выбухание стенки матки",
    "preoperative_bleeding": "отрицательный контекст: кровотечение",
    "invasion_type": "отрицательный контекст: врастание плаценты",
}


def normalize_text(text: str, config: dict[str, Any]) -> str:
    text = str(text)

    normalization = config.get("normalization", {})

    if normalization.get("lowercase", True):
        text = text.lower()

    for old, new in normalization.get("replace", {}).items():
        text = text.replace(old, new)

    return text


def get_match_context(
    text: str,
    match: re.Match,
    window_chars: int = NEGATION_WINDOW_CHARS,
) -> str:
    start = max(0, match.start() - window_chars)
    end = min(len(text), match.end() + window_chars)

    return text[start:end]


def get_scoped_match_context(
    text: str,
    match: re.Match,
    window_chars: int = NEGATION_WINDOW_CHARS,
) -> str:
    start = max(0, match.start() - window_chars)
    end = min(len(text), match.end() + window_chars)

    boundary_chars = ".;\n"
    scoped_start = start
    scoped_end = end

    for boundary in boundary_chars:
        before_boundary = text.rfind(boundary, start, match.start())
        if before_boundary >= scoped_start:
            scoped_start = before_boundary + 1

        after_boundary = text.find(boundary, match.end(), end)
        if after_boundary != -1:
            scoped_end = min(scoped_end, after_boundary + 1)

    return text[scoped_start:scoped_end]


def is_uncertain_context(context: str) -> bool:
    return any(re.search(pattern, context) for pattern in UNCERTAIN_PATTERNS)


def has_negation_context(context: str) -> bool:
    return any(re.search(pattern, context) for pattern in NEGATION_PATTERNS)


def is_negated_match(text: str, match: re.Match) -> bool:
    context = get_scoped_match_context(text, match)

    return has_negation_context(context)


def add_negative_evidence(features: dict[str, Any], feature_name: str) -> None:
    label = NEGATIVE_EVIDENCE_LABELS.get(feature_name)
    if not label:
        return

    negative_findings = features.setdefault("_negative_findings", [])
    if label not in negative_findings:
        negative_findings.append(label)


def parse_previous_cs_count(text: str) -> int | None:
    for pattern in PREVIOUS_CS_COUNT_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue

        return PREVIOUS_CS_COUNT_VALUES.get(match.group("count"))

    return None


def extract_numeric_features(
    text: str,
    config: dict[str, Any],
    features: dict[str, Any],
) -> None:
    for feature_name, rule in config.get("regex_numeric", {}).items():
        match = re.search(rule["pattern"], text)

        if not match:
            continue

        value = match.group(rule.get("group", 1))

        if rule.get("type") == "int":
            value = int(value)

        features[feature_name] = value

    previous_cs_count = parse_previous_cs_count(text)
    if previous_cs_count is not None:
        features["previous_cs_count"] = previous_cs_count


def extract_regex_features(
    text: str,
    config: dict[str, Any],
    features: dict[str, Any],
) -> None:
    for feature_name, status_rules in config.get("regex_features", {}).items():
        for status, pattern in status_rules.items():
            matches = list(re.finditer(pattern, text))
            negated_matches = []
            uncertain_matches = []
            positive_matches = []

            for match in matches:
                if is_negated_match(text, match):
                    negated_matches.append(match)
                elif status == "present" and is_uncertain_context(
                    get_match_context(text, match)
                ):
                    uncertain_matches.append(match)
                else:
                    positive_matches.append(match)

            if negated_matches and not positive_matches:
                add_negative_evidence(features, feature_name)
                continue

            if uncertain_matches and not positive_matches:
                features[feature_name] = "possible"
                break

            if positive_matches:
                features[feature_name] = status
                break


def extract_invasion_type(
    text: str,
    config: dict[str, Any],
    features: dict[str, Any],
) -> None:
    matched_type = "none"
    matched_priority = 0

    for invasion_type, rule in config.get("invasion_type_rules", {}).items():
        priority = rule.get("priority", 0)

        for pattern in rule.get("patterns", []):
            matches = list(re.finditer(pattern, text))
            negated_matches = [
                match for match in matches if is_negated_match(text, match)
            ]
            positive_matches = [
                match for match in matches if not is_negated_match(text, match)
            ]

            if negated_matches and not positive_matches:
                add_negative_evidence(features, "invasion_type")
                continue

            if positive_matches and priority > matched_priority:
                matched_type = invasion_type
                matched_priority = priority

    features["invasion_type"] = matched_type


def extract_invasion_confidence(
    text: str,
    config: dict[str, Any],
    features: dict[str, Any],
) -> None:
    confidence_rules = config.get("invasion_confidence_rules", {})

    if features.get("invasion_type") == "none":
        features["invasion_confidence"] = confidence_rules.get(
            "default_if_no_invasion",
            "absent",
        )
        return

    for confidence in ["possible", "probable"]:
        for pattern in confidence_rules.get(confidence, []):
            if re.search(pattern, text):
                features["invasion_confidence"] = confidence
                return

    features["invasion_confidence"] = confidence_rules.get(
        "default_if_invasion_present",
        "definite",
    )


def build_evidence(features: dict[str, Any]) -> dict[str, list[str]]:
    positive = []
    uncertain = []
    negative = list(features.get("_negative_findings", []))

    positive_labels = {
        "placenta_previa": "предлежание/перекрытие внутреннего зева",
        "anterior_placenta": "плацента по передней стенке",
        "retroplacental_vessels": "расширенные/ретроплацентарные сосуды",
        "lacunae": "плацентарные лакуны",
        "uterine_wall_thinning": "истончение миометрия/рубца",
        "uterine_hernia_or_bulging": "выбухание/грыжевидная деформация",
        "preoperative_bleeding": "кровотечение",
        "parametrium_involvement": "вовлечение параметрия",
        "posterior_wall_involvement": "вовлечение задней стенки",
    }

    if features.get("invasion_type") != "none":
        positive.append(f"тип врастания: {features['invasion_type']}")

    for key, label in positive_labels.items():
        value = features.get(key)

        if value == "present":
            positive.append(label)
        elif value in ["possible", "probable"]:
            uncertain.append(label)

    bladder = features.get("bladder_involvement")

    if bladder == "present":
        positive.append("вовлечение мочевого пузыря")
    elif bladder in ["possible", "probable"]:
        uncertain.append("возможное вовлечение мочевого пузыря")

    return {
        "positive_findings": positive,
        "uncertain_findings": uncertain,
        "negative_findings": negative,
    }


def old_features_to_new_result(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_info": {
            "gestational_week": features.get("gestational_week"),
            "previous_cs_count": features.get("previous_cs_count"),
        },
        "extracted_features": {
            "invasion": {
                "type": features.get("invasion_type", "none"),
                "confidence": features.get("invasion_confidence", "absent"),
            },
            "anatomy": {
                "bladder_involvement": features.get(
                    "bladder_involvement",
                    "absent",
                ),
                "parametrium_involvement": features.get(
                    "parametrium_involvement",
                    "absent",
                ),
                "posterior_wall_involvement": features.get(
                    "posterior_wall_involvement",
                    "absent",
                ),
            },
            "placenta_location": {
                "placenta_previa": features.get(
                    "placenta_previa",
                    "absent",
                ),
                "anterior_placenta": features.get(
                    "anterior_placenta",
                    "absent",
                ),
            },
            "mri_signs": {
                "retroplacental_vessels": features.get(
                    "retroplacental_vessels",
                    "absent",
                ),
                "lacunae": features.get("lacunae", "absent"),
                "uterine_wall_thinning": features.get(
                    "uterine_wall_thinning",
                    "absent",
                ),
                "uterine_hernia_or_bulging": features.get(
                    "uterine_hernia_or_bulging",
                    "absent",
                ),
            },
            "clinical_context": {
                "preoperative_bleeding": features.get(
                    "preoperative_bleeding",
                    "absent",
                ),
            },
        },
        "evidence": build_evidence(features),
    }


def rule_extract_features(
    mri_text: str,
    rules_config_name: str = "rules.yaml",
) -> MRIExtractionResult:
    config = load_config(rules_config_name)
    text = normalize_text(mri_text, config)

    features = dict(config.get("default_features", {}))

    extract_regex_features(text, config, features)
    extract_numeric_features(text, config, features)
    extract_invasion_type(text, config, features)
    extract_invasion_confidence(text, config, features)

    result = old_features_to_new_result(features)

    return MRIExtractionResult.model_validate(result)
