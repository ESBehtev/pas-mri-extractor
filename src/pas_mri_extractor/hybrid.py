"""
Hybrid merge для LLM-результата и rule-based извлечения.

Идея:
- LLM остаётся основным источником интерпретации.
- Regex rules уточняют простые фактические признаки.
- Все изменения фиксируются в audit.
"""

from copy import deepcopy
from typing import Any

from .rules import rule_extract_features


MERGE_FEATURE_PATHS = [
    (
        "extracted_features",
        "placenta_location",
        "placenta_previa",
    ),
    (
        "extracted_features",
        "placenta_location",
        "anterior_placenta",
    ),
    (
        "extracted_features",
        "mri_signs",
        "retroplacental_vessels",
    ),
    (
        "extracted_features",
        "mri_signs",
        "lacunae",
    ),
    (
        "extracted_features",
        "mri_signs",
        "uterine_wall_thinning",
    ),
    (
        "extracted_features",
        "mri_signs",
        "uterine_hernia_or_bulging",
    ),
    (
        "extracted_features",
        "clinical_context",
        "preoperative_bleeding",
    ),
]

CASE_INFO_PATHS = [
    (
        "case_info",
        "previous_cs_count",
    ),
    (
        "case_info",
        "gestational_week",
    ),
]

VALUE_RANK = {
    "absent": 0,
    "possible": 1,
    "probable": 2,
    "present": 3,
}


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current = data

    for key in path:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def set_nested(
    data: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current = data

    for key in path[:-1]:
        current = current.setdefault(key, {})

    current[path[-1]] = value


def should_override_feature(
    llm_value: str,
    rule_value: str,
) -> bool:
    llm_rank = VALUE_RANK.get(llm_value, 0)
    rule_rank = VALUE_RANK.get(rule_value, 0)

    return rule_rank > llm_rank


def merge_evidence(
    merged: dict[str, Any],
    rule_result: dict[str, Any],
) -> None:
    evidence = merged.setdefault("evidence", {})
    rule_evidence = rule_result.get("evidence", {})

    for key in [
        "positive_findings",
        "uncertain_findings",
        "negative_findings",
    ]:
        current_values = evidence.setdefault(key, [])
        rule_values = rule_evidence.get(key, [])

        for value in rule_values:
            if value not in current_values:
                current_values.append(value)


def merge_with_rules(
    llm_result: dict[str, Any],
    mri_text: str,
) -> dict[str, Any]:
    merged = deepcopy(llm_result)

    rule_result = rule_extract_features(mri_text)

    if hasattr(rule_result, "model_dump"):
        rule_result = rule_result.model_dump()

    audit = {
        "rules_applied": True,
        "overrides": [],
        "conflicts": [],
    }

    for path in MERGE_FEATURE_PATHS:
        rule_value = get_nested(rule_result, path) or "absent"
        llm_value = get_nested(merged, path) or "absent"

        if should_override_feature(llm_value, rule_value):
            set_nested(merged, path, rule_value)

            audit["overrides"].append(
                {
                    "field": ".".join(path),
                    "llm_value": llm_value,
                    "rule_value": rule_value,
                    "final_value": rule_value,
                }
            )

        elif llm_value != rule_value and rule_value != "absent":
            audit["conflicts"].append(
                {
                    "field": ".".join(path),
                    "llm_value": llm_value,
                    "rule_value": rule_value,
                    "final_value": llm_value,
                }
            )

    for path in CASE_INFO_PATHS:
        rule_value = get_nested(rule_result, path)
        llm_value = get_nested(merged, path)

        if llm_value is None and rule_value is not None:
            set_nested(merged, path, rule_value)

            audit["overrides"].append(
                {
                    "field": ".".join(path),
                    "llm_value": llm_value,
                    "rule_value": rule_value,
                    "final_value": rule_value,
                }
            )

    merge_evidence(merged, rule_result)

    merged["audit"] = audit

    return merged