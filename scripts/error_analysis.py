"""Анализ ошибок LLM risk prediction по результатам benchmark JSONL.
Зачем нужен:
- считать ошибки кровопотери и readiness;
- искать признаки и паттерны завышения/занижения риска;
- писать machine-readable JSON и компактный Markdown отчёт.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "llm_risk_eval_20_v2.jsonl"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "outputs" / "error_analysis.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "outputs" / "error_analysis.md"

FEATURE_NAMES = [
    "placenta_previa",
    "anterior_placenta",
    "retroplacental_vessels",
    "lacunae",
    "uterine_wall_thinning",
    "uterine_hernia_or_bulging",
    "bladder_involvement",
    "percreta_suspicion",
    "previous_cs_count",
    "invasion.type",
]
FREQUENT_DRIVER_FEATURE_PREFIXES = {
    "placenta_previa",
    "anterior_placenta",
}
ERROR_GROUPS = ["overestimation", "neutral", "underestimation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM PAS risk errors.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL.")
    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_JSON_OUTPUT),
        help="Output JSON report.",
    )
    parser.add_argument(
        "--md-output",
        default=str(DEFAULT_MD_OUTPUT),
        help="Output Markdown report.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}")
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(record)
    return records


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def nested_get(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def rule_result(record: dict[str, Any]) -> dict[str, Any]:
    rule_based = record.get("rule_based")
    if not isinstance(rule_based, dict):
        return {}

    result = rule_based.get("result")
    if isinstance(result, dict):
        return result

    case_context = rule_based.get("case_context")
    extracted_result = (
        case_context.get("extracted_result")
        if isinstance(case_context, dict)
        else None
    )
    if isinstance(extracted_result, dict):
        return extracted_result

    return rule_based


def extract_rule_features(record: dict[str, Any]) -> dict[str, Any]:
    result = rule_result(record)
    extracted = result.get("extracted_features") or {}
    placenta = extracted.get("placenta_location") or {}
    signs = extracted.get("mri_signs") or {}
    anatomy = extracted.get("anatomy") or {}
    invasion = extracted.get("invasion") or {}
    suspicion = result.get("suspicion") or {}
    case_info = result.get("case_info") or {}

    return {
        "placenta_previa": placenta.get("placenta_previa"),
        "anterior_placenta": placenta.get("anterior_placenta"),
        "retroplacental_vessels": signs.get("retroplacental_vessels"),
        "lacunae": signs.get("lacunae"),
        "uterine_wall_thinning": signs.get("uterine_wall_thinning"),
        "uterine_hernia_or_bulging": signs.get("uterine_hernia_or_bulging"),
        "bladder_involvement": anatomy.get("bladder_involvement"),
        "percreta_suspicion": suspicion.get("percreta_suspicion"),
        "previous_cs_count": case_info.get("previous_cs_count"),
        "invasion.type": invasion.get("type"),
    }


def feature_tokens(features: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for name in FEATURE_NAMES:
        value = features.get(name)
        if value is None:
            continue
        if name == "previous_cs_count":
            count = as_int(value)
            if count is None:
                continue
            tokens.append("previous_cs_count>=2" if count >= 2 else f"{name}={count}")
            continue
        if value in {"absent", "none", ""}:
            continue
        tokens.append(f"{name}={value}")
    return sorted(tokens)


def analyze_case(record: dict[str, Any]) -> dict[str, Any]:
    actual = record.get("actual") or {}
    llm_risk = record.get("llm_risk") or {}
    risk_assessment = llm_risk.get("risk_assessment") or {}
    readiness = llm_risk.get("readiness") or {}

    actual_blood_loss = as_int(actual.get("blood_loss_ml"))
    predicted_blood_loss = as_int(risk_assessment.get("estimated_blood_loss_ml"))
    blood_loss_error = (
        predicted_blood_loss - actual_blood_loss
        if predicted_blood_loss is not None and actual_blood_loss is not None
        else None
    )
    abs_blood_loss_error = (
        abs(blood_loss_error) if blood_loss_error is not None else None
    )

    actual_readiness = as_int(actual.get("readiness_level"))
    predicted_readiness = as_int(readiness.get("level"))
    readiness_error = (
        predicted_readiness - actual_readiness
        if predicted_readiness is not None and actual_readiness is not None
        else None
    )

    features = extract_rule_features(record)
    category = None
    error_direction = None
    if abs_blood_loss_error is not None:
        category = (
            "good_predictions"
            if abs_blood_loss_error <= 500
            else "bad_predictions"
        )
        if blood_loss_error > 500:
            error_direction = "overestimation"
        elif blood_loss_error < -500:
            error_direction = "underestimation"
        else:
            error_direction = "neutral"

    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "actual_blood_loss_ml": actual_blood_loss,
        "predicted_blood_loss_ml": predicted_blood_loss,
        "blood_loss_error": blood_loss_error,
        "abs_blood_loss_error": abs_blood_loss_error,
        "actual_readiness": actual_readiness,
        "predicted_readiness": predicted_readiness,
        "readiness_error": readiness_error,
        "prediction_group": category,
        "error_direction": error_direction,
        "features": features,
        "feature_tokens": feature_tokens(features),
    }


def summarize_feature(feature: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    actual_values = [
        case["actual_blood_loss_ml"]
        for case in cases
        if case.get("actual_blood_loss_ml") is not None
    ]
    predicted_values = [
        case["predicted_blood_loss_ml"]
        for case in cases
        if case.get("predicted_blood_loss_ml") is not None
    ]
    errors = [
        case["blood_loss_error"]
        for case in cases
        if case.get("blood_loss_error") is not None
    ]
    absolute_errors = [
        case["abs_blood_loss_error"]
        for case in cases
        if case.get("abs_blood_loss_error") is not None
    ]

    return {
        "feature": feature,
        "n_cases": len(cases),
        "mean_actual_blood_loss": mean(actual_values),
        "mean_predicted_blood_loss": mean(predicted_values),
        "mean_error": mean(errors),
        "mean_absolute_error": mean(absolute_errors),
    }


def summarize_features(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        for token in case.get("feature_tokens") or []:
            grouped[token].append(case)

    return sorted(
        [summarize_feature(feature, items) for feature, items in grouped.items()],
        key=lambda item: (-item["n_cases"], item["feature"]),
    )


def top_patterns(
    cases: list[dict[str, Any]],
    direction: str,
    min_abs_error: int = 500,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if direction == "over":
        selected = [
            case for case in cases if (case.get("blood_loss_error") or 0) > min_abs_error
        ]
    elif direction == "under":
        selected = [
            case
            for case in cases
            if (case.get("blood_loss_error") or 0) < -min_abs_error
        ]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in selected:
        tokens = case.get("feature_tokens") or []
        for pattern in itertools.combinations(tokens, 2):
            grouped[pattern].append(case)

    patterns = []
    for pattern, pattern_cases in grouped.items():
        errors = [
            case["blood_loss_error"]
            for case in pattern_cases
            if case.get("blood_loss_error") is not None
        ]
        patterns.append(
            {
                "pattern": " + ".join(pattern),
                "features": list(pattern),
                "n": len(pattern_cases),
                "mean_error": mean(errors),
            }
        )

    return sorted(
        patterns,
        key=lambda item: (-item["n"], -abs(item["mean_error"] or 0), item["pattern"]),
    )[:limit]


def feature_base_name(token: str) -> str:
    return token.split("=", 1)[0]


def is_frequent_driver_feature(token: str, overall_frequency: float) -> bool:
    return (
        feature_base_name(token) in FREQUENT_DRIVER_FEATURE_PREFIXES
        and overall_frequency > 0.8
    )


def discriminative_feature_analysis(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable = [case for case in cases if case.get("error_direction") in ERROR_GROUPS]
    total = len(evaluable)
    grouped_cases = {
        group: [case for case in evaluable if case.get("error_direction") == group]
        for group in ERROR_GROUPS
    }
    feature_set = sorted(
        {
            token
            for case in evaluable
            for token in (case.get("feature_tokens") or [])
        }
    )

    features: list[dict[str, Any]] = []
    for token in feature_set:
        overall_count = sum(
            1 for case in evaluable if token in (case.get("feature_tokens") or [])
        )
        overall_frequency = overall_count / total if total else 0
        item: dict[str, Any] = {
            "feature": token,
            "n_cases": overall_count,
            "overall_frequency": overall_frequency,
            "ignored_as_driver": is_frequent_driver_feature(
                token,
                overall_frequency,
            ),
        }

        for group in ERROR_GROUPS:
            group_cases = grouped_cases[group]
            group_count = sum(
                1
                for case in group_cases
                if token in (case.get("feature_tokens") or [])
            )
            group_frequency = group_count / len(group_cases) if group_cases else 0
            item[f"presence_in_{group}"] = group_count
            item[f"frequency_in_{group}"] = group_frequency
            item[f"enrichment_in_{group}"] = (
                group_frequency / overall_frequency if overall_frequency else None
            )

        features.append(item)

    driver_candidates = [
        item for item in features if not item["ignored_as_driver"]
    ]

    return {
        "group_counts": {
            group: len(grouped_cases[group])
            for group in ERROR_GROUPS
        },
        "features": sorted(
            features,
            key=lambda item: (
                item["ignored_as_driver"],
                -item["overall_frequency"],
                item["feature"],
            ),
        ),
        "top_enriched_overestimation_drivers": top_enriched(
            driver_candidates,
            "overestimation",
        ),
        "top_enriched_underestimation_drivers": top_enriched(
            driver_candidates,
            "underestimation",
        ),
    }


def top_enriched(
    features: list[dict[str, Any]],
    group: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    enrichment_key = f"enrichment_in_{group}"
    presence_key = f"presence_in_{group}"
    return sorted(
        [
            item
            for item in features
            if item.get(enrichment_key) is not None and item.get(presence_key, 0) > 0
        ],
        key=lambda item: (
            -item[enrichment_key],
            -item[presence_key],
            item["feature"],
        ),
    )[:limit]


def blood_loss_subset_summary(
    cases: list[dict[str, Any]],
    predicate,
) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case.get("actual_blood_loss_ml") is not None and predicate(case)
    ]
    predictions = [
        case["predicted_blood_loss_ml"]
        for case in selected
        if case.get("predicted_blood_loss_ml") is not None
    ]
    errors = [
        case["blood_loss_error"]
        for case in selected
        if case.get("blood_loss_error") is not None
    ]
    return {
        "n_cases": len(selected),
        "mean_prediction": mean(predictions),
        "mean_error": mean(errors),
    }


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [analyze_case(record) for record in records]
    evaluable = [
        case for case in cases if case.get("abs_blood_loss_error") is not None
    ]
    readiness_cases = [
        case for case in cases if case.get("readiness_error") is not None
    ]
    exact_readiness = sum(
        1 for case in readiness_cases if case.get("readiness_error") == 0
    )
    absolute_errors = [case["abs_blood_loss_error"] for case in evaluable]

    feature_summary = summarize_features(evaluable)
    over_patterns = top_patterns(evaluable, "over")
    under_patterns = top_patterns(evaluable, "under")
    discriminative = discriminative_feature_analysis(evaluable)
    severe_cases = blood_loss_subset_summary(
        evaluable,
        lambda case: case["actual_blood_loss_ml"] >= 2500,
    )
    low_risk_cases = blood_loss_subset_summary(
        evaluable,
        lambda case: case["actual_blood_loss_ml"] <= 1000,
    )

    return {
        "summary": {
            "n_cases": len(records),
            "n_evaluable_blood_loss": len(evaluable),
            "blood_loss_mae_ml": mean(absolute_errors),
            "readiness_exact_match": (
                exact_readiness / len(readiness_cases) if readiness_cases else None
            ),
            "readiness_exact_match_n": len(readiness_cases),
            "n_good_predictions": sum(
                1 for case in evaluable if case["prediction_group"] == "good_predictions"
            ),
            "n_bad_predictions": sum(
                1 for case in evaluable if case["prediction_group"] == "bad_predictions"
            ),
        },
        "cases": cases,
        "feature_summary": feature_summary,
        "discriminative_analysis": discriminative,
        "severe_cases": severe_cases,
        "low_risk_cases": low_risk_cases,
        "top_overestimation_patterns": over_patterns,
        "top_underestimation_patterns": under_patterns,
        "recommendations": build_recommendations(
            feature_summary,
            discriminative,
        ),
    }


def format_ml(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} мл" if isinstance(value, float) else f"{value} мл"


def build_recommendations(
    feature_summary: list[dict[str, Any]],
    discriminative: dict[str, Any],
) -> list[str]:
    over = [
        item["feature"]
        for item in discriminative.get("top_enriched_overestimation_drivers", [])[:5]
    ]
    under = [
        item["feature"]
        for item in discriminative.get("top_enriched_underestimation_drivers", [])[:5]
    ]

    recommendations: list[str] = []
    if over:
        features = ", ".join(over)
        recommendations.append(
            f"Check whether these features overweight risk: {features}."
        )
    if under:
        features = ", ".join(under)
        recommendations.append(
            f"Check whether these features are underestimated: {features}."
        )
    if not recommendations:
        recommendations.append(
            "No single feature group crossed the +/-500 ml mean error threshold."
        )
    return recommendations


def markdown_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    lines = [
        "# Error analysis",
        "",
        f"{summary['n_cases']} cases",
        "",
        f"Blood loss MAE: {format_ml(summary['blood_loss_mae_ml'])}",
        f"Readiness exact match: {format_percent(summary['readiness_exact_match'])}",
        f"Good predictions: {summary['n_good_predictions']}",
        f"Bad predictions: {summary['n_bad_predictions']}",
        "",
        "## Most common overestimation patterns",
        "",
    ]

    lines.extend(pattern_lines(analysis["top_overestimation_patterns"]))
    lines.extend(["", "## Most common underestimation patterns", ""])
    lines.extend(pattern_lines(analysis["top_underestimation_patterns"]))
    lines.extend(["", "## Top enriched overestimation drivers", ""])
    lines.extend(
        enriched_lines(
            analysis["discriminative_analysis"][
                "top_enriched_overestimation_drivers"
            ],
            "overestimation",
        )
    )
    lines.extend(["", "## Top enriched underestimation drivers", ""])
    lines.extend(
        enriched_lines(
            analysis["discriminative_analysis"][
                "top_enriched_underestimation_drivers"
            ],
            "underestimation",
        )
    )
    lines.extend(["", "## Severe cases", ""])
    lines.extend(subset_lines(analysis["severe_cases"]))
    lines.extend(["", "## Low-risk cases", ""])
    lines.extend(subset_lines(analysis["low_risk_cases"]))
    lines.extend(["", "## Feature summary", ""])

    for item in analysis["feature_summary"][:20]:
        lines.append(
            f"- {item['feature']}: n={item['n_cases']}, "
            f"mean_error={format_signed_ml(item['mean_error'])}, "
            f"MAE={format_ml(item['mean_absolute_error'])}"
        )

    lines.extend(["", "## Recommendations", ""])
    for recommendation in analysis["recommendations"]:
        lines.append(f"- {recommendation}")

    return "\n".join(lines) + "\n"


def enriched_lines(items: list[dict[str, Any]], group: str) -> list[str]:
    if not items:
        return ["- none"]
    return [
        f"- {item['feature']}: presence={item[f'presence_in_{group}']}, "
        f"enrichment={item[f'enrichment_in_{group}']:.2f}, "
        f"overall_frequency={item['overall_frequency']:.2f}"
        for item in items
    ]


def subset_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"- n={summary['n_cases']}",
        f"- mean prediction={format_ml(summary['mean_prediction'])}",
        f"- mean error={format_signed_ml(summary['mean_error'])}",
    ]


def pattern_lines(patterns: list[dict[str, Any]]) -> list[str]:
    if not patterns:
        return ["- none"]
    return [
        f"- {item['pattern']}: n={item['n']}, "
        f"mean_error={format_signed_ml(item['mean_error'])}"
        for item in patterns
    ]


def format_signed_ml(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.0f} мл"


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    json_output_path = resolve_path(args.json_output)
    md_output_path = resolve_path(args.md_output)

    records = load_jsonl(input_path)
    analysis = analyze_records(records)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)

    with json_output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    md_output_path.write_text(markdown_report(analysis), encoding="utf-8")

    print(markdown_report(analysis))


if __name__ == "__main__":
    main()
