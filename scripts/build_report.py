"""Build a Markdown report from benchmark metric artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_utils import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    HIGH_IMPACT_FIELDS,
    build_mri_text,
    index_by_case_id,
    load_records,
    normalize_gold,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PAS benchmark Markdown report.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL/CSV.")
    parser.add_argument("--cases", required=True, help="Original benchmark cases JSON/JSONL/CSV.")
    parser.add_argument("--baseline", required=True, help="Normalized baseline predictions.")
    parser.add_argument("--full", required=True, help="Normalized full predictions.")
    parser.add_argument(
        "--description-only",
        required=True,
        help="Normalized description-only predictions.",
    )
    parser.add_argument("--full-metrics", required=True, help="Full-vs-gold metrics JSON.")
    parser.add_argument(
        "--description-only-metrics",
        required=True,
        help="Description-only-vs-gold metrics JSON.",
    )
    parser.add_argument("--baseline-metrics", required=True, help="Baseline-vs-gold metrics JSON.")
    parser.add_argument("--mode-comparison", required=True, help="Full-vs-desc comparison JSON.")
    parser.add_argument(
        "--baseline-comparison",
        required=True,
        help="Baseline/full/desc comparison JSON.",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmark_report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def load_json(path_value: str) -> dict[str, Any]:
    with resolve_path(path_value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def model_summary_row(name: str, metrics: dict[str, Any]) -> str:
    fields = metrics.get("fields", {})
    values = [
        item.get("f1", item.get("accuracy", 0.0))
        for item in fields.values()
        if item.get("support", 0) > 0
    ]
    average = sum(values) / len(values) if values else 0.0
    case_count = metrics.get("case_count", 0)
    return f"| {name} | {case_count} | {average:.3f} |"


def field_metric(metrics: dict[str, Any], field: str) -> Any:
    field_metrics = metrics.get("fields", {}).get(field, {})
    if field in BINARY_FIELDS:
        return field_metrics.get("f1", "")
    return field_metrics.get("accuracy", "")


def build_error_rows(
    cases: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    desc: dict[str, dict[str, Any]],
) -> list[str]:
    rows = []
    case_records = index_by_case_id(
        [
            {**record, "case_id": record.get("case_id") or record.get("id")}
            for record in cases
        ]
    )
    for case_id, gold_record in gold.items():
        for field in HIGH_IMPACT_FIELDS:
            gold_value = gold_record.get(field, "")
            if not gold_value:
                continue
            full_value = full.get(case_id, {}).get(field, "")
            desc_value = desc.get(case_id, {}).get(field, "")
            if full_value == gold_value and desc_value == gold_value:
                continue
            baseline_value = baseline.get(case_id, {}).get(field, "")
            source_record = case_records.get(case_id, {})
            mri_text = build_mri_text(source_record, "full") if source_record else ""
            reason = "Prediction differs from gold on a high-impact field."
            rows.append(
                "\n".join(
                    [
                        f"### {case_id} / {field}",
                        "",
                        f"- Gold: `{gold_value}`",
                        f"- Baseline: `{baseline_value}`",
                        f"- Qwen35B Full: `{full_value}`",
                        f"- Qwen35B Description Only: `{desc_value}`",
                        f"- Explanation: {reason}",
                        "",
                        "```text",
                        mri_text[:4000],
                        "```",
                    ]
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_json(args.baseline_metrics)
    full_metrics = load_json(args.full_metrics)
    desc_metrics = load_json(args.description_only_metrics)
    mode_comparison = load_json(args.mode_comparison)
    baseline_comparison = load_json(args.baseline_comparison)

    gold = index_by_case_id([normalize_gold(record) for record in load_records(resolve_path(args.gold))])
    baseline = index_by_case_id(load_records(resolve_path(args.baseline)))
    full = index_by_case_id(load_records(resolve_path(args.full)))
    desc = index_by_case_id(load_records(resolve_path(args.description_only)))
    cases = load_records(resolve_path(args.cases))

    lines = [
        "# PAS MRI Qwen35B Benchmark Report",
        "",
        "## Summary",
        "",
        "| Metric | Baseline | Qwen35B Full | Qwen35B Description Only |",
        "|---|---:|---:|---:|",
        (
            "| Average field score | "
            f"{fmt(sum(field_metric(baseline_metrics, f) or 0 for f in BINARY_FIELDS + CATEGORICAL_FIELDS) / len(BINARY_FIELDS + CATEGORICAL_FIELDS))} | "
            f"{fmt(sum(field_metric(full_metrics, f) or 0 for f in BINARY_FIELDS + CATEGORICAL_FIELDS) / len(BINARY_FIELDS + CATEGORICAL_FIELDS))} | "
            f"{fmt(sum(field_metric(desc_metrics, f) or 0 for f in BINARY_FIELDS + CATEGORICAL_FIELDS) / len(BINARY_FIELDS + CATEGORICAL_FIELDS))} |"
        ),
        "",
        "## Full vs Gold",
        "",
        model_summary_row("Qwen35B Full", full_metrics),
        "",
        "## DescriptionOnly vs Gold",
        "",
        model_summary_row("Qwen35B Description Only", desc_metrics),
        "",
        "## Baseline vs New Model",
        "",
        "| Field | Baseline | Full | DescOnly |",
        "|---|---:|---:|---:|",
    ]

    for row in baseline_comparison.get("field_table", []):
        lines.append(
            f"| {row['Field']} | {fmt(row['Baseline'])} | {fmt(row['Full'])} | {fmt(row['DescOnly'])} |"
        )

    lines.extend(
        [
            "",
            "## Full vs DescriptionOnly",
            "",
            f"- Case count: {mode_comparison.get('case_count', 0)}",
            f"- Unchanged cases: {mode_comparison.get('summary', {}).get('unchanged_cases', 0)}",
            f"- PAS type changed cases: {mode_comparison.get('summary', {}).get('pas_type_changed_cases', 0)}",
            f"- Readiness level changed cases: {mode_comparison.get('summary', {}).get('readiness_level_changed_cases', 0)}",
            f"- Bladder involvement changed cases: {mode_comparison.get('summary', {}).get('bladder_involvement_changed_cases', 0)}",
            f"- Highest suspected extent changed cases: {mode_comparison.get('summary', {}).get('highest_suspected_extent_changed_cases', 0)}",
            "",
            "## Field Metrics",
            "",
            "| Field | Baseline F1/Accuracy | Full F1/Accuracy | DescOnly F1/Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )

    for field in BINARY_FIELDS + CATEGORICAL_FIELDS:
        lines.append(
            f"| {field} | {fmt(field_metric(baseline_metrics, field))} | "
            f"{fmt(field_metric(full_metrics, field))} | {fmt(field_metric(desc_metrics, field))} |"
        )

    lines.extend(["", "## Error Analysis", ""])
    error_rows = build_error_rows(cases, gold, baseline, full, desc)
    if error_rows:
        lines.extend(error_rows)
    else:
        lines.append("No high-impact errors found in supplied artifacts.")

    lines.extend(
        [
            "",
            "## Interpretation Questions",
            "",
            "1. Does MRI conclusion improve model quality?",
            "2. Which features depend most on conclusion?",
            "3. Which features are stable from description alone?",
            "4. Does Qwen35B improve over baseline?",
            "5. Which error types remain most frequent?",
            "6. Can description-only mode be used without substantial quality loss?",
            "7. Which cases have maximum Full vs DescriptionOnly divergence?",
            "8. Which extraction prompt areas should be improved?",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
