"""Compare baseline, full, and description-only predictions against gold."""

from __future__ import annotations

import argparse

from benchmark_utils import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    FLAT_FIELDS,
    evaluate_records,
    index_by_case_id,
    load_records,
    macro_average,
    normalize_gold,
    resolve_path,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline/new benchmark metrics.")
    parser.add_argument("baseline", help="Normalized baseline predictions JSON/JSONL/CSV.")
    parser.add_argument("full", help="Normalized Qwen full predictions JSON/JSONL/CSV.")
    parser.add_argument(
        "description_only",
        help="Normalized Qwen description-only predictions JSON/JSONL/CSV.",
    )
    parser.add_argument("gold", help="Gold JSON/JSONL/CSV.")
    parser.add_argument("--output", required=True, help="Comparison JSON output.")
    parser.add_argument("--csv-output", default=None, help="Optional field metrics CSV.")
    return parser.parse_args()


def summarize_model(name: str, gold_records: list[dict], pred_records: list[dict]) -> dict:
    metrics = evaluate_records(gold_records, pred_records)
    return {
        "name": name,
        "metrics": metrics,
        "macro_binary_f1": macro_average(metrics, "f1"),
    }


def metric_value(model_metrics: dict, field: str) -> float | str:
    field_metrics = model_metrics["metrics"]["fields"].get(field, {})
    if field in BINARY_FIELDS:
        return field_metrics.get("f1", "")
    return field_metrics.get("accuracy", "")


def decision_change_summary(
    baseline_records: list[dict],
    full_records: list[dict],
    desc_records: list[dict],
) -> dict:
    baseline_by_id = index_by_case_id(baseline_records)
    full_by_id = index_by_case_id(full_records)
    desc_by_id = index_by_case_id(desc_records)
    common_case_ids = sorted(set(baseline_by_id) & set(full_by_id) & set(desc_by_id))

    field_changes = {}
    desc_only_changes = {}
    for field in FLAT_FIELDS:
        full_changes = 0
        desc_changes = 0
        only_desc_changes = 0
        for case_id in common_case_ids:
            baseline_value = str(baseline_by_id[case_id].get(field, ""))
            full_value = str(full_by_id[case_id].get(field, ""))
            desc_value = str(desc_by_id[case_id].get(field, ""))
            full_changed = full_value != baseline_value
            desc_changed = desc_value != baseline_value
            if full_changed:
                full_changes += 1
            if desc_changed:
                desc_changes += 1
            if desc_changed and not full_changed:
                only_desc_changes += 1
        field_changes[field] = {
            "full_vs_baseline": full_changes,
            "description_only_vs_baseline": desc_changes,
        }
        desc_only_changes[field] = only_desc_changes

    return {
        "case_count": len(common_case_ids),
        "field_changes_vs_baseline": field_changes,
        "changes_only_after_conclusion_removed": desc_only_changes,
    }


def main() -> None:
    args = parse_args()
    gold_records = [normalize_gold(record) for record in load_records(resolve_path(args.gold))]
    baseline_records = load_records(resolve_path(args.baseline))
    full_records = load_records(resolve_path(args.full))
    desc_records = load_records(resolve_path(args.description_only))
    models = [
        summarize_model("baseline", gold_records, baseline_records),
        summarize_model("full", gold_records, full_records),
        summarize_model("description_only", gold_records, desc_records),
    ]

    field_rows = []
    for field in BINARY_FIELDS + CATEGORICAL_FIELDS:
        field_rows.append(
            {
                "Field": field,
                "Baseline": metric_value(models[0], field),
                "Full": metric_value(models[1], field),
                "DescOnly": metric_value(models[2], field),
            }
        )

    output = {
        "models": models,
        "field_table": field_rows,
        "decision_change_summary": decision_change_summary(
            baseline_records,
            full_records,
            desc_records,
        ),
        "metric_note": "Binary fields use F1; categorical fields use accuracy.",
    }
    write_json(resolve_path(args.output), output)
    if args.csv_output:
        write_csv(
            resolve_path(args.csv_output),
            field_rows,
            ["Field", "Baseline", "Full", "DescOnly"],
        )


if __name__ == "__main__":
    main()
