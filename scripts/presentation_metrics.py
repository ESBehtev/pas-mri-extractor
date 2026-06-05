"""Build presentation-ready PAS20 benchmark metrics.

The script reads existing normalized predictions and gold labels. It does not
run inference and does not modify prediction or gold files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any, Callable

from benchmark_utils import (
    index_by_case_id,
    load_records,
    normalize_value,
    resolve_path,
    write_json,
)


MRI_FEATURE_FIELDS = [
    "placenta_previa",
    "anterior_placenta",
    "retroplacental_vessels",
    "lacunae",
    "uterine_wall_thinning",
    "uterine_hernia_or_bulging",
    "bladder_involvement",
]

METRIC_GROUP_LABELS = {
    "mri_feature_extraction_quality": "Качество извлечения МРТ-признаков",
    "pas_detection": "Выявление PAS",
    "advanced_pas_detection": "Выявление advanced PAS",
    "percreta_detection": "Выявление percreta",
    "clinical_risk_outputs": "Клинические и риск-выходы",
}

CLINICAL_FIELDS = [
    "pas_type",
    "invasion_type",
    "highest_suspected_extent",
    "readiness_level",
    "risk_group",
]

GOLD_FIELD_MAP = {
    "pas_type": ["gold_pas_type", "gold_invasion_type"],
    "invasion_type": ["gold_invasion_type", "gold_pas_type"],
    "invasion_confidence": ["gold_invasion_confidence"],
    "bladder_involvement": ["gold_bladder_involvement"],
    "parametrium_involvement": ["gold_parametrium_involvement"],
    "posterior_wall_involvement": ["gold_posterior_wall_involvement"],
    "placenta_previa": ["gold_placenta_previa"],
    "anterior_placenta": ["gold_anterior_placenta"],
    "retroplacental_vessels": ["gold_retroplacental_vessels"],
    "lacunae": ["gold_lacunae"],
    "uterine_wall_thinning": ["gold_uterine_wall_thinning"],
    "uterine_hernia_or_bulging": ["gold_uterine_hernia_or_bulging"],
    "highest_suspected_extent": ["gold_highest_suspected_extent"],
    "readiness_level": ["gold_readiness_level"],
    "risk_group": ["gold_risk_group"],
}

POSITIVE_STATUS = {"possible", "probable", "present", "true", "1", "yes", "да"}
PAS_PRESENT = {"accreta", "increta", "percreta"}
ADVANCED_PAS = {"increta", "percreta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build presentation PAS20 metrics.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL/CSV path.")
    parser.add_argument("--full", required=True, help="Full normalized predictions.")
    parser.add_argument(
        "--description-only",
        required=True,
        help="Description-only normalized predictions.",
    )
    parser.add_argument("--output-json", required=True, help="Metrics JSON output.")
    parser.add_argument("--output-md", required=True, help="Metrics Markdown output.")
    return parser.parse_args()


def first_gold_value(record: dict[str, Any], field: str) -> str:
    for gold_field in GOLD_FIELD_MAP[field]:
        value = normalize_value(record.get(gold_field))
        if value:
            return value
    return ""


def pred_value(record: dict[str, Any] | None, field: str) -> str:
    if not record:
        return ""
    return normalize_value(record.get(field))


def binary_metric_from_pairs(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for gold_positive, pred_positive in pairs:
        if gold_positive and pred_positive:
            tp += 1
        elif not gold_positive and pred_positive:
            fp += 1
        elif gold_positive and not pred_positive:
            fn += 1
        else:
            tn += 1
    support = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / support if support else 0.0
    return {
        "support": support,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def binary_field_metrics(
    gold_by_id: dict[str, dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    field: str,
    positive_fn: Callable[[str], bool],
) -> dict[str, Any]:
    pairs = []
    skipped_missing_gold = 0
    for case_id in sorted(gold_by_id):
        gold = first_gold_value(gold_by_id[case_id], field)
        if not gold:
            skipped_missing_gold += 1
            continue
        pred = pred_value(pred_by_id.get(case_id), field)
        pairs.append((positive_fn(gold), positive_fn(pred)))
    metrics = binary_metric_from_pairs(pairs)
    metrics["skipped_missing_gold"] = skipped_missing_gold
    return metrics


def micro_binary_metrics(field_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    tp = sum(item["tp"] for item in field_metrics.values())
    fp = sum(item["fp"] for item in field_metrics.values())
    fn = sum(item["fn"] for item in field_metrics.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def aggregate_mri_metrics(field_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = [item for item in field_metrics.values() if item["support"] > 0]
    micro = micro_binary_metrics(field_metrics)
    return {
        "mean_accuracy": sum(item["accuracy"] for item in values) / len(values)
        if values
        else 0.0,
        "macro_f1": sum(item["f1"] for item in values) / len(values) if values else 0.0,
        **micro,
    }


def categorical_metrics(
    gold_by_id: dict[str, dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    pairs = []
    skipped_missing_gold = 0
    for case_id in sorted(gold_by_id):
        gold = first_gold_value(gold_by_id[case_id], field)
        if not gold:
            skipped_missing_gold += 1
            continue
        pred = pred_value(pred_by_id.get(case_id), field)
        pairs.append((gold, pred))

    support = len(pairs)
    correct = sum(1 for gold, pred in pairs if gold == pred)
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    gold_class_counts: Counter[str] = Counter()
    gold_class_correct: Counter[str] = Counter()
    for gold, pred in pairs:
        matrix[gold][pred] += 1
        gold_class_counts[gold] += 1
        if gold == pred:
            gold_class_correct[gold] += 1

    recalls = [
        gold_class_correct[label] / count
        for label, count in gold_class_counts.items()
        if count > 0
    ]
    return {
        "support": support,
        "accuracy": correct / support if support else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "confusion_matrix": {
            gold: dict(preds) for gold, preds in sorted(matrix.items())
        },
        "skipped_missing_gold": skipped_missing_gold,
    }


def positive_status(value: str) -> bool:
    return normalize_value(value) in POSITIVE_STATUS


def pas_present(value: str) -> bool:
    return normalize_value(value) in PAS_PRESENT


def advanced_pas(value: str) -> bool:
    return normalize_value(value) in ADVANCED_PAS


def percreta(value: str) -> bool:
    return normalize_value(value) == "percreta"


def build_mode_metrics(
    gold_by_id: dict[str, dict[str, Any]],
    pred_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mri_fields = {
        field: binary_field_metrics(gold_by_id, pred_by_id, field, positive_status)
        for field in MRI_FEATURE_FIELDS
    }
    return {
        "mri_feature_extraction_quality": {
            "fields": mri_fields,
            "aggregate": aggregate_mri_metrics(mri_fields),
        },
        "pas_detection": {
            field: binary_field_metrics(gold_by_id, pred_by_id, field, pas_present)
            for field in ["pas_type", "invasion_type"]
        },
        "advanced_pas_detection": {
            field: binary_field_metrics(gold_by_id, pred_by_id, field, advanced_pas)
            for field in ["pas_type", "invasion_type"]
        },
        "percreta_detection": {
            field: binary_field_metrics(gold_by_id, pred_by_id, field, percreta)
            for field in ["pas_type", "invasion_type"]
        },
        "clinical_risk_outputs": {
            field: categorical_metrics(gold_by_id, pred_by_id, field)
            for field in CLINICAL_FIELDS
        },
    }


def metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric_name, key in [
        ("Макро-F1 МРТ-признаков", ("mri_feature_extraction_quality", "aggregate", "macro_f1")),
        ("Средняя accuracy МРТ-признаков", ("mri_feature_extraction_quality", "aggregate", "mean_accuracy")),
        ("F1 выявления PAS / pas_type", ("pas_detection", "pas_type", "f1")),
        ("F1 выявления PAS / invasion_type", ("pas_detection", "invasion_type", "f1")),
        ("F1 advanced PAS / pas_type", ("advanced_pas_detection", "pas_type", "f1")),
        ("F1 advanced PAS / invasion_type", ("advanced_pas_detection", "invasion_type", "f1")),
        ("F1 percreta / pas_type", ("percreta_detection", "pas_type", "f1")),
        ("F1 percreta / invasion_type", ("percreta_detection", "invasion_type", "f1")),
        ("Accuracy типа PAS", ("clinical_risk_outputs", "pas_type", "accuracy")),
        ("Accuracy уровня готовности", ("clinical_risk_outputs", "readiness_level", "accuracy")),
        ("Accuracy группы риска", ("clinical_risk_outputs", "risk_group", "accuracy")),
    ]:
        full = nested(metrics["full"], key)
        desc = nested(metrics["description_only"], key)
        rows.append(
            {
                "metric_group": key[0],
                "metric": metric_name,
                "full": full,
                "description_only": desc,
                "delta": desc - full,
                "best_mode": best_mode(full, desc),
            }
        )
    return rows


def nested(data: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = data
    for key in path:
        current = current[key]
    return float(current)


def best_mode(full: float, desc: float) -> str:
    if desc > full:
        return "description_only"
    if full > desc:
        return "full"
    return "tie"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics["comparison_rows"]
    better_desc = [row for row in rows if row["best_mode"] == "description_only"]
    better_full = [row for row in rows if row["best_mode"] == "full"]
    ties = [row for row in rows if row["best_mode"] == "tie"]

    lines = [
        "# Презентационные метрики PAS20",
        "",
        "## Full vs только описание",
        "",
        "| Группа метрик | Метрика | Full | Только описание | Разница |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {METRIC_GROUP_LABELS.get(row['metric_group'], row['metric_group'])} | "
            f"{row['metric']} | {fmt(row['full'])} | "
            f"{fmt(row['description_only'])} | {fmt(row['delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Лучший режим по метрикам",
            "",
            f"- Лучше только описание: {len(better_desc)}",
            f"- Лучше Full: {len(better_full)}",
            f"- Одинаково: {len(ties)}",
            "",
            "### Лучше только описание",
            "",
        ]
    )
    lines.extend([f"- {row['metric']}: +{fmt(row['delta'])}" for row in better_desc] or ["- None"])
    lines[-1:] = ["- Нет"] if lines[-1:] == ["- None"] else lines[-1:]
    lines.extend(["", "### Лучше Full", ""])
    lines.extend([f"- {row['metric']}: {fmt(row['delta'])}" for row in better_full] or ["- None"])
    lines[-1:] = ["- Нет"] if lines[-1:] == ["- None"] else lines[-1:]
    lines.extend(["", "### Одинаково", ""])
    lines.extend([f"- {row['metric']}" for row in ties] or ["- None"])
    lines[-1:] = ["- Нет"] if lines[-1:] == ["- None"] else lines[-1:]

    lines.extend(["", "## Метрики по МРТ-признакам", ""])
    for mode in ["full", "description_only"]:
        mode_label = "Full" if mode == "full" else "Только описание"
        lines.extend(
            [
                f"### {mode_label}",
                "",
                "| Поле | Support | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        field_metrics = metrics[mode]["mri_feature_extraction_quality"]["fields"]
        for field in MRI_FEATURE_FIELDS:
            item = field_metrics[field]
            lines.append(
                f"| {field} | {item['support']} | {fmt(item['accuracy'])} | "
                f"{fmt(item['precision'])} | {fmt(item['recall'])} | {fmt(item['f1'])} | "
                f"{item['tp']} | {item['fp']} | {item['tn']} | {item['fn']} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gold_by_id = index_by_case_id(load_records(resolve_path(args.gold)))
    full_by_id = index_by_case_id(load_records(resolve_path(args.full)))
    desc_by_id = index_by_case_id(load_records(resolve_path(args.description_only)))

    metrics = {
        "case_count": len(gold_by_id),
        "full": build_mode_metrics(gold_by_id, full_by_id),
        "description_only": build_mode_metrics(gold_by_id, desc_by_id),
    }
    metrics["comparison_rows"] = metric_rows(metrics)

    write_json(resolve_path(args.output_json), metrics)
    write_markdown(resolve_path(args.output_md), metrics)


if __name__ == "__main__":
    main()
