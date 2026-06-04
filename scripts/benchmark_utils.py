"""
Shared helpers for the PAS MRI benchmark scripts.

The helpers intentionally avoid loading models. Model inference is isolated in
scripts/run_benchmark.py.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FLAT_FIELDS = [
    "pas_type",
    "invasion_type",
    "bladder_involvement",
    "parametrium_involvement",
    "posterior_wall_involvement",
    "placenta_previa",
    "anterior_placenta",
    "retroplacental_vessels",
    "lacunae",
    "uterine_wall_thinning",
    "uterine_hernia_or_bulging",
    "preoperative_bleeding",
    "highest_suspected_extent",
    "percreta_suspicion",
    "bladder_serosa_suspicion",
    "readiness_level",
    "risk_group",
    "confidence",
]

BINARY_FIELDS = [
    "bladder_involvement",
    "placenta_previa",
    "anterior_placenta",
    "retroplacental_vessels",
    "lacunae",
    "uterine_wall_thinning",
    "uterine_hernia_or_bulging",
    "posterior_wall_involvement",
    "percreta_suspicion",
    "preoperative_bleeding",
]

CATEGORICAL_FIELDS = [
    "pas_type",
    "invasion_type",
    "highest_suspected_extent",
    "readiness_level",
    "risk_group",
]

HIGH_IMPACT_FIELDS = [
    "pas_type",
    "invasion_type",
    "highest_suspected_extent",
    "bladder_involvement",
    "percreta_suspicion",
    "readiness_level",
]

DESCRIPTION_COLUMNS = [
    "МРТ_Описание",
    "МРТ Описание",
    "Описание МРТ",
    "MRI_description",
    "mri_description",
]

CONCLUSION_COLUMNS = [
    "МРТ_Заключение",
    "МРТ Заключение",
    "Заключение МРТ",
    "MRI_conclusion",
    "mri_conclusion",
]

CASE_ID_COLUMNS = ["case_id", "id", "case", "номер", "номер_случая", "№"]
MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "-"}
POSITIVE_STATUS_VALUES = {"possible", "probable", "present", "true", "1", "yes", "да"}
NEGATIVE_STATUS_VALUES = {"absent", "none", "false", "0", "no", "нет"}
SEVERITY = {"none": 0, "accreta": 1, "increta": 2, "percreta": 3}


def resolve_path(path_value: str | Path) -> Path:
    return Path(path_value).expanduser().resolve()


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", "_", text)


def find_column(record: dict[str, Any], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_column_name(key): key for key in record}
    for candidate in candidates:
        key = normalized.get(normalize_column_name(candidate))
        if key is not None:
            return key
    return None


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return [dict(item) for item in data["cases"]]
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return [dict(item) for item in data["records"]]
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def case_id_for_record(record: dict[str, Any], index: int) -> str:
    column = find_column(record, CASE_ID_COLUMNS)
    if column:
        value = value_to_text(record.get(column))
        if value:
            return value
    return f"case_{index + 1:06d}"


def build_mri_text(record: dict[str, Any], mode: str) -> str:
    description_column = find_column(record, DESCRIPTION_COLUMNS)
    conclusion_column = find_column(record, CONCLUSION_COLUMNS)
    description = value_to_text(record.get(description_column)) if description_column else ""
    conclusion = value_to_text(record.get(conclusion_column)) if conclusion_column else ""

    if mode == "description_only":
        return description
    if mode == "full":
        parts = []
        if description:
            parts.append(f"МРТ_Описание:\n{description}")
        if conclusion:
            parts.append(f"МРТ_Заключение:\n{conclusion}")
        return "\n\n".join(parts)
    raise ValueError(f"Unsupported mode: {mode}")


def nested_get(data: dict[str, Any], path: list[str], default: Any = "") -> Any:
    current: Any = data
    for item in path:
        if not isinstance(current, dict):
            return default
        current = current.get(item)
    return default if current is None else current


def normalize_value(value: Any) -> str:
    text = value_to_text(value)
    if text.lower() in MISSING_VALUES:
        return ""
    return text.lower()


def normalize_prediction(case_id: str, payload: dict[str, Any]) -> dict[str, str]:
    extracted = payload.get("extracted_features", {})
    score = payload.get("score", {})
    recommendation = payload.get("recommendation", {})
    suspicion = payload.get("suspicion", {})

    invasion_type = normalize_value(nested_get(extracted, ["invasion", "type"]))
    confidence = normalize_value(nested_get(extracted, ["invasion", "confidence"]))

    return {
        "case_id": case_id,
        "pas_type": invasion_type,
        "invasion_type": invasion_type,
        "bladder_involvement": normalize_value(
            nested_get(extracted, ["anatomy", "bladder_involvement"])
        ),
        "parametrium_involvement": normalize_value(
            nested_get(extracted, ["anatomy", "parametrium_involvement"])
        ),
        "posterior_wall_involvement": normalize_value(
            nested_get(extracted, ["anatomy", "posterior_wall_involvement"])
        ),
        "placenta_previa": normalize_value(
            nested_get(extracted, ["placenta_location", "placenta_previa"])
        ),
        "anterior_placenta": normalize_value(
            nested_get(extracted, ["placenta_location", "anterior_placenta"])
        ),
        "retroplacental_vessels": normalize_value(
            nested_get(extracted, ["mri_signs", "retroplacental_vessels"])
        ),
        "lacunae": normalize_value(nested_get(extracted, ["mri_signs", "lacunae"])),
        "uterine_wall_thinning": normalize_value(
            nested_get(extracted, ["mri_signs", "uterine_wall_thinning"])
        ),
        "uterine_hernia_or_bulging": normalize_value(
            nested_get(extracted, ["mri_signs", "uterine_hernia_or_bulging"])
        ),
        "preoperative_bleeding": normalize_value(
            nested_get(extracted, ["clinical_context", "preoperative_bleeding"])
        ),
        "highest_suspected_extent": normalize_value(
            suspicion.get("highest_suspected_extent")
        ),
        "percreta_suspicion": normalize_value(suspicion.get("percreta_suspicion")),
        "bladder_serosa_suspicion": normalize_value(
            suspicion.get("bladder_serosa_suspicion")
        ),
        "readiness_level": normalize_value(recommendation.get("readiness_level")),
        "risk_group": normalize_value(score.get("risk_group")),
        "confidence": confidence,
    }


def normalize_gold(record: dict[str, Any]) -> dict[str, str]:
    case_id = value_to_text(record.get("case_id"))
    invasion_type = normalize_value(
        record.get("gold_pas_type") or record.get("gold_invasion_type")
    )
    return {
        "case_id": case_id,
        "pas_type": invasion_type,
        "invasion_type": normalize_value(record.get("gold_invasion_type")),
        "bladder_involvement": normalize_value(record.get("gold_bladder_involvement")),
        "parametrium_involvement": normalize_value(
            record.get("gold_parametrium_involvement")
        ),
        "posterior_wall_involvement": normalize_value(
            record.get("gold_posterior_wall_involvement")
        ),
        "placenta_previa": normalize_value(record.get("gold_placenta_previa")),
        "anterior_placenta": normalize_value(record.get("gold_anterior_placenta")),
        "retroplacental_vessels": normalize_value(
            record.get("gold_retroplacental_vessels")
        ),
        "lacunae": normalize_value(record.get("gold_lacunae")),
        "uterine_wall_thinning": normalize_value(
            record.get("gold_uterine_wall_thinning")
        ),
        "uterine_hernia_or_bulging": normalize_value(
            record.get("gold_uterine_hernia_or_bulging")
        ),
        "preoperative_bleeding": normalize_value(record.get("gold_preoperative_bleeding")),
        "highest_suspected_extent": normalize_value(
            record.get("gold_highest_suspected_extent")
        ),
        "percreta_suspicion": normalize_value(record.get("gold_percreta_suspicion")),
        "bladder_serosa_suspicion": normalize_value(
            record.get("gold_bladder_serosa_suspicion")
        ),
        "readiness_level": normalize_value(record.get("gold_readiness_level")),
        "risk_group": normalize_value(record.get("gold_risk_group")),
        "confidence": normalize_value(record.get("gold_confidence")),
    }


def index_by_case_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("case_id")): record for record in records if record.get("case_id")}


def is_missing(value: Any) -> bool:
    return normalize_value(value) == ""


def is_positive_status(value: Any) -> bool:
    return normalize_value(value) in POSITIVE_STATUS_VALUES


def binary_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for gold, pred in pairs:
        gold_pos = is_positive_status(gold)
        pred_pos = is_positive_status(pred)
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "support": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def categorical_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    total = len(pairs)
    correct = sum(1 for gold, pred in pairs if normalize_value(gold) == normalize_value(pred))
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, pred in pairs:
        matrix[normalize_value(gold)][normalize_value(pred)] += 1
    return {
        "support": total,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": {
            gold: dict(pred_counts) for gold, pred_counts in sorted(matrix.items())
        },
    }


def evaluate_records(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = index_by_case_id(gold_records)
    pred_by_id = index_by_case_id(pred_records)
    common_case_ids = sorted(set(gold_by_id) & set(pred_by_id))
    field_metrics: dict[str, Any] = {}

    for field in BINARY_FIELDS + CATEGORICAL_FIELDS:
        pairs = []
        skipped_missing_gold = 0
        for case_id in common_case_ids:
            gold_value = gold_by_id[case_id].get(field, "")
            pred_value = pred_by_id[case_id].get(field, "")
            if is_missing(gold_value):
                skipped_missing_gold += 1
                continue
            pairs.append((gold_value, pred_value))

        if field in BINARY_FIELDS:
            metrics = binary_metrics(pairs)
            metrics["type"] = "binary"
        else:
            metrics = categorical_metrics(pairs)
            metrics["type"] = "categorical"
        metrics["skipped_missing_gold"] = skipped_missing_gold
        field_metrics[field] = metrics

    return {
        "case_count": len(common_case_ids),
        "missing_predictions": sorted(set(gold_by_id) - set(pred_by_id)),
        "extra_predictions": sorted(set(pred_by_id) - set(gold_by_id)),
        "fields": field_metrics,
    }


def macro_average(metrics: dict[str, Any], key: str = "f1") -> float:
    values = [
        field_metrics[key]
        for field_metrics in metrics.get("fields", {}).values()
        if key in field_metrics and field_metrics.get("support", 0) > 0
    ]
    return sum(values) / len(values) if values else 0.0


def accuracy_average(metrics: dict[str, Any]) -> float:
    values = [
        field_metrics["accuracy"]
        for field_metrics in metrics.get("fields", {}).values()
        if "accuracy" in field_metrics and field_metrics.get("support", 0) > 0
    ]
    return sum(values) / len(values) if values else 0.0
