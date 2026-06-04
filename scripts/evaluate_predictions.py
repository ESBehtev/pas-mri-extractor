"""
Run PAS extraction on an exported clinical dataset and compute gold metrics.

This script is intended for the server/runtime environment. It refuses to load
an LLM unless --run-llm is passed explicitly. Use --use-rules for a lightweight
baseline that does not load model weights.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pas_mri_extractor.models import LoadedModel, ModelConfigError, PROJECT_ROOT
from pas_mri_extractor.pipeline import (
    extract_features_with_artifacts,
    get_cached_model,
    unload_current_model,
)
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_PREDICTIONS_NAME = "predictions.jsonl"
DEFAULT_METRICS_NAME = "metrics.json"

MRI_TEXT_CANDIDATES = [
    "МРТ_Описание",
    "МРТ Описание",
    "Описание МРТ",
    "Описание",
    "MRI_description",
    "mri_description",
    "МРТ_Заключение",
    "МРТ Заключение",
    "Заключение МРТ",
    "Заключение",
    "MRI_conclusion",
    "mri_conclusion",
]

DESCRIPTION_CANDIDATES = [
    "МРТ_Описание",
    "МРТ Описание",
    "Описание МРТ",
    "Описание",
    "MRI_description",
    "mri_description",
]

CONCLUSION_CANDIDATES = [
    "МРТ_Заключение",
    "МРТ Заключение",
    "Заключение МРТ",
    "Заключение",
    "MRI_conclusion",
    "mri_conclusion",
]

GOLD_FIELDS = [
    "gold_invasion_type",
    "gold_invasion_confidence",
    "gold_blood_loss_ml",
    "gold_blood_loss_class",
    "gold_massive_blood_loss",
    "gold_bladder_involvement",
    "gold_parametrium_involvement",
    "gold_posterior_wall_involvement",
    "gold_placenta_previa",
    "gold_anterior_placenta",
    "gold_retroplacental_vessels",
    "gold_lacunae",
    "gold_uterine_wall_thinning",
    "gold_uterine_hernia_or_bulging",
    "gold_preoperative_bleeding",
    "gold_highest_suspected_extent",
    "gold_percreta_suspicion",
    "gold_bladder_serosa_suspicion",
    "gold_vascular_intervention",
    "gold_pas_type",
    "gold_readiness_level",
    "gold_risk_group",
    "gold_confidence",
    "gold_rationale",
]

EVAL_TASKS = {
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
    "preoperative_bleeding": ["gold_preoperative_bleeding"],
    "highest_suspected_extent": ["gold_highest_suspected_extent"],
    "percreta_suspicion": ["gold_percreta_suspicion"],
    "bladder_serosa_suspicion": ["gold_bladder_serosa_suspicion"],
    "massive_blood_loss": ["gold_massive_blood_loss"],
    "blood_loss_class": ["gold_blood_loss_class"],
    "vascular_intervention": ["gold_vascular_intervention"],
    "readiness_level": ["gold_readiness_level"],
    "risk_group": ["gold_risk_group"],
}

CASE_ID_CANDIDATES = [
    "case_id",
    "id",
    "case",
    "номер",
    "номер_случая",
    "№",
]

MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "-"}
TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "да",
    "истина",
    "present",
    "positive",
    "possible",
    "probable",
}
FALSE_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "нет",
    "ложь",
    "absent",
    "negative",
    "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PAS extraction predictions against gold fields.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to exported CSV or JSONL. Defaults to data/evaluation/dataset_sheet3.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for predictions.jsonl and metrics.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name from configs/models.yaml when --run-llm is used.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Allow loading the configured LLM and running inference.",
    )
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Use the regex rule baseline instead of LLM inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=None,
        help="Explicit text columns to join as the MRI report text.",
    )
    parser.add_argument(
        "--print-raw-output",
        action="store_true",
        help="Print raw LLM output during inference.",
    )
    return parser.parse_args()


def resolve_dataset(path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path.resolve()

    candidates = [
        DEFAULT_DATASET_DIR / "dataset_sheet3_gold.csv",
        DEFAULT_DATASET_DIR / "dataset_sheet3_gold.jsonl",
        DEFAULT_DATASET_DIR / "dataset_sheet3.csv",
        DEFAULT_DATASET_DIR / "dataset_sheet3.jsonl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No exported dataset found. Run scripts/export_excel_dataset.py first."
    )


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_map = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]

    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error

    return records


def read_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
        return ""
    return str(value).strip()


def get_case_id(row: dict[str, Any], index: int) -> str:
    case_column = find_column(list(row.keys()), CASE_ID_CANDIDATES)
    if case_column:
        value = text_value(row.get(case_column))
        if value:
            return value

    return f"case_{index + 1:06d}"


def get_text_columns(row: dict[str, Any], explicit_columns: list[str] | None) -> list[str]:
    columns = list(row.keys())

    if explicit_columns:
        missing = [column for column in explicit_columns if column not in row]
        if missing:
            raise ValueError(f"Text columns not found: {', '.join(missing)}")
        return explicit_columns

    description = find_column(columns, DESCRIPTION_CANDIDATES)
    conclusion = find_column(columns, CONCLUSION_CANDIDATES)
    selected = []
    if description:
        selected.append(description)
    if conclusion and conclusion not in selected:
        selected.append(conclusion)

    if selected:
        return selected

    fallback = find_column(columns, MRI_TEXT_CANDIDATES)
    if fallback:
        return [fallback]

    raise ValueError(
        "MRI text column not found. Pass --text-columns with one or more column names."
    )


def build_mri_text(row: dict[str, Any], text_columns: list[str]) -> str:
    parts = []
    for column in text_columns:
        value = text_value(row.get(column))
        if value:
            parts.append(value)

    return "\n\n".join(parts).strip()


def normalize_missing(value: Any) -> str | None:
    text = text_value(value)
    if text.lower() in MISSING_VALUES:
        return None
    return text


def normalize_bool_label(value: Any) -> str | None:
    text = normalize_missing(value)
    if text is None:
        return None
    normalized = text.strip().lower()
    if normalized in TRUE_VALUES:
        return "positive"
    if normalized in FALSE_VALUES:
        return "negative"
    return normalized


def normalize_pas_type(value: Any) -> str | None:
    text = normalize_missing(value)
    if text is None:
        return None
    normalized = text.strip().lower()
    if "percreta" in normalized:
        return "percreta"
    if "increta" in normalized:
        return "increta"
    if "accreta" in normalized:
        return "accreta"
    if normalized in {"none", "no", "нет", "absent", "negative"}:
        return "none"
    return normalized


def normalize_readiness(value: Any) -> str | None:
    text = normalize_missing(value)
    if text is None:
        return None
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)
    return text.strip().lower()


def normalize_blood_loss_ml(value: Any) -> int | None:
    text = normalize_missing(value)
    if text is None:
        return None

    normalized = text.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", normalized)
    if not match:
        return None

    return int(float(match.group(0)))


def blood_loss_class_from_ml(value: int) -> str:
    if value >= 1500:
        return ">=1500"
    if value >= 1000:
        return "1000-1499"
    return "<1000"


def normalize_blood_loss_class(value: Any) -> str | None:
    text = normalize_missing(value)
    if text is None:
        return None

    normalized = text.strip().lower().replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", normalized)

    if any(marker in compact for marker in ["massive", "массив", "high", "высок"]):
        return ">=1500"
    if any(marker in compact for marker in ["moderate", "сред", "умерен"]):
        return "1000-1499"
    if any(marker in compact for marker in ["low", "низк"]):
        return "<1000"

    numbers = re.findall(r"\d+(?:\.\d+)?", compact)
    if numbers:
        return blood_loss_class_from_ml(int(float(numbers[0])))

    return compact


def predicted_massive_blood_loss(result: dict[str, Any]) -> str:
    risks = result.get("predicted_risks") or {}
    percent = risks.get("massive_blood_loss_over_1500_ml_percent", 0)
    try:
        return "positive" if float(percent) >= 50 else "negative"
    except (TypeError, ValueError):
        return "negative"


def predicted_bladder_involvement(result: dict[str, Any]) -> str:
    anatomy = (result.get("extracted_features") or {}).get("anatomy") or {}
    return str(anatomy.get("bladder_involvement", "absent")).lower()


def predicted_anatomy_status(result: dict[str, Any], field_name: str) -> str | None:
    anatomy = (result.get("extracted_features") or {}).get("anatomy") or {}
    value = anatomy.get(field_name)
    return None if value is None else str(value).lower()


def predicted_placenta_location_status(
    result: dict[str, Any],
    field_name: str,
) -> str | None:
    placenta_location = (
        (result.get("extracted_features") or {}).get("placenta_location") or {}
    )
    value = placenta_location.get(field_name)
    return None if value is None else str(value).lower()


def predicted_mri_sign_status(result: dict[str, Any], field_name: str) -> str | None:
    mri_signs = (result.get("extracted_features") or {}).get("mri_signs") or {}
    value = mri_signs.get(field_name)
    return None if value is None else str(value).lower()


def predicted_clinical_context_status(
    result: dict[str, Any],
    field_name: str,
) -> str | None:
    clinical_context = (
        (result.get("extracted_features") or {}).get("clinical_context") or {}
    )
    value = clinical_context.get(field_name)
    return None if value is None else str(value).lower()


def predicted_suspicion_status(result: dict[str, Any], field_name: str) -> str | None:
    suspicion = result.get("suspicion") or {}
    value = suspicion.get(field_name)
    return None if value is None else str(value).lower()


def predicted_blood_loss_class(result: dict[str, Any]) -> str | None:
    risks = result.get("predicted_risks") or {}
    return normalize_blood_loss_class(risks.get("estimated_blood_loss_ml_range"))


def predicted_pas_type(result: dict[str, Any]) -> str | None:
    invasion = (result.get("extracted_features") or {}).get("invasion") or {}
    return normalize_pas_type(invasion.get("type"))


def predicted_invasion_confidence(result: dict[str, Any]) -> str | None:
    invasion = (result.get("extracted_features") or {}).get("invasion") or {}
    value = invasion.get("confidence")
    return None if value is None else str(value).lower()


def predicted_readiness(result: dict[str, Any]) -> str | None:
    recommendation = result.get("recommendation") or {}
    return normalize_readiness(recommendation.get("readiness_level"))


def predicted_risk_group(result: dict[str, Any]) -> str | None:
    score = result.get("score") or {}
    value = score.get("risk_group")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "moderate":
        return "medium"
    return normalized


def predicted_vascular_intervention(result: dict[str, Any]) -> str:
    risks = result.get("predicted_risks") or {}
    percent = risks.get("vascular_intervention_percent", 0)
    try:
        return "positive" if float(percent) >= 50 else "negative"
    except (TypeError, ValueError):
        return "negative"


def task_prediction(task_name: str, result: dict[str, Any]) -> str | None:
    if task_name == "invasion_type":
        return predicted_pas_type(result)
    if task_name == "invasion_confidence":
        return predicted_invasion_confidence(result)
    if task_name == "massive_blood_loss":
        return predicted_massive_blood_loss(result)
    if task_name == "blood_loss_class":
        return predicted_blood_loss_class(result)
    if task_name == "bladder_involvement":
        return predicted_bladder_involvement(result)
    if task_name == "parametrium_involvement":
        return predicted_anatomy_status(result, "parametrium_involvement")
    if task_name == "posterior_wall_involvement":
        return predicted_anatomy_status(result, "posterior_wall_involvement")
    if task_name == "placenta_previa":
        return predicted_placenta_location_status(result, "placenta_previa")
    if task_name == "anterior_placenta":
        return predicted_placenta_location_status(result, "anterior_placenta")
    if task_name == "retroplacental_vessels":
        return predicted_mri_sign_status(result, "retroplacental_vessels")
    if task_name == "lacunae":
        return predicted_mri_sign_status(result, "lacunae")
    if task_name == "uterine_wall_thinning":
        return predicted_mri_sign_status(result, "uterine_wall_thinning")
    if task_name == "uterine_hernia_or_bulging":
        return predicted_mri_sign_status(result, "uterine_hernia_or_bulging")
    if task_name == "preoperative_bleeding":
        return predicted_clinical_context_status(result, "preoperative_bleeding")
    if task_name == "highest_suspected_extent":
        return normalize_pas_type(
            predicted_suspicion_status(result, "highest_suspected_extent")
        )
    if task_name == "percreta_suspicion":
        return predicted_suspicion_status(result, "percreta_suspicion")
    if task_name == "bladder_serosa_suspicion":
        return predicted_suspicion_status(result, "bladder_serosa_suspicion")
    if task_name == "pas_classification":
        return predicted_pas_type(result)
    if task_name == "readiness_level":
        return predicted_readiness(result)
    if task_name == "risk_group":
        return predicted_risk_group(result)
    if task_name == "vascular_intervention":
        return predicted_vascular_intervention(result)
    raise ValueError(f"Unknown task: {task_name}")


def normalize_gold(task_name: str, value: Any) -> str | None:
    if task_name in {"massive_blood_loss", "vascular_intervention"}:
        return normalize_bool_label(value)
    if task_name in {"invasion_type", "pas_classification", "highest_suspected_extent"}:
        return normalize_pas_type(value)
    if task_name == "readiness_level":
        return normalize_readiness(value)
    if task_name == "blood_loss_class":
        return normalize_blood_loss_class(value)
    if task_name == "risk_group":
        text = normalize_missing(value)
        if text is None:
            return None
        normalized = text.strip().lower()
        if normalized == "moderate":
            return "medium"
        return normalized

    text = normalize_missing(value)
    if text is None:
        return None
    return text.strip().lower()


def get_gold_value(row: dict[str, Any], gold_fields: str | list[str]) -> Any:
    candidates = [gold_fields] if isinstance(gold_fields, str) else gold_fields
    for gold_field in candidates:
        column = find_column(list(row.keys()), [gold_field])
        if column is None:
            continue
        value = row.get(column)
        if normalize_missing(value) is not None:
            return value

    return None


def get_gold_for_task(
    row: dict[str, Any],
    task_name: str,
    gold_fields: str | list[str],
) -> str | None:
    gold = normalize_gold(task_name, get_gold_value(row, gold_fields))
    if gold is not None:
        return gold

    blood_loss_ml = normalize_blood_loss_ml(get_gold_value(row, "gold_blood_loss_ml"))
    if blood_loss_ml is None:
        return None

    if task_name == "massive_blood_loss":
        return "positive" if blood_loss_ml >= 1500 else "negative"

    if task_name == "blood_loss_class":
        return blood_loss_class_from_ml(blood_loss_ml)

    return None


def run_rules_inference(text: str) -> dict[str, Any]:
    validated = rule_extract_features(text)
    result = normalize_mri_result(validated)
    return {
        "raw_output": json.dumps(validated.model_dump(), ensure_ascii=False),
        "validated": validated.model_dump(),
        "result": result.model_dump(),
    }


def run_inference(
    text: str,
    model_name: str | None,
    loaded_model: LoadedModel | None,
    use_rules: bool,
    print_raw_output: bool,
) -> dict[str, Any]:
    if use_rules:
        return run_rules_inference(text)

    return extract_features_with_artifacts(
        text=text,
        model_name=model_name,
        loaded_model=loaded_model,
        print_raw_output=print_raw_output,
    )


def update_metrics_state(
    state: dict[str, list[tuple[str, str]]],
    row: dict[str, Any],
    result: dict[str, Any],
) -> None:
    for task_name, gold_field in EVAL_TASKS.items():
        gold = get_gold_for_task(row, task_name, gold_field)
        if gold is None:
            continue

        prediction = task_prediction(task_name, result)
        if prediction is None:
            continue

        state[task_name].append((str(gold), str(prediction)))


def binary_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any] | None:
    labels = {label for pair in pairs for label in pair}
    if not labels.issubset({"positive", "negative"}):
        return None

    tp = sum(1 for gold, pred in pairs if gold == "positive" and pred == "positive")
    tn = sum(1 for gold, pred in pairs if gold == "negative" and pred == "negative")
    fp = sum(1 for gold, pred in pairs if gold == "negative" and pred == "positive")
    fn = sum(1 for gold, pred in pairs if gold == "positive" and pred == "negative")
    total = len(pairs)

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_label": "positive",
    }


def macro_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    accuracy = sum(1 for gold, pred in pairs if gold == pred) / total if total else None
    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / len(labels) if labels else None,
        "recall": sum(recalls) / len(labels) if labels else None,
        "f1": sum(f1s) / len(labels) if labels else None,
        "average": "macro",
    }


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    labels = sorted({label for pair in pairs for label in pair})
    matrix: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }
    for gold, prediction in pairs:
        matrix[gold][prediction] += 1

    return matrix


def compute_metrics(state: dict[str, list[tuple[str, str]]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for task_name in EVAL_TASKS:
        pairs = state.get(task_name, [])
        if not pairs:
            metrics[task_name] = {
                "support": 0,
                "skipped": True,
                "reason": "No filled gold labels.",
            }
            continue

        base_metrics = binary_metrics(pairs) or macro_metrics(pairs)
        metrics[task_name] = {
            "support": len(pairs),
            **base_metrics,
            "confusion_matrix": confusion_matrix(pairs),
        }

    return metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.run_llm and args.use_rules:
        raise ValueError("Use either --run-llm or --use-rules, not both.")

    if not args.run_llm and not args.use_rules:
        raise ValueError(
            "Choose an inference mode: pass --use-rules for baseline or --run-llm "
            "to allow model loading."
        )

    dataset_path = resolve_dataset(args.dataset)
    rows = read_dataset(dataset_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / DEFAULT_PREDICTIONS_NAME
    metrics_path = output_dir / DEFAULT_METRICS_NAME
    predictions_path.write_text("", encoding="utf-8")

    loaded_model = None
    if args.run_llm:
        try:
            loaded_model = get_cached_model(args.model)
        except ModelConfigError as error:
            print(str(error), file=sys.stderr)
            sys.exit(2)

    metrics_state: dict[str, list[tuple[str, str]]] = defaultdict(list)
    summaries = []

    try:
        for index, row in enumerate(rows):
            case_id = get_case_id(row, index)
            text_columns = None
            try:
                text_columns = get_text_columns(row, args.text_columns)
                mri_text = build_mri_text(row, text_columns)
                if not mri_text:
                    raise ValueError("Empty MRI text.")

                artifacts = run_inference(
                    text=mri_text,
                    model_name=args.model,
                    loaded_model=loaded_model,
                    use_rules=args.use_rules,
                    print_raw_output=args.print_raw_output,
                )
                result = artifacts["result"]
                update_metrics_state(metrics_state, row, result)
                error = None
            except Exception as exc:
                result = None
                artifacts = {}
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

            record = {
                "case_id": case_id,
                "row_index": index,
                "text_columns": text_columns,
                "prediction": result,
                "validated": artifacts.get("validated"),
                "error": error,
                "gold": {
                    field: get_gold_value(row, field)
                    for field in GOLD_FIELDS
                },
            }
            append_jsonl(predictions_path, record)
            summaries.append(
                {
                    "case_id": case_id,
                    "ok": error is None,
                    "error_type": None if error is None else error["type"],
                }
            )
    finally:
        if args.run_llm:
            unload_current_model()

    metrics = {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "total_rows": len(rows),
        "completed": sum(1 for item in summaries if item["ok"]),
        "failed": sum(1 for item in summaries if not item["ok"]),
        "tasks": compute_metrics(metrics_state),
    }
    write_json(metrics_path, metrics)

    print(
        json.dumps(
            {
                "predictions": str(predictions_path),
                "metrics": str(metrics_path),
                "total_rows": len(rows),
                "failed": metrics["failed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
