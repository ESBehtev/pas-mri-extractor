"""
Evaluate experimental LLMRiskPredictionStage against gold JSONL outcomes.

This script is intended for server-side terminal runs. Use --dry-run locally to
verify JSONL reading and field mapping without loading an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pas_mri_extractor.orchestrator import (  # noqa: E402
    run_case_pipeline,
    run_risk_prediction_experiment,
)
from pas_mri_extractor.pipeline import get_cached_model, unload_current_model  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "pas20.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "llm_risk_eval.jsonl"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "llm_risk_eval_summary.json"

MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "-"}
TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "да",
    "present",
    "positive",
    "probable",
}
FALSE_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "нет",
    "absent",
    "negative",
}

CASE_ID_FIELDS = ["case_id", "id", "case", "record_id", "номер", "№"]
TEXT_FIELDS = [
    "text",
    "report_text",
    "mri_text",
    "MRI_text",
    "MRI_description",
    "mri_description",
    "МРТ_Описание",
    "МРТ Описание",
    "Описание МРТ",
    "Описание",
    "MRI_conclusion",
    "mri_conclusion",
    "МРТ_Заключение",
    "МРТ Заключение",
    "Заключение МРТ",
    "Заключение",
]

ACTUAL_FIELDS = {
    "blood_loss_ml": [
        "blood_loss_ml",
        "gold_blood_loss_ml",
        "actual_blood_loss_ml",
        "Кровопотеря",
        "КровопотеряРоды",
        "КровопотеряОперация",
    ],
    "transfusion": [
        "transfusion",
        "gold_transfusion",
        "actual_transfusion",
        "blood_transfusion",
        "переливание",
        "гемотрансфузия",
    ],
    "hysterectomy": [
        "hysterectomy",
        "gold_hysterectomy",
        "actual_hysterectomy",
        "гистерэктомия",
        "экстирпация",
    ],
    "bladder_involvement": [
        "bladder_involvement",
        "bladder_injury",
        "gold_bladder_involvement",
        "actual_bladder_involvement",
        "повреждение_мочевого_пузыря",
    ],
    "vascular_intervention": [
        "vascular_intervention",
        "vascular_stage",
        "gold_vascular_intervention",
        "actual_vascular_intervention",
        "сосудистый_этап",
        "эмболизация",
    ],
    "final_pas": [
        "final_pas",
        "pas_type",
        "gold_pas_type",
        "gold_invasion_type",
        "actual_pas_type",
    ],
    "readiness_level": [
        "readiness_level",
        "gold_readiness_level",
        "actual_readiness_level",
    ],
}

BINARY_TASKS = {
    "transfusion": "transfusion_risk_percent",
    "hysterectomy": "hysterectomy_risk_percent",
    "vascular_intervention": "vascular_intervention_risk_percent",
    "bladder_involvement": "bladder_involvement_risk_percent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate experimental LLM PAS risk prediction on JSONL.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL.")
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Summary metrics JSON.",
    )
    parser.add_argument("--model", default="qwen3_6_35b_a3b_gguf")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--text-field", default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and map fields only. Does not load an LLM.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    return False


def clean_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def coerce_int(value: Any) -> int | None:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None
    return None


def coerce_bool(value: Any) -> bool | None:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
    return None


def nested_actual_sources(record: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [record]
    for key in ["actual", "actual_outcome", "gold", "outcome"]:
        value = record.get(key)
        if isinstance(value, dict):
            sources.append(value)
        elif isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                sources.append(parsed)
    return sources


def first_present(
    sources: list[dict[str, Any]],
    field_names: list[str],
) -> tuple[Any, str | None]:
    for source in sources:
        for field_name in field_names:
            if field_name in source and not is_missing(source[field_name]):
                return source[field_name], field_name
    return None, None


def extract_text(
    record: dict[str, Any],
    text_field: str,
    warnings: list[str],
) -> str | None:
    if text_field != "auto":
        value = clean_value(record.get(text_field))
        if value is None:
            warnings.append(f"text field not found or empty: {text_field}")
            return None
        return str(value)

    value, field_name = first_present([record], TEXT_FIELDS)
    if value is None:
        warnings.append("MRI text field not found")
        return None
    if field_name not in {"text", "report_text", "mri_text", "MRI_text"}:
        warnings.append(f"MRI text mapped from field: {field_name}")
    return str(clean_value(value))


def extract_case_fields(
    record: dict[str, Any],
    index: int,
    text_field: str = "auto",
) -> tuple[str, str | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    sources = nested_actual_sources(record)

    case_id_value, case_id_field = first_present(sources, CASE_ID_FIELDS)
    case_id = str(clean_value(case_id_value) or f"case_{index + 1:06d}")
    if case_id_field is None:
        warnings.append("case_id not found; generated from row index")

    text = extract_text(record, text_field, warnings)

    actual: dict[str, Any] = {}
    for output_field, field_names in ACTUAL_FIELDS.items():
        value, field_name = first_present(sources, field_names)
        if output_field == "blood_loss_ml":
            actual[output_field] = coerce_int(value)
        elif output_field in BINARY_TASKS:
            actual[output_field] = coerce_bool(value)
        else:
            actual[output_field] = clean_value(value)

        if field_name is None:
            warnings.append(f"actual field not found: {output_field}")

    return case_id, text, actual, warnings


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


def stage_by_name(stage_results: list[Any], stage_name: str) -> Any | None:
    for result in stage_results:
        if getattr(result, "stage_name", None) == stage_name:
            return result
    return None


def strip_debug(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_debug(item)
            for key, item in value.items()
            if key != "debug_artifacts"
        }
    if isinstance(value, list):
        return [strip_debug(item) for item in value]
    return value


def process_record(
    record: dict[str, Any],
    index: int,
    model_id: str,
    text_field: str,
    dry_run: bool,
    shared_model: Any = None,
    run_case_pipeline_fn: Callable[..., list[Any]] = run_case_pipeline,
    run_risk_prediction_fn: Callable[..., Any] = run_risk_prediction_experiment,
) -> dict[str, Any]:
    case_id, text, actual, warnings = extract_case_fields(record, index, text_field)
    output = {
        "case_id": case_id,
        "status": "success",
        "actual": actual,
        "rule_based": None,
        "llm_risk": None,
        "errors": [],
        "warnings": warnings,
    }

    if text is None:
        output["status"] = "failed"
        output["errors"].append("Missing MRI text")
        return output

    if dry_run:
        output["warnings"].append("dry-run: LLM inference skipped")
        return output

    try:
        stage_results = run_case_pipeline_fn(text, model_id)
        extractor_result = stage_by_name(stage_results, "ExtractorStage")
        risk_result = stage_by_name(stage_results, "RiskPredictionStage")

        if extractor_result is None or extractor_result.status.value != "success":
            output["status"] = "failed"
            output["errors"].append("ExtractorStage failed")
            if extractor_result is not None and extractor_result.error:
                output["errors"].append(extractor_result.error)
            return output

        extracted_result = extractor_result.output.get("extracted_result")
        if risk_result is not None and risk_result.status.value == "success":
            output["rule_based"] = strip_debug(risk_result.output)
        elif risk_result is not None:
            output["warnings"].append("RiskPredictionStage did not succeed")

        llm_result = run_risk_prediction_fn(
            text=text,
            extracted_result=extracted_result,
            model_id=model_id,
            loaded_model=shared_model,
        )
        if llm_result.status.value != "success":
            output["status"] = "failed"
            output["errors"].append("LLMRiskPredictionStage failed")
            if llm_result.error:
                output["errors"].append(llm_result.error)
            return output

        output["llm_risk"] = strip_debug(llm_result.output)
        return output
    except Exception as error:
        output["status"] = "failed"
        output["errors"].append(str(error))
        return output


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(1 for actual, predicted in pairs if actual and predicted)
    tn = sum(1 for actual, predicted in pairs if not actual and not predicted)
    fp = sum(1 for actual, predicted in pairs if not actual and predicted)
    fn = sum(1 for actual, predicted in pairs if actual and not predicted)
    total = len(pairs)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    blood_loss_errors: list[int] = []
    readiness_pairs: list[tuple[str, str]] = []
    binary_pairs: dict[str, list[tuple[bool, bool]]] = {
        task_name: [] for task_name in BINARY_TASKS
    }

    for record in successful:
        actual = record.get("actual") or {}
        llm_risk = record.get("llm_risk") or {}
        risk_assessment = llm_risk.get("risk_assessment") or {}
        readiness = llm_risk.get("readiness") or {}

        actual_blood_loss = actual.get("blood_loss_ml")
        predicted_blood_loss = risk_assessment.get("estimated_blood_loss_ml")
        if actual_blood_loss is not None and predicted_blood_loss is not None:
            blood_loss_errors.append(abs(actual_blood_loss - predicted_blood_loss))

        actual_readiness = actual.get("readiness_level")
        predicted_readiness = readiness.get("level")
        if actual_readiness is not None and predicted_readiness is not None:
            readiness_pairs.append((str(actual_readiness), str(predicted_readiness)))

        for task_name, probability_field in BINARY_TASKS.items():
            actual_value = actual.get(task_name)
            probability = risk_assessment.get(probability_field)
            if actual_value is None or probability is None:
                continue
            binary_pairs[task_name].append((bool(actual_value), int(probability) >= 50))

    readiness_matches = sum(1 for actual, predicted in readiness_pairs if actual == predicted)

    return {
        "n_total": len(records),
        "n_success": len(successful),
        "n_failed": len(records) - len(successful),
        "blood_loss_mae_ml": (
            sum(blood_loss_errors) / len(blood_loss_errors)
            if blood_loss_errors
            else None
        ),
        "blood_loss_mae_n": len(blood_loss_errors),
        "readiness_exact_match": (
            readiness_matches / len(readiness_pairs) if readiness_pairs else None
        ),
        "readiness_exact_match_n": len(readiness_pairs),
        "binary_metrics": {
            task_name: binary_metrics(pairs)
            for task_name, pairs in binary_pairs.items()
        },
    }


def print_case_summary(record: dict[str, Any]) -> None:
    actual = record.get("actual") or {}
    llm_risk = record.get("llm_risk") or {}
    risk_assessment = llm_risk.get("risk_assessment") or {}
    readiness = llm_risk.get("readiness") or {}

    actual_blood_loss = actual.get("blood_loss_ml")
    predicted_blood_loss = risk_assessment.get("estimated_blood_loss_ml")
    absolute_error = (
        abs(actual_blood_loss - predicted_blood_loss)
        if actual_blood_loss is not None and predicted_blood_loss is not None
        else None
    )

    def pred_bool(field: str) -> bool | None:
        value = risk_assessment.get(field)
        return None if value is None else int(value) >= 50

    print(
        " | ".join(
            [
                f"case_id={record.get('case_id')}",
                f"actual_blood_loss={actual_blood_loss}",
                f"pred_blood_loss={predicted_blood_loss}",
                f"abs_error={absolute_error}",
                f"transfusion={actual.get('transfusion')}/"
                f"{pred_bool('transfusion_risk_percent')}",
                f"hysterectomy={actual.get('hysterectomy')}/"
                f"{pred_bool('hysterectomy_risk_percent')}",
                f"vascular={actual.get('vascular_intervention')}/"
                f"{pred_bool('vascular_intervention_risk_percent')}",
                f"bladder={actual.get('bladder_involvement')}/"
                f"{pred_bool('bladder_involvement_risk_percent')}",
                f"readiness={actual.get('readiness_level')}/"
                f"{readiness.get('level')}",
                f"status={record.get('status')}",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)

    records = load_jsonl(input_path)
    selected = records[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]

    shared_model = None
    if not args.dry_run:
        shared_model = get_cached_model(args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated: list[dict[str, Any]] = []
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(selected, start=args.offset):
                evaluated_record = process_record(
                    record=record,
                    index=index,
                    model_id=args.model,
                    text_field=args.text_field,
                    dry_run=args.dry_run,
                    shared_model=shared_model,
                )
                evaluated.append(evaluated_record)
                print_case_summary(evaluated_record)
                json.dump(evaluated_record, handle, ensure_ascii=False)
                handle.write("\n")
    finally:
        if not args.dry_run:
            unload_current_model()

    summary = calculate_summary(evaluated)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
