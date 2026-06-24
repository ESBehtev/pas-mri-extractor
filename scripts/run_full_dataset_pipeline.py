"""Batch-прогон полного dataset_sheet3 через текущий PAS pipeline.
Зачем нужен:
- запускать extractor, rule-based scoring и optional LLM risk prediction;
- сохранять original.json и итоговый result.json по каждому case;
- считать summary metrics по blood loss для серверного прогона.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.llm_risk_helpers import (  # noqa: E402
    build_combined_result_json,
    build_extracted_result_for_llm_risk,
    build_rule_based_risk_json,
    strip_debug_artifacts,
)
from pas_mri_extractor.orchestrator import run_risk_prediction_experiment  # noqa: E402
from pas_mri_extractor.pipeline import (  # noqa: E402
    extract_features_with_artifacts,
    get_cached_model,
    unload_current_model,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "dataset_sheet3.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_dataset_run"
DESCRIPTION_FIELD = "МРТ_Описание"
CONCLUSION_FIELD = "МРТ_Заключение"
BIRTH_BLOOD_LOSS_FIELD = "КровопотеряРоды"
OPERATION_BLOOD_LOSS_FIELD = "КровопотеряОперация"
INTERVENTION_TEXT_FIELD = "Ход Вмешательства"
TOTAL_BLOOD_LOSS_RE = re.compile(
    r"общая\s+кровопотеря[^\d]{0,40}(\d+(?:[\s.,]\d+)?)\s*мл",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extractor + rule-based + LLM risk on dataset_sheet3 JSONL.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-case folders and summary.json.",
    )
    parser.add_argument("--model", default="qwen3_6_35b_a3b_gguf")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-llm-risk",
        action="store_true",
        help="Run extractor/rule-based only; skip second LLM risk prediction.",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def case_id_for_record(record: dict[str, Any], index: int) -> str:
    value = record.get("case_id")
    if value is None or str(value).strip() == "":
        return f"case_{index + 1:06d}"
    return str(value).strip()


def build_mri_text(record: dict[str, Any]) -> str:
    description = record.get(DESCRIPTION_FIELD) or ""
    conclusion = record.get(CONCLUSION_FIELD) or ""
    return (
        "Описание:\n"
        f"{description}\n\n"
        "Заключение:\n"
        f"{conclusion}"
    )


def parse_ml_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        match = re.search(r"\d+(?:[\s.,]\d+)?", stripped)
        if not match:
            return None
        numeric = match.group(0).replace(" ", "").replace(",", ".")
        try:
            return int(float(numeric))
        except ValueError:
            return None
    return None


def extract_total_blood_loss_from_text(text: Any) -> int | None:
    if not isinstance(text, str):
        return None
    match = TOTAL_BLOOD_LOSS_RE.search(text)
    if not match:
        return None
    return parse_ml_value(match.group(1))


def extract_actual_blood_loss(record: dict[str, Any]) -> dict[str, Any]:
    birth = parse_ml_value(record.get(BIRTH_BLOOD_LOSS_FIELD))
    operation = parse_ml_value(record.get(OPERATION_BLOOD_LOSS_FIELD))
    actual = {
        "birth_blood_loss_ml": birth,
        "operation_blood_loss_ml": operation,
        "actual_total_blood_loss_ml": None,
        "actual_total_blood_loss_from_text_ml": extract_total_blood_loss_from_text(
            record.get(INTERVENTION_TEXT_FIELD),
        ),
    }
    if birth is not None and operation is not None:
        actual["actual_total_blood_loss_ml"] = birth + operation
    return actual


def stage_status_value(stage_result: Any) -> str | None:
    status = getattr(stage_result, "status", None)
    return getattr(status, "value", status)


def stage_error(stage_result: Any) -> str | None:
    error = getattr(stage_result, "error", None)
    return str(error) if error else None


def make_failed_result(
    case_id: str,
    actual_blood_loss: dict[str, Any],
    errors: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "failed",
        "actual_blood_loss": actual_blood_loss,
        "errors": errors,
        "warnings": warnings or [],
        "metadata": {
            "has_llm_risk": False,
            "llm_risk_status": "failed",
        },
    }


def build_success_result(
    extraction_result: dict[str, Any],
    llm_risk: dict[str, Any] | None,
    llm_risk_status: str,
    actual_blood_loss: dict[str, Any],
    llm_errors: list[str] | None = None,
    llm_warnings: list[str] | None = None,
) -> dict[str, Any]:
    rule_based_risk = build_rule_based_risk_json(extraction_result)
    combined = build_combined_result_json(
        extraction_result=extraction_result,
        rule_based_risk=rule_based_risk,
        llm_risk=llm_risk,
        llm_risk_status=llm_risk_status,
        llm_risk_errors=llm_errors,
        llm_risk_warnings=llm_warnings,
    )
    combined["status"] = "success"
    combined["actual_blood_loss"] = actual_blood_loss
    return strip_debug_artifacts(combined)


def process_case(
    record: dict[str, Any],
    index: int,
    output_dir: Path,
    model_id: str,
    loaded_model: Any,
    run_llm_risk: bool = True,
    overwrite: bool = False,
    skip_existing: bool = False,
    extract_fn: Callable[..., dict[str, Any]] = extract_features_with_artifacts,
    risk_fn: Callable[..., Any] = run_risk_prediction_experiment,
) -> dict[str, Any]:
    case_id = case_id_for_record(record, index)
    case_dir = output_dir / case_id
    original_path = case_dir / "original.json"
    result_path = case_dir / "result.json"

    if skip_existing and result_path.exists():
        return {
            "case_id": case_id,
            "status": "skipped",
            "case_dir": str(case_dir),
            "errors": [],
            "warnings": ["skip-existing: result.json already exists"],
        }

    if overwrite and case_dir.exists():
        shutil.rmtree(case_dir)

    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(original_path, record)

    actual_blood_loss = extract_actual_blood_loss(record)
    warnings: list[str] = []
    errors: list[str] = []

    try:
        text = build_mri_text(record)
        artifacts = extract_fn(
            text=text,
            model_name=model_id,
            loaded_model=loaded_model,
        )
        extraction_result = strip_debug_artifacts(artifacts["result"])

        llm_risk = None
        llm_status = "disabled"
        llm_errors: list[str] = []
        llm_warnings: list[str] = []

        if run_llm_risk:
            extracted_result = build_extracted_result_for_llm_risk(extraction_result)
            llm_stage_result = risk_fn(
                text=text,
                extracted_result=extracted_result,
                model_id=model_id,
                loaded_model=loaded_model,
            )
            llm_status = stage_status_value(llm_stage_result) or "failed"
            llm_warnings = list(getattr(llm_stage_result, "warnings", []) or [])

            if llm_status == "success":
                llm_risk = strip_debug_artifacts(getattr(llm_stage_result, "output", None))
            else:
                llm_error = stage_error(llm_stage_result)
                llm_errors.append("LLMRiskPredictionStage failed")
                if llm_error:
                    llm_errors.append(llm_error)
                errors.extend(llm_errors)

        if errors:
            result = make_failed_result(
                case_id=case_id,
                actual_blood_loss=actual_blood_loss,
                errors=errors,
                warnings=warnings + llm_warnings,
            )
        else:
            result = build_success_result(
                extraction_result=extraction_result,
                llm_risk=llm_risk,
                llm_risk_status=llm_status,
                actual_blood_loss=actual_blood_loss,
                llm_errors=llm_errors,
                llm_warnings=llm_warnings,
            )
            result["case_id"] = case_id
            result["warnings"] = warnings + llm_warnings

        write_json(result_path, result)
        return {
            "case_id": case_id,
            "status": result.get("status", "failed"),
            "case_dir": str(case_dir),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
        }
    except Exception as error:
        result = make_failed_result(
            case_id=case_id,
            actual_blood_loss=actual_blood_loss,
            errors=[str(error)],
            warnings=warnings,
        )
        write_json(result_path, result)
        return {
            "case_id": case_id,
            "status": "failed",
            "case_dir": str(case_dir),
            "errors": result["errors"],
            "warnings": result["warnings"],
        }


def blood_loss_prediction(result: dict[str, Any]) -> int | None:
    llm_risk = result.get("llm_risk") or {}
    risk_assessment = llm_risk.get("risk_assessment") or {}
    return parse_ml_value(risk_assessment.get("estimated_blood_loss_ml"))


def calculate_blood_loss_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[int] = []
    squared_errors: list[int] = []
    ape_values: list[float] = []
    within_250 = 0
    within_500 = 0
    within_1000 = 0

    for result in results:
        if result.get("status") != "success":
            continue
        actual = (result.get("actual_blood_loss") or {}).get(
            "actual_total_blood_loss_ml"
        )
        predicted = blood_loss_prediction(result)
        if actual is None or predicted is None:
            continue

        actual_int = int(actual)
        error = int(predicted) - actual_int
        abs_error = abs(error)
        errors.append(abs_error)
        squared_errors.append(error * error)
        if abs_error <= 250:
            within_250 += 1
        if abs_error <= 500:
            within_500 += 1
        if abs_error <= 1000:
            within_1000 += 1
        if actual_int != 0:
            ape_values.append(abs_error / actual_int * 100)

    n = len(errors)
    return {
        "n": n,
        "mae_ml": sum(errors) / n if n else None,
        "mape_percent": sum(ape_values) / len(ape_values) if ape_values else None,
        "rmse_ml": math.sqrt(sum(squared_errors) / n) if n else None,
        "within_250_ml": within_250 / n if n else None,
        "within_500_ml": within_500 / n if n else None,
        "within_1000_ml": within_1000 / n if n else None,
    }


def load_case_result(case_dir: Path) -> dict[str, Any] | None:
    result_path = case_dir / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    return result if isinstance(result, dict) else None


def calculate_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in case_results if result.get("status") == "success"]
    failed = [result for result in case_results if result.get("status") == "failed"]
    return {
        "n_total": len(case_results),
        "n_success": len(successful),
        "n_failed": len(failed),
        "failed_cases": [
            {
                "case_id": result.get("case_id"),
                "errors": result.get("errors") or [],
                "warnings": result.get("warnings") or [],
            }
            for result in failed
        ],
        "blood_loss_metrics": calculate_blood_loss_metrics(successful),
    }


def selected_records(
    records: list[dict[str, Any]],
    offset: int,
    limit: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    selected = indexed[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    records = load_jsonl(input_path)
    selected = selected_records(records, args.offset, args.limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    shared_model = get_cached_model(args.model)
    progress_records: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    try:
        for index, record in selected:
            case_id = case_id_for_record(record, index)
            print(f"case_id={case_id} | status=running")
            progress = process_case(
                record=record,
                index=index,
                output_dir=output_dir,
                model_id=args.model,
                loaded_model=shared_model,
                run_llm_risk=not args.no_llm_risk,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            )
            progress_records.append(progress)

            if progress["status"] == "skipped":
                print(f"case_id={case_id} | status=skipped")
                existing = load_case_result(output_dir / case_id)
                if existing is not None:
                    case_results.append(existing)
                continue

            result = load_case_result(output_dir / case_id)
            if result is not None:
                case_results.append(result)
            print(f"case_id={case_id} | status={progress['status']}")
            for error in progress.get("errors") or []:
                print(f"  error: {error}")
            for warning in progress.get("warnings") or []:
                print(f"  warning: {warning}")
    finally:
        unload_current_model()

    summary = calculate_summary(case_results)
    write_json(output_dir / "summary.json", summary)
    print("=== SUMMARY ===")
    print(f"n_total: {summary['n_total']}")
    print(f"n_success: {summary['n_success']}")
    print(f"n_failed: {summary['n_failed']}")
    print(f"blood_loss_metrics: {summary['blood_loss_metrics']}")


if __name__ == "__main__":
    main()
