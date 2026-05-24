"""
Batch evaluation CLI for MRI extraction.

Reads a YAML eval config, runs model/rules/fixture inference, validates the
canonical extraction schema, and stores per-case artifacts.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pas_mri_extractor.extractor import postprocess_extraction
from pas_mri_extractor.json_utils import extract_json_object
from pas_mri_extractor.models import LoadedModel, ModelConfigError, PROJECT_ROOT
from pas_mri_extractor.pipeline import (
    extract_features_with_artifacts,
    get_cached_model,
)
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.schemas import MRIExtractionResult
from pas_mri_extractor.scoring import normalize_mri_result


CASE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch PAS MRI extraction eval from a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to eval YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from YAML.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from YAML.",
    )
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Use regex rules instead of LLM for all cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases.",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 if any case has validation errors or expected diffs.",
    )
    return parser.parse_args()


def load_eval_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Eval config must be a YAML mapping.")

    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Eval config must define a non-empty 'cases' list.")

    return config


def resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    config_relative = config_path.parent / path
    if config_relative.exists():
        return config_relative

    return PROJECT_ROOT / path


def read_case_text(case: dict[str, Any], config_path: Path, case_id: str) -> str:
    has_text = "text" in case and case.get("text") is not None
    has_text_file = "text_file" in case and case.get("text_file")

    if has_text and has_text_file:
        raise ValueError(
            f"Case {case_id}: specify either 'text' or 'text_file', not both."
        )

    if has_text:
        return str(case["text"])

    if has_text_file:
        text_path = resolve_path(str(case["text_file"]), config_path)
        with text_path.open("r", encoding="utf-8") as file:
            return file.read()

    return ""


def safe_case_id(value: object, index: int) -> str:
    raw_id = str(value or f"case_{index + 1:03d}").strip()
    safe_id = CASE_ID_PATTERN.sub("_", raw_id).strip("._-")
    return safe_id or f"case_{index + 1:03d}"


def make_output_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or config.get("output_dir") or "outputs/eval"
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    run_name = config.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    return output_path / str(run_name)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def diff_expected(
    expected: Any,
    actual: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    if isinstance(expected, dict):
        differences: list[dict[str, Any]] = []
        if not isinstance(actual, dict):
            return [
                {
                    "path": path or "$",
                    "expected": expected,
                    "actual": actual,
                }
            ]

        for key, expected_value in expected.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                differences.append(
                    {
                        "path": child_path,
                        "expected": expected_value,
                        "actual": "<missing>",
                    }
                )
                continue
            differences.extend(
                diff_expected(expected_value, actual[key], child_path)
            )

        return differences

    if expected != actual:
        return [
            {
                "path": path or "$",
                "expected": expected,
                "actual": actual,
            }
        ]

    return []


def parse_fixture_output(raw_output: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_output)
    parsed = postprocess_extraction(parsed)
    validated = MRIExtractionResult.model_validate(parsed)
    full_result = normalize_mri_result(validated)

    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "validated": validated.model_dump(),
        "result": full_result.model_dump(),
    }


def run_rules_inference(text: str) -> dict[str, Any]:
    validated = rule_extract_features(text)
    parsed = validated.model_dump()
    full_result = normalize_mri_result(validated)

    return {
        "raw_output": json.dumps(parsed, ensure_ascii=False, indent=2),
        "parsed": parsed,
        "validated": parsed,
        "result": full_result.model_dump(),
    }


def evaluate_case(
    case: dict[str, Any],
    case_id: str,
    config_path: Path,
    model_name: str | None,
    loaded_model: LoadedModel | None,
    use_rules: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = case.get("expected", {})
    if expected is None:
        expected = {}
    if not isinstance(expected, dict):
        raise ValueError(f"Case {case_id}: 'expected' must be a mapping if provided.")

    artifacts: dict[str, Any] = {}
    differences: list[dict[str, Any]] = []
    error = None
    schema_valid = False

    try:
        if case.get("raw_output") is not None:
            artifacts = parse_fixture_output(str(case["raw_output"]))
        else:
            text = read_case_text(case, config_path, case_id)
            if not text.strip():
                raise ValueError(f"Case {case_id}: empty MRI text.")

            if use_rules:
                artifacts = run_rules_inference(text)
            else:
                artifacts = extract_features_with_artifacts(
                    text=text,
                    model_name=model_name,
                    loaded_model=loaded_model,
                )

        MRIExtractionResult.model_validate(artifacts["validated"])
        schema_valid = True
        differences = diff_expected(expected, artifacts["result"])
    except (ValidationError, ValueError, RuntimeError) as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    ok = error is None and schema_valid and not differences
    summary = {
        "case_id": case_id,
        "ok": ok,
        "schema_valid": schema_valid,
        "diff_count": len(differences),
        "error_type": None if error is None else error["type"],
        "error_message": None if error is None else error["message"],
    }

    details = {
        "case_id": case_id,
        "summary": summary,
        "expected": expected,
        "differences": differences,
        "error": error,
        "artifacts": artifacts,
    }

    return summary, details


def write_case_artifacts(
    output_dir: Path,
    case_id: str,
    details: dict[str, Any],
) -> None:
    artifacts = details.get("artifacts") or {}

    write_text(
        output_dir / "raw" / f"{case_id}.txt",
        str(artifacts.get("raw_output", "")),
    )
    write_json(
        output_dir / "parsed" / f"{case_id}.json",
        {
            "case_id": case_id,
            "parsed": artifacts.get("parsed"),
            "validated": artifacts.get("validated"),
            "result": artifacts.get("result"),
            "error": details.get("error"),
        },
    )
    write_json(
        output_dir / "diff" / f"{case_id}.json",
        {
            "case_id": case_id,
            "expected": details.get("expected", {}),
            "differences": details.get("differences", []),
            "error": details.get("error"),
        },
    )


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    total = len(summaries)
    ok_count = sum(1 for item in summaries if item["ok"])
    failed_count = total - ok_count

    payload = {
        "total": total,
        "ok": ok_count,
        "failed": failed_count,
        "cases": summaries,
    }
    write_json(output_dir / "summary.json", payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "ok",
                "schema_valid",
                "diff_count",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)


def needs_model(cases: list[dict[str, Any]], use_rules: bool) -> bool:
    if use_rules:
        return False

    return any(case.get("raw_output") is None for case in cases)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_eval_config(config_path)

    cases = config["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]

    use_rules = bool(args.use_rules or config.get("use_rules", False))
    model_name = args.model or config.get("model")
    loaded_model = None

    try:
        if needs_model(cases, use_rules):
            loaded_model = get_cached_model(model_name)
    except ModelConfigError as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)

    output_dir = make_output_dir(config, args)
    summaries = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index + 1} must be a mapping.")

        case_id = safe_case_id(case.get("id"), index)
        summary, details = evaluate_case(
            case=case,
            case_id=case_id,
            config_path=config_path,
            model_name=model_name,
            loaded_model=loaded_model,
            use_rules=use_rules,
        )
        write_case_artifacts(output_dir, case_id, details)
        summaries.append(summary)

    write_summary(output_dir, summaries)

    failed_count = sum(1 for item in summaries if not item["ok"])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total": len(summaries),
                "failed": failed_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.fail_on_diff and failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
