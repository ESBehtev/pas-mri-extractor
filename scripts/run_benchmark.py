"""
Run extractor inference for PAS benchmark cases.

This script loads the configured LLM only when --run-llm is passed. It is meant
to be executed on the runtime server, not during local code-editing sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_utils import build_mri_text, case_id_for_record, load_records, resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pas_mri_extractor.pipeline import (  # noqa: E402
    extract_features_with_artifacts,
    get_cached_model,
    unload_current_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAS MRI benchmark inference.")
    parser.add_argument("--input", required=True, help="Input cases JSON/JSONL/CSV.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full", "description_only"],
        help="full uses MRI description + conclusion; description_only excludes conclusion.",
    )
    parser.add_argument("--output", required=True, help="Output directory for raw/json files.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name from configs/models.yaml. Defaults to configured default_model.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Required guard flag that allows model loading and inference.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N limit.")
    parser.add_argument(
        "--print-raw-output",
        action="store_true",
        help="Forward raw model output to stderr during extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_llm:
        raise SystemExit(
            "Refusing to load an LLM without --run-llm. "
            "Run this script on the server with --run-llm when ready."
        )

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    loaded_model = get_cached_model(args.model)
    try:
        for index, record in enumerate(records):
            case_id = case_id_for_record(record, index)
            mri_text = build_mri_text(record, args.mode)
            if not mri_text.strip():
                raise ValueError(f"{case_id}: empty MRI text for mode={args.mode}")

            artifacts = extract_features_with_artifacts(
                text=mri_text,
                loaded_model=loaded_model,
                print_raw_output=args.print_raw_output,
            )
            raw_text = artifacts.get("raw_output", "")
            result = artifacts.get("result", {})

            (output_dir / f"{case_id}_raw.txt").write_text(raw_text, encoding="utf-8")
            with (output_dir / f"{case_id}.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
    finally:
        unload_current_model()


if __name__ == "__main__":
    main()
