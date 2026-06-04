"""Normalize extractor JSON outputs into a flat benchmark JSON/JSONL table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from benchmark_utils import (
    FLAT_FIELDS,
    normalize_prediction,
    resolve_path,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize PAS benchmark predictions.")
    parser.add_argument(
        "--input",
        required=True,
        help="Directory with case_XXXXX.json files, or JSON/JSONL with extractor payloads.",
    )
    parser.add_argument("--output", required=True, help="Output normalized JSON path.")
    parser.add_argument("--jsonl-output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def case_id_from_path(path: Path) -> str:
    return re.sub(r"_raw$", "", path.stem)


def load_prediction_payloads(path: Path) -> list[tuple[str, dict]]:
    if path.is_dir():
        payloads = []
        for json_path in sorted(path.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as handle:
                payloads.append((case_id_from_path(json_path), json.load(handle)))
        return payloads

    if path.suffix.lower() == ".jsonl":
        payloads = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = str(record.get("case_id") or record.get("id") or "")
                payload = record.get("prediction") if isinstance(record.get("prediction"), dict) else record
                payloads.append((case_id, payload))
        return payloads

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            payloads = []
            for item in data:
                case_id = str(item.get("case_id") or item.get("id") or "")
                payload = item.get("prediction") if isinstance(item.get("prediction"), dict) else item
                payloads.append((case_id, payload))
            return payloads
        if isinstance(data, dict) and isinstance(data.get("predictions"), list):
            payloads = []
            for item in data["predictions"]:
                case_id = str(item.get("case_id") or item.get("id") or "")
                payload = item.get("prediction") if isinstance(item.get("prediction"), dict) else item
                payloads.append((case_id, payload))
            return payloads
        if isinstance(data, dict):
            case_id = str(data.get("case_id") or path.stem)
            return [(case_id, data)]

    raise ValueError(f"Unsupported prediction input: {path}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    records = [
        normalize_prediction(case_id, payload)
        for case_id, payload in load_prediction_payloads(input_path)
    ]

    write_json(output_path, records)
    if args.jsonl_output:
        write_jsonl(resolve_path(args.jsonl_output), records)
    if args.csv_output:
        write_csv(resolve_path(args.csv_output), records, ["case_id", *FLAT_FIELDS])


if __name__ == "__main__":
    main()
