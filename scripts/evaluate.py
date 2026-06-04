"""Evaluate normalized PAS benchmark predictions against gold labels."""

from __future__ import annotations

import argparse

from benchmark_utils import evaluate_records, load_records, normalize_gold, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate normalized predictions vs gold.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL/CSV path.")
    parser.add_argument("--pred", required=True, help="Normalized prediction JSON/JSONL/CSV path.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = [normalize_gold(record) for record in load_records(resolve_path(args.gold))]
    pred_records = load_records(resolve_path(args.pred))
    metrics = evaluate_records(gold_records, pred_records)
    write_json(resolve_path(args.output), metrics)


if __name__ == "__main__":
    main()
