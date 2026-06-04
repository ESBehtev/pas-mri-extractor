"""Compare full MRI predictions with description-only predictions."""

from __future__ import annotations

import argparse
from collections import Counter

from benchmark_utils import (
    FLAT_FIELDS,
    SEVERITY,
    index_by_case_id,
    load_records,
    resolve_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark prediction modes.")
    parser.add_argument("--full", required=True, help="Normalized full-mode predictions.")
    parser.add_argument(
        "--description-only",
        required=True,
        help="Normalized description-only predictions.",
    )
    parser.add_argument("--output", required=True, help="Mode comparison JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_by_id = index_by_case_id(load_records(resolve_path(args.full)))
    desc_by_id = index_by_case_id(load_records(resolve_path(args.description_only)))
    common_case_ids = sorted(set(full_by_id) & set(desc_by_id))

    case_deltas = []
    field_change_counts: Counter[str] = Counter()
    unchanged_cases = 0
    pas_type_changed = 0
    readiness_changed = 0
    bladder_changed = 0
    highest_changed = 0
    percreta_changed = 0

    for case_id in common_case_ids:
        deltas = []
        changed_count = 0
        for field in FLAT_FIELDS:
            full_value = str(full_by_id[case_id].get(field, ""))
            desc_value = str(desc_by_id[case_id].get(field, ""))
            changed = full_value != desc_value
            if changed:
                changed_count += 1
                field_change_counts[field] += 1
            deltas.append(
                {
                    "field": field,
                    "full": full_value,
                    "description_only": desc_value,
                    "changed": changed,
                }
            )

        if changed_count == 0:
            unchanged_cases += 1
        if full_by_id[case_id].get("pas_type") != desc_by_id[case_id].get("pas_type"):
            pas_type_changed += 1
        if (
            full_by_id[case_id].get("readiness_level")
            != desc_by_id[case_id].get("readiness_level")
        ):
            readiness_changed += 1
        if (
            full_by_id[case_id].get("bladder_involvement")
            != desc_by_id[case_id].get("bladder_involvement")
        ):
            bladder_changed += 1
        if (
            full_by_id[case_id].get("highest_suspected_extent")
            != desc_by_id[case_id].get("highest_suspected_extent")
        ):
            highest_changed += 1
        if (
            full_by_id[case_id].get("percreta_suspicion")
            != desc_by_id[case_id].get("percreta_suspicion")
        ):
            percreta_changed += 1

        case_deltas.append(
            {
                "case_id": case_id,
                "changed_field_count": changed_count,
                "pas_severity_delta": SEVERITY.get(str(desc_by_id[case_id].get("pas_type")), 0)
                - SEVERITY.get(str(full_by_id[case_id].get("pas_type")), 0),
                "field_level_delta": deltas,
            }
        )

    output = {
        "case_count": len(common_case_ids),
        "missing_description_only": sorted(set(full_by_id) - set(desc_by_id)),
        "extra_description_only": sorted(set(desc_by_id) - set(full_by_id)),
        "summary": {
            "field_change_counts": dict(field_change_counts),
            "unchanged_cases": unchanged_cases,
            "pas_type_changed_cases": pas_type_changed,
            "readiness_level_changed_cases": readiness_changed,
            "bladder_involvement_changed_cases": bladder_changed,
            "highest_suspected_extent_changed_cases": highest_changed,
            "percreta_suspicion_changed_cases": percreta_changed,
        },
        "cases": case_deltas,
    }
    write_json(resolve_path(args.output), output)


if __name__ == "__main__":
    main()
