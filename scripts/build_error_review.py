"""Build PAS20 benchmark error-review artifacts for prompt iteration.

This script does not run inference and does not modify predictions. It compares
saved normalized predictions with gold labels and writes JSONL + Markdown files
that are convenient for extraction prompt review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_utils import (
    CONCLUSION_COLUMNS,
    DESCRIPTION_COLUMNS,
    case_id_for_record,
    find_column,
    index_by_case_id,
    load_records,
    normalize_value,
    resolve_path,
    value_to_text,
)


FIELD_MAP = {
    "pas_type": "gold_pas_type",
    "invasion_type": "gold_invasion_type",
    "invasion_confidence": "gold_invasion_confidence",
    "bladder_involvement": "gold_bladder_involvement",
    "parametrium_involvement": "gold_parametrium_involvement",
    "posterior_wall_involvement": "gold_posterior_wall_involvement",
    "placenta_previa": "gold_placenta_previa",
    "anterior_placenta": "gold_anterior_placenta",
    "retroplacental_vessels": "gold_retroplacental_vessels",
    "lacunae": "gold_lacunae",
    "uterine_wall_thinning": "gold_uterine_wall_thinning",
    "uterine_hernia_or_bulging": "gold_uterine_hernia_or_bulging",
    "preoperative_bleeding": "gold_preoperative_bleeding",
    "highest_suspected_extent": "gold_highest_suspected_extent",
    "percreta_suspicion": "gold_percreta_suspicion",
    "bladder_serosa_suspicion": "gold_bladder_serosa_suspicion",
    "readiness_level": "gold_readiness_level",
    "risk_group": "gold_risk_group",
}

PAS_SEVERITY = {"none": 0, "accreta": 1, "increta": 2, "percreta": 3}
STATUS_SEVERITY = {"absent": 0, "possible": 1, "probable": 2, "present": 3}
RISK_SEVERITY = {"low": 0, "medium": 1, "moderate": 1, "high": 2, "critical": 3}
MRI_SIGN_FIELDS = {
    "lacunae",
    "retroplacental_vessels",
    "uterine_wall_thinning",
    "placenta_previa",
    "anterior_placenta",
    "uterine_hernia_or_bulging",
    "posterior_wall_involvement",
}
EXCERPT_RE = re.compile(
    r"accreta|increta|percreta|врастан|мочев|пузыр|миометр|лакун|"
    r"ретроплацент|выбух|грыж|задн",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PAS20 benchmark error review.")
    parser.add_argument("--cases", required=True, help="Source PAS20 cases JSON/JSONL/CSV.")
    parser.add_argument("--gold", required=True, help="Gold labels JSON/JSONL/CSV.")
    parser.add_argument("--full", required=True, help="Full-mode normalized predictions.")
    parser.add_argument(
        "--description-only",
        required=True,
        help="Description-only normalized predictions.",
    )
    parser.add_argument("--full-raw-dir", required=True, help="Full-mode raw output directory.")
    parser.add_argument(
        "--description-raw-dir",
        required=True,
        help="Description-only raw output directory.",
    )
    parser.add_argument("--output", required=True, help="Error review JSONL output.")
    parser.add_argument("--markdown", required=True, help="Error review Markdown output.")
    return parser.parse_args()


def indexed_cases(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case_id_for_record(record, index): record for index, record in enumerate(records)}


def gold_value(record: dict[str, Any], gold_field: str, field: str) -> str:
    if field == "pas_type":
        return normalize_value(record.get("gold_pas_type") or record.get("gold_invasion_type"))
    return normalize_value(record.get(gold_field))


def prediction_value(record: dict[str, Any] | None, field: str) -> str:
    if not record:
        return ""
    if field == "invasion_confidence":
        return normalize_value(record.get("invasion_confidence") or record.get("confidence"))
    return normalize_value(record.get(field))


def description_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    column = find_column(record, DESCRIPTION_COLUMNS)
    return value_to_text(record.get(column)) if column else ""


def conclusion_text(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    column = find_column(record, CONCLUSION_COLUMNS)
    return value_to_text(record.get(column)) if column else ""


def excerpt(text: str, max_len: int = 900) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    match = EXCERPT_RE.search(text)
    if not match:
        return text[:max_len]
    start = max(0, match.start() - 320)
    end = min(len(text), match.end() + 520)
    return text[start:end][:max_len]


def int_value(value: str) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def severity_for(field: str, gold: str, full: str, desc: str) -> str:
    values = [full, desc]
    if field in {"pas_type", "invasion_type", "percreta_suspicion", "bladder_involvement"}:
        return "critical"
    if field == "readiness_level":
        gold_int = int_value(gold)
        deltas = [abs((int_value(value) or gold_int or 0) - (gold_int or 0)) for value in values]
        return "critical" if max(deltas or [0]) >= 2 else "high"
    if field in {
        "highest_suspected_extent",
        "posterior_wall_involvement",
        "uterine_hernia_or_bulging",
    }:
        return "high"
    if field in {
        "lacunae",
        "retroplacental_vessels",
        "uterine_wall_thinning",
        "placenta_previa",
        "anterior_placenta",
    }:
        return "medium"
    if field in {"risk_group", "invasion_confidence", "preoperative_bleeding"}:
        return "low"
    return "medium"


def representative_prediction(gold: str, full: str, desc: str) -> str:
    if full != gold:
        return full
    return desc


def error_type_for(field: str, gold: str, full: str, desc: str) -> str:
    pred = representative_prediction(gold, full, desc)

    if field in {"pas_type", "invasion_type", "highest_suspected_extent"}:
        gold_level = PAS_SEVERITY.get(gold, 0)
        pred_level = PAS_SEVERITY.get(pred, 0)
        if gold_level > 0 and pred_level == 0:
            return "missed_pas"
        if gold_level == 0 and pred_level > 0:
            return "overcalled_pas"
        if pred_level < gold_level:
            return "underestimated_depth"
        if pred_level > gold_level:
            return "overestimated_depth"

    if field == "percreta_suspicion":
        gold_level = STATUS_SEVERITY.get(gold, 0)
        pred_level = STATUS_SEVERITY.get(pred, 0)
        if pred_level < gold_level:
            return "missed_percreta_suspicion"
        if pred_level > gold_level:
            return "overcalled_percreta_suspicion"

    if field == "bladder_involvement":
        gold_level = STATUS_SEVERITY.get(gold, 0)
        pred_level = STATUS_SEVERITY.get(pred, 0)
        if pred_level < gold_level:
            return "missed_bladder"
        if pred_level > gold_level:
            return "overcalled_bladder"

    if field == "readiness_level":
        gold_level = int_value(gold)
        pred_level = int_value(pred)
        if gold_level is not None and pred_level is not None:
            if pred_level < gold_level:
                return "readiness_underestimated"
            if pred_level > gold_level:
                return "readiness_overestimated"

    if field in MRI_SIGN_FIELDS:
        gold_level = STATUS_SEVERITY.get(gold, 0)
        pred_level = STATUS_SEVERITY.get(pred, 0)
        if pred_level < gold_level:
            return "missed_mri_sign"
        if pred_level > gold_level:
            return "overcalled_mri_sign"

    if field == "risk_group":
        gold_level = RISK_SEVERITY.get(gold, 0)
        pred_level = RISK_SEVERITY.get(pred, 0)
        if pred_level < gold_level:
            return "readiness_underestimated"
        if pred_level > gold_level:
            return "readiness_overestimated"

    return "other"


def raw_path(raw_dir: Path, case_id: str) -> str:
    return str(raw_dir / f"{case_id}_raw.txt")


def build_errors(
    cases: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    desc: dict[str, dict[str, Any]],
    full_raw_dir: Path,
    desc_raw_dir: Path,
) -> list[dict[str, Any]]:
    errors = []
    for case_id in sorted(gold):
        case_record = cases.get(case_id, {})
        gold_record = gold[case_id]
        full_record = full.get(case_id)
        desc_record = desc.get(case_id)
        description = description_text(case_record)
        conclusion = conclusion_text(case_record)

        for field, gold_field in FIELD_MAP.items():
            gold_label = gold_value(gold_record, gold_field, field)
            if not gold_label:
                continue
            full_label = prediction_value(full_record, field)
            desc_label = prediction_value(desc_record, field)
            full_correct = full_label == gold_label
            desc_correct = desc_label == gold_label
            if full_correct and desc_correct:
                continue

            severity = severity_for(field, gold_label, full_label, desc_label)
            errors.append(
                {
                    "case_id": case_id,
                    "field": field,
                    "gold": gold_label,
                    "full": full_label,
                    "description_only": desc_label,
                    "full_correct": full_correct,
                    "description_only_correct": desc_correct,
                    "error_type": error_type_for(field, gold_label, full_label, desc_label),
                    "severity": severity,
                    "mri_description_excerpt": excerpt(description),
                    "mri_conclusion": conclusion,
                    "gold_rationale": value_to_text(gold_record.get("gold_rationale")),
                    "full_raw_path": raw_path(full_raw_dir, case_id),
                    "description_raw_path": raw_path(desc_raw_dir, case_id),
                }
            )
    return errors


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_error(error: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {error['case_id']} / {error['field']}",
            "",
            f"- Уровень значимости: `{error['severity']}`",
            f"- Тип ошибки: `{error['error_type']}`",
            f"- Gold / Full / только описание: `{error['gold']}` / `{error['full']}` / `{error['description_only']}`",
            f"- Full корректен: `{str(error['full_correct']).lower()}`",
            f"- Только описание корректно: `{str(error['description_only_correct']).lower()}`",
            f"- Raw-ответ Full: `{error['full_raw_path']}`",
            f"- Raw-ответ только описание: `{error['description_raw_path']}`",
            "",
            "Обоснование gold:",
            "",
            f"> {error['gold_rationale']}",
            "",
            "МРТ-заключение:",
            "",
            "```text",
            error["mri_conclusion"],
            "```",
            "",
            "Фрагмент МРТ-описания:",
            "",
            "```text",
            error["mri_description_excerpt"],
            "```",
            "",
        ]
    )


def write_markdown(path: Path, errors: list[dict[str, Any]], total_cases: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    severity_counts = Counter(error["severity"] for error in errors)
    field_counts = Counter(error["field"] for error in errors)
    full_correct = sum(1 for error in errors if error["full_correct"])
    desc_correct = sum(1 for error in errors if error["description_only_correct"])

    lines = [
        "# Разбор ошибок PAS20",
        "",
        "## Сводка",
        "",
        f"- всего кейсов: {total_cases}",
        f"- всего несовпадений: {len(errors)}",
        f"- critical: {severity_counts.get('critical', 0)}",
        f"- high: {severity_counts.get('high', 0)}",
        f"- medium: {severity_counts.get('medium', 0)}",
        f"- low: {severity_counts.get('low', 0)}",
        f"- Full корректен в строках с несовпадениями: {full_correct}",
        f"- Только описание корректно в строках с несовпадениями: {desc_correct}",
        "",
        "Несовпадения по полям:",
        "",
    ]
    for field, count in field_counts.most_common():
        lines.append(f"- {field}: {count}")

    lines.extend(["", "## Critical ошибки", ""])
    critical = [error for error in errors if error["severity"] == "critical"]
    lines.extend([markdown_error(error) for error in critical] or ["Critical ошибок нет."])

    lines.extend(["", "## High ошибки", ""])
    high = [error for error in errors if error["severity"] == "high"]
    lines.extend([markdown_error(error) for error in high] or ["High ошибок нет."])

    lines.extend(
        [
            "",
            "## Таблица ошибок по полям",
            "",
            "| Поле | Всего | Critical | High | Medium | Low |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field, count in field_counts.most_common():
        by_severity = Counter(
            error["severity"] for error in errors if error["field"] == field
        )
        lines.append(
            f"| {field} | {count} | {by_severity.get('critical', 0)} | "
            f"{by_severity.get('high', 0)} | {by_severity.get('medium', 0)} | "
            f"{by_severity.get('low', 0)} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases_path = resolve_path(args.cases)
    gold_path = resolve_path(args.gold)
    full_path = resolve_path(args.full)
    desc_path = resolve_path(args.description_only)
    full_raw_dir = resolve_path(args.full_raw_dir)
    desc_raw_dir = resolve_path(args.description_raw_dir)

    cases = indexed_cases(load_records(cases_path))
    gold = index_by_case_id(load_records(gold_path))
    full = index_by_case_id(load_records(full_path))
    desc = index_by_case_id(load_records(desc_path))

    errors = build_errors(cases, gold, full, desc, full_raw_dir, desc_raw_dir)
    write_jsonl(resolve_path(args.output), errors)
    write_markdown(resolve_path(args.markdown), errors, total_cases=len(gold))


if __name__ == "__main__":
    main()
