"""Build presentation Markdown reports from existing benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_utils import resolve_path


KEY_METRICS = [
    ("Макро-F1 МРТ-признаков", ("mri_feature_extraction_quality", "aggregate", "macro_f1")),
    ("F1 выявления PAS", ("pas_detection", "pas_type", "f1")),
    ("F1 advanced PAS", ("advanced_pas_detection", "pas_type", "f1")),
    ("F1 percreta", ("percreta_detection", "pas_type", "f1")),
    ("Accuracy типа PAS", ("clinical_risk_outputs", "pas_type", "accuracy")),
    ("Accuracy уровня готовности", ("clinical_risk_outputs", "readiness_level", "accuracy")),
    ("Accuracy группы риска", ("clinical_risk_outputs", "risk_group", "accuracy")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PAS20 presentation reports.")
    parser.add_argument("--metrics", required=True, help="presentation_metrics.json path.")
    parser.add_argument("--error-review", required=True, help="error_review.jsonl path.")
    parser.add_argument("--output-dir", required=True, help="Report output directory.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nested(data: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = data
    for key in path:
        current = current[key]
    return float(current)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def key_metric_rows(metrics: dict[str, Any]) -> list[str]:
    rows = [
        "| Метрика | Full | Только описание | Разница |",
        "|---|---:|---:|---:|",
    ]
    for label, path in KEY_METRICS:
        full = nested(metrics["full"], path)
        desc = nested(metrics["description_only"], path)
        rows.append(f"| {label} | {fmt(full)} | {fmt(desc)} | {fmt(desc - full)} |")
    return rows


def executive_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Итоги PAS20 benchmark",
            "",
            "## Датасет",
            "",
            "- 20 вручную проверенных PAS-кейсов.",
            "- Два режима: полный МРТ-текст и только описание без заключения.",
            "",
            "## Ключевые количественные результаты",
            "",
            *key_metric_rows(metrics),
            "",
            "## Основные выводы",
            "",
            "- Извлечение МРТ-признаков работает устойчиво.",
            "- Классификация хирургического исхода PAS слабее, чем извлечение признаков.",
            "- Режим только описания может быть сопоставим с Full или лучше по части метрик.",
            "- Текст заключения в некоторых кейсах может подавлять подозрительные признаки из описания.",
            "",
            "## Ограничения",
            "",
            "- Только 20 кейсов.",
            "- Gold-разметка отражает хирургическую/клиническую истину, а не чисто МРТ-истину.",
            "- Разделение accreta и increta по одному МРТ-тексту принципиально сложное.",
            "- `readiness_level` и `risk_group` являются outcome-like метками и не должны интерпретироваться как чистое качество extraction.",
            "",
            "## Рекомендуемый следующий шаг",
            "",
            "- Двухэтапный пайплайн: МРТ-текст -> структурированные МРТ-признаки -> predictor хирургического/клинического исхода.",
            "",
        ]
    )


def management_summary(metrics: dict[str, Any]) -> str:
    mri_full = nested(metrics["full"], ("mri_feature_extraction_quality", "aggregate", "macro_f1"))
    mri_desc = nested(
        metrics["description_only"],
        ("mri_feature_extraction_quality", "aggregate", "macro_f1"),
    )
    pas_full = nested(metrics["full"], ("pas_detection", "pas_type", "f1"))
    pas_desc = nested(metrics["description_only"], ("pas_detection", "pas_type", "f1"))
    return "\n".join(
        [
            "# Краткое резюме PAS20 benchmark",
            "",
            "## Цель",
            "",
            "Оценить качество извлечения PAS-признаков новой моделью Qwen35B в двух режимах: полный МРТ-текст и только описание.",
            "",
            "## Данные",
            "",
            "Использован набор из 20 вручную разобранных PAS-кейсов с gold-разметкой по клиническим и операционным данным.",
            "",
            "## Основные результаты",
            "",
            f"- Макро-F1 МРТ-признаков: Full {fmt(mri_full)}, только описание {fmt(mri_desc)}.",
            f"- F1 выявления PAS: Full {fmt(pas_full)}, только описание {fmt(pas_desc)}.",
            "",
            "## Что работает хорошо",
            "",
            "Модель уверенно извлекает многие МРТ-признаки из описательной части исследования.",
            "",
            "## Где ограничения",
            "",
            "Классификация хирургического исхода PAS и уровней готовности менее стабильна, потому что gold отражает клиническую истину, а не только МРТ-картину.",
            "",
            "## Следующий этап",
            "",
            "Разделить пайплайн на два этапа: извлечение МРТ-признаков и отдельное прогнозирование клинического/операционного исхода.",
            "",
        ]
    )


def summarize_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    severity = Counter(error.get("severity", "") for error in errors)
    fields = Counter(error.get("field", "") for error in errors)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in errors:
        by_case[error.get("case_id", "")].append(error)
    critical_cases = sorted(
        by_case.items(),
        key=lambda item: sum(1 for error in item[1] if error.get("severity") == "critical"),
        reverse=True,
    )
    high_cases = sorted(
        by_case.items(),
        key=lambda item: sum(1 for error in item[1] if error.get("severity") == "high"),
        reverse=True,
    )
    return {
        "severity": severity,
        "fields": fields,
        "critical_cases": critical_cases,
        "high_cases": high_cases,
    }


def case_line(case_id: str, errors: list[dict[str, Any]], severity: str) -> str:
    selected = [error for error in errors if error.get("severity") == severity]
    fields = ", ".join(error.get("field", "") for error in selected[:8])
    return f"- {case_id}: {len(selected)} ошибок уровня {severity} ({fields})"


def error_analysis_summary(errors: list[dict[str, Any]]) -> str:
    summary = summarize_errors(errors)
    severity = summary["severity"]
    fields = summary["fields"]
    critical_cases = [
        (case_id, case_errors)
        for case_id, case_errors in summary["critical_cases"]
        if any(error.get("severity") == "critical" for error in case_errors)
    ][:5]
    high_cases = [
        (case_id, case_errors)
        for case_id, case_errors in summary["high_cases"]
        if any(error.get("severity") == "high" for error in case_errors)
    ][:5]

    lines = [
        "# Сводка анализа ошибок PAS20",
        "",
        "## Количество ошибок",
        "",
        f"- всего несовпадений: {len(errors)}",
        f"- critical: {severity.get('critical', 0)}",
        f"- high: {severity.get('high', 0)}",
        f"- medium: {severity.get('medium', 0)}",
        f"- low: {severity.get('low', 0)}",
        "",
        "## Поля с наибольшим числом несовпадений",
        "",
    ]
    lines.extend([f"- {field}: {count}" for field, count in fields.most_common(10)])
    lines.extend(["", "## Топ-5 critical кейсов", ""])
    lines.extend([case_line(case_id, case_errors, "critical") for case_id, case_errors in critical_cases] or ["- None"])
    lines[-1:] = ["- Нет"] if lines[-1:] == ["- None"] else lines[-1:]
    lines.extend(["", "## Топ-5 high кейсов", ""])
    lines.extend([case_line(case_id, case_errors, "high") for case_id, case_errors in high_cases] or ["- None"])
    lines[-1:] = ["- Нет"] if lines[-1:] == ["- None"] else lines[-1:]
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- Недооценка между accreta и increta остаётся важным источником клинически значимых ошибок.",
            "- `highest_suspected_extent` часто пропускается или занижается при неопределённых формулировках.",
            "- `preoperative_bleeding` извлекается нестабильно из доступного текста.",
            "- `posterior_wall_involvement` пропускается в части кейсов; этот признак стоит сделать более явным в prompt.",
            "- `readiness_level` и `risk_group` нестабильны, потому что смешивают MRI extraction с outcome-like допущениями.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics = load_json(resolve_path(args.metrics))
    errors = load_jsonl(resolve_path(args.error_review))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "executive_summary.md").write_text(
        executive_summary(metrics),
        encoding="utf-8",
    )
    (output_dir / "management_summary.md").write_text(
        management_summary(metrics),
        encoding="utf-8",
    )
    (output_dir / "error_analysis_summary.md").write_text(
        error_analysis_summary(errors),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
