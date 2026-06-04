"""
Build retrospective gold labels from the exported clinical evaluation dataset.

This is a deterministic annotation helper for scientific evaluation, not a
clinical conclusion. It reads data/evaluation/dataset_sheet3.csv and writes new
gold files without changing the source dataset.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from pas_mri_extractor.models import PROJECT_ROOT


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "dataset_sheet3.csv"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "dataset_sheet3_gold.csv"
DEFAULT_JSONL_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "dataset_sheet3_gold.jsonl"
GOLD_LABELING_PROTOCOL = PROJECT_ROOT / "GOLD_LABELING.md"

SOURCE_COLUMNS = {
    "mri_description": ["МРТ_Описание", "МРТ Описание", "Описание МРТ"],
    "mri_conclusion": ["МРТ_Заключение", "МРТ Заключение", "Заключение МРТ"],
    "diagnoses": ["ДиагнозыВыпЭпикриза", "Диагнозы Вып Эпикриза"],
    "operation_indications": ["ПоказанияКОперации", "Показания К Операции"],
    "operation_course": ["Ход Вмешательства", "ХодВмешательства"],
    "blood_loss_delivery": ["КровопотеряРоды", "Кровопотеря Роды"],
    "blood_loss_operation": ["КровопотеряОперация", "Кровопотеря Операция"],
}

CASE_ID_CANDIDATES = ["case_id", "id", "case", "номер", "номер_случая", "№"]

GOLD_FIELDS = [
    "gold_invasion_type",
    "gold_invasion_confidence",
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
    "gold_readiness_level",
    "gold_risk_group",
    "gold_blood_loss_ml",
    "gold_massive_blood_loss",
    "gold_blood_loss_class",
    "gold_vascular_intervention",
    "gold_pas_type",
    "gold_confidence",
    "gold_rationale",
]

UNCERTAIN_RE = re.compile(
    r"нельзя\s+исключ|не\s+исключ|возможн|сомнитель|подозр|под\s+вопрос",
    re.IGNORECASE,
)
PROBABLE_RE = re.compile(
    r"соответствует|наиболее\s+соответствует|вероятн|типичн",
    re.IGNORECASE,
)
DEFINITE_RE = re.compile(
    r"достовер|убедитель|выявлен|определя|прорастан",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"не\s+выяв|не\s+определ|нет|без\s+признак|без\s+убедительн|"
    r"без\s+достоверн|данн\w*\s+за.{0,80}нет",
    re.IGNORECASE,
)
LATIN_CONFUSABLES = str.maketrans(
    {
        "а": "a",
        "А": "A",
        "е": "e",
        "Е": "E",
        "о": "o",
        "О": "O",
        "р": "p",
        "Р": "P",
        "с": "c",
        "С": "C",
        "х": "x",
        "Х": "X",
        "у": "y",
        "У": "Y",
    },
)
PAS_SEVERITY = {"none": 0, "accreta": 1, "increta": 2, "percreta": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create gold-label CSV/JSONL files from dataset_sheet3.csv.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Source CSV path. The file is read-only.",
    )
    parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_CSV_OUTPUT),
        help="Output gold CSV path.",
    )
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help="Output gold JSONL path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of random annotated cases printed for manual review.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for review sample selection.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", "_", text)


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_map = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]
    return None


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
        return ""
    return str(value).strip()


def get_column_value(row: pd.Series, aliases: list[str]) -> str:
    column = find_column(list(row.index), aliases)
    if column is None:
        return ""
    return text_value(row.get(column))


def get_case_id(row: pd.Series, index: int) -> str:
    case_id = get_column_value(row, CASE_ID_CANDIDATES)
    return case_id or f"case_{index + 1:06d}"


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def normalize_latin_tokens(text: str) -> str:
    return normalize_text(text).translate(LATIN_CONFUSABLES)


def source_texts(row: pd.Series) -> dict[str, str]:
    return {
        name: get_column_value(row, aliases)
        for name, aliases in SOURCE_COLUMNS.items()
    }


def join_sources(sources: dict[str, str], names: list[str]) -> str:
    return "\n".join(sources[name] for name in names if sources.get(name))


def compact_snippet(text: str, max_len: int = 140) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= max_len:
        return snippet
    return f"{snippet[: max_len - 1]}..."


def first_context(text: str, pattern: str, window: int = 70) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return compact_snippet(text[start:end])


def first_context_from_match(text: str, start: int, end: int, window: int = 70) -> str:
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    return compact_snippet(text[context_start:context_end])


def add_rationale(rationale: list[str], field_name: str, text: str) -> None:
    snippet = compact_snippet(text)
    if snippet:
        rationale.append(f"{field_name}: {snippet}")


def status_from_patterns(
    text: str,
    positive_patterns: list[str],
    negative_patterns: list[str],
    probable_patterns: list[str] | None = None,
) -> tuple[str, str]:
    if not text.strip():
        return "", ""

    probable_patterns = probable_patterns or []
    normalized = normalize_text(text)

    for pattern in positive_patterns:
        context = first_context(normalized, pattern)
        if not context:
            continue
        if UNCERTAIN_RE.search(context):
            return "possible", context
        if NEGATION_RE.search(context):
            return "absent", context
        if PROBABLE_RE.search(context):
            return "probable", context
        return "present", context

    for pattern in probable_patterns:
        context = first_context(normalized, pattern)
        if context:
            return "probable", context

    for pattern in negative_patterns:
        context = first_context(normalized, pattern)
        if context:
            return "absent", context

    return "", ""


def classify_invasion(text: str) -> tuple[str, str, str]:
    if not text.strip():
        return "", "", ""

    normalized = normalize_text(text)
    normalized_latin = normalize_latin_tokens(text)

    typed_candidates = [
        ("percreta", r"(?<![a-z])(?:p\s*l\W*|placenta\W*)percreta(?![a-z])"),
        ("increta", r"(?<![a-z])(?:p\s*l\W*|placenta\W*)increta(?![a-z])"),
        ("accreta", r"(?<![a-z])(?:p\s*l\W*|placenta\W*)acc?reta(?![a-z])"),
    ]
    for invasion_type, pattern in typed_candidates:
        match = re.search(pattern, normalized_latin, flags=re.IGNORECASE)
        if not match:
            continue
        context = first_context_from_match(normalized, match.start(), match.end())
        if NEGATION_RE.search(context) and not UNCERTAIN_RE.search(context):
            continue
        if UNCERTAIN_RE.search(context):
            return invasion_type, "possible", context
        return invasion_type, "definite", context

    bladder_percreta_context = first_context(
        normalized,
        r"врастан\w*\s+плацент\w*.{0,120}стенк\w*\s+мочев\w*\s+пузыр|"
        r"стенк\w*\s+мочев\w*\s+пузыр\w*.{0,120}врастан\w*\s+плацент|"
        r"плацент\w*.{0,120}врастан\w*.{0,120}мочев\w*\s+пузыр",
    )
    if bladder_percreta_context and not NEGATION_RE.search(bladder_percreta_context):
        return "percreta", "definite", bladder_percreta_context

    candidates = [
        ("percreta", r"percreta|перкрет|прорастан\w*.{0,80}(сероз|мочев|параметр|за\s+предел)|инвази\w*.{0,80}(мочев|параметр)"),
        ("increta", r"increta|инкрет|глубок\w*.{0,80}(врастан|инвази)|миометри\w*.{0,80}не\s+прослеж"),
        ("accreta", r"accreta|акрет|плотн\w*\s+прикреп|поверхностн\w*.{0,80}врастан"),
    ]
    for invasion_type, pattern in candidates:
        context = first_context(normalized, pattern)
        if not context:
            continue
        if NEGATION_RE.search(context) and not UNCERTAIN_RE.search(context):
            continue
        if UNCERTAIN_RE.search(context):
            return invasion_type, "possible", context
        if PROBABLE_RE.search(context):
            return invasion_type, "probable", context
        if DEFINITE_RE.search(context):
            return invasion_type, "definite", context
        return invasion_type, "probable", context

    negative_context = first_context(
        normalized,
        r"признак\w*.{0,80}(pas|врастан|инвази).{0,80}(не\s+выяв|нет)|"
        r"данн\w*\s+за.{0,80}(pas|врастан|инвази).{0,80}нет|"
        r"(pas|врастан|инвази).{0,80}(не\s+выяв|нет)",
    )
    if negative_context:
        return "none", "absent", negative_context

    return "", "", ""


def classify_general_invasion(text: str) -> tuple[str, str, str]:
    if not text.strip():
        return "", "", ""

    normalized = normalize_text(text)
    context = first_context(
        normalized,
        r"врастани\w*\s+плацент|плацент\w*.{0,80}врастан|placenta\s+accreta\s+spectrum|pas",
    )
    if not context:
        return "", "", ""
    if NEGATION_RE.search(context) and not UNCERTAIN_RE.search(context):
        return "none", "absent", context
    if UNCERTAIN_RE.search(context):
        return "accreta", "possible", context
    return "accreta", "probable", context


def classify_invasion_by_priority(
    sources: dict[str, str],
) -> tuple[str, str, str, str]:
    confirmed_sources = [
        ("Ход Вмешательства", sources.get("operation_course", "")),
        ("ДиагнозыВыпЭпикриза", sources.get("diagnoses", "")),
    ]
    confirmed_candidates: list[tuple[str, str, str, str]] = []
    for source_name, text in confirmed_sources:
        invasion_type, confidence, context = classify_invasion(text)
        if invasion_type and PAS_SEVERITY.get(invasion_type, 0) > 0:
            confirmed_candidates.append((invasion_type, confidence, context, source_name))
    if confirmed_candidates:
        return max(
            confirmed_candidates,
            key=lambda item: PAS_SEVERITY.get(item[0], 0),
        )

    priority = [
        ("Ход Вмешательства", sources.get("operation_course", "")),
        ("ДиагнозыВыпЭпикриза", sources.get("diagnoses", "")),
        ("ПоказанияКОперации", sources.get("operation_indications", "")),
        (
            "МРТ",
            join_sources(sources, ["mri_description", "mri_conclusion"]),
        ),
    ]

    for source_name, text in priority:
        invasion_type, confidence, context = classify_invasion(text)
        if invasion_type:
            return invasion_type, confidence, context, source_name

        invasion_type, confidence, context = classify_general_invasion(text)
        if invasion_type:
            return invasion_type, confidence, context, source_name

    return "", "", "", ""


def classify_bladder_involvement(sources: dict[str, str]) -> tuple[str, str]:
    operation_course = sources.get("operation_course", "")
    priority_text = join_sources(
        sources,
        [
            "operation_course",
            "diagnoses",
            "operation_indications",
            "mri_description",
            "mri_conclusion",
        ],
    )
    normalized_operation = normalize_text(operation_course)
    normalized_all = normalize_text(priority_text)

    explicit_context = first_context(
        normalized_operation,
        r"резекци\w*.{0,80}мочев\w*\s+пузыр|"
        r"мочев\w*\s+пузыр\w*.{0,80}резецир|"
        r"ушиван\w*.{0,80}мочев\w*\s+пузыр|"
        r"мочев\w*\s+пузыр\w*.{0,80}ушит|"
        r"цистотом|"
        r"поврежден\w*.{0,80}мочев\w*\s+пузыр",
    )
    if explicit_context:
        return "present", explicit_context

    explicit_context = first_context(
        normalized_all,
        r"инвази\w*.{0,80}мочев\w*\s+пузыр|"
        r"мочев\w*\s+пузыр\w*.{0,80}инвази|"
        r"прорастан\w*.{0,80}мочев\w*\s+пузыр|"
        r"мочев\w*\s+пузыр\w*.{0,80}прорастан",
    )
    if explicit_context and not NEGATION_RE.search(explicit_context):
        return "present", explicit_context

    negative_context = first_context(
        normalized_all,
        r"инвази\w*.{0,80}мочев\w*\s+пузыр\w*.{0,80}(не\s+выяв|нет|не\s+получ)|"
        r"вовлеч\w*.{0,80}(стенк\w*)?\s*пузыр\w*.{0,80}(не\s+выяв|нет|не\s+получ)|"
        r"убедительн\w*.{0,80}вовлеч\w*.{0,80}пузыр\w*.{0,80}(не\s+выяв|нет|не\s+получ)|"
        r"пузыр\w*.{0,80}вовлеч\w*.{0,80}(не\s+выяв|нет|не\s+получ)|"
        r"мочев\w*\s+пузыр\w*.{0,80}(не\s+вовлеч|без\s+признак|без\s+убедительн|без\s+достоверн)|"
        r"убедительн\w*.{0,80}вовлечен\w*.{0,80}мочев\w*\s+пузыр\w*.{0,80}нет",
    )
    if negative_context:
        return "absent", negative_context

    probable_context = first_context(
        normalized_all,
        r"мочев\w*\s+пузыр\w*.{0,80}вовлеч|"
        r"вовлеч\w*.{0,80}мочев\w*\s+пузыр",
    )
    if probable_context:
        if UNCERTAIN_RE.search(probable_context):
            return "possible", probable_context
        return "probable", probable_context

    possible_context = first_context(
        normalized_all,
        r"стенк\w*\s+пузыр\w*.{0,80}деформ|"
        r"мочев\w*\s+пузыр\w*.{0,80}деформ|"
        r"маточно-пузырн\w*.{0,80}(не\s+дифференц|плохо\s+дифференц|пространств)|"
        r"пузыр\w*.{0,80}прилеж",
    )
    if possible_context:
        return "possible", possible_context

    return "", ""


def classify_mri_features(sources: dict[str, str], gold: dict[str, str], rationale: list[str]) -> None:
    mri_text = join_sources(
        sources,
        ["mri_description", "mri_conclusion", "diagnoses", "operation_indications"],
    )

    invasion_type, confidence, context, source_name = classify_invasion_by_priority(
        sources,
    )
    gold["gold_invasion_type"] = invasion_type
    gold["gold_invasion_confidence"] = confidence
    gold["gold_pas_type"] = invasion_type
    if context:
        add_rationale(rationale, f"PAS ({source_name})", context)

    feature_rules = {
        "gold_parametrium_involvement": (
            [r"параметр\w*.{0,80}(инвази|вовлеч|прораст)"],
            [r"параметр\w*.{0,80}(не\s+вовлеч|не\s+выяв|без\s+признак|нет)"],
            [],
        ),
        "gold_posterior_wall_involvement": (
            [r"задн\w*\s+стенк\w*.{0,80}(инвази|вовлеч|прораст)"],
            [r"задн\w*\s+стенк\w*.{0,80}(не\s+вовлеч|не\s+выяв|без\s+признак|нет)"],
            [],
        ),
        "gold_placenta_previa": (
            [r"предлежани\w*\s+плацент|placenta\s+previa|перекрыва\w*.{0,80}внутренн\w*\s+зев"],
            [r"предлежани\w*.{0,80}(нет|не\s+выяв|не\s+определ)|внутренн\w*\s+зев.{0,80}не\s+перекры"],
            [],
        ),
        "gold_anterior_placenta": (
            [r"плацент\w*.{0,80}передн\w*\s+стенк|по\s+передн\w*\s+стенк\w*.{0,80}плацент"],
            [r"плацент\w*.{0,80}не\s+по\s+передн"],
            [],
        ),
        "gold_retroplacental_vessels": (
            [r"ретроплацентарн\w*.{0,80}сосуд|расширенн\w*.{0,80}сосуд|гиперваскуляр"],
            [r"ретроплацентарн\w*.{0,80}сосуд\w*.{0,80}(не\s+выяв|нет|без)"],
            [],
        ),
        "gold_lacunae": (
            [r"лакун"],
            [r"лакун\w*.{0,80}(не\s+выяв|нет|без)"],
            [],
        ),
        "gold_uterine_wall_thinning": (
            [r"истонч\w*.{0,80}(миометри|рубц|стенк)|миометри\w*.{0,80}не\s+прослеж"],
            [r"истонч\w*.{0,80}(не\s+выяв|нет|без)"],
            [],
        ),
        "gold_uterine_hernia_or_bulging": (
            [r"выбухан|грыжевидн|деформаци\w*.{0,80}контур|bulging"],
            [r"выбухан\w*.{0,80}(нет|не\s+выяв)|грыжевидн\w*.{0,80}(нет|не\s+выяв)"],
            [],
        ),
        "gold_preoperative_bleeding": (
            [r"дородов\w*.{0,80}кровотеч|кровянисты\w*\s+выдел|кровотечен\w*.{0,80}до\s+операц"],
            [r"кровотечен\w*.{0,80}(нет|не\s+было|не\s+отмеч)"],
            [],
        ),
    }

    bladder_status, bladder_context = classify_bladder_involvement(sources)
    gold["gold_bladder_involvement"] = bladder_status
    if bladder_context:
        add_rationale(rationale, "gold_bladder_involvement", bladder_context)

    for field, (positive, negative, probable) in feature_rules.items():
        status, context = status_from_patterns(mri_text, positive, negative, probable)
        gold[field] = status
        if context:
            add_rationale(rationale, field, context)


def classify_suspicion(gold: dict[str, str], sources: dict[str, str], rationale: list[str]) -> None:
    mri_text = join_sources(
        sources,
        ["mri_description", "mri_conclusion", "diagnoses", "operation_indications"],
    )
    normalized = normalize_text(mri_text)

    gold["gold_highest_suspected_extent"] = ""
    gold["gold_percreta_suspicion"] = ""
    gold["gold_bladder_serosa_suspicion"] = ""

    if gold.get("gold_invasion_type") == "percreta":
        gold["gold_highest_suspected_extent"] = "percreta"
        gold["gold_percreta_suspicion"] = "present"
        if gold.get("gold_bladder_involvement") in {"possible", "probable", "present"}:
            gold["gold_bladder_serosa_suspicion"] = gold["gold_bladder_involvement"]
        return

    percreta_context = first_context(
        normalized,
        r"(нельзя\s+исключ|не\s+исключ|возможн|подозр).{0,80}(percreta|перкрет|сероз|мочев|параметр)",
    )
    if percreta_context:
        gold["gold_highest_suspected_extent"] = "percreta"
        gold["gold_percreta_suspicion"] = "possible"
        add_rationale(rationale, "percreta suspicion", percreta_context)

    bladder_serosa_context = first_context(
        normalized,
        r"(нельзя\s+исключ|не\s+исключ|возможн|подозр).{0,80}(сероз|мочев|маточно-пузырн)",
    )
    if bladder_serosa_context:
        gold["gold_bladder_serosa_suspicion"] = "possible"
        if not gold["gold_highest_suspected_extent"]:
            gold["gold_highest_suspected_extent"] = "percreta"
        if not gold["gold_percreta_suspicion"]:
            gold["gold_percreta_suspicion"] = "possible"
        add_rationale(rationale, "bladder/serosa suspicion", bladder_serosa_context)

    if gold.get("gold_invasion_type") == "none" and not gold["gold_highest_suspected_extent"]:
        gold["gold_highest_suspected_extent"] = "none"
        gold["gold_percreta_suspicion"] = "absent"
        gold["gold_bladder_serosa_suspicion"] = "absent"


def parse_blood_loss_ml(text: str) -> int | None:
    if not text.strip():
        return None
    normalized = text.replace(",", ".").replace(" ", "")
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if not numbers:
        return None
    if "+" in normalized:
        return int(sum(numbers))
    return int(max(numbers))


def blood_loss_class(value: int) -> str:
    if value <= 500:
        return "0-500"
    if value <= 1000:
        return "500-1000"
    if value <= 1500:
        return "1000-1500"
    if value <= 2000:
        return "1500-2000"
    return "2000+"


def parse_gold_blood_loss_ml(value: str) -> int | None:
    text = text_value(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def severity_rank(value: str) -> int:
    return {
        "none": 0,
        "accreta": 1,
        "increta": 2,
        "percreta": 3,
    }.get(text_value(value), -1)


def risk_group_for_readiness(value: str) -> str:
    return {
        "1": "low",
        "2": "medium",
        "3": "high",
        "4": "critical",
    }.get(text_value(value), "")


def classify_vascular_intervention(sources: dict[str, str]) -> tuple[str, str]:
    operation_course = sources.get("operation_course", "")
    combined = join_sources(sources, ["operation_course", "operation_indications", "diagnoses"])
    normalized_operation = normalize_text(operation_course)
    negative_context = first_context(
        normalized_operation,
        r"от\s+перевязк\w*.{0,80}(воздерж|отказ)|"
        r"решено.{0,80}(воздерж|отказ).{0,80}перевязк|"
        r"перевязк\w*.{0,80}(воздерж|отказ)",
    )
    if negative_context:
        return "absent", negative_context

    performed_patterns = [
        r"эмболизац",
        r"окклюзи\w*.{0,80}(артери|аорт|баллон)",
        r"баллон\w*.{0,80}(окклюзи|аорт|подвздошн|артери)",
        r"катетериз\w*.{0,80}маточн\w*\s+артери",
        r"перевязк\w*.{0,80}(внутренн\w*)?\s*подвздошн\w*\s+артери",
        r"эндоваскуляр|рентгенэндоваскуляр",
    ]
    planned_patterns = [
        r"план\w*.{0,80}(эмболизац|окклюзи|баллон|сосудист)",
        r"сосудист\w*\s+хирург",
    ]
    negative_patterns = [
        r"(эмболизац|окклюзи|баллон).{0,80}(не\s+провод|не\s+выполн|нет)",
    ]

    status, context = status_from_patterns(operation_course, performed_patterns, negative_patterns)
    if status:
        return status, context

    status, context = status_from_patterns(combined, planned_patterns, negative_patterns)
    if status:
        return "possible" if status == "present" else status, context

    return "", ""


def has_bladder_injury(sources: dict[str, str]) -> tuple[bool, str]:
    context = first_context(
        normalize_text(sources.get("operation_course", "")),
        r"поврежден\w*.{0,80}мочев\w*\s+пузыр|цистотом|ушиван\w*.{0,80}мочев\w*\s+пузыр|резекци\w*.{0,80}мочев\w*\s+пузыр",
    )
    return bool(context), context


def has_critical_outcome(sources: dict[str, str]) -> tuple[bool, str]:
    context = first_context(
        normalize_text(join_sources(sources, ["operation_course", "diagnoses"])),
        r"летальн|смерт|геморрагическ\w*\s+шок|двс|пациентк\w*.{0,80}(орит|реанимац)|"
        r"(переведен|переведена).{0,80}(орит|реанимац)",
    )
    if re.search(r"новорожден|ребен|неонат|апгар|плод", context):
        return False, ""
    return bool(context), context


def classify_outcomes(gold: dict[str, str], sources: dict[str, str], rationale: list[str]) -> None:
    operation_loss = parse_blood_loss_ml(sources.get("blood_loss_operation", ""))
    delivery_loss = parse_blood_loss_ml(sources.get("blood_loss_delivery", ""))
    blood_loss = operation_loss if operation_loss is not None else delivery_loss

    if blood_loss is not None:
        gold["gold_blood_loss_ml"] = str(blood_loss)
        gold["gold_massive_blood_loss"] = "true" if blood_loss > 1500 else "false"
        gold["gold_blood_loss_class"] = blood_loss_class(blood_loss)
        source_name = (
            "КровопотеряОперация"
            if operation_loss is not None
            else "КровопотеряРоды"
        )
        add_rationale(rationale, source_name, str(blood_loss))
    else:
        gold["gold_blood_loss_ml"] = ""
        gold["gold_massive_blood_loss"] = ""
        gold["gold_blood_loss_class"] = ""

    vascular_status, vascular_context = classify_vascular_intervention(sources)
    gold["gold_vascular_intervention"] = vascular_status
    if vascular_context:
        add_rationale(rationale, "vascular intervention", vascular_context)

    bladder_injury, bladder_context = has_bladder_injury(sources)
    critical_outcome, critical_context = has_critical_outcome(sources)
    if bladder_context:
        add_rationale(rationale, "bladder injury", bladder_context)
    if critical_context:
        add_rationale(rationale, "critical outcome", critical_context)

    readiness = ""
    if critical_outcome or (blood_loss is not None and blood_loss > 3000):
        readiness = "4"
    elif (
        bladder_injury
        or vascular_status == "present"
        or (blood_loss is not None and 1500 < blood_loss <= 3000)
    ):
        readiness = "3"
    elif blood_loss is not None and 1000 < blood_loss <= 1500:
        readiness = "2"
    elif (
        blood_loss is not None
        and blood_loss <= 1000
        and not bladder_injury
        and vascular_status != "present"
    ):
        readiness = "1"

    if gold.get("gold_bladder_involvement") == "present" and readiness in {"", "1", "2"}:
        readiness = "3"
        add_rationale(
            rationale,
            "readiness override",
            "gold_bladder_involvement=present -> readiness_level минимум 3",
        )

    gold["gold_readiness_level"] = readiness
    gold["gold_risk_group"] = {
        "1": "low",
        "2": "medium",
        "3": "high",
        "4": "critical",
    }.get(readiness, "")


def classify_gold_confidence(gold: dict[str, str], rationale: list[str]) -> str:
    filled_core = sum(
        bool(gold.get(field))
        for field in [
            "gold_invasion_type",
            "gold_invasion_confidence",
            "gold_blood_loss_ml",
            "gold_readiness_level",
        ]
    )
    if filled_core >= 3 and len(rationale) >= 3:
        return "high"
    if filled_core >= 1 or rationale:
        return "medium"
    return "low"


def self_review_case(gold: dict[str, str]) -> list[str]:
    warnings = []
    rationale = normalize_text(gold.get("gold_rationale", ""))

    if gold.get("gold_bladder_involvement") == "present":
        bladder_markers = [
            "инвази",
            "прорастан",
            "резекци",
            "ушив",
            "ушит",
            "цистотом",
            "поврежден",
            "percreta",
        ]
        has_bladder_source = any(marker in rationale for marker in bladder_markers)
        has_bladder_context = "пузыр" in rationale or "bladder" in rationale
        if not (has_bladder_source and has_bladder_context):
            gold["gold_bladder_involvement"] = "probable"
            warnings.append(
                "downgraded bladder present: no explicit PAS-related bladder source"
            )

    if (
        gold.get("gold_bladder_involvement") == "present"
        and gold.get("gold_readiness_level") in {"", "1", "2"}
    ):
        gold["gold_readiness_level"] = "3"
        gold["gold_risk_group"] = "high"
        warnings.append("raised readiness to 3 because bladder involvement is present")

    if gold.get("gold_invasion_type") == "percreta":
        percreta_markers = [
            "за предел",
            "сероз",
            "мочев",
            "пузыр",
            "параметр",
            "percreta",
            "резекци",
            "ушив",
            "цистотом",
        ]
        if not any(marker in rationale for marker in percreta_markers):
            warnings.append("percreta lacks explicit rationale source")

    blood_loss = parse_gold_blood_loss_ml(gold.get("gold_blood_loss_ml", ""))
    if blood_loss is not None:
        expected_massive = "true" if blood_loss > 1500 else "false"
        if gold.get("gold_massive_blood_loss") != expected_massive:
            gold["gold_massive_blood_loss"] = expected_massive
            warnings.append("fixed massive blood loss flag")

        expected_class = blood_loss_class(blood_loss)
        if gold.get("gold_blood_loss_class") != expected_class:
            gold["gold_blood_loss_class"] = expected_class
            warnings.append("fixed blood loss class")

    invasion_type = gold.get("gold_invasion_type", "")
    suspected_extent = gold.get("gold_highest_suspected_extent", "")
    if invasion_type and severity_rank(suspected_extent) < severity_rank(invasion_type):
        gold["gold_highest_suspected_extent"] = invasion_type
        if invasion_type == "percreta" and not gold.get("gold_percreta_suspicion"):
            gold["gold_percreta_suspicion"] = "present"
        warnings.append("raised highest suspected extent to invasion type")

    expected_risk_group = risk_group_for_readiness(gold.get("gold_readiness_level", ""))
    if expected_risk_group and gold.get("gold_risk_group") != expected_risk_group:
        gold["gold_risk_group"] = expected_risk_group
        warnings.append("fixed risk group from readiness level")

    return warnings


def annotate_row(row: pd.Series) -> dict[str, str]:
    gold = {field: "" for field in GOLD_FIELDS}
    rationale: list[str] = []
    sources = source_texts(row)

    classify_mri_features(sources, gold, rationale)
    classify_suspicion(gold, sources, rationale)
    classify_outcomes(gold, sources, rationale)

    gold["gold_confidence"] = classify_gold_confidence(gold, rationale)
    gold["gold_rationale"] = "; ".join(dict.fromkeys(rationale))
    review_warnings = self_review_case(gold)
    if review_warnings:
        gold["gold_rationale"] = (
            f"{gold['gold_rationale']}; self-review: "
            + "; ".join(review_warnings)
        ).strip("; ")
    return gold


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_jsonl(dataframe: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in dataframe.to_dict(orient="records"):
            safe_record = {
                str(key): json_safe_value(value)
                for key, value in record.items()
            }
            file.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def value_counts(dataframe: pd.DataFrame, field: str) -> dict[str, int]:
    values = [
        text_value(value)
        for value in dataframe[field].tolist()
        if text_value(value)
    ]
    return dict(Counter(values))


def filled_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    return {
        field: int(sum(bool(text_value(value)) for value in dataframe[field].tolist()))
        for field in GOLD_FIELDS
    }


def review_sample(dataframe: pd.DataFrame, sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or dataframe.empty:
        return []

    sample_count = min(sample_size, len(dataframe))
    sampled = dataframe.sample(n=sample_count, random_state=seed)
    sample = []
    for index, row in sampled.iterrows():
        sample.append(
            {
                "row_index": int(index),
                "case_id": get_case_id(row, int(index)),
                "gold_invasion_type": text_value(row.get("gold_invasion_type")),
                "gold_invasion_confidence": text_value(row.get("gold_invasion_confidence")),
                "gold_bladder_involvement": text_value(row.get("gold_bladder_involvement")),
                "gold_blood_loss_ml": text_value(row.get("gold_blood_loss_ml")),
                "gold_blood_loss_class": text_value(row.get("gold_blood_loss_class")),
                "gold_readiness_level": text_value(row.get("gold_readiness_level")),
                "gold_risk_group": text_value(row.get("gold_risk_group")),
                "gold_confidence": text_value(row.get("gold_confidence")),
                "gold_rationale": text_value(row.get("gold_rationale")),
            }
        )
    return sample


def validate_saved_case(gold: dict[str, str]) -> list[str]:
    warnings = []
    rationale = normalize_text(gold.get("gold_rationale", ""))

    if gold.get("gold_bladder_involvement") == "present":
        has_bladder = "пузыр" in rationale or "bladder" in rationale
        has_pas_related_source = any(
            marker in rationale
            for marker in [
                "инвази",
                "прорастан",
                "резекци",
                "ушив",
                "ушит",
                "цистотом",
                "поврежден",
                "percreta",
            ]
        )
        if not (has_bladder and has_pas_related_source):
            warnings.append("bladder present without explicit PAS-related source")

        if gold.get("gold_readiness_level") in {"1", "2"}:
            warnings.append("bladder present with readiness 1/2")

    if gold.get("gold_invasion_type") == "percreta":
        if not any(
            marker in rationale
            for marker in [
                "за предел",
                "сероз",
                "мочев",
                "пузыр",
                "параметр",
                "percreta",
                "резекци",
                "ушив",
                "цистотом",
            ]
        ):
            warnings.append("percreta without required source")

    blood_loss = parse_gold_blood_loss_ml(gold.get("gold_blood_loss_ml", ""))
    if blood_loss is not None:
        if blood_loss > 1500 and gold.get("gold_massive_blood_loss") != "true":
            warnings.append("blood loss >1500 but massive flag is not true")
        if blood_loss <= 1500 and gold.get("gold_massive_blood_loss") != "false":
            warnings.append("blood loss <=1500 but massive flag is not false")
        if gold.get("gold_blood_loss_class") != blood_loss_class(blood_loss):
            warnings.append("blood loss class mismatch")

    if (
        gold.get("gold_invasion_type")
        and severity_rank(gold.get("gold_highest_suspected_extent", ""))
        < severity_rank(gold.get("gold_invasion_type", ""))
    ):
        warnings.append("highest suspected extent below invasion type")

    expected_risk_group = risk_group_for_readiness(gold.get("gold_readiness_level", ""))
    if expected_risk_group and gold.get("gold_risk_group") != expected_risk_group:
        warnings.append("risk group does not match readiness")

    return warnings


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    csv_output = resolve_path(args.csv_output)
    jsonl_output = resolve_path(args.jsonl_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if not GOLD_LABELING_PROTOCOL.exists():
        raise FileNotFoundError(f"Gold labeling protocol not found: {GOLD_LABELING_PROTOCOL}")
    protocol_text = GOLD_LABELING_PROTOCOL.read_text(encoding="utf-8")
    if "Mandatory Self-Review" not in protocol_text:
        raise ValueError("GOLD_LABELING.md must define Mandatory Self-Review.")

    dataframe = pd.read_csv(input_path, dtype=object, keep_default_na=False)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    gold_dataframe = dataframe.copy()
    for field in GOLD_FIELDS:
        gold_dataframe[field] = ""

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    remaining_warnings: list[dict[str, Any]] = []

    for row_index, row in dataframe.iterrows():
        annotation = annotate_row(row)
        for field in GOLD_FIELDS:
            gold_dataframe.at[row_index, field] = annotation[field]

        case_warnings = validate_saved_case(annotation)
        if case_warnings:
            remaining_warnings.append(
                {
                    "row_index": int(row_index),
                    "case_id": get_case_id(row, int(row_index)),
                    "warnings": case_warnings,
                }
            )

        gold_dataframe.to_csv(csv_output, index=False, encoding="utf-8")
        write_jsonl(gold_dataframe, jsonl_output)

    summary = {
        "warning": "Retrospective labels for scientific evaluation; not clinical conclusions.",
        "protocol": str(GOLD_LABELING_PROTOCOL),
        "input": str(input_path),
        "csv_output": str(csv_output),
        "jsonl_output": str(jsonl_output),
        "rows_processed": int(len(gold_dataframe)),
        "filled_counts": filled_counts(gold_dataframe),
        "gold_invasion_type_distribution": value_counts(
            gold_dataframe,
            "gold_invasion_type",
        ),
        "gold_blood_loss_class_distribution": value_counts(
            gold_dataframe,
            "gold_blood_loss_class",
        ),
        "gold_readiness_level_distribution": value_counts(
            gold_dataframe,
            "gold_readiness_level",
        ),
        "gold_bladder_involvement_distribution": value_counts(
            gold_dataframe,
            "gold_bladder_involvement",
        ),
        "bladder_present_cases": [
            str(value)
            for value in gold_dataframe.loc[
                gold_dataframe["gold_bladder_involvement"] == "present",
                "case_id",
            ].tolist()
        ],
        "percreta_cases": [
            str(value)
            for value in gold_dataframe.loc[
                gold_dataframe["gold_invasion_type"] == "percreta",
                "case_id",
            ].tolist()
        ],
        "random_review_cases": review_sample(
            gold_dataframe,
            sample_size=args.sample_size,
            seed=args.seed,
        ),
        "self_review_warnings": remaining_warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
