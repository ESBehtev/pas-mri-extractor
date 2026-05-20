import html
import re
from dataclasses import dataclass
from typing import Any


COLORS = {
    "positive": "#ea580c",
    "uncertain": "#f59e0b",
    "negative": "#16a34a",
}

POSITIVE_PATTERNS = [
    r"\bpas\b",
    r"\bplacenta\s+(?:accreta|increta|percreta)\b",
    r"\bврастан\w*",
    r"\bинвази\w*",
    r"\bпрорастан\w*",
    r"\bлакун\w*",
    r"\bретроплацентарн\w*",
    r"\bсосуд\w*",
    r"\bгиперваскуляризаци\w*",
    r"\bистонч\w*",
    r"\bмиометр\w*\s+не\s+прослеж",
    r"\bвовлеч\w*",
    r"\bмочев\w*\s+пузыр",
    r"\bперекрывает\s+внутренн\w*\s+зев",
    r"\bпередн\w*\s+стенк",
]

UNCERTAIN_PATTERNS = [
    r"\bнельзя\s+исключ",
    r"\bне\s+исключ",
    r"\bвозможн",
    r"\bчастично\s+не\s+дифференц",
    r"\bподозр",
    r"\bсомнитель",
]

NEGATIVE_PATTERNS = [
    r"\bнет\b",
    r"\bне\s+выяв\w*",
    r"\bне\s+получен\w*",
    r"\bне\s+определя\w*",
    r"\bне\s+подтвержд\w*",
    r"\bсохранен\w*",
    r"\bбез\s+убедительн\w*\s+данн\w*",
    r"\bбез\s+достоверн\w*",
    r"\bпризнак\w*\b.{0,80}\bнет\b",
    r"\bданн\w*\s+за\b.{0,80}\bнет\b",
]

WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int
    polarity: str | None


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").split())


def normalized_words(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {word for word in WORD_RE.findall(normalized) if len(word) >= 4}


def has_any_pattern(text: str, patterns: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def split_report_sentences(report_text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    start = 0

    boundary_pattern = re.compile(r"(?<=[.!?])\s+|\n+|(?=однако\b)")

    for match in boundary_pattern.finditer(report_text):
        end = match.start()
        add_sentence_span(report_text, start, end, spans)
        start = match.end()

    add_sentence_span(report_text, start, len(report_text), spans)

    return spans


def add_sentence_span(
    report_text: str,
    start: int,
    end: int,
    spans: list[SentenceSpan],
) -> None:
    while start < end and report_text[start].isspace():
        start += 1

    while end > start and report_text[end - 1].isspace():
        end -= 1

    if start >= end:
        return

    text = report_text[start:end]
    spans.append(
        SentenceSpan(
            text=text,
            start=start,
            end=end,
            polarity=classify_sentence(text),
        )
    )


def classify_sentence(sentence: str) -> str | None:
    has_positive = has_any_pattern(sentence, POSITIVE_PATTERNS)
    has_uncertain = has_any_pattern(sentence, UNCERTAIN_PATTERNS)
    has_negative = has_any_pattern(sentence, NEGATIVE_PATTERNS)

    if has_negative and (has_positive or has_uncertain):
        return "negative"

    if has_uncertain:
        return "uncertain"

    if has_positive:
        return "positive"

    if has_negative:
        return "negative"

    return None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str):
        if ";" in value:
            return [part.strip() for part in value.split(";") if part.strip()]
        return [value.strip()] if value.strip() else []

    return [str(value)]


def collect_evidence(result: dict[str, Any]) -> list[dict[str, str]]:
    evidence = result.get("evidence", {}) if isinstance(result, dict) else {}
    items: list[dict[str, str]] = []

    for polarity, key in [
        ("positive", "positive_findings"),
        ("uncertain", "uncertain_findings"),
        ("negative", "negative_findings"),
    ]:
        for phrase in as_list(evidence.get(key)):
            items.append({"polarity": polarity, "phrase": phrase})

    return items


def evidence_matches_sentence(evidence_phrase: str, sentence: str) -> bool:
    normalized_phrase = normalize_text(evidence_phrase)
    normalized_sentence = normalize_text(sentence)

    if not normalized_phrase:
        return False

    if normalized_phrase in normalized_sentence:
        return True

    phrase_words = normalized_words(evidence_phrase)
    sentence_words = normalized_words(sentence)

    if not phrase_words:
        return False

    overlap = len(phrase_words & sentence_words)
    required_overlap = min(3, len(phrase_words))

    return overlap >= required_overlap and overlap / len(phrase_words) >= 0.45


def sentence_polarity_from_evidence(
    sentence: SentenceSpan,
    evidence_items: list[dict[str, str]],
) -> tuple[str | None, set[int]]:
    matched_indices = set()
    matched_polarities = []

    for index, item in enumerate(evidence_items):
        if evidence_matches_sentence(item["phrase"], sentence.text):
            matched_indices.add(index)
            matched_polarities.append(item["polarity"])

    if "negative" in matched_polarities and has_any_pattern(
        sentence.text,
        NEGATIVE_PATTERNS,
    ):
        return "negative", matched_indices

    if "uncertain" in matched_polarities:
        return "uncertain", matched_indices

    if "positive" in matched_polarities:
        if has_any_pattern(sentence.text, NEGATIVE_PATTERNS):
            return "negative", matched_indices
        return "positive", matched_indices

    return sentence.polarity, matched_indices


def build_report_highlighting(
    report_text: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    sentences = split_report_sentences(report_text)
    evidence_items = collect_evidence(result)
    matched_evidence_indices: set[int] = set()
    highlighted_ranges: list[tuple[int, int, str]] = []
    counts = {"positive": 0, "uncertain": 0, "negative": 0}

    for sentence in sentences:
        polarity, matched_indices = sentence_polarity_from_evidence(
            sentence,
            evidence_items,
        )
        matched_evidence_indices.update(matched_indices)

        if not polarity:
            continue

        counts[polarity] += 1
        highlighted_ranges.append((sentence.start, sentence.end, polarity))

    unmatched_evidence = [
        item["phrase"]
        for index, item in enumerate(evidence_items)
        if index not in matched_evidence_indices
    ]

    return {
        "html": render_highlighted_report(report_text, highlighted_ranges),
        "counts": counts,
        "unmatched_evidence": unmatched_evidence,
        "sentences": sentences,
    }


def render_highlighted_report(
    report_text: str,
    highlighted_ranges: list[tuple[int, int, str]],
) -> str:
    parts = []
    cursor = 0

    for start, end, polarity in sorted(highlighted_ranges, key=lambda item: item[0]):
        color = COLORS[polarity]
        parts.append(html.escape(report_text[cursor:start]))
        parts.append(
            (
                f'<span style="background-color: {color}33; '
                f'border-bottom: 2px solid {color}; '
                f'padding: 1px 2px; border-radius: 3px;">'
                f'{html.escape(report_text[start:end])}</span>'
            )
        )
        cursor = end

    parts.append(html.escape(report_text[cursor:]))
    return "".join(parts)
