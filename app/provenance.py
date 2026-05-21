"""Evidence-only provenance highlighting helpers.

This module must stay a visualization layer. It does not infer clinical
findings, classify sentences, or modify extraction output. The source of truth
is always result["evidence"].
"""

import html
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


Polarity = Literal["positive", "uncertain", "negative"]
MatchMethod = Literal["exact", "normalized", "token-overlap"]


class MatchPayload(TypedDict):
    polarity: Polarity
    phrase: str
    start: int
    end: int
    method: MatchMethod


class HighlightPayload(TypedDict):
    html: str
    matches: list[MatchPayload]
    unmatched_evidence: list[str]


COLORS = {
    "positive": "#ea580c",
    "uncertain": "#f59e0b",
    "negative": "#16a34a",
}

POLARITY_PRIORITY = {
    "uncertain": 3,
    "positive": 2,
    "negative": 1,
}

METHOD_PRIORITY = {
    "exact": 3,
    "normalized": 2,
    "token-overlap": 1,
}

WORD_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceItem:
    index: int
    polarity: Polarity
    phrase: str


@dataclass(frozen=True)
class EvidenceMatch:
    polarity: Polarity
    phrase: str
    start: int
    end: int
    method: MatchMethod


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int


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


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").split())


def normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    normalized_chars = []
    index_map = []
    previous_was_space = False

    for index, char in enumerate(text):
        normalized_char = char.lower().replace("ё", "е")

        if normalized_char.isspace():
            if previous_was_space:
                continue

            normalized_chars.append(" ")
            index_map.append(index)
            previous_was_space = True
            continue

        normalized_chars.append(normalized_char)
        index_map.append(index)
        previous_was_space = False

    return "".join(normalized_chars), index_map


def normalize_token(token: str) -> str:
    return token.lower().replace("ё", "е")


def tokenize(text: str) -> list[Token]:
    tokens = []

    for match in WORD_RE.finditer(text):
        value = normalize_token(match.group(0))
        if len(value) < 4:
            continue

        tokens.append(Token(value=value, start=match.start(), end=match.end()))

    return tokens


def collect_evidence(result: dict[str, Any]) -> list[EvidenceItem]:
    evidence = result.get("evidence", {}) if isinstance(result, dict) else {}
    items: list[EvidenceItem] = []
    evidence_sources: list[tuple[Polarity, str]] = [
        ("positive", "positive_findings"),
        ("uncertain", "uncertain_findings"),
        ("negative", "negative_findings"),
    ]

    for polarity, key in evidence_sources:
        for phrase in as_list(evidence.get(key)):
            items.append(
                EvidenceItem(
                    index=len(items),
                    polarity=polarity,
                    phrase=phrase,
                )
            )

    return items


def find_exact_ranges(report_text: str, phrase: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(phrase), report_text)
    ]


def find_normalized_ranges(report_text: str, phrase: str) -> list[tuple[int, int]]:
    normalized_report, index_map = normalize_with_index_map(report_text)
    normalized_phrase = normalize_text(phrase)

    if not normalized_phrase or not index_map:
        return []

    ranges = []
    start = 0

    while True:
        match_start = normalized_report.find(normalized_phrase, start)
        if match_start == -1:
            break

        match_end = match_start + len(normalized_phrase) - 1
        original_start = index_map[match_start]
        original_end = index_map[match_end] + 1
        ranges.append((original_start, original_end))
        start = match_start + 1

    return ranges


def find_token_overlap_ranges(report_text: str, phrase: str) -> list[tuple[int, int]]:
    """Conservative fallback for near-literal phrases.

    This is intentionally narrow: it requires several matching normalized words
    within a short window and refuses spans crossing sentence/line boundaries.
    """
    if ":" in phrase:
        return []

    report_tokens = tokenize(report_text)
    phrase_tokens = tokenize(phrase)
    phrase_values = {token.value for token in phrase_tokens}

    if len(phrase_values) < 3:
        return []

    required_overlap = len(phrase_values) if len(phrase_values) <= 4 else 3
    max_window = max(8, min(len(phrase_values) + 4, 14))
    ranges = []

    for left in range(len(report_tokens)):
        seen = set()
        first_matched_start = None
        right_limit = min(len(report_tokens), left + max_window)

        for right in range(left, right_limit):
            token = report_tokens[right]

            if token.value in phrase_values:
                seen.add(token.value)
                if first_matched_start is None:
                    first_matched_start = token.start

            overlap = len(seen)
            if overlap < required_overlap:
                continue

            if overlap / len(phrase_values) < 0.6:
                continue

            original_start = first_matched_start or token.start
            original_end = token.end
            matched_text = report_text[original_start:original_end]

            if "\n" in matched_text or "." in matched_text:
                continue

            ranges.append((original_start, original_end))
            break

    return ranges


def evidence_phrase_variants(phrase: str, polarity: Polarity) -> list[str]:
    """Split mixed evidence clauses only enough to avoid misleading spans."""
    parts = [
        part.strip(" ,.;")
        for part in re.split(r"\bоднако\b", phrase, flags=re.IGNORECASE)
        if part.strip(" ,.;")
    ]

    if len(parts) > 1:
        if polarity == "uncertain":
            return [parts[0]]

        return parts

    return [phrase]


def find_evidence_ranges(
    report_text: str,
    phrase: str,
    polarity: Polarity,
) -> list[tuple[int, int, MatchMethod]]:
    ranges_with_methods: list[tuple[int, int, MatchMethod]] = []

    for variant in evidence_phrase_variants(phrase, polarity):
        exact_ranges = find_exact_ranges(report_text, variant)
        if exact_ranges:
            ranges_with_methods.extend(
                (start, end, "exact") for start, end in exact_ranges
            )
            continue

        normalized_ranges = find_normalized_ranges(report_text, variant)
        if normalized_ranges:
            ranges_with_methods.extend(
                (start, end, "normalized") for start, end in normalized_ranges
            )
            continue

        ranges_with_methods.extend(
            (start, end, "token-overlap")
            for start, end in find_token_overlap_ranges(report_text, variant)
        )

    return ranges_with_methods


def ranges_overlap(left: EvidenceMatch, right: EvidenceMatch) -> bool:
    return left.start < right.end and left.end > right.start


def select_non_overlapping_matches(
    candidates: list[EvidenceMatch],
) -> list[EvidenceMatch]:
    selected: list[EvidenceMatch] = []

    sorted_candidates = sorted(
        candidates,
        key=lambda match: (
            -POLARITY_PRIORITY[match.polarity],
            -METHOD_PRIORITY[match.method],
            -(match.end - match.start),
            match.start,
        ),
    )

    for candidate in sorted_candidates:
        if any(ranges_overlap(candidate, existing) for existing in selected):
            continue

        selected.append(candidate)

    return sorted(selected, key=lambda match: match.start)


def build_report_highlighting(
    report_text: str,
    result: dict[str, Any],
) -> HighlightPayload:
    """Build safe HTML by highlighting only evidence phrases found in report."""
    evidence_items = collect_evidence(result)
    candidates: list[EvidenceMatch] = []
    found_evidence_indices = set()

    for item in evidence_items:
        ranges = find_evidence_ranges(report_text, item.phrase, item.polarity)

        if ranges:
            found_evidence_indices.add(item.index)

        for start, end, method in ranges:
            candidates.append(
                EvidenceMatch(
                    polarity=item.polarity,
                    phrase=item.phrase,
                    start=start,
                    end=end,
                    method=method,
                )
            )

    selected_matches = select_non_overlapping_matches(candidates)
    unmatched_evidence = [
        item.phrase
        for item in evidence_items
        if item.index not in found_evidence_indices
    ]

    return {
        "html": render_highlighted_report(report_text, selected_matches),
        "matches": [
            {
                "polarity": match.polarity,
                "phrase": match.phrase,
                "start": match.start,
                "end": match.end,
                "method": match.method,
            }
            for match in selected_matches
        ],
        "unmatched_evidence": unmatched_evidence,
    }


def render_highlighted_report(
    report_text: str,
    matches: list[EvidenceMatch],
) -> str:
    parts = []
    cursor = 0

    for match in matches:
        color = COLORS[match.polarity]
        parts.append(html.escape(report_text[cursor:match.start]))
        parts.append(
            (
                f'<span style="background-color: {color}33; '
                f'border-bottom: 2px solid {color}; '
                f'padding: 1px 2px; border-radius: 3px;">'
                f'{html.escape(report_text[match.start:match.end])}</span>'
            )
        )
        cursor = match.end

    parts.append(html.escape(report_text[cursor:]))
    return "".join(parts)
