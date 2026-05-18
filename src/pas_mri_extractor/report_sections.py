from dataclasses import dataclass
import re


@dataclass
class ReportSections:
    body: str
    conclusion: str | None
    has_conclusion: bool


CONCLUSION_PATTERNS = [
    r"\bЗАКЛЮЧЕНИЕ\s*:",
    r"\bЗаключение\s*:",
    r"\bзаключение\s*:",
]


def split_report_sections(text: str) -> ReportSections:
    text = text.strip()

    if not text:
        return ReportSections(
            body="",
            conclusion=None,
            has_conclusion=False,
        )

    for pattern in CONCLUSION_PATTERNS:
        match = re.search(pattern, text)

        if match:
            body = text[: match.start()].strip()
            conclusion = text[match.end() :].strip()

            return ReportSections(
                body=body,
                conclusion=conclusion or None,
                has_conclusion=bool(conclusion),
            )

    return ReportSections(
        body=text,
        conclusion=None,
        has_conclusion=False,
    )