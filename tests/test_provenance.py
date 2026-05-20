import html
import unittest

from app.examples import STREAMLIT_EXAMPLES
from app.provenance import build_report_highlighting
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


def example_by_name(name: str) -> dict:
    for example in STREAMLIT_EXAMPLES:
        if example["name"] == name:
            return example

    raise AssertionError(f"Example not found: {name}")


def provenance_for_example(name: str) -> tuple[dict, str]:
    example = example_by_name(name)
    extraction = rule_extract_features(example["report_text"])
    result = normalize_mri_result(extraction)

    return build_report_highlighting(example["report_text"], result.model_dump()), example[
        "report_text"
    ]


def matched_texts(provenance: dict, report_text: str, polarity: str | None = None) -> list[str]:
    matches = provenance["matches"]
    if polarity:
        matches = [match for match in matches if match["polarity"] == polarity]

    return [report_text[match["start"] : match["end"]].lower() for match in matches]


class ProvenanceTest(unittest.TestCase):
    def test_full_report_is_rendered_and_line_breaks_are_preserved(self) -> None:
        report = "Первая строка <unsafe>\nВторая строка без evidence."
        result = {
            "evidence": {
                "positive_findings": [],
                "uncertain_findings": [],
                "negative_findings": [],
            }
        }

        provenance = build_report_highlighting(report, result)

        self.assertEqual(provenance["html"], html.escape(report))
        self.assertIn("\n", provenance["html"])
        self.assertEqual(provenance["matches"], [])
        self.assertEqual(provenance["unmatched_evidence"], [])
        self.assertNotIn("counts", provenance)

    def test_unmatched_evidence_is_stable(self) -> None:
        provenance = build_report_highlighting(
            "Плацента по задней стенке.",
            {
                "evidence": {
                    "positive_findings": ["placenta increta"],
                    "uncertain_findings": [],
                    "negative_findings": [],
                }
            },
        )

        self.assertEqual(provenance["matches"], [])
        self.assertEqual(provenance["unmatched_evidence"], ["placenta increta"])

    def test_overlap_priority_is_uncertain_positive_negative(self) -> None:
        report = "Нельзя исключить вовлечение мочевого пузыря."
        result = {
            "evidence": {
                "positive_findings": ["вовлечение мочевого пузыря"],
                "uncertain_findings": [
                    "Нельзя исключить вовлечение мочевого пузыря"
                ],
                "negative_findings": ["вовлечение мочевого пузыря"],
            }
        }

        provenance = build_report_highlighting(report, result)

        self.assertEqual(len(provenance["matches"]), 1)
        self.assertEqual(provenance["matches"][0]["polarity"], "uncertain")

    def test_token_overlap_is_conservative_fallback(self) -> None:
        report = "Определяется достоверная инвазия стенки мочевого пузыря."
        result = {
            "evidence": {
                "positive_findings": [
                    "достоверной инвазии стенки мочевого пузыря"
                ],
                "uncertain_findings": [],
                "negative_findings": [],
            }
        }

        provenance = build_report_highlighting(report, result)

        self.assertEqual(len(provenance["matches"]), 1)
        self.assertEqual(provenance["matches"][0]["method"], "token-overlap")
        self.assertIn("стенки мочевого пузыря", provenance["html"].lower())

    def test_no_pas_examples_have_no_accidental_positive_pas_highlight(self) -> None:
        for name in [
            "Без PAS — простой случай",
            "Без PAS — сложная отрицательная формулировка",
        ]:
            with self.subTest(example=name):
                provenance, report_text = provenance_for_example(name)
                positive_text = "\n".join(matched_texts(provenance, report_text, "positive"))

                self.assertNotIn("placenta increta", positive_text)
                self.assertNotIn("инвазии мочевого пузыря", positive_text)
                self.assertNotIn("врастание плаценты", positive_text)

    def test_increta_easy_highlights_only_evidence_phrases(self) -> None:
        provenance, report_text = provenance_for_example("Increta — простой случай")
        highlights = "\n".join(matched_texts(provenance, report_text))

        self.assertIn("множественные лакуны", highlights)
        self.assertIn("ретроплацентарные сосуды расширены", highlights)
        self.assertIn("истончение миометрия", highlights)
        self.assertIn("placenta increta", highlights)
        self.assertNotIn("состояние после двух кесаревых сечений", highlights)

    def test_hard_increta_highlights_evidence_without_sentence_inference(self) -> None:
        provenance, report_text = provenance_for_example(
            "Increta — сложный случай с возможным вовлечением"
        )
        uncertain_text = "\n".join(matched_texts(provenance, report_text, "uncertain"))
        negative_text = "\n".join(matched_texts(provenance, report_text, "negative"))

        self.assertIn(
            "маточно-пузырное пространство частично не дифференцируется",
            uncertain_text,
        )
        self.assertIn(
            "достоверных признаков инвазии стенки мочевого пузыря не получено",
            negative_text,
        )

    def test_percreta_examples_highlight_bladder_wall_invasion(self) -> None:
        for name, phrase in [
            (
                "Percreta — простой случай",
                "достоверная инвазия стенки мочевого пузыря",
            ),
            (
                "Percreta — сложный случай со смешанными формулировками",
                "достоверной инвазии стенки мочевого пузыря",
            ),
        ]:
            with self.subTest(example=name):
                provenance, report_text = provenance_for_example(name)
                positive_text = "\n".join(
                    matched_texts(provenance, report_text, "positive")
                )

                self.assertIn(phrase, positive_text)


if __name__ == "__main__":
    unittest.main()
