import unittest

from app.examples import STREAMLIT_EXAMPLES
from app.provenance import (
    build_report_highlighting,
    evidence_matches_sentence,
    split_report_sentences,
)
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


def example_by_name(name: str) -> dict:
    for example in STREAMLIT_EXAMPLES:
        if example["name"] == name:
            return example

    raise AssertionError(f"Example not found: {name}")


def provenance_for_example(name: str) -> dict:
    example = example_by_name(name)
    extraction = rule_extract_features(example["report_text"])
    result = normalize_mri_result(extraction)

    return build_report_highlighting(example["report_text"], result.model_dump())


def sentence_polarities(report_text: str) -> dict[str, str | None]:
    return {
        sentence.text: sentence.polarity
        for sentence in split_report_sentences(report_text)
    }


class ProvenanceTest(unittest.TestCase):
    def test_token_overlap_matches_evidence_to_sentence(self) -> None:
        self.assertTrue(
            evidence_matches_sentence(
                "достоверной инвазии стенки мочевого пузыря",
                "По передней стенке матки имеется участок достоверной инвазии "
                "стенки мочевого пузыря.",
            )
        )

    def test_negative_phrases_are_not_positive_sentences(self) -> None:
        negative_examples = [
            "Без PAS — простой случай",
            "Без PAS — сложная отрицательная формулировка",
        ]

        for name in negative_examples:
            example = example_by_name(name)
            polarities = sentence_polarities(example["report_text"])

            for sentence, polarity in polarities.items():
                normalized = sentence.lower().replace("ё", "е")
                if (
                    "не выяв" in normalized
                    or "нет" in normalized
                    or "не определяется" in normalized
                    or "не подтверждается" in normalized
                    or "не получено" in normalized
                ):
                    with self.subTest(example=name, sentence=sentence):
                        self.assertNotEqual(polarity, "positive")

    def test_increta_easy_highlights_core_positive_findings(self) -> None:
        provenance = provenance_for_example("Increta — простой случай")
        html = provenance["html"].lower()

        self.assertGreaterEqual(provenance["counts"]["positive"], 4)
        self.assertIn("множественные лакуны", html)
        self.assertIn("ретроплацентарные сосуды расширены", html)
        self.assertIn("истончение миометрия", html)
        self.assertIn("placenta increta", html)

    def test_hard_increta_splits_uncertain_and_negative_bladder_context(self) -> None:
        provenance = provenance_for_example(
            "Increta — сложный случай с возможным вовлечением"
        )
        html = provenance["html"].lower()

        self.assertGreaterEqual(provenance["counts"]["uncertain"], 1)
        self.assertGreaterEqual(provenance["counts"]["negative"], 1)
        self.assertIn("маточно-пузырное пространство частично не дифференцируется", html)
        self.assertIn(
            "достоверных признаков инвазии стенки мочевого пузыря не получено",
            html,
        )

    def test_percreta_highlights_bladder_wall_invasion(self) -> None:
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
                provenance = provenance_for_example(name)

                self.assertGreaterEqual(provenance["counts"]["positive"], 1)
                self.assertIn(phrase, provenance["html"].lower())


if __name__ == "__main__":
    unittest.main()
