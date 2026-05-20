import unittest

from pas_mri_extractor.prompts import build_prompt


SYNTHETIC_REPORT = """
МРТ малого таза. Беременность 34 недели.
Плацента по передней стенке, перекрывает внутренний зев.
В области рубца миометрий местами не прослеживается.
Определяются множественные плацентарные лакуны.
Ретроплацентарные сосуды расширены.
Достоверных признаков инвазии стенки мочевого пузыря не получено.
Нельзя исключить начальное вовлечение серозной оболочки.

Заключение: МР-картина соответствует placenta increta.
""".strip()


class PromptEvidenceContractTest(unittest.TestCase):
    def test_prompt_requires_source_phrase_evidence_for_present_features(self) -> None:
        prompt = build_prompt(SYNTHETIC_REPORT)

        self.assertIn("для каждого признака со статусом", prompt.lower())
        self.assertIn("source phrase", prompt)
        self.assertIn("не выдумывай evidence", prompt.lower())
        self.assertIn("не только заключение", prompt.lower())

        expected_source_phrases = [
            "МР-картина соответствует placenta increta",
            "множественные плацентарные лакуны",
            "ретроплацентарные сосуды расширены",
            "миометрий местами не прослеживается",
        ]

        for phrase in expected_source_phrases:
            self.assertIn(phrase, prompt)

    def test_prompt_keeps_uncertain_and_negative_evidence_separate(self) -> None:
        prompt = build_prompt(SYNTHETIC_REPORT)

        self.assertIn(
            "достоверных признаков инвазии стенки мочевого пузыря не получено",
            prompt,
        )
        self.assertIn("нельзя исключить начальное вовлечение", prompt)
        self.assertIn("negative_findings должен содержать", prompt)
        self.assertIn("uncertain_findings должен содержать", prompt)


if __name__ == "__main__":
    unittest.main()
