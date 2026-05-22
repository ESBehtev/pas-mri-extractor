import unittest
from unittest.mock import patch

from pas_mri_extractor.extractor import parse_json_with_retry
from pas_mri_extractor.models import LoadedModel, build_llama_cpp_prompt


class JsonRetryTest(unittest.TestCase):
    def make_loaded_model(self) -> LoadedModel:
        return LoadedModel(
            name="fake",
            model_id="fake",
            backend="llama_cpp",
            tokenizer=None,
            model=None,
            generation_config={},
            tokenizer_config={},
        )

    def test_retry_runs_once_for_missing_json(self) -> None:
        loaded_model = self.make_loaded_model()

        with patch(
            "pas_mri_extractor.extractor.generate_text",
            return_value='{"schema_version": "1.0"}',
        ) as generate_mock:
            parsed, raw_output = parse_json_with_retry(
                loaded_model,
                "prompt",
                "not json",
            )

        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(raw_output, '{"schema_version": "1.0"}')
        generate_mock.assert_called_once()
        self.assertEqual(generate_mock.call_args.kwargs["retry_json"], True)
        self.assertEqual(
            generate_mock.call_args.kwargs["generation_overrides"]["max_new_tokens"],
            3500,
        )
        self.assertEqual(
            generate_mock.call_args.kwargs["generation_overrides"]["temperature"],
            0.0,
        )

    def test_retry_error_includes_raw_output_debug(self) -> None:
        loaded_model = self.make_loaded_model()

        with patch(
            "pas_mri_extractor.extractor.generate_text",
            return_value="still not json",
        ):
            with self.assertRaises(ValueError) as context:
                parse_json_with_retry(loaded_model, "prompt", "not json")

        message = str(context.exception)
        self.assertIn("raw_output_length=", message)
        self.assertIn("raw_output_first_1500", message)
        self.assertIn("still not json", message)

    def test_llama_cpp_plain_prompt_has_strict_json_instruction(self) -> None:
        prompt = build_llama_cpp_prompt("clinical prompt")

        self.assertTrue(prompt.startswith("Return ONLY one valid JSON object."))
        self.assertIn("The first character of the response must be {.", prompt)
        self.assertIn("clinical prompt", prompt)


if __name__ == "__main__":
    unittest.main()
