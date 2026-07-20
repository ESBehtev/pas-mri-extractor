import unittest

from pas_mri_extractor.llm.client import MockLLMClient
from pas_mri_extractor.services.extractor import parse_json_with_retry


class JsonRetryTest(unittest.TestCase):
    def test_retry_runs_once_for_missing_json(self) -> None:
        client = MockLLMClient(['{"schema_version": "1.0"}'])

        parsed, raw_output = parse_json_with_retry(client, "prompt", "not json")

        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(raw_output, '{"schema_version": "1.0"}')
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["retry_json"])
        self.assertEqual(client.calls[0]["generation_overrides"]["max_tokens"], 3500)
        self.assertEqual(client.calls[0]["generation_overrides"]["temperature"], 0.0)

    def test_retry_error_includes_raw_output_debug(self) -> None:
        client = MockLLMClient(["still not json"])

        with self.assertRaises(ValueError) as context:
            parse_json_with_retry(client, "prompt", "not json")

        message = str(context.exception)
        self.assertIn("raw_output_length=", message)
        self.assertIn("raw_output_first_1500", message)
        self.assertIn("still not json", message)


if __name__ == "__main__":
    unittest.main()
