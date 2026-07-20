import unittest
from unittest.mock import patch

from pas_mri_extractor.llm import client as models
from pas_mri_extractor.llm.client import LLMConfig, MockLLMClient, OpenAICompatibleClient


class _Completions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": '{"schema_version": "1.0"}'})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _SDK:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _Completions()})()

    def close(self) -> None:
        return None


class _UnsupportedResponseFormatError(Exception):
    pass


class _FallbackCompletions(_Completions):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _UnsupportedResponseFormatError("response_format is unsupported")
        message = type("Message", (), {"content": '{"schema_version": "1.0"}'})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class ApiClientTest(unittest.TestCase):
    def test_mock_client_is_deterministic_and_network_free(self) -> None:
        client = MockLLMClient(["first", "second"])

        self.assertEqual(client.generate("a"), "first")
        self.assertEqual(client.generate("b", retry_json=True), "second")
        self.assertEqual([call["prompt"] for call in client.calls], ["a", "b"])

    def test_openai_client_uses_chat_completions_and_json_mode(self) -> None:
        config = LLMConfig(
            name="test",
            base_url="http://api.example/v1",
            api_key="test-key",
            model="test-model",
            timeout=10,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            fallback_on_unsupported_response_format=False,
        )
        sdk = _SDK()
        client = OpenAICompatibleClient(config, sdk_client=sdk)

        result = client.generate("extract this", generation_overrides={"max_tokens": 256})

        self.assertEqual(result, '{"schema_version": "1.0"}')
        request = sdk.chat.completions.calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["max_tokens"], 256)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["messages"][1]["content"], "extract this")

    def test_response_format_fallback_is_explicitly_opt_in(self) -> None:
        config = LLMConfig(
            name="test",
            base_url="http://api.example/v1",
            api_key="test-key",
            model="test-model",
            timeout=10,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            fallback_on_unsupported_response_format=True,
        )
        sdk = _SDK()
        sdk.chat.completions = _FallbackCompletions()
        client = OpenAICompatibleClient(config, sdk_client=sdk)

        with self.assertLogs("pas_mri_extractor.llm.client", level="WARNING"):
            with patch.object(models, "BadRequestError", _UnsupportedResponseFormatError):
                result = client.generate("extract this")

        self.assertEqual(result, '{"schema_version": "1.0"}')
        self.assertEqual(len(sdk.chat.completions.calls), 2)
        self.assertIn("response_format", sdk.chat.completions.calls[0])
        self.assertNotIn("response_format", sdk.chat.completions.calls[-1])


if __name__ == "__main__":
    unittest.main()
