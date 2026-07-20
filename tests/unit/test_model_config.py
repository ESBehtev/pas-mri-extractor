import os
import unittest
from unittest.mock import patch

from pas_mri_extractor.config import config_overrides
from pas_mri_extractor.llm.client import (
    ModelConfigError,
    dry_run_model_config,
    get_available_models,
    get_default_model_name,
    resolve_llm_config,
)


class ModelConfigTest(unittest.TestCase):
    def test_openai_compatible_profile_is_default(self) -> None:
        with patch.dict(os.environ, {"PAS_API_KEY": "test-key"}, clear=True):
            self.assertEqual(get_default_model_name(), "openai_compatible")
            self.assertIn("openai_compatible", get_available_models())

    def test_environment_overrides_yaml_runtime_values(self) -> None:
        environment = {
            "PAS_API_KEY": "test-key",
            "PAS_API_BASE_URL": "http://vllm.example/v1/",
            "PAS_MODEL": "Qwen/test",
            "PAS_TIMEOUT": "12.5",
            "PAS_TEMPERATURE": "0.2",
            "PAS_MAX_TOKENS": "1024",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = resolve_llm_config()

        self.assertEqual(config.base_url, "http://vllm.example/v1")
        self.assertEqual(config.model, "Qwen/test")
        self.assertEqual(config.timeout, 12.5)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.max_tokens, 1024)

    def test_missing_api_key_has_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ModelConfigError, "PAS_API_KEY"):
                resolve_llm_config()

    def test_dry_run_never_returns_api_key(self) -> None:
        with patch.dict(os.environ, {"PAS_API_KEY": "secret-value"}, clear=True):
            dry_run = dry_run_model_config()

        self.assertEqual(dry_run["provider"], "openai_compatible")
        self.assertTrue(dry_run["api_key_configured"])
        self.assertNotIn("secret-value", repr(dry_run))
        self.assertNotIn("api_key", dry_run)

    def test_response_format_fallback_is_opt_in(self) -> None:
        override = {
            "models.yaml": {
                "default_model": "test",
                "models": {
                    "test": {
                        "base_url": "http://api.example/v1",
                        "api_key_env": "PAS_API_KEY",
                        "model": "test-model",
                        "output": {"enforce_json": True, "response_format": {"type": "json_object"}},
                    }
                },
            }
        }
        with config_overrides(override), patch.dict(os.environ, {"PAS_API_KEY": "test-key"}, clear=True):
            config = resolve_llm_config()

        self.assertFalse(config.fallback_on_unsupported_response_format)


if __name__ == "__main__":
    unittest.main()
