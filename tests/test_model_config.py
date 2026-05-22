import os
import unittest
from unittest.mock import patch

from pas_mri_extractor.models import (
    dry_run_model_config,
    get_available_models,
    get_default_model_name,
)


class ModelConfigTest(unittest.TestCase):
    def test_qwen_3_6_gguf_is_default_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_default_model_name(), "qwen_3_6_35b_gguf")

    def test_pas_model_env_overrides_default_model(self) -> None:
        with patch.dict(os.environ, {"PAS_MODEL": "qwen_2_5_7b"}, clear=True):
            self.assertEqual(get_default_model_name(), "qwen_2_5_7b")

    def test_qwen_2_5_7b_fallback_model_remains_registered(self) -> None:
        models = get_available_models()

        self.assertIn("qwen_2_5_7b", models)
        self.assertEqual(models["qwen_2_5_7b"]["model_id"], "Qwen/Qwen2.5-7B-Instruct")

    def test_dry_run_qwen_3_6_gguf_does_not_load_model(self) -> None:
        dry_run = dry_run_model_config("qwen_3_6_35b_gguf")

        self.assertEqual(dry_run["model"], "qwen_3_6_35b_gguf")
        self.assertEqual(dry_run["backend"], "llama_cpp")
        self.assertEqual(
            dry_run["hf_repo"],
            "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
        )
        self.assertEqual(
            dry_run["expected_path"],
            "models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        )
        self.assertIn(
            "hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF "
            "Qwen3.6-35B-A3B-Q4_K_M.gguf",
            dry_run["download_command"],
        )
        self.assertIn("--model qwen_2_5_7b", dry_run["fallback_command"])

    def test_transformers_qwen_3_6_remains_registered_as_experimental(self) -> None:
        models = get_available_models()

        self.assertIn("qwen_3_6_35b", models)
        self.assertEqual(models["qwen_3_6_35b"]["backend"], "transformers")


if __name__ == "__main__":
    unittest.main()
