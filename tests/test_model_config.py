import os
import unittest
from unittest.mock import patch

from pas_mri_extractor.models import (
    dry_run_model_config,
    get_available_models,
    get_default_model_name,
)


class ModelConfigTest(unittest.TestCase):
    def test_qwen3_6_35b_a3b_gguf_is_default_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_default_model_name(), "qwen3_6_35b_a3b_gguf")

    def test_pas_model_env_overrides_default_model(self) -> None:
        with patch.dict(os.environ, {"PAS_MODEL": "qwen_2_5_7b"}, clear=True):
            self.assertEqual(get_default_model_name(), "qwen_2_5_7b")

    def test_qwen_2_5_7b_fallback_model_remains_registered(self) -> None:
        models = get_available_models()

        self.assertIn("qwen_2_5_7b", models)
        self.assertEqual(models["qwen_2_5_7b"]["model_id"], "Qwen/Qwen2.5-7B-Instruct")

    def test_dry_run_qwen3_6_35b_a3b_gguf_does_not_load_model(self) -> None:
        dry_run = dry_run_model_config("qwen3_6_35b_a3b_gguf")

        self.assertEqual(dry_run["model"], "qwen3_6_35b_a3b_gguf")
        self.assertEqual(dry_run["backend"], "llama_cpp")
        self.assertEqual(
            dry_run["hf_repo"],
            "unsloth/Qwen3.6-35B-A3B-GGUF",
        )
        self.assertEqual(
            dry_run["hf_filename"],
            "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        )
        self.assertTrue(
            dry_run["expected_path"].endswith(
                "/models/Qwen3.6-35B-A3B-GGUF/"
                "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            )
        )
        self.assertIn(
            "hf download unsloth/Qwen3.6-35B-A3B-GGUF "
            "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            dry_run["download_command"],
        )
        self.assertIn("--model qwen_2_5_7b", dry_run["fallback_command"])

        model_cfg = get_available_models()["qwen3_6_35b_a3b_gguf"]
        self.assertEqual(model_cfg["generation"]["temperature"], 0.0)
        self.assertEqual(model_cfg["generation"]["top_p"], 0.8)
        self.assertEqual(model_cfg["generation"]["max_new_tokens"], 2048)
        self.assertEqual(model_cfg["runtime"]["n_ctx"], 8192)
        self.assertTrue(model_cfg["tokenizer"]["use_chat_template"])

    def test_qwen3_27b_gguf_remains_registered(self) -> None:
        models = get_available_models()

        self.assertIn("qwen3_27b_q4_k_m_gguf", models)
        self.assertEqual(models["qwen3_27b_q4_k_m_gguf"]["backend"], "llama_cpp")


if __name__ == "__main__":
    unittest.main()
