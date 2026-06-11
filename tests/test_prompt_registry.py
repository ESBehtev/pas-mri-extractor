import unittest

from pas_mri_extractor.config import load_config
from pas_mri_extractor.prompt_registry import (
    PromptRegistry,
    PromptRegistryError,
    load_stage_prompt,
)


class PromptRegistryTest(unittest.TestCase):
    def test_registry_loads_extractor_prompt_from_current_config(self) -> None:
        registry = PromptRegistry()

        prompt_config = registry.load("extractor")
        legacy_config = load_config("prompt.yaml")

        self.assertEqual(prompt_config["template"], legacy_config["template"])
        self.assertEqual(prompt_config["stage"], "extractor")
        self.assertEqual(prompt_config["prompt_config_name"], "prompt.yaml")
        self.assertEqual(
            prompt_config["registry_metadata"]["source_of_truth"],
            "configs/prompt.yaml",
        )

    def test_missing_stage_raises_clear_error(self) -> None:
        registry = PromptRegistry()

        with self.assertRaises(PromptRegistryError) as context:
            registry.load("validation")

        message = str(context.exception)
        self.assertIn("validation", message)
        self.assertIn("Available stages", message)

    def test_legacy_prompt_yaml_remains_compatible(self) -> None:
        legacy_config = load_config("prompt.yaml")

        self.assertIn("template", legacy_config)
        self.assertIn("{mri_text}", legacy_config["template"])

    def test_extractor_prompt_alias_documents_active_config(self) -> None:
        alias_config = load_config("prompts/extractor.yaml")

        self.assertEqual(alias_config["stage"], "extractor")
        self.assertEqual(alias_config["status"], "alias")
        self.assertEqual(alias_config["active_config"], "prompt.yaml")
        self.assertFalse(alias_config["runtime_active"])

    def test_planned_prompt_can_be_loaded_without_runtime_use(self) -> None:
        prompt_config = load_stage_prompt("risk_prediction")

        self.assertEqual(prompt_config["stage"], "risk_prediction")
        self.assertEqual(prompt_config["status"], "planned_example")
        self.assertIn("template", prompt_config)


if __name__ == "__main__":
    unittest.main()
