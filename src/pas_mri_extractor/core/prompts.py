"""
Сборка prompt для LLM.

Берёт production-шаблон из configs/prompts/extractor.yaml и подставляет MRI-текст.
"""

from ..config import load_config


def build_prompt(
    mri_text: str,
    prompt_config_name: str = "prompts/extractor.yaml",
) -> str:
    config = load_config(prompt_config_name)

    template = config.get("template")
    if not template:
        raise ValueError("extractor prompt config must contain 'template' field")

    return template.format(mri_text=mri_text.strip())
