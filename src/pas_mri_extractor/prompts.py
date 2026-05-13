"""
Сборка prompt для LLM.

Берёт шаблон из configs/prompt.yaml и подставляет MRI-текст.
"""

from .config import load_config


def build_prompt(mri_text: str, prompt_config_name: str = "prompt.yaml") -> str:
    config = load_config(prompt_config_name)

    template = config.get("template")
    if not template:
        raise ValueError("prompt.yaml must contain 'template' field")

    return template.format(mri_text=mri_text.strip())