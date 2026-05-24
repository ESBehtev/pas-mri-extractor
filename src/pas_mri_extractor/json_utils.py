"""
Утилиты для работы с JSON-ответом модели.

Модель иногда возвращает текст до или после JSON,
поэтому здесь извлекается первый валидный JSON-объект.
"""

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    decoder = json.JSONDecoder()

    try:
        parsed, end = decoder.raw_decode(text)
        if isinstance(parsed, dict) and not text[end:].strip():
            return parsed
    except json.JSONDecodeError:
        pass

    start_positions = [
        index for index, char in enumerate(text) if char == "{"
    ]

    if not start_positions:
        raise ValueError("No JSON object found in model output")

    last_error: json.JSONDecodeError | None = None
    for start in start_positions:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as error:
            last_error = error
            continue

        if isinstance(parsed, dict):
            return parsed

    if text.rfind("}") < start_positions[0]:
        raise ValueError("Incomplete JSON object in model output")

    message = "No valid JSON object found in model output"
    if last_error is not None:
        message = f"{message}: {last_error.msg}"
    raise ValueError(message)
