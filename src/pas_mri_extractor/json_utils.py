"""
Утилиты для работы с JSON-ответом модели.

Модель иногда возвращает текст до или после JSON,
поэтому здесь извлекается первый валидный JSON-объект.
"""

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    # сначала пробуем распарсить ответ как есть
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ищем первый JSON-объект вручную
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model output")

    brace_count = 0
    end = None

    for i in range(start, len(text)):
        char = text[i]

        if char == "{":
            brace_count += 1

        elif char == "}":
            brace_count -= 1

            if brace_count == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("Incomplete JSON object in model output")

    json_text = text[start:end]

    return json.loads(json_text)