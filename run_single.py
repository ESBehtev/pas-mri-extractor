"""
Запуск извлечения признаков для одного MRI-текста.

По умолчанию использует LLM. Если модель не вернула валидный результат,
можно включить fallback на regex-правила.
"""

import argparse
import json
import sys

from pas_mri_extractor.extractor import extract_mri_features
from pas_mri_extractor.rules import rule_extract_features
from pas_mri_extractor.scoring import normalize_mri_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=None,
        help="Model name from configs/models.yaml",
    )

    parser.add_argument(
        "--text",
        required=False,
        help="MRI text as a command line argument",
    )

    parser.add_argument(
        "--text-file",
        required=False,
        help="Path to a text file with MRI description",
    )

    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Use regex rules instead of LLM",
    )

    return parser.parse_args()


def read_mri_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read()

    return sys.stdin.read()


def main() -> None:
    args = parse_args()
    mri_text = read_mri_text(args)

    if not mri_text.strip():
        raise ValueError("MRI text is empty. Use --text, --text-file, or stdin.")

    if args.use_rules:
        extracted = rule_extract_features(mri_text)
    else:
        extracted = extract_mri_features(
            mri_text=mri_text,
            model_name=args.model,
        )

    result = normalize_mri_result(extracted)

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()