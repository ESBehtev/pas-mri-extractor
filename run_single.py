"""
Запуск извлечения признаков для одного MRI-текста.

По умолчанию использует LLM. Если модель не вернула валидный результат,
можно включить fallback на regex-правила.
"""

import argparse
import json
import sys

from pas_mri_extractor.extractor import extract_mri_features
from pas_mri_extractor.models import ModelConfigError, dry_run_model_config
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

    parser.add_argument(
        "--dry-run-model-config",
        action="store_true",
        help="Validate selected model config and local path without loading weights.",
    )

    parser.add_argument(
        "--print-raw-output",
        action="store_true",
        help="Print raw model output before JSON parsing.",
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

    if args.dry_run_model_config:
        try:
            config_check = dry_run_model_config(args.model)
        except ModelConfigError as error:
            print(str(error), file=sys.stderr)
            sys.exit(2)

        print(
            json.dumps(
                config_check,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    mri_text = read_mri_text(args)

    if not mri_text.strip():
        raise ValueError("MRI text is empty. Use --text, --text-file, or stdin.")

    try:
        if args.use_rules:
            extracted = rule_extract_features(mri_text)
        else:
            extracted = extract_mri_features(
                mri_text=mri_text,
                model_name=args.model,
                print_raw_output=args.print_raw_output,
            )
    except ModelConfigError as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)

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
