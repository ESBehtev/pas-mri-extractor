"""
Server-side smoke test for llama.cpp GGUF models.

This command intentionally loads the configured model and runs one tiny JSON
generation. Do not run it on local machines without the GGUF model.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from pas_mri_extractor.extractor import postprocess_extraction
from pas_mri_extractor.json_utils import extract_json_object
from pas_mri_extractor.models import (
    ModelConfigError,
    generate_text,
    get_model_config,
    load_llm,
)
from pas_mri_extractor.schemas import MRIExtractionResult


SMOKE_PROMPT = """
Extract PAS MRI features from this short report and return only canonical JSON.

MRI report:
Pregnancy 34 weeks. Placenta is anterior and low lying. No convincing signs of
placenta accreta spectrum. Bladder wall is intact. No parametrial invasion.

Use this exact top-level JSON structure:
{
  "case_info": {
    "gestational_week": null,
    "previous_cs_count": null
  },
  "extracted_features": {
    "invasion": {
      "type": "none",
      "confidence": "absent"
    },
    "anatomy": {
      "bladder_involvement": "absent",
      "parametrium_involvement": "absent",
      "posterior_wall_involvement": "absent"
    },
    "placenta_location": {
      "placenta_previa": "absent",
      "anterior_placenta": "absent"
    },
    "mri_signs": {
      "retroplacental_vessels": "absent",
      "lacunae": "absent",
      "uterine_wall_thinning": "absent",
      "uterine_hernia_or_bulging": "absent"
    },
    "clinical_context": {
      "preoperative_bleeding": "absent"
    }
  },
  "evidence": {
    "positive_findings": [],
    "uncertain_findings": [],
    "negative_findings": []
  }
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a configured llama_cpp GGUF model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from configs/models.yaml.",
    )
    parser.add_argument(
        "--raw-output-file",
        default="outputs/smoke/llama_cpp_raw.txt",
        help="Where to save raw model output.",
    )
    return parser.parse_args()


def count_tokens(model: Any, text: str) -> int | None:
    if not hasattr(model, "tokenize"):
        return None

    try:
        return len(model.tokenize(text.encode("utf-8")))
    except (TypeError, ValueError, RuntimeError):
        return None


def write_raw_output(path_value: str, raw_output: str) -> str:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_output, encoding="utf-8")
    return str(path)


def main() -> None:
    args = parse_args()

    try:
        model_name, model_cfg = get_model_config(args.model)
        if model_cfg.get("backend") != "llama_cpp":
            raise ModelConfigError(
                f"Model {model_name} uses backend '{model_cfg.get('backend')}', "
                "expected 'llama_cpp'."
            )

        load_start = perf_counter()
        loaded_model = load_llm(model_name)
        load_seconds = perf_counter() - load_start

        generation_start = perf_counter()
        raw_output = generate_text(
            loaded_model,
            SMOKE_PROMPT,
            generation_overrides={
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        generation_seconds = perf_counter() - generation_start

        raw_output_file = write_raw_output(args.raw_output_file, raw_output)

        parsed = extract_json_object(raw_output)
        parsed = postprocess_extraction(parsed)
        validated = MRIExtractionResult.model_validate(parsed)

        prompt_tokens = count_tokens(loaded_model.model, SMOKE_PROMPT)
        output_tokens = count_tokens(loaded_model.model, raw_output)

    except (ModelConfigError, ValidationError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "model": loaded_model.name,
                "model_path": loaded_model.model_id,
                "raw_output_file": raw_output_file,
                "load_seconds": round(load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "schema_version": validated.schema_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
