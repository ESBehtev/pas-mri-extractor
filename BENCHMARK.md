# PAS MRI Qwen35B Benchmark Pipeline

This document describes the server-side benchmark workflow for comparing the
new Qwen35B extractor against baseline predictions and gold labels on 20 PAS
MRI cases.

Do not run model inference locally. Run `scripts/run_benchmark.py` only on the
server/runtime environment with model weights available.

## Inputs

Expected benchmark input:

- `data/pas20.json`: 20 benchmark cases.
- `data/evaluation/gold.jsonl`: gold labels in `gold_*` format.
- `outputs/baseline/`: baseline extractor JSON files, or an already normalized
  `outputs/baseline_normalized.json`.

Each case in `data/pas20.json` may be a JSON object with at least:

```json
{
  "case_id": "case_000001",
  "МРТ_Описание": "...",
  "МРТ_Заключение": "..."
}
```

The scripts also accept JSONL/CSV inputs and common column aliases.

## Outputs

Inference outputs:

- `outputs/full/case_XXXXX_raw.txt`
- `outputs/full/case_XXXXX.json`
- `outputs/description_only/case_XXXXX_raw.txt`
- `outputs/description_only/case_XXXXX.json`

Normalized outputs:

- `outputs/full_normalized.json`
- `outputs/description_only_normalized.json`
- `outputs/baseline_normalized.json`

Metric/comparison outputs:

- `outputs/full_metrics.json`
- `outputs/description_only_metrics.json`
- `outputs/baseline_metrics.json`
- `outputs/mode_comparison.json`
- `outputs/baseline_comparison.json`
- `reports/benchmark_report.md`

## Commands

Inference, full MRI:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --input data/pas20.json \
  --mode full \
  --model qwen3_6_35b_a3b_gguf \
  --output outputs/full \
  --run-llm
```

Inference, description only:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --input data/pas20.json \
  --mode description_only \
  --model qwen3_6_35b_a3b_gguf \
  --output outputs/description_only \
  --run-llm
```

Normalize predictions:

```bash
python scripts/normalize_predictions.py \
  --input outputs/full \
  --output outputs/full_normalized.json \
  --jsonl-output outputs/full_normalized.jsonl \
  --csv-output outputs/full_normalized.csv

python scripts/normalize_predictions.py \
  --input outputs/description_only \
  --output outputs/description_only_normalized.json \
  --jsonl-output outputs/description_only_normalized.jsonl \
  --csv-output outputs/description_only_normalized.csv

python scripts/normalize_predictions.py \
  --input outputs/baseline \
  --output outputs/baseline_normalized.json \
  --jsonl-output outputs/baseline_normalized.jsonl \
  --csv-output outputs/baseline_normalized.csv
```

Evaluate against gold:

```bash
python scripts/evaluate.py \
  --gold data/evaluation/gold.jsonl \
  --pred outputs/full_normalized.json \
  --output outputs/full_metrics.json

python scripts/evaluate.py \
  --gold data/evaluation/gold.jsonl \
  --pred outputs/description_only_normalized.json \
  --output outputs/description_only_metrics.json

python scripts/evaluate.py \
  --gold data/evaluation/gold.jsonl \
  --pred outputs/baseline_normalized.json \
  --output outputs/baseline_metrics.json
```

Compare modes:

```bash
python scripts/compare_modes.py \
  --full outputs/full_normalized.json \
  --description-only outputs/description_only_normalized.json \
  --output outputs/mode_comparison.json
```

Compare baseline with Qwen35B:

```bash
python scripts/compare_baseline.py \
  outputs/baseline_normalized.json \
  outputs/full_normalized.json \
  outputs/description_only_normalized.json \
  data/evaluation/gold.jsonl \
  --output outputs/baseline_comparison.json \
  --csv-output outputs/baseline_comparison.csv
```

Build report:

```bash
python scripts/build_report.py \
  --gold data/evaluation/gold.jsonl \
  --cases data/pas20.json \
  --baseline outputs/baseline_normalized.json \
  --full outputs/full_normalized.json \
  --description-only outputs/description_only_normalized.json \
  --baseline-metrics outputs/baseline_metrics.json \
  --full-metrics outputs/full_metrics.json \
  --description-only-metrics outputs/description_only_metrics.json \
  --mode-comparison outputs/mode_comparison.json \
  --baseline-comparison outputs/baseline_comparison.json \
  --output reports/benchmark_report.md
```

## Presentation metrics and reports

Presentation metrics:

```bash
python scripts/presentation_metrics.py \
  --gold data/evaluation/pas20.jsonl \
  --full outputs/full_normalized.json \
  --description-only outputs/description_only_normalized.json \
  --output-json reports/presentation_metrics.json \
  --output-md reports/presentation_metrics.md
```

Presentation reports:

```bash
python scripts/build_presentation_reports.py \
  --metrics reports/presentation_metrics.json \
  --error-review reports/error_review.jsonl \
  --output-dir reports
```

## Notes

- `mode=full` joins `МРТ_Описание` and `МРТ_Заключение`.
- `mode=description_only` uses only `МРТ_Описание`; conclusion is excluded.
- Binary metrics treat `possible`, `probable`, and `present` as positive.
- Categorical metrics use exact-match accuracy and confusion matrices.
- Missing gold labels are skipped per field.
