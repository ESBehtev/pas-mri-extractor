# AGENTS.md

## Project goal

PAS MRI extractor converts free-text pelvic MRI reports into a structured JSON
description of placenta accreta spectrum (PAS) findings, then computes a
research risk/readiness score and supports retrospective evaluation.

Current end-to-end project stage:

```text
MRI report text
-> Qwen3.6-35B-A3B
-> structured PAS JSON
-> risk calculation
-> quality evaluation on a retrospective clinical sample
```

This is research/prototyping software and is not a medical device.

## Medical context

PAS means placenta accreta spectrum. The project focuses on MRI signs relevant
to abnormal placental adherence/invasion:

- accreta: superficial abnormal adherence;
- increta: deeper myometrial invasion without convincing extrauterine spread;
- percreta: invasion through uterine serosa and/or convincing extension toward
  bladder, parametrium, or beyond the uterus.

The extractor must preserve uncertainty. Positive, negative, and uncertain
evidence are separated in the JSON. Do not invent findings absent from the MRI
report and do not overstate diagnostic confidence.

## Current status

- Main extraction path is implemented.
- LLM output is validated with Pydantic before scoring.
- Scoring is deterministic Python logic backed by `configs/risk_score.yaml`.
- Streamlit review UI exists.
- YAML-based fixture/batch eval exists in `src/pas_mri_extractor/batch_eval.py`.
- Tabular retrospective evaluation is being added through `scripts/`.
- Runtime testing, real inference, Streamlit checks, and dataset processing are
  performed on the server, not locally.

## Development rules for agents

This repository is edited locally, but runtime testing is performed on a
separate server.

Before any retrospective gold labeling, the agent must read
`GOLD_LABELING.md` and follow it. If `GOLD_LABELING.md` contradicts older
instructions or earlier gold-labeling scripts, `GOLD_LABELING.md` has priority
for gold labeling.

Do not run locally:

- `streamlit run`;
- model inference;
- training scripts;
- GPU workloads;
- heavy GGUF model loading;
- long-running scripts;
- data processing over real MRI datasets;
- commands that require real medical data;
- commands that download model weights;
- commands that modify remote servers.

Allowed locally:

- read files;
- edit files;
- inspect project structure;
- update documentation;
- update configs;
- make small code changes;
- dry-run model config checks only.

Ask before running:

- tests;
- linting;
- formatting;
- dependency installation;
- scripts that create, delete, or rewrite files.

Expected workflow:

1. Agent edits code locally.
2. User reviews changes.
3. User commits and pushes to GitHub.
4. Server pulls the branch.
5. User runs tests/app/inference/eval on the server.
6. User sends logs/results back.
7. Agent proposes the next change.

## Stack

- Python 3.11+
- Streamlit
- PyTorch
- Transformers
- llama-cpp-python for GGUF models
- Pandas + openpyxl for retrospective Excel export
- Pydantic
- PyYAML

## Architecture

Core package: `src/pas_mri_extractor/`.

- `schemas.py`: canonical Pydantic schemas for extraction output and full
  scored output.
- `prompts.py`: builds the LLM prompt from `configs/prompt.yaml`.
- `models.py`: resolves model configs, loads transformers or llama.cpp models,
  supports dry-run model config checks, and configures JSON mode where possible.
- `extractor.py`: builds prompt, calls the model, extracts/repairs JSON, and
  validates `MRIExtractionResult`.
- `pipeline.py`: process-level model lifecycle wrapper and main reusable
  `extract_features_with_artifacts()` entry point.
- `rules.py`: regex/rule baseline used as fallback and lightweight eval mode.
- `scoring.py`: deterministic scoring, predicted risks, readiness level, and
  rationale generation.
- `json_utils.py`: robust JSON extraction from raw model output.
- `batch_eval.py`: existing YAML-driven eval/fixture runner.

App/UI package: `app/`.

- `streamlit_app.py`: main Streamlit app.
- `state.py`, `components.py`, `provenance.py`, `examples.py`,
  `config_studio.py`: UI state, rendering, provenance helpers, examples, and
  risk-score config controls.

New tabular eval scripts:

- `scripts/export_excel_dataset.py`: export third Excel sheet to CSV/JSONL.
- `scripts/evaluate_predictions.py`: run extraction over exported dataset and
  compute metrics for filled gold labels.

## Directory structure

- `app/`: Streamlit review interface.
- `configs/`: prompts, model registry, rules, scoring, eval YAML examples.
- `examples/`: small non-sensitive example MRI text.
- `notebooks/`: exploratory notebooks.
- `scripts/`: dataset export and retrospective evaluation scripts.
- `src/pas_mri_extractor/`: package source.
- `tests/`: unit/regression tests.
- `data/`: local clinical/evaluation data, ignored by git.
- `outputs/`: generated outputs and predictions, ignored by git.
- `models/`: local model weights, ignored by git.
- `runtime_configs/`: local runtime overrides, ignored by git.

## Models

Models are configured in `configs/models.yaml`.

Current default:

- `qwen3_6_35b_a3b_gguf`: Qwen3.6-35B-A3B GGUF UD-Q4_K_M via llama.cpp.

Additional configured models:

- `qwen3_27b_q4_k_m_gguf`: Qwen3.6-27B GGUF Q4_K_M via llama.cpp.
- `qwen_2_5_7b`: Qwen2.5-7B-Instruct via transformers fallback.

Both GGUF configs use deterministic generation (`temperature: 0.0`) and
backend-level JSON mode when supported:

```yaml
output:
  response_format:
    type: json_object
  enforce_json: true
```

Dry-run config checks do not load model weights and must not be described as
real model validation:

```bash
PYTHONPATH=src python run_single.py --model qwen3_6_35b_a3b_gguf --dry-run-model-config
PYTHONPATH=src python run_single.py --model qwen3_27b_q4_k_m_gguf --dry-run-model-config
```

## Input format

Primary input is a plain-text MRI report. It may be passed as:

- `--text` to `run_single.py`;
- `--text-file` to `run_single.py`;
- stdin to `run_single.py`;
- pasted/uploaded text in Streamlit;
- one or more table columns in retrospective eval.

For tabular evaluation, `scripts/evaluate_predictions.py` looks for MRI text in
columns such as:

- `МРТ_Описание`, `МРТ Описание`, `Описание МРТ`, `Описание`;
- `МРТ_Заключение`, `МРТ Заключение`, `Заключение МРТ`, `Заключение`;
- English analogs: `MRI_description`, `mri_description`,
  `MRI_conclusion`, `mri_conclusion`.

Use `--text-columns` when the dataset uses different names.

## Canonical output JSON

The LLM must return only:

```json
{
  "schema_version": "1.0",
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
  "suspicion": {
    "highest_suspected_extent": "none",
    "percreta_suspicion": "absent",
    "bladder_serosa_suspicion": "absent",
    "rationale": []
  },
  "evidence": {
    "positive_findings": [],
    "uncertain_findings": [],
    "negative_findings": []
  }
}
```

After validation, `scoring.normalize_mri_result()` adds:

- `score.clinical_score`;
- `score.risk_group`;
- `score.red_flag`;
- `score.score_reasons`;
- `predicted_risks.massive_blood_loss_over_1500_ml_percent`;
- `predicted_risks.estimated_blood_loss_ml_range`;
- `predicted_risks.vascular_intervention_percent`;
- `predicted_risks.bladder_involvement_percent`;
- `recommendation.readiness_level`;
- `recommendation.readiness_text`;
- `computed_rationale`.

## PAS feature schema

Enums:

- `invasion.type`: `none`, `accreta`, `increta`, `percreta`.
- `invasion.confidence`: `absent`, `possible`, `probable`, `definite`,
  `unclear`.
- Feature statuses: `absent`, `possible`, `probable`, `present`.

Feature groups:

- `case_info`: gestational week and previous C-section count.
- `extracted_features.invasion`: primary PAS type and confidence.
- `extracted_features.anatomy`: bladder, parametrium, posterior wall.
- `extracted_features.placenta_location`: previa and anterior placenta.
- `extracted_features.mri_signs`: retroplacental vessels, lacunae, wall
  thinning, hernia/bulging.
- `extracted_features.clinical_context`: preoperative bleeding.
- `suspicion`: safety block for uncertain higher-grade disease such as
  "cannot exclude percreta"; it does not overwrite `invasion.type`.
- `evidence`: positive, uncertain, and negative source phrases.

Do not reintroduce the legacy flat `features` payload. It is explicitly
rejected by schema/scoring code.

## Current scoring system

Config-driven weights live in `configs/risk_score.yaml`; hard overrides live in
`src/pas_mri_extractor/scoring.py`.

Base score weights:

- invasion type: accreta +1, increta +3, percreta +5;
- invasion confidence: definite +2, probable/possible +1;
- bladder involvement: present +3, probable/possible +2;
- parametrium involvement present +3;
- uterine wall thinning present +1;
- uterine hernia/bulging present +2;
- retroplacental vessels present +1;
- lacunae present +1;
- placenta previa present +1;
- anterior placenta present +1;
- preoperative bleeding present +2;
- previous C-section count >=2 adds +1.

Risk groups:

- low: 0-3;
- moderate: 4-9;
- high: 10-100.

Important hard rules:

- percreta forces high risk and minimum score 8;
- percreta plus possible/probable/present bladder involvement forces minimum
  score 9 and red flag;
- bladder present/probable is a red flag;
- increta without bladder involvement is forced to moderate;
- possible/probable/present percreta suspicion prevents low-risk downgrade.

Predicted risks and readiness levels are deterministic outputs derived from the
risk group plus post-processing caps/overrides in `scoring.py`.

## Running the project

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run Streamlit on server:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py --server.port 8501
```

Run single report on server:

```bash
PYTHONPATH=src python run_single.py \
  --model qwen3_6_35b_a3b_gguf \
  --text-file examples/sample_mri.txt
```

Rule-only local/server baseline:

```bash
PYTHONPATH=src python run_single.py \
  --use-rules \
  --text-file examples/sample_mri.txt
```

Existing YAML batch eval on server:

```bash
PYTHONPATH=src python -m pas_mri_extractor.batch_eval \
  --config configs/eval_qwen.yaml \
  --model qwen3_6_35b_a3b_gguf
```

## Retrospective evaluation pipeline

The clinical workbook is expected as an `.xlsx` file in the project root or as
an explicit `--xlsx` path. Do not commit the workbook.

Export third sheet to working artifacts:

```bash
PYTHONPATH=src python scripts/export_excel_dataset.py
```

Explicit workbook path:

```bash
PYTHONPATH=src python scripts/export_excel_dataset.py \
  --xlsx "Копия Итог 2 этап_v2 (00000002).xlsx"
```

Outputs:

- `data/evaluation/dataset_sheet3.csv`;
- `data/evaluation/dataset_sheet3.jsonl`.

The export script:

- reads the third sheet by default (`--sheet-index 2`);
- preserves all source columns;
- creates/fills stable `case_id`;
- adds gold annotation placeholders if missing for PAS JSON features,
  suspicion fields, outcomes, readiness/risk, confidence, and rationale;
- handles Russian column names and empty values.

Build retrospective gold labels in new files without changing
`dataset_sheet3.csv`:

```bash
PYTHONPATH=src python scripts/build_gold_dataset.py
```

Outputs:

- `data/evaluation/dataset_sheet3_gold.csv`;
- `data/evaluation/dataset_sheet3_gold.jsonl`.

The gold builder uses available clinical fields such as `МРТ_Описание`,
`МРТ_Заключение`, `ДиагнозыВыпЭпикриза`, `ПоказанияКОперации`,
`Ход Вмешательства`, `КровопотеряРоды`, and `КровопотеряОперация`.
It is a deterministic retrospective helper for scientific evaluation, not a
clinical conclusion.

Rule baseline evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --use-rules \
  --limit 10
```

LLM evaluation on server:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --run-llm \
  --model qwen3_6_35b_a3b_gguf \
  --limit 10
```

Outputs:

- `outputs/evaluation/predictions.jsonl`;
- `outputs/evaluation/metrics.json`.

Evaluation tasks:

- invasion type and confidence;
- anatomy, placenta-location, MRI-sign, clinical-context, and suspicion fields;
- massive blood loss;
- blood loss class;
- bladder involvement;
- vascular intervention;
- readiness level;
- risk group.

Metrics are computed only for rows where the corresponding `gold_*` field is
filled. Empty gold labels are skipped per task. For blood-loss tasks,
`gold_blood_loss_ml` is used as a fallback when class/boolean labels are empty.
`evaluate_predictions.py` prefers `dataset_sheet3_gold.csv/jsonl` when present
and falls back to the unannotated export. Current metrics are accuracy,
precision, recall, F1, and confusion matrix.

## Data and privacy rules

Never commit:

- MRI data;
- DICOM/NIfTI files;
- personal medical data;
- Excel/CSV/JSONL clinical tables;
- `.env`;
- API keys or tokens;
- model weights;
- cache folders;
- local virtual environments;
- generated outputs containing patient data.

Ignored folders/artifacts include:

- `.venv/`;
- `data/`;
- `outputs/`;
- `.cache/`;
- `models/`;
- `runtime_configs/`;
- `__pycache__/`;
- `*.xlsx`, `*.xls`, medical image formats, model weights.

## Model lifecycle and VRAM safety

- Keep only one heavyweight model loaded per Python process.
- When switching models, unload the previous model before loading the next one.
- For `llama_cpp` objects, call `.close()` when available.
- Clear local references, run `gc.collect()`, and call
  `torch.cuda.empty_cache()` when CUDA is available.
- Do not add hidden `st.cache_resource` or global singleton paths that can keep
  old GGUF, tokenizer, or transformers objects alive.
- Treat Streamlit multi-session model switching as a server-side VRAM risk;
  lifecycle changes require careful review.
- Do not claim a GGUF model is tested or verified if only
  `--dry-run-model-config` was executed.

## Known limitations

- Scoring is heuristic/research logic, not clinically validated.
- Blood loss and vascular intervention are predicted risk proxies, not direct
  outcome extraction from operative records.
- Thresholds for binary evaluation of predicted risk percentages are simple
  defaults and may need calibration.
- MRI text column names in retrospective tables may vary; use `--text-columns`
  when auto-detection fails.
- Gold fields are placeholders until manual or semi-automatic annotation is
  completed.
- LLM JSON mode depends on installed `llama-cpp-python` support; fallback
  prompt/postprocessing remains in place.
- Local agents must not process real clinical spreadsheets or run inference.

## Current Development Roadmap

1. Export third Excel sheet into stable CSV/JSONL evaluation artifacts.
2. Add and document gold annotation fields for retrospective labels.
3. Run rule-baseline eval on a small limit to verify table/text plumbing.
4. Run Qwen3.6-35B-A3B eval on server with `--limit 10`, inspect failures.
5. Fill or map gold labels for blood loss, bladder involvement, PAS type, and
   readiness.
6. Calibrate outcome label normalization and risk thresholds against the
   retrospective sample.
7. Expand metrics reporting with per-class supports and review-friendly error
   summaries.
8. Feed systematic extraction errors back into prompt/rules only with schema and
   scoring compatibility preserved.

## Near-term tasks

- Run `scripts/export_excel_dataset.py` on the server or approved secure
  environment.
- Inspect exported columns and pass `--text-columns` if automatic MRI text
  detection misses the correct fields.
- Complete/fill gold annotation columns.
- Run `scripts/evaluate_predictions.py --use-rules --limit 10` as a plumbing
  check.
- Run `scripts/evaluate_predictions.py --run-llm --model qwen3_6_35b_a3b_gguf
  --limit 10` on the server.
- Review `outputs/evaluation/predictions.jsonl` and
  `outputs/evaluation/metrics.json` before scaling to the full sample.

## Code style

- Prefer simple, readable Python.
- Prefer `pathlib.Path` for filesystem paths.
- Keep functions small and testable.
- Avoid unnecessary abstractions.
- Do not rewrite large parts of the project without explicit approval.
- Do not change public JSON schemas without an explicit migration/backward
  compatibility plan.
- Do not silently change clinical scoring logic.
- Do not break existing eval configs or output artifact structure.
- Do not change prompt enums without an explicit schema/eval update.
- Preserve backward compatibility for older extraction payloads unless a
  deliberate migration is documented.
