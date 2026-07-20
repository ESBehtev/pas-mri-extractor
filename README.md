# PAS MRI Extractor

Research/prototype application that converts free-text pelvic MRI reports into
validated PAS JSON and deterministic research risk/readiness outputs. It is
not a medical device and does not replace clinical judgement.

## Production flow

```text
Streamlit / CLI -> pipeline -> extractor -> OpenAI-compatible API -> schema -> scoring
```

The production package is intentionally limited to one extraction scenario.
Benchmarking, retrospective evaluation, reports, and historical experiments
are kept under `legacy/` and are not imported by the application.

## Configuration

`configs/models.yaml` defines the default OpenAI-compatible API profile. YAML
values can be overridden with environment variables:

- `PAS_API_PROFILE`
- `PAS_API_BASE_URL`
- `PAS_API_KEY`
- `PAS_MODEL`
- `PAS_TIMEOUT`
- `PAS_TEMPERATURE`
- `PAS_MAX_TOKENS`

The API key is never emitted by the CLI dry-run or application logs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

PAS_API_KEY=... PYTHONPATH=src streamlit run app/streamlit_app.py
```

CLI:

```bash
PAS_API_KEY=... PYTHONPATH=src python scripts/run_single.py \
  --text-file path/to/report.txt

PAS_API_KEY=... PYTHONPATH=src python scripts/run_single.py \
  --dry-run-model-config
```

Rule-only fallback for local deterministic checks:

```bash
PYTHONPATH=src python scripts/run_single.py --use-rules --text "MRI report text"
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit
```

## Repository layout

- `src/pas_mri_extractor/core/`: prompts, rules, JSON parsing and scoring.
- `src/pas_mri_extractor/llm/`: OpenAI-compatible client and test mock.
- `src/pas_mri_extractor/schemas/`: canonical Pydantic schemas.
- `src/pas_mri_extractor/services/`: extraction pipeline.
- `app/`: Streamlit interface.
- `legacy/`: isolated historical research material; see `legacy/README.md`.
