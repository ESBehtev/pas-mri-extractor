# AGENTS.md

## Scope

PAS MRI Extractor is research/prototyping software, not a medical device. The
production flow is:

```text
Streamlit / CLI -> pipeline -> extractor -> OpenAI-compatible API -> schema -> scoring
```

Do not alter clinical schemas, prompt semantics, rules, or deterministic
scoring without explicit approval. Preserve positive, negative and uncertain
MRI evidence separately; never invent findings absent from the report.

## Production code

- `core/`: prompts, JSON parsing, rules and scoring.
- `llm/client.py`: OpenAI-compatible client, errors and test mock.
- `schemas/`: Pydantic contracts.
- `services/`: extractor and single production pipeline.
- `api/`: reserved for future FastAPI work.

API configuration comes from `configs/models.yaml` and `PAS_*` environment
variables. Never log API keys.

## Legacy material

`legacy/` contains retrospective evaluation, benchmarks, reports, notebooks,
historical data/artifacts and experimental branches. Production modules must
not import from it. Do not process its clinical tables or run benchmark/eval
scripts locally.

## Local workflow

Allowed: focused code edits, import checks and unit tests. Ask before package
installation, formatting, linting or any non-unit runtime workload. Do not run
model inference, Streamlit, benchmarks, datasets, notebooks or GPU workloads
locally.

Run only the focused unit suite when requested:

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit
```

Data, outputs, models and runtime configuration stay untracked. Use
`pathlib.Path`, simple readable Python, and avoid compatibility wrappers.
