# AGENTS.md

## Project

PAS MRI extractor project.

The project extracts structured clinical features from MRI reports and supports Streamlit-based review, analysis, and future segmentation/classification workflows.

## Current development mode

This repository is edited locally, but runtime testing is performed on a separate server.

The coding agent should inspect, edit, refactor, and reason about the code locally, but must not run the application, models, or long-running commands on the local machine.

## Strict rules for the coding agent

Do not run locally:

- `streamlit run`
- model inference
- training scripts
- GPU workloads
- long-running scripts
- data processing over real MRI datasets
- commands that require real medical data
- commands that download model weights
- commands that modify remote servers

Allowed locally:

- read files
- edit files
- inspect project structure
- update documentation
- update configs
- make small code changes
- suggest commands for the user to run on the server

Ask before running:

- tests
- linting
- formatting
- dependency installation
- scripts that create, delete, or rewrite files

## Deployment/testing workflow

Expected workflow:

1. Agent edits code locally.
2. User reviews changes.
3. User commits and pushes to GitHub.
4. Server pulls the branch.
5. User runs tests/app/inference on the server.
6. User sends logs/results back.
7. Agent proposes the next change.

## Stack

- Python 3.11+
- Streamlit
- PyTorch
- Transformers
- Pandas
- Pydantic

## Main entry points

- Streamlit app: `app/streamlit_app.py`
- Single-report inference: `run_single.py`
- Examples: `examples/`

## Data and privacy rules

Never commit:

- MRI data
- DICOM/NIfTI files
- personal medical data
- `.env`
- API keys
- tokens
- model weights
- cache folders
- local virtual environments
- generated outputs containing patient data

Expected ignored folders:

- `.venv/`
- `data/`
- `outputs/`
- `.cache/`
- `models/`
- `__pycache__/`

## Code style

- Prefer simple, readable Python.
- Prefer `pathlib.Path` for filesystem paths.
- Keep functions small and testable.
- Avoid unnecessary abstractions.
- Do not rewrite large parts of the project without explicit approval.
- Do not change public JSON schemas unless explicitly requested.
- Do not silently change clinical scoring logic.

## Medical logic rules

This project is research/prototyping software and not a medical device.

When editing clinical extraction or scoring logic:

- Preserve traceability from extracted features to source report text.
- Keep positive, negative, and uncertain evidence separated.
- Avoid inventing findings not present in the report.
- Represent uncertainty explicitly.
- Do not overstate diagnostic confidence.
- Do not remove conservative safety checks.

## Useful server commands for the user

Run Streamlit on server:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Run single example on server:

```bash
python run_single.py --model qwen_7b --text-file examples/sample_mri.txt
```

Check git state:

```bash
git status
```

Check changed files:

```bash
git diff --stat
```
