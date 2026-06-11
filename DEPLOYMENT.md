# Deployment

This project is currently packaged as a Docker baseline for the Streamlit
extractor workflow. It does not add FastAPI, an agent graph, or a separate
inference service yet.

## Local venv workflow

The existing local workflow remains supported:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Docker Streamlit baseline

Create a local environment file from the example:

```bash
cp .env.example .env
docker compose up --build streamlit
```

The Streamlit UI will listen on:

```text
http://localhost:8501
```

The container runs only the current extractor UI. It uses the existing
`app/streamlit_app.py`, `configs/models.yaml`, `configs/prompt.yaml`, Pydantic
validation, rules, and scoring code.

## Volumes

Model weights and clinical data are not copied into the image.

```text
./models          -> /app/models:ro
./data            -> /app/data:ro
./outputs         -> /app/outputs
./configs         -> /app/configs:ro
./runtime_configs -> /app/runtime_configs
```

GGUF files must be placed under `PAS_MODELS_DIR`. In Docker this is `/app/models`
through the mounted `./models:/app/models:ro` volume. A server deployment can
use a host path such as:

```bash
PAS_MODELS_DIR=/home/mirea/pas-mri-extractor/models
```

The old `~/pas-mri-extractor/models` style is not recommended because it
expands to `/root/pas-mri-extractor/models` inside Docker containers. The image
excludes `models/`, `data/`, `outputs/`, `reports/`, `.env`, GGUF files, and
common DICOM/NIfTI file extensions.

## Runtime environment

The baseline environment variables are documented in `.env.example`:

```text
PAS_APP_ENV=local
PAS_APP_MODE=streamlit
PAS_ACTIVE_STAGE=extractor
PAS_MODEL_CONFIG=configs/models.yaml
PAS_PROMPT_CONFIG=configs/prompt.yaml
PAS_OUTPUT_DIR=outputs
PAS_DATA_DIR=data
PAS_MODELS_DIR=models
PAS_RUNTIME_CONFIG_DIR=runtime_configs
PAS_LOG_LEVEL=INFO
PAS_ENABLE_AUDIT_LOG=false
PAS_ENABLE_PARALLEL_REVIEW=false
```

`src/pas_mri_extractor/runtime.py` reads these values with safe local defaults.
The current extraction pipeline is not rewritten to use a stage graph.

## Future production architecture

The planned production architecture can evolve without changing the Docker
baseline contract:

- `streamlit`: clinical review UI;
- `api`: future FastAPI backend;
- `inference`: future model-loading service or worker;
- `nginx`: future reverse proxy and TLS termination;
- mounted `models/`, `data/`, `outputs/`, `configs/`, and `runtime_configs/`
  volumes.

`configs/stages.example.yaml` documents a future multi-stage structure for
extractor, validation, prognosis, and clinical review stages. It is an example
only and is not loaded by the current runtime.

## Medical data safety

For deployments with real medical data, assume a closed network or VPN, HTTPS,
authentication, audit logging, no raw PHI in logs, controlled access to mounted
volumes, and an explicit data retention policy for `outputs/`.
