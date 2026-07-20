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

## Docker

The Compose stack has two containers: Streamlit `app` and an OpenAI-compatible
local `vllm` server. They communicate on the private Compose network; vLLM is
not exposed on a host port.

Create the local runtime configuration once:

```bash
cp .env.example .env
# Set HF_TOKEN if the selected Hugging Face model needs it.
```

Start the stack:

```bash
docker compose up -d --build
docker compose ps
curl -f http://localhost:8501/_stcore/health
```

The first vLLM start downloads the configured model into the named
`huggingface-cache` volume, so its healthcheck can take several minutes. Open
http://localhost:8501 only after both services are healthy.

Switch the local model with one `.env` line, for example:

```bash
VLLM_MODEL=Qwen/Qwen3-14B
```

The same change works for a compatible Llama or Mistral Hugging Face model;
then restart the stack:

```bash
docker compose up -d --force-recreate vllm app
```

No Python code changes are needed: `PAS_MODEL=${VLLM_MODEL}` selects the API
model name, while `PAS_API_BASE_URL=http://vllm:8000/v1` routes requests to the
local server. To use another OpenAI-compatible server, set
`PAS_API_BASE_URL` and `PAS_MODEL` directly in `.env`.

Update containers:

```bash
docker compose pull
docker compose up -d --build
```

Clear downloaded Hugging Face models only when the stack is stopped:

```bash
docker compose down
docker volume rm pas-mri-extractor_huggingface-cache
```

Stop the local service without deleting model cache:

```bash
docker compose down
```

Publish only after authenticating to Docker Hub:

```bash
docker tag pas-mri-extractor:local <dockerhub_username>/pas-mri-extractor:latest
docker push <dockerhub_username>/pas-mri-extractor:latest
```

On the server, place `.env` next to `compose.yaml`, then pull and run:

```bash
docker pull <dockerhub_username>/pas-mri-extractor:latest
DOCKER_IMAGE=<dockerhub_username>/pas-mri-extractor:latest docker compose up -d
docker compose ps
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
