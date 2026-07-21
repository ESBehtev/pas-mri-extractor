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

## Deployment

### Docker development container

The image includes the CUDA build of `llama-server`, Python and the app, but
contains no model weights. It starts idle and launches neither `llama-server`
nor Streamlit automatically. This lets one process keep a model in VRAM while
the other is restarted during application development.

```bash
cp .env.example .env
# Set PAS_API_BASE_URL, PAS_API_KEY and PAS_MODEL for your provider.

docker build -t pas-mri-extractor:local .
docker run -d --name pas-mri-dev --gpus all -p 8501:8501 --env-file .env \
  pas-mri-extractor:local
```

Enter the running container in two terminals:

```bash
docker exec -it pas-mri-dev bash
```

In the first terminal, start the model manually and leave it running:

```bash
llama-server -m /models/your-model.gguf --host 127.0.0.1 --port 8080 \
  --api-key local-token -ngl 999
```

In the second terminal, start or restart only Streamlit after editing code:

```bash
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

Mount a host directory with GGUF files when starting the container, for example:

```bash
docker run -d --name pas-mri-dev --gpus all -p 8501:8501 \
  --env-file .env -v /host/models:/models:ro pas-mri-extractor:local
```

The existing Compose file uses the same idle container and `.env`; it requires
an NVIDIA-capable Docker host:

```bash
docker compose up -d --build
docker compose exec app bash
```

### External OpenAI-compatible API

The application supports OpenAI, separately deployed vLLM, LM Studio,
Ollama's OpenAI-compatible API, and any endpoint compatible with OpenAI Chat
Completions. For an external vLLM deployment, change only this line in `.env`:

```dotenv
PAS_API_BASE_URL=http://your-vllm-host:8000/v1
```

Set `PAS_MODEL` to the model name served by that endpoint. Switching Qwen,
Llama or Mistral requires no image rebuild and no Python-code change.

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
