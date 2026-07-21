# vLLM provides the CUDA runtime and OpenAI-compatible server implementation.
FROM --platform=linux/amd64 vllm/vllm-openai:latest

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/huggingface/hub \
    MODEL_PATH=/models \
    VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8000 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git python3-venv \
    && python3 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies before application code to maximize layer reuse.
COPY requirements.txt pyproject.toml README.md /opt/pas-mri-extractor/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/pas-mri-extractor/requirements.txt

# Backup only; editable application code may be cloned manually into /workspace.
COPY src /opt/pas-mri-extractor/src
COPY app /opt/pas-mri-extractor/app
COPY configs /opt/pas-mri-extractor/configs
COPY .streamlit /opt/pas-mri-extractor/.streamlit
COPY scripts/run_single.py /opt/pas-mri-extractor/scripts/run_single.py
COPY start_vllm.sh /workspace/start_vllm.sh

RUN pip install --no-cache-dir --no-deps /opt/pas-mri-extractor \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /workspace /models /data/huggingface/hub \
    && chmod +x /workspace/start_vllm.sh \
    && chown -R appuser:appuser /workspace /models /data /opt/pas-mri-extractor

USER appuser

EXPOSE 8000 8501

# Manual workflow only: run ./start_vllm.sh and Streamlit explicitly after exec.
CMD ["sleep", "infinity"]
