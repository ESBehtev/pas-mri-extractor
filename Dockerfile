# This image is intentionally Linux AMD64 only: llama.cpp is built with CUDA.
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-devel-ubuntu24.04 AS llama-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /opt/llama.cpp/build --config Release -j"$(nproc)"

# Runtime image: CUDA libraries, Python tooling, llama-server and a backup app copy.
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib/llama \
    HF_HOME=/data/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/huggingface/hub

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        python3 \
        python3-venv \
    && python3 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md /opt/pas-mri-extractor/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/pas-mri-extractor/requirements.txt

COPY --from=llama-builder /opt/llama.cpp/build/bin/ /usr/local/lib/llama/
RUN ln -s /usr/local/lib/llama/llama-server /usr/local/bin/llama-server

# Backup only; development work happens in /workspace/pas-mri-extractor after manual git clone.
COPY src /opt/pas-mri-extractor/src
COPY app /opt/pas-mri-extractor/app
COPY configs /opt/pas-mri-extractor/configs
COPY .streamlit /opt/pas-mri-extractor/.streamlit
COPY scripts/run_single.py /opt/pas-mri-extractor/scripts/run_single.py
RUN pip install --no-cache-dir --no-deps /opt/pas-mri-extractor \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /workspace /models /data/huggingface/hub \
    && chown -R appuser:appuser /workspace /models /data /opt/pas-mri-extractor

USER appuser

EXPOSE 8080 8501

# Manual workflow only: do not start llama-server, Streamlit, or git clone here.
CMD ["sleep", "infinity"]
