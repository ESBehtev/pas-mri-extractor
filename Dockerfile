# CUDA build stage for llama.cpp. No model weights are included in the image.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS llama-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /opt/llama.cpp/build --config Release -j"$(nproc)"

# Runtime image contains Python, the application and the manually-invoked llama-server.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib/llama

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && python3 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies before application code to maximize layer reuse.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --from=llama-builder /opt/llama.cpp/build/bin/ /usr/local/lib/llama/
RUN ln -s /usr/local/lib/llama/llama-server /usr/local/bin/llama-server

# The Docker build context is limited to production files by .dockerignore.
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts/run_single.py ./scripts/run_single.py

RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501 8080

# Start neither llama-server nor Streamlit automatically. Use docker exec for both.
CMD ["sleep", "infinity"]
