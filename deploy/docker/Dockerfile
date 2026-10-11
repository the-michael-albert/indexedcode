# Inference worker for AMD/Intel GPUs via llama.cpp's Vulkan backend.
# Build context is the repository root (see docker-compose.yml).
FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev cmake build-essential git \
        libvulkan-dev glslc libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
# llama-cpp-python is compiled against Vulkan (same flags as
# `indexedcode install-runtime --backend vulkan`)
RUN CMAKE_ARGS="-DGGML_VULKAN=on" FORCE_CMAKE=1 \
        /opt/venv/bin/pip install --no-cache-dir llama-cpp-python huggingface-hub \
    && /opt/venv/bin/pip install --no-cache-dir /src

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libvulkan1 mesa-vulkan-drivers vulkan-tools libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models \
    INDEXEDCODE_GPU=amd \
    INDEXEDCODE_AUTO_INSTALL=0 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# Role, model and secret come from the compose file.
ENTRYPOINT ["python", "-m", "indexedcode.workers.serve", \
            "--backend", "vulkan", "--host", "0.0.0.0", "--port", "8000", "--warmup"]
