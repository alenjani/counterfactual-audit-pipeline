# CUDA 12.4 + PyTorch 2.4 + Python 3.11 — tuned for L4 (Ada Lovelace, sm_89)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    git curl wget ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install torch first against the right CUDA build
RUN pip install --upgrade pip && \
    pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
RUN pip install -e ".[dev]"

# Create cache mount points
RUN mkdir -p /cache/huggingface /cache/torch /workspace/data /workspace/outputs
VOLUME ["/cache", "/workspace/data", "/workspace/outputs"]

# Default: drop into a shell. Override with `cap-generate ...` etc.
CMD ["/bin/bash"]
