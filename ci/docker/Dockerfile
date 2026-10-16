# One Dockerfile, N runtimes. The base image houses the torch/CUDA/ROCm
# runtime; this layer adds only the app env, hermetic assets, and entrypoint.
# A variant is just (BASE_IMAGE, REQUIREMENTS_FILE) at build time:
#
#   cuda:   BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime   \
#           REQUIREMENTS_FILE=ci/requirements-cuda.txt
#   runpod: BASE_IMAGE=runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404 \
#           REQUIREMENTS_FILE=ci/requirements-cuda.txt
#   rocm:   BASE_IMAGE=ubuntu:24.04                                    \
#           REQUIREMENTS_FILE=ci/requirements-rocm.txt
#
# Requirement files carry their own pip index directives (the ROCm freeze
# points at AMD's wheel index). Model downloads at runtime on first use
# (from_pretrained falls back to the HF hub); the 22MB wikitext token cache
# is baked in so smoke mode skips datasets-regen. Env pins come from the
# uv freeze of the validated build (ci/requirements-cuda.lock.txt).
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE=ci/requirements-cuda.txt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/littlebit
WORKDIR /workspace/littlebit

# --system-site-packages: the venv sees the base's torch; pip adds only the
# delta from the variant's freeze — never reinstalls the runtime stack.
RUN python3 -m venv --system-site-packages /opt/lbvenv && \
    /opt/lbvenv/bin/pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

COPY ci/assets/wikitext2_tokens.pt /workspace/assets/wikitext2_tokens.pt
COPY ci/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
COPY ci/scripts /workspace/scripts

ENV MODE=smoke \
    PATH=/opt/lbvenv/bin:$PATH \
    HF_HOME=/workspace/hf \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
