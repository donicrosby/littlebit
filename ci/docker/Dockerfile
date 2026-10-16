# One Dockerfile, N runtimes. The base image houses the torch/CUDA/ROCm
# runtime; this layer adds only the app env, hermetic assets, and entrypoint.
# A variant is just (BASE_IMAGE, REQUIREMENTS_FILE) at build time — CI
# generates requirements.txt from uv.lock per variant (torch filtered out
# for base-image variants; AMD-index pins kept for the rocm variant):
#
#   cuda:   BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
#   runpod: BASE_IMAGE=runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404
#   rocm:   BASE_IMAGE=ubuntu:24.04  (+ AMD wheel index from the lock pins)
#
# Local dev: `uv sync --extra cuda` (or --extra rocm). Model downloads at
# runtime on first use (from_pretrained falls back to the HF hub); the 22MB
# wikitext token cache is baked in so smoke mode skips datasets-regen.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG REQUIREMENTS_FILE=build/requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
# python3 is for bare-ubuntu bases (rocm variant); a harmless no-op on the
# pytorch bases, which ship their own interpreter and resolve `python3` first.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 libgomp1 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# App code lives under /opt (FHS: self-contained add-on software), NOT under
# /workspace — on RunPod /workspace is the conventional volume mountpoint, and
# an attached volume would shadow code stored there. /workspace holds runtime
# data only (HF cache, smoke outputs).
COPY . /opt/littlebit
WORKDIR /opt/littlebit

# --system-site-packages: the venv sees the base's torch; pip adds only the
# delta — never reinstalls the runtime stack the base provides.
RUN python3 -m venv --system-site-packages /opt/lbvenv && \
    /opt/lbvenv/bin/pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

# ci/ scripts stay INSIDE the repo copy — never relocate them. The smoke
# script resolves the repo root relative to its own in-repo path; relocating
# it (e.g. to /workspace/scripts) breaks that invariant silently.
# The image carries NO dataset artifacts: the smoke token cache lives under
# /workspace/cache (keyed by tokenizer) and is regenerated from the HF hub
# (Salesforce/wikitext, wikitext-2-raw-v1) when absent — see ci/scripts.
ENV MODE=smoke \
    PATH=/opt/lbvenv/bin:$PATH \
    HF_HOME=/workspace/hf \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
