#!/bin/bash
# LittleBit container entrypoint — mode-switched via $MODE
set -euo pipefail
# venv when present (CUDA variant), else system python (ROCm variant)
if [ -f /opt/lbvenv/bin/activate ]; then
  export VIRTUAL_ENV=/opt/lbvenv
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
cd /opt/littlebit

# Expose SSH when the image ships it (RunPod startSsh/ports 22/tcp map to it).
# Our ENTRYPOINT replaces the base image's entrypoint, which does TWO things:
# injects $PUBLIC_KEY into root's authorized_keys AND starts sshd. Replicate
# both — starting sshd alone ships a daemon that rejects every key (found the
# hard way on the 50k-step run: sshd up, auth dead).
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  grep -qF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null || \
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
if command -v sshd >/dev/null 2>&1; then
  mkdir -p /run/sshd
  /usr/sbin/sshd || echo "WARN: sshd failed to start (continuing without SSH)"
fi

echo "=== littlebit container: MODE=${MODE:-unset} HF_MODEL=${HF_MODEL:-unset} MODEL_PATH=${MODEL_PATH:-unset} ==="

case "${MODE}" in
  p0)
    python tests/run_p0.py
    ;;
  smoke)
    exec python ci/scripts/qwen3_smoke_full.py
    ;;
  train)
    HF_MODEL="${HF_MODEL:?HF_MODEL must be set for MODE=train}"
    SAVE_DIR="${SAVE_DIR:-/workspace/out}"
    mkdir -p "${SAVE_DIR}"
    exec python main.py \
      --model_id "${HF_MODEL}" \
      --save_dir "${SAVE_DIR}" \
      --quant_func STEBinary --quant_mod LittleBitLinear \
      --split_dim 1024 --eff_bit 1.0 --residual --kv_factor 1.0 --min_split_dim 8 \
      --use_itq --itq_n_iter 50 \
      --exp_name littlebit --run_name "${RUN_NAME:-qwen3-smoke}" \
      --report none \
      --num_train_epochs "${EPOCHS:-1}" \
      --per_device_train_batch_size "${BATCH:-4}" \
      --gradient_accumulation_steps "${GRAD_ACC:-4}" \
      --lr "${LR:-2e-5}" \
      --data_root /workspace/data \
      --dataset "${DATASET:-wikitext2}" \
      --dataset_prepared /workspace/data/prepared \
      --seed 42
    ;;
  eval)
    MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set for MODE=eval}"
    exec python eval.py \
      --model_id "${MODEL_PATH}" \
      --ppl_task "${PPL_TASK:-wikitext2}" \
      --zeroshot_task "${ZEROSHOT_TASK:-}" \
      --batch_size "${BATCH:-8}" \
      --seqlen "${SEQLEN:-2048}"
    ;;
  *)
    echo "unknown MODE '${MODE}' (want: p0 | smoke | train | eval)"; exit 2
    ;;
esac
