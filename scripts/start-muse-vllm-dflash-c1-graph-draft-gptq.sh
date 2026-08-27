#!/bin/bash
# Candidate-only launcher: GPTQ target + GPTQ DFlash assistant, XPU graph mode.
# The BF16 assistant launcher remains start-muse-vllm-dflash-c1-graph.sh.
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
MODEL="${MODEL:-/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128}"
DRAFT="${DRAFT:-/home/mike/inference/models/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128}"
SPEC="${SPEC:-/tmp/muse-dflash-spec-gptq.json}"
PATCH="${PATCH:-/tmp/patch-vllm-dflash-gptq-context-kv.py}"
DFLASH_KV_MODE="${DFLASH_KV_MODE:-none}"
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
NAME="${NAME:-muse-vllm-xpu-c1}"
PORT="${PORT:-8000}"

test -f "$MODEL/config.json"
test -f "$DRAFT/config.json"
test -f "$DRAFT/quantize_config.json"
test -f "$SPEC"
test -f "$PATCH"

/usr/bin/docker rm -f glimmer-sycl glimmer-dflash2-sycl "$NAME" >/dev/null 2>&1 || true

/usr/bin/docker run -d --name "$NAME" -p "0.0.0.0:${PORT}:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/model:ro" \
  -v "$DRAFT:/draft:ro" \
  -v "$SPEC:/spec.json:ro" \
  -v "$PATCH:/patch-vllm-dflash-gptq-context-kv.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e DFLASH_KV_MODE="${DFLASH_KV_MODE}" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; pip install -q vllm-xpu-kernels==0.1.13.2; DFLASH_KV_MODE="$DFLASH_KV_MODE" python /patch-vllm-dflash-gptq-context-kv.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --served-model-name muse-glimmer-gptq --language-model-only --reasoning-parser muse_glimmer --speculative-config "$(cat /spec.json)"'

echo started "$NAME" graph-mode GPTQ draft on "$PORT" "(DFLASH_KV_MODE=$DFLASH_KV_MODE)"
/usr/bin/docker ps
