#!/bin/bash
set -euo pipefail
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL=/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128
DRAFT=/home/mike/inference/models/Muse-Glimmer-30B-assistant
SPEC=/tmp/muse-dflash-spec.json
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
NAME=muse-vllm-xpu-c1
PORT=8000

test -f "$DRAFT/config.json"
test -f "$SPEC"

/usr/bin/docker rm -f glimmer-sycl glimmer-dflash2-sycl "$NAME" >/dev/null 2>&1 || true

/usr/bin/docker run -d --name "$NAME" -p "0.0.0.0:${PORT}:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/model:ro" \
  -v "$DRAFT:/draft:ro" \
  -v "$SPEC:/spec.json:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; pip install -q vllm-xpu-kernels==0.1.13.2; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --served-model-name muse-glimmer-gptq --language-model-only --reasoning-parser muse_glimmer --speculative-config "$(cat /spec.json)"'

echo started "$NAME" dflash xpu-graph on "$PORT"
/usr/bin/docker ps
