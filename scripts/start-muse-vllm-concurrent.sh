#!/bin/bash
# B70 concurrency cell: GPTQ target/draft, XPU graphs, eight sequences, DFlash K4.
# Separate from the published C1/K20 recipe. Stop this container before Qwen.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL="${MODEL:-/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128}"
DRAFT="${DRAFT:-/home/mike/inference/models/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128}"
PATCH="${PATCH:-$HERE/patch-vllm-dflash-gptq-context-kv.py}"
NAME="${NAME:-muse-vllm-xpu-concurrent}"
PORT="${PORT:-18080}"
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"

test -f "$MODEL/config.json"
test -f "$DRAFT/config.json"
test -f "$DRAFT/quantize_config.json"
test -f "$PATCH"
# No automatic deletion of existing containers; fail if NAME is already in use.
/usr/bin/docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/model:ro" -v "$DRAFT:/draft:ro" \
  -v "$PATCH:/patch-vllm-dflash-gptq-context-kv.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e DFLASH_KV_MODE=none \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    pip install -q vllm-xpu-kernels==0.1.13.2
    python /patch-vllm-dflash-gptq-context-kv.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
    exec vllm serve /model --quantization gptq --dtype float16 \
      --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
      --port 8000 --max-num-seqs 8 --max-num-batched-tokens 2048 \
      --no-enable-prefix-caching --served-model-name muse-glimmer-gptq \
      --language-model-only --reasoning-parser muse_glimmer \
      --speculative-config '\''{"method":"dflash","model":"/draft","num_speculative_tokens":4,"quantization":"gptq"}'\''
  '
echo "Started $NAME: http://127.0.0.1:$PORT/v1 (C8, DFlash K4, 131072 context)"
