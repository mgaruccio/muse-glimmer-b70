#!/bin/bash
# Wait for the candidate vLLM server to become healthy.
# Usage: bash wait-vllm-health.sh [timeout_s] [port]
set -u
TIMEOUT="${1:-360}"
PORT="${2:-8000}"
URL="http://127.0.0.1:${PORT}/v1/models"
i=0
while [ "$i" -lt "$TIMEOUT" ]; do
  if curl -sf -m 3 "$URL" >/dev/null 2>&1; then
    echo "HEALTH_OK after ${i}s"
    curl -s "$URL" | head -c 200
    echo
    exit 0
  fi
  sleep 5
  i=$((i + 5))
done
echo "HEALTH_TIMEOUT after ${TIMEOUT}s"
exit 1