# Glimmer B70: C8 / DFlash K4 with native 128k context

Measured 2026-09-05 on one Intel Arc Pro B70. This extends the existing vLLM-XPU
GPTQ target + GPTQ assistant stack; it is not a llama.cpp or CUDA recipe.

For the later **experimental C8–C128 shortlist/K3 sweep** (peak observed
840.810 aggregate tok/s at C96), see [the separate report](concurrency-sweep.md).
It uses a different draft head and is not a replacement for this C8/K4 recipe.

## Result: 278.1 aggregate tok/s, not per-stream decode

The retained **C8/K4** profile measured **278.104 aggregate e2e tok/s** at eight
clients with native 131072 context configured. The original C1/K20 scheduler
measured **48.080** on the same short-input workload and client load: **5.78×**.
The eight-request wave fell from **42.60s to 7.36s**.

Controlled screening medians (aggregate e2e tok/s):

| Server profile | 1 client | 4 clients | 8 clients |
|---|---:|---:|---:|
| Original C1 / DFlash K20 | 48.110 | 48.085 | 48.080 |
| C8 / DFlash K20 | 52.377 | 93.975 | 152.629 |
| C8 / no speculation | 31.846 | 122.725 | 236.485 |
| C8 / DFlash K4 | 51.964 | 158.345 | 278.255 |

Screening used 8192 configured context. The final native-context C8/K4 replay
was **278.104** (277.809–278.473), preserving the gain.

Protocol: the same short sky prompt, temperature 1, top_p .95, top_k 64, seed 42,
256 output tokens, one shape-matched warmup and three measured waves per load.
Aggregate e2e is sum of completion tokens / concurrent wave wall time, including
queueing, prefill, reasoning, and answer tokens. These are capped performance
probes, **not completed answers, per-stream decode, or a benchmark quality score**.
Do not compare 278 directly with the README's original C1 89/101/43 decode table.

Separate completed-answer smoke checks at native context passed **8/8** mixed
GSM8K Janet / HumanEval/0 / MT-Bench Hawaii requests: correct $18, the existing two
code assertions, or >=400-character writing, with visible content and natural
stop. This small cohort does not establish broad quality equivalence.

## Native context, resident capacity, and dynamic batching

**131072 tokens is the per-request prompt-plus-output limit**, not eight reserved
131k slots. `max-num-seqs=8` caps active requests; memory admission and chunked
prefill determine which requests run or wait.

- **130940 prompt + 128 output = 131068** completed successfully.
- A **128896-token** synthetic archive recovered all three beginning/middle/end
  checkpoint codes and stopped naturally. This is a retrieval check, not a broad
  long-context reasoning benchmark.
- **8×65532 prompt + 2048 output per request:** all completed, eight active,
  **84.25% peak KV usage**, zero preemptions. More than 524k prompt tokens were
  concurrently resident.
- **8×81912 prompt + 2048 output:** all completed; peak seven active / 89.86% KV
  use, with excess work queued and zero preemptions. This does not prove eight
  80k contexts resident together.
- **12 staggered requests:** eight active, waiting increased, replacements began
  streaming before the initial batch all finished, and the queue drained with
  zero preemptions. Continuous batching and queueing were observed, not inferred
  merely from successful startup.
- **Six ordinary 128994-token prompts:** all returned the correct codes and
  stopped naturally. Peak three active, five waiting, zero preemptions; all
  drained in 587.92s. This is full-context queue service, not six resident slots.

### Why the startup capacity changed from 180k to 609k

The same **5.18 GiB KV allocation** reported **180022 effective tokens** at 8k
configured context and **609193** at 131072 (4.65 full-length requests). vLLM
calculates this as estimated concurrency × configured context, using shared
cache-group blocks. It is **not a fixed flat token pool**. Glimmer has 39
sliding-window layers (window 2048), 13 full-attention layers, and a five-layer
sliding-window draft; group padding and draft head geometry also affect capacity.
Use the filled-request measurements above, not a universal "609k tokens" promise.

### Unresolved high-pressure failure — do not omit

A forced-length test of six **128994-token prompts + 2048 output tokens**, with
`ignore_eos=true`, reached **97.46% KV use**. Two responses reported output tokens
but emitted no visible text. The queue drained and preemptions stayed at zero;
**the response-validity gate still failed**.

Single-request streaming/non-streaming × normal-EOS/ignore-EOS replays all returned
visible, correct retrieval text. The subsequent six-request normal-EOS test also
passed, but with lower occupancy. Neither result resolves the near-capacity
failure or proves it harmless. This remains a **research profile**, not a claim
of unrestricted production-safe full-context concurrency. Dynamic batching does
work; the remaining question is response validity under that stress condition.

## Reproduce the retained profile

Use the same downloaded GPTQ trees and checksum checks from
[README → Reproduce](../README.md#reproduce). Run on the B70 host with no other
GPU model resident. No re-quantization, driver update, or power-setting change is
part of this recipe.

```bash
export MODEL="$PWD/models/target"
export DRAFT="$PWD/models/draft"
bash scripts/start-muse-vllm-concurrent.sh
bash scripts/wait-vllm-health.sh 420 18080
curl -fsS http://127.0.0.1:18080/v1/models
```

The launcher pins the existing image digest and kernels 0.1.13.2, uses graphs,
FP8 KV, memory utilization .90, `max-num-batched-tokens=2048`, no prefix caching,
and the packed-QKV fallback with `DFLASH_KV_MODE=none`. DFlash depth is **4**.
It binds **127.0.0.1:18080** and refuses an existing container name rather than
deleting it. Stop the original C1 container explicitly before launching this one.
There is no automatic service installation or client-roster change.

Shape-matched warmup, then three C8 waves:

```bash
python3 scripts/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18080/v1 --model muse-glimmer-gptq \
  --concurrency 8 --reps 1 --max-tokens 256 \
  --label c8-k4-warmup --out /tmp/c8-k4-warmup.json --log ''
python3 scripts/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18080/v1 --model muse-glimmer-gptq \
  --concurrency 8 --reps 3 --max-tokens 256 \
  --label c8-k4 --out /tmp/c8-k4.json --log ''
```

Read `waves[].aggregate_e2e_tok_s` and check `errors` and all eight `ok` results.
The client recognizes both `delta.reasoning` and legacy `delta.reasoning_content`.
Its historical client decode ratio is approximate for multi-token speculative
chunks; use aggregate e2e for the comparison above. The optional llama-server log
fields are not vLLM acceptance measurements.

For the original completed-answer suite as a **C1 smoke check on this server**:

```bash
DFLASH_SHARE_BASE=http://127.0.0.1:18080 \
  python3 scripts/vllm-dflash-share-suite.py 3 2048 /tmp/c8-profile-c1-smoke.json
# Free the shared B70 afterward:
docker rm -f muse-vllm-xpu-concurrent
```

That last command runs the suite serially; it does not reproduce the separate
8-request mixed-answer smoke test. The
[full research handoff](https://github.com/mgaruccio/b70-inference/blob/main/docs/glimmer-b70-concurrency-20260905.md)
records exact host-side load/context probe commands, metrics, raw-response paths,
and failed as well as successful cases. Raw test artifacts remain local and are
not included in this repository. Source launcher:
[`scripts/start-muse-vllm-concurrent.sh`](../scripts/start-muse-vllm-concurrent.sh).

## Primary references

- [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
  — test speculation under load; C1 latency gains do not imply throughput gains.
- [Hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
  — sliding/full-attention groups, shared blocks, padding, and eviction.
- [Muse model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) — native context.
- [Chunked prefill](https://docs.vllm.ai/en/v0.27.0/configuration/optimization/)
  and [metrics](https://docs.vllm.ai/en/stable/usage/metrics/) — admission/drain checks.
