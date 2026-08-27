# DFlash share suite (v1)

The characterization numbers from 2026-08-26 (eval 86.5 / work 100.9 tok/s) are **not** this suite. Those were 256-token windows that never left the reasoning channel.

This suite is the one we can quote. The post is the [README](../README.md).

## Protocol

Script: `scripts/vllm-dflash-share-suite.py`

| Knob | Value |
|---|---|
| Boundary | public OpenAI streaming `/v1/chat/completions` + `/metrics` |
| Model id | `muse-glimmer-gptq` |
| Concurrency | 1 |
| Temperature | **0** (greedy) |
| Max tokens | **2048** cap, not a target |
| Stop | `finish_reason=stop` required |
| Warmup | 1 per prompt, excluded |
| Measured reps | 3 |
| Decode tok/s | `completion_tokens / (last_chunk − first_generated_chunk)` |
| E2E tok/s | `completion_tokens / wall_to_stop` (includes TTFT) |
| Acceptance | vLLM `accepted_draft_tokens / draft_runs` (no bonus token) |
| Quoteable row | stop + non-empty **content** + task check pass |

Prompts:

1. **gsm8k-janet** — GSM8K test[0], zero-shot chat. Content must yield **18**.
2. **humaneval-0** — HumanEval/0 doctest, chat-wrapped. Content must pass the two canonical asserts.
3. **mtbench-81** — MT-Bench question 81 turn 1. Content must be ≥400 characters.

A headline number is allowed only if **all three prompts are quoteable on all measured reps**. Otherwise print the per-prompt table and do not average.

This is **not** Spec-Bench (480 prompts) and **not** a spec vs no-spec A/B. It measures the live DFlash C1 cell as served.

## Run

With the server up:
```
python3 scripts/vllm-dflash-share-suite.py 3 2048 share-suite.json
```
## Results — 2026-08-27
vLLM-XPU + GPTQ DFlash n=20, graphs, one stream. Quoteable **3/3**.
| Prompt | Finish | Tokens | Time to first content | Decode tok/s | E2E tok/s | Accepted drafts / verify | Check |
|---|---|---:|---:|---:|---:|---:|---|
| gsm8k-janet | stop | 586 | 6.10 s | **89.1** | 86.7 | 3.37 | **18 PASS** |
| humaneval-0 | stop | 887 | 8.44 s | **101.1** | 99.3 | 4.14 | **doctest PASS** |
| mtbench-81 | stop | 1585 | 8.53 s | **42.6** | 42.3 | 1.22 | 4670 chars |
Quote as the table, not a single average. Decode is post-first-token generation rate; E2E includes TTFT. Both count all generated tokens (thinking + answer) until stop. Writing is the slow cluster; math/code speculate better.
GSM8K and MT-Bench were bit-stable across the 3 greedy reps. HumanEval still passed all 3 doctests but completion length moved (1074 / 882 / 887).
