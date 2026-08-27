# Agent instructions

Reproduce **Muse Glimmer 30B** on **one Intel Arc Pro B70** with vLLM-XPU and DFlash. This is not a CUDA recipe.

Follow **README.md → Reproduce** in order. Do not improvise a llama.cpp or OpenVINO path unless the user asks.

## Hard requirements

- Intel Arc Pro B70 (32 GB), Linux `xe` driver, Docker, `/dev/dri`
- Do **not** use NVIDIA/CUDA images
- Pin `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- Install `vllm-xpu-kernels==0.1.13.2` in the container (the launcher does this)
- `DFLASH_KV_MODE=none` — never `timing` on the serving path (`torch.xpu.synchronize` perturbs sampling)
- One stream (`--max-num-seqs 1`), `--max-model-len 8192`

## Weights

Download the GPTQ trees (do not requantize unless the user asks):

- Target: `mgaruccio/Muse-Glimmer-30B-GPTQ-Int4-sym-G128`
- Draft: `mgaruccio/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128`

Verify shards against `docs/checksums.md`. Base model license is Apache 2.0 ([meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)).

## Success

1. `curl -sf http://127.0.0.1:8000/v1/models` returns `muse-glimmer-gptq`
2. Logs show `XPUwNa16LinearKernel`, `num_spec_tokens=20`, patch `mode=base`
3. `python3 scripts/vllm-dflash-share-suite.py 3 2048 share-suite.json` exits 0

Headline tok/s will only match the README if you used **these** GPTQ trees, n=20, and graphs. A from-scratch requant is a different artifact.

## Do not

- Call `torch.xpu.synchronize` on the server
- Retry vLLM `extract_hidden_states` on this Muse/XPU build
- Overwrite the downloaded GPTQ directories
- Report 128-token thinking-window numbers as the share suite
