#!/usr/bin/env python3
"""Guarded overlay for vLLM DFlash GPTQ QKV context precompute (v2).

DFlash normally fuses the BF16 K/V weights from every draft QKV projection and
uses F.linear once per verify. GPTQ QKV modules intentionally expose qweight,
not .weight, so the existing fast path cannot build that fusion. This patch
keeps the fast path unchanged and adds, for packed-QKV drafts:

  * base fallback (always): run each already-configured QKV module on
    normalized context states, discard Q, stack K/V, then follow the original
    normalization/RoPE/cache-store path.
  * timing (DFLASH_KV_MODE=timing, default): wrap
    precompute_and_store_context_kv / _project_context_kv with device-synced
    perf_counter instrumentation (they run eagerly every step, outside CUDA
    graphs), logging num_ctx distribution and phase budgets every 100 calls.
  * kvonly (DFLASH_KV_MODE=kvonly / kvonly+timing): replace the per-layer loop
    with one fused int4_gemm_w4a16 over the concatenated K/V output rows of
    every layer (Q is never computed), built lazily from the processed packed
    params. A one-shot numerical self-check compares the fused path against
    the per-layer fallback at first use.

Usage inside the vLLM container:
  DFLASH_KV_MODE=timing python /patch-vllm-dflash-gptq-context-kv.py \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py

Modes: none (base only, v1 behavior) | timing | kvonly | kvonly+timing
"""

import os
from pathlib import Path
import sys

MODE = os.environ.get("DFLASH_KV_MODE", "timing").strip().lower()
if MODE not in ("none", "timing", "kvonly", "kvonly+timing"):
    raise SystemExit(f"unknown DFLASH_KV_MODE={MODE!r} (none|timing|kvonly|kvonly+timing)")

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} VLLM_QWEN3_DFLASH_PY")

path = Path(sys.argv[1])
orig_backup = path.with_suffix(".py.orig")
if not orig_backup.exists():
    orig_backup.write_text(path.read_text())
    print(f"backup: saved pristine original to {orig_backup}")

if MODE == "none":
    # Restore the pristine vLLM file, then re-apply only the base fallback so
    # the candidate keeps v1 behavior (per-layer packed-QKV loop), with all
    # timing/fused markers stripped.
    path.write_text(orig_backup.read_text())
    print(f"restored pristine original from {orig_backup}")
    MODE = "base"  # apply base patch below from the clean file

source = path.read_text()
applied = []

# ---------------------------------------------------------------- base patch
base_marker = "# DFLASH_GPTQ_CONTEXT_KV_FALLBACK"
if base_marker in source:
    print(f"base: already applied ({path})")
else:
    old_build = '''        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
'''
    new_build = '''        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._context_kv_attn_layers = layers_attn

        # DFLASH_GPTQ_CONTEXT_KV_FALLBACK
        # AutoGPTQ QKV modules expose packed qweight/scales rather than a
        # materialized .weight. Preserve the one-GEMM BF16 path when all
        # weights exist; packed QKV falls back to its own quantized projection
        # in _project_context_kv below.
        self._packed_context_qkv = any(
            not hasattr(attn.qkv_proj, "weight") for attn in layers_attn
        )
        if not self._packed_context_qkv:
            # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
            if has_bias:
                kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
                self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
            else:
                self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
'''
    if source.count(old_build) != 1:
        raise RuntimeError("qwen3_dflash.py: build-context anchor missing or ambiguous")
    source = source.replace(old_build, new_build)

    old_project = '''        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v
'''
    new_project = '''        if self._packed_context_qkv:
            # A quantized QKV module has no materialized weight for the fast
            # fused F.linear path. Use its existing quantized kernel per layer
            # and retain only K/V. This runs only for packed-QKV drafts.
            all_k = []
            all_v = []
            for attn in self._context_kv_attn_layers:
                qkv, _ = attn.qkv_proj(normed_context_states)
                _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                all_k.append(k.view(num_ctx, num_kv_heads, head_dim))
                all_v.append(v.view(num_ctx, num_kv_heads, head_dim))
            return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)

        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v
'''
    if source.count(old_project) != 1:
        raise RuntimeError("qwen3_dflash.py: project-context anchor missing or ambiguous")
    source = source.replace(old_project, new_project)
    applied.append("base")

# ----------------------------------------------------------------- timing
timing_marker = "# DFLASH_GPTQ_CONTEXT_KV_TIMING"
if MODE in ("timing", "kvonly+timing"):
    if timing_marker in source:
        print("timing: already applied")
    else:
        if "import time" not in source:
            old_imp = "import io\n"
            assert source.count(old_imp) == 1, "import anchor ambiguous"
            source = source.replace(old_imp, "import io\nimport time\n")

        for old_name, new_name in (
            (
                "def precompute_and_store_context_kv(\n"
                "        self,\n"
                "        context_states: torch.Tensor,\n"
                "        context_positions: torch.Tensor,\n"
                "        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,\n"
                "    ) -> None:\n"
                "        \"\"\"Precompute K/V for context states write them into each layer's KV cache.",
                "def _orig_precompute_and_store_context_kv(\n"
                "        self,\n"
                "        context_states: torch.Tensor,\n"
                "        context_positions: torch.Tensor,\n"
                "        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,\n"
                "    ) -> None:\n"
                "        \"\"\"Precompute K/V for context states write them into each layer's KV cache.",
            ),
            ("def _project_context_kv(", "def _orig_project_context_kv("),
        ):
            if new_name in source:
                print(f"timing: {old_name!r} already renamed")
            elif source.count(old_name) == 1:
                source = source.replace(old_name, new_name)
            else:
                raise RuntimeError(f"timing: {old_name!r} anchor missing or ambiguous")

        wrapper = '''    def precompute_and_store_context_kv(self, *args, **kwargs):
        # DFLASH_GPTQ_CONTEXT_KV_TIMING
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        try:
            out = self._orig_precompute_and_store_context_kv(*args, **kwargs)
        finally:
            torch.xpu.synchronize()
            t1 = time.perf_counter()
            ctx = kwargs.get("context_states", args[0] if args else None)
            num_ctx = ctx.shape[0] if ctx is not None else -1
            t = getattr(self, "_ctxkv_timing", None)
            if t is None:
                t = self._ctxkv_timing = {
                    "calls": 0, "tot_s": 0.0, "proj_s": 0.0,
                    "ctx_sum": 0, "ctx_max": 0, "hist": {},
                }
            t["calls"] += 1
            t["tot_s"] += t1 - t0
            t["proj_s"] += getattr(self, "_ctxkv_last_proj_s", 0.0)
            t["ctx_sum"] += num_ctx
            t["ctx_max"] = max(t["ctx_max"], num_ctx)
            h = t["hist"]
            for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64),
                           (64, 128), (128, 256), (256, 512), (512, 1 << 62)):
                if num_ctx < hi:
                    h[f"{lo}-{hi}"] = h.get(f"{lo}-{hi}", 0) + 1
                    break
            else:
                h["other"] = h.get("other", 0) + 1
            if t["calls"] % 100 == 0:
                n = t["calls"]
                logger.info(
                    "CTXKV timing calls=%d avg_ctx=%.1f max_ctx=%d "
                    "proj_avg_ms=%.2f rest_avg_ms=%.2f tot_avg_ms=%.2f hist=%s",
                    n, t["ctx_sum"] / n, t["ctx_max"],
                    1e3 * t["proj_s"] / n, 1e3 * (t["tot_s"] - t["proj_s"]) / n,
                    1e3 * t["tot_s"] / n, sorted(t["hist"].items()),
                )
                t["calls"] = 0
                t["tot_s"] = 0.0
                t["proj_s"] = 0.0
                t["ctx_sum"] = 0
                t["ctx_max"] = 0
        return out

    def _project_context_kv(self, context_states, num_ctx, num_layers,
                            num_kv_heads, head_dim):
        # DFLASH_GPTQ_CONTEXT_KV_TIMING
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = self._orig_project_context_kv(
            context_states, num_ctx, num_layers, num_kv_heads, head_dim
        )
        torch.xpu.synchronize()
        self._ctxkv_last_proj_s = time.perf_counter() - t0
        return out

'''
        anchor = "    def _orig_project_context_kv("
        if source.count(anchor) != 1:
            raise RuntimeError("timing: renamed project anchor missing")
        source = source.replace(anchor, wrapper + anchor)
        applied.append("timing")

# ------------------------------------------------------------------ kvonly
kv_marker = "# DFLASH_GPTQ_CONTEXT_KV_FUSED"
if MODE in ("kvonly", "kvonly+timing"):
    if kv_marker in source:
        print("kvonly: already applied")
    else:
        old_loop = '''        if self._packed_context_qkv:
            # A quantized QKV module has no materialized weight for the fast
            # fused F.linear path. Use its existing quantized kernel per layer
            # and retain only K/V. This runs only for packed-QKV drafts.
            all_k = []
            all_v = []
            for attn in self._context_kv_attn_layers:
                qkv, _ = attn.qkv_proj(normed_context_states)
                _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                all_k.append(k.view(num_ctx, num_kv_heads, head_dim))
                all_v.append(v.view(num_ctx, num_kv_heads, head_dim))
            return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)
'''
        new_fused = '''        if self._packed_context_qkv:
            # DFLASH_GPTQ_CONTEXT_KV_FUSED
            # Fused KV-only packed projection: one int4_gemm_w4a16 over the
            # concatenated K/V output rows of every layer (Q never computed).
            if getattr(self, "_fused_kv_qweight", None) is None:
                self._build_fused_kv_only_qkv()
                self._ctxkv_fused_check(normed_context_states)
            all_kv_flat = torch.ops._xpu_C.int4_gemm_w4a16(
                normed_context_states,
                self._fused_kv_qweight,
                self._fused_kv_bias,
                self._fused_kv_scales,
                self._fused_kv_zero_point,
                self._fused_kv_group_size,
                None,
            )
            all_kv = (
                all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            return all_kv[0], all_kv[1]
'''
        if source.count(old_loop) != 1:
            raise RuntimeError("kvonly: packed-loop anchor missing or ambiguous")
        source = source.replace(old_loop, new_fused)

        helpers = '''    def _build_fused_kv_only_qkv(self) -> None:
        """Build one fused KV-only W4A16 projection from the packed QKV params.

        Slices the K/V output rows of each layer's qkv_proj (Q rows dropped)
        and concatenates them across layers so the per-step context precompute
        runs a single GEMM. Layout matches XPUwNa16LinearKernel.apply_weights:
        qweight [K/8, N], scales [K/128, N], zero-point scalar, group_size.
        """
        layers = self._context_kv_attn_layers
        qw_parts, sc_parts, bias_parts = [], [], []
        group_size = None
        for attn in layers:
            m = attn.qkv_proj
            qw = m.qweight.data
            sc = m.scales.data
            # After process_weights_after_loading qweight is [N, K/8]; before
            # processing it is [K/8, N]. Detect by comparing with scales
            # [K/128, N] (same trick as XPUwNa16LinearKernel).
            if qw.shape[0] != sc.shape[0]:
                qw = qw.t().contiguous()
            o_off = attn.q_size
            qw_parts.append(qw[o_off:].contiguous())
            sc_parts.append(sc[:, o_off:].contiguous())
            bias_parts.append(
                m.bias[o_off:].contiguous() if m.bias is not None else None
            )
            if group_size is None:
                kern = getattr(getattr(m, "quant_method", None), "kernel", None)
                group_size = (
                    getattr(getattr(kern, "config", None), "group_size", None)
                    or 128
                )
        self._fused_kv_qweight = torch.cat(qw_parts, dim=0).t().contiguous()
        self._fused_kv_scales = torch.cat(sc_parts, dim=1).contiguous()
        fused_bias = (
            None if any(b is None for b in bias_parts)
            else torch.cat(bias_parts, dim=0)
        )
        self._fused_kv_bias = fused_bias
        self._fused_kv_zero_point = torch.tensor(
            [8], dtype=torch.int8, device=self._fused_kv_scales.device
        )
        self._fused_kv_group_size = group_size
        logger.info(
            "DFLASH_GPTQ fused KV-only: qweight %s scales %s bias=%s group=%d",
            tuple(self._fused_kv_qweight.shape),
            tuple(self._fused_kv_scales.shape),
            fused_bias is not None, group_size,
        )

    def _ctxkv_fused_check(self, x: torch.Tensor) -> None:
        """Equivalence gate: fused KV-only path vs per-layer fallback."""
        try:
            attn0 = self._context_kv_attn_layers[0]
            ref_k, ref_v = [], []
            for attn in self._context_kv_attn_layers:
                qkv, _ = attn.qkv_proj(x)
                _, k, v = qkv.split(
                    [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
                )
                ref_k.append(k.view(x.shape[0], attn.num_kv_heads, attn.head_dim))
                ref_v.append(v.view(x.shape[0], attn.num_kv_heads, attn.head_dim))
            ref_k = torch.stack(ref_k, dim=0)
            ref_v = torch.stack(ref_v, dim=0)
            fused = torch.ops._xpu_C.int4_gemm_w4a16(
                x, self._fused_kv_qweight, self._fused_kv_bias,
                self._fused_kv_scales, self._fused_kv_zero_point,
                self._fused_kv_group_size, None,
            )
            L = len(self._context_kv_attn_layers)
            fused_kv = fused.view(
                x.shape[0], L, 2, attn0.num_kv_heads, attn0.head_dim
            ).permute(2, 1, 0, 3, 4)
            dk = (fused_kv[0] - ref_k).abs().max().item()
            dv = (fused_kv[1] - ref_v).abs().max().item()
            logger.info(
                "DFLASH_GPTQ fused self-check: max|dK|=%.3e max|dV|=%.3e "
                "(ref max magnitude %.3f)",
                dk, dv, max(ref_k.abs().max().item(), ref_v.abs().max().item()),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("DFLASH_GPTQ fused self-check failed: %s", e)

'''
        anchor = "    def _build_fused_kv_buffers(self) -> None:"
        if source.count(anchor) != 1:
            raise RuntimeError("kvonly: helpers anchor missing")
        source = source.replace(anchor, helpers + anchor)
        applied.append("kvonly")

path.write_text(source)
print(f"patched: {path} (mode={MODE}, applied={applied or ['none']})")