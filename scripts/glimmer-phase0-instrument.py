#!/usr/bin/env python3
"""Phase 0 Glimmer instrumentation against the live llama-server public API.

Splits TTFT / prefill / decode, reasoning vs content, and (when the server log
is readable) per-slot draft acceptance. Samples Xe hwmon energy + GT freq.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPT_SKY = (
    "Explain in detail why the sky is blue. Cover Rayleigh scattering, "
    "wavelength dependence, and why sunsets look red. Write complete sentences."
)
PAD = (
    "The Battlemage G31 decode path streams quantized transformer weights "
    "through a memory-bound GEMV. This paragraph exists only to occupy "
    "prompt tokens for a context-length sweep. "
)

TIMING_RE = re.compile(
    r"slot print_timing: id\s+(\d+)\s+\| task\s+(\d+)\s+\|\s+"
    r"(prompt eval time|eval time|total time|graphs reused|draft acceptance)\s+=\s+(.*)$"
)
SPEC_RE = re.compile(r"spec statistics:.*$")
ACCEPT_RE = re.compile(
    r"(?:draft acceptance = )?([0-9.]+)\s+\(\s*(\d+)\s+accepted\s+/\s+(\d+)\s+generated\),\s+mean len =\s+([0-9.]+)"
)
EVAL_RE = re.compile(
    r"([0-9.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s*([0-9.]+)\s+ms per token,\s+([0-9.]+)\s+tokens per second\)"
)


def pad_prompt(base: str, approx_tokens: int) -> str:
    if approx_tokens <= 0:
        return base
    need = max(0, approx_tokens * 4 - len(base))
    reps = (need + len(PAD) - 1) // len(PAD)
    return base + "\n\n" + (PAD * reps)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


class GpuSampler:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.energy1_path = "/sys/class/drm/card0/device/hwmon/hwmon2/energy1_input"
        self.energy2_path = "/sys/class/drm/card0/device/hwmon/hwmon2/energy2_input"
        self.freq0_path = "/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq"
        self.freq1_path = "/sys/class/drm/card0/device/tile0/gt1/freq0/cur_freq"
        self.temp_path = "/sys/class/drm/card0/device/hwmon/hwmon2/temp2_input"

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if len(self.samples) < 2:
            return {"samples": self.samples, "card_j": None, "pkg_j": None}
        t0 = self.samples[0]["t"]
        t1 = self.samples[-1]["t"]
        dt = max(t1 - t0, 1e-6)
        e1_0 = self.samples[0].get("energy1_uj")
        e1_1 = self.samples[-1].get("energy1_uj")
        e2_0 = self.samples[0].get("energy2_uj")
        e2_1 = self.samples[-1].get("energy2_uj")
        freqs = [s["freq0_mhz"] for s in self.samples if s.get("freq0_mhz") is not None]
        return {
            "n": len(self.samples),
            "wall_s": dt,
            "card_j": None if e1_0 is None or e1_1 is None else (e1_1 - e1_0) / 1e6,
            "pkg_j": None if e2_0 is None or e2_1 is None else (e2_1 - e2_0) / 1e6,
            "card_w": None
            if e1_0 is None or e1_1 is None
            else (e1_1 - e1_0) / 1e6 / dt,
            "median_freq0_mhz": statistics.median(freqs) if freqs else None,
            "max_freq0_mhz": max(freqs) if freqs else None,
            "samples": self.samples,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = {
                "t": time.monotonic(),
                "energy1_uj": _int_or_none(read_text(self.energy1_path)),
                "energy2_uj": _int_or_none(read_text(self.energy2_path)),
                "freq0_mhz": _int_or_none(read_text(self.freq0_path)),
                "freq1_mhz": _int_or_none(read_text(self.freq1_path)),
                "temp2_mc": _int_or_none(read_text(self.temp_path)),
            }
            self.samples.append(sample)
            self._stop.wait(self.interval_s)


def _int_or_none(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def stream_once(
    base: str,
    model: str,
    idx: int,
    prompt: str,
    max_tokens: int,
    timeout: int,
    seed: int,
    reasoning_strength: str | None,
) -> dict:
    body: dict = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": False,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": seed,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_strength:
        body["chat_template_kwargs"] = {"reasoning_strength": reasoning_strength}
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_any = None
    first_content = None
    first_reason = None
    content_parts: list[str] = []
    reason_parts: list[str] = []
    usage: dict = {}
    timings: dict = {}
    finish = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if data.get("usage"):
                usage = data["usage"]
            if data.get("timings"):
                timings = data["timings"]
            choice = (data.get("choices") or [{}])[0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            now = time.monotonic()
            if reason:
                if first_any is None:
                    first_any = now
                if first_reason is None:
                    first_reason = now
                reason_parts.append(reason)
                chunks += 1
            if content:
                if first_any is None:
                    first_any = now
                if first_content is None:
                    first_content = now
                content_parts.append(content)
                chunks += 1
    ended = time.monotonic()
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    elapsed = ended - started
    decode_s = None if first_any is None else ended - first_any
    reason_text = "".join(reason_parts)
    content_text = "".join(content_parts)
    pred_n = timings.get("predicted_n")
    pred_ms = timings.get("predicted_ms")
    prompt_n = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    decode_tok_s = None
    if pred_n and pred_ms:
        decode_tok_s = (float(pred_n) * 1000.0) / float(pred_ms)
    elif decode_s and completion:
        decode_tok_s = completion / decode_s
    prefill_tok_s = None
    if prompt_n and prompt_ms:
        prefill_tok_s = (float(prompt_n) * 1000.0) / float(prompt_ms)
    elif first_any and prompt_tokens:
        prefill_tok_s = prompt_tokens / (first_any - started)
    return {
        "idx": idx,
        "ok": True,
        "seed": seed,
        "ttft_s": None if first_any is None else first_any - started,
        "ttft_reasoning_s": None if first_reason is None else first_reason - started,
        "ttft_content_s": None if first_content is None else first_content - started,
        "elapsed_s": elapsed,
        "decode_s": decode_s,
        "chunks": chunks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "e2e_tok_s": (completion / elapsed) if elapsed and completion else 0.0,
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "reasoning_chars": len(reason_text),
        "content_chars": len(content_text),
        "reasoning_frac_chars": (
            len(reason_text) / max(1, len(reason_text) + len(content_text))
        ),
        "finish_reason": finish,
        "usage": usage,
        "timings": timings,
        "content_head": content_text[:240],
        "reasoning_head": reason_text[:240],
    }


def parse_log_slice(text: str) -> list[dict]:
    slots: dict[tuple[str, str], dict] = {}
    extras: list[str] = []
    for line in text.splitlines():
        spec = SPEC_RE.search(line)
        if spec:
            extras.append(spec.group(0))
        match = TIMING_RE.search(line)
        if not match:
            continue
        slot_id, task_id, kind, rest = match.groups()
        row = slots.setdefault(
            (slot_id, task_id),
            {"slot_id": int(slot_id), "task_id": int(task_id)},
        )
        if kind == "draft acceptance":
            acc = ACCEPT_RE.search(rest)
            if acc:
                row["draft_acceptance"] = float(acc.group(1))
                row["draft_accepted"] = int(acc.group(2))
                row["draft_generated"] = int(acc.group(3))
                row["draft_mean_len"] = float(acc.group(4))
            row["draft_acceptance_raw"] = rest.strip()
            continue
        if kind == "graphs reused":
            row["graphs_reused"] = rest.strip()
            continue
        ev = EVAL_RE.search(rest)
        if not ev:
            row[kind.replace(" ", "_") + "_raw"] = rest.strip()
            continue
        ms, ntok, ms_tok, tps = ev.groups()
        prefix = {
            "prompt eval time": "prompt",
            "eval time": "eval",
            "total time": "total",
        }[kind]
        row[f"{prefix}_ms"] = float(ms)
        row[f"{prefix}_tokens"] = int(ntok)
        row[f"{prefix}_ms_per_token"] = float(ms_tok)
        row[f"{prefix}_tok_s"] = float(tps)
    out = list(slots.values())
    if extras:
        for row in out:
            row.setdefault("spec_statistics", extras)
        if not out:
            out.append({"spec_statistics": extras})
    return out


def run_wave(
    *,
    base: str,
    model: str,
    n: int,
    prompt: str,
    max_tokens: int,
    timeout: int,
    seed: int,
    reasoning_strength: str | None,
    log_path: Path | None,
) -> dict:
    log_offset = log_path.stat().st_size if log_path and log_path.exists() else 0
    sampler = GpuSampler()
    sampler.start()
    t0 = time.monotonic()
    rows: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [
            pool.submit(
                stream_once,
                base,
                model,
                i,
                prompt,
                max_tokens,
                timeout,
                seed,
                reasoning_strength,
            )
            for i in range(n)
        ]
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
    wall = time.monotonic() - t0
    gpu = sampler.stop()
    log_rows: list[dict] = []
    log_slice = ""
    if log_path and log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(log_offset)
            log_slice = fh.read()
        log_rows = parse_log_slice(log_slice)
    toks = sum(int(r.get("completion_tokens") or 0) for r in rows)
    decode_rates = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    accept = [r["draft_acceptance"] for r in log_rows if r.get("draft_acceptance") is not None]
    eval_rates = [r["eval_tok_s"] for r in log_rows if r.get("eval_tok_s") is not None]
    return {
        "concurrency": n,
        "ok": len(rows),
        "errors": errors,
        "wall_s": wall,
        "aggregate_e2e_tok_s": toks / wall if wall else 0.0,
        "median_client_decode_tok_s": statistics.median(decode_rates) if decode_rates else None,
        "median_log_eval_tok_s": statistics.median(eval_rates) if eval_rates else None,
        "log_acceptance": accept,
        "acceptance_spread": None if len(accept) < 2 else (max(accept) / max(min(accept), 1e-6)),
        "gpu": {k: v for k, v in gpu.items() if k != "samples"},
        "gpu_samples": gpu.get("samples"),
        "rows": sorted(rows, key=lambda r: r.get("idx", 0)),
        "log_rows": sorted(log_rows, key=lambda r: (r.get("slot_id", -1), r.get("task_id", -1))),
        "log_slice": log_slice,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:18099/v1")
    p.add_argument("--model", default="muse-glimmer-30b")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="~/inference/logs/muse-glimmer.log")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-tokens", type=int, default=0, help="Approximate pad tokens after the base prompt")
    p.add_argument("--reasoning-strength", default="", help="low|medium|high|xhigh or empty=template default")
    p.add_argument("--warmup", action="store_true")
    args = p.parse_args()
    prompt = pad_prompt(PROMPT_SKY, args.prompt_tokens)
    strength = args.reasoning_strength or None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else None

    models = json.load(urllib.request.urlopen(f"{args.base.rstrip('/')}/models", timeout=10))
    if args.warmup:
        try:
            run_wave(
                base=args.base,
                model=args.model,
                n=1,
                prompt=PROMPT_SKY,
                max_tokens=16,
                timeout=args.timeout,
                seed=args.seed,
                reasoning_strength=strength,
                log_path=log_path,
            )
        except Exception as exc:  # noqa: BLE001
            print("warmup failed", repr(exc), file=sys.stderr)

    waves = []
    for rep in range(args.reps):
        wave = run_wave(
            base=args.base,
            model=args.model,
            n=args.concurrency,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=args.seed,
            reasoning_strength=strength,
            log_path=log_path,
        )
        wave["rep"] = rep
        print(
            json.dumps(
                {
                    "label": args.label,
                    "rep": rep,
                    "concurrency": args.concurrency,
                    "ok": wave["ok"],
                    "errors": wave["errors"],
                    "wall_s": round(wave["wall_s"], 3),
                    "aggregate_e2e_tok_s": round(wave["aggregate_e2e_tok_s"], 3),
                    "median_client_decode_tok_s": wave["median_client_decode_tok_s"],
                    "median_log_eval_tok_s": wave["median_log_eval_tok_s"],
                    "log_acceptance": wave["log_acceptance"],
                    "acceptance_spread": wave["acceptance_spread"],
                    "gpu_card_w": (wave.get("gpu") or {}).get("card_w"),
                    "gpu_freq0": (wave.get("gpu") or {}).get("median_freq0_mhz"),
                },
                indent=2,
            )
        )
        # Keep log_slice only on disk, not in stdout.
        waves.append(wave)

    payload = {
        "label": args.label,
        "created_unix": time.time(),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "argv": sys.argv,
        "request": {
            "base": args.base,
            "model": args.model,
            "concurrency": args.concurrency,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "prompt_tokens_pad": args.prompt_tokens,
            "reasoning_strength": strength,
            "prompt_head": prompt[:200],
            "prompt_chars": len(prompt),
        },
        "models": models,
        "waves": waves,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out_path)
    return 1 if any(w["errors"] for w in waves) else 0


if __name__ == "__main__":
    sys.exit(main())
