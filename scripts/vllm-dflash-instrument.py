#!/usr/bin/env python3
"""DFlash instrumentation at the public boundary: client decode tok/s +
server-side spec-decode counters scraped from /metrics around a measured run.

Usage: python3 vllm-dflash-instrument.py [num_reps] [max_tokens]
"""
import json
import re
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "muse-glimmer-gptq"

PROMPT = (
    "Explain in detail why the sky is blue. Cover Rayleigh scattering, "
    "wavelength dependence, and why sunsets look red. Write complete sentences."
)


def chat(max_tokens, max_completion_tokens=None):
    body = json.dumps({
        "model": MODEL,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or "") \
                    + (d.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    return {
        "ttft_s": None if first is None else round(first - t0, 4),
        "completion_tokens": ct,
        "decode_tok_s": None if not (first and ct) else round(ct / (t1 - first), 3),
        "e2e_tok_s": None if not ct else round(ct / (t1 - t0), 3),
        # wall time of the whole streaming window (first->last chunk)
        "decode_wall_s": None if first is None else round(t1 - first, 4),
    }


def scrape_spec_counters():
    """Return {metric_name: total_value} for spec-decode-ish counters,
    summed across label sets."""
    want = re.compile(
        r"spec_decode|accept", re.I)
    totals = {}
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if not want.search(name):
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    print("== warmup x2 ==")
    for _ in range(2):
        chat(max_tokens)

    pre = scrape_spec_counters()
    print("== measured reps ==")
    rows = [chat(max_tokens) for _ in range(reps)]
    post = scrape_spec_counters()

    for i, r in enumerate(rows):
        print(i, r)
    dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    e2e = [r["e2e_tok_s"] for r in rows if r["e2e_tok_s"]]
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    print("median_decode_tok_s", round(statistics.median(dec), 3) if dec else None)
    print("median_e2e_tok_s", round(statistics.median(e2e), 3) if e2e else None)
    print("median_ttft_s", round(statistics.median(ttfts), 4) if ttfts else None)

    print("== spec counter deltas ==")
    for name in sorted(set(pre) | set(post)):
        delta = post.get(name, 0.0) - pre.get(name, 0.0)
        if abs(delta) > 0:
            print(f"{name} delta={delta:.0f}")

    out = {
        "reps": reps,
        "max_tokens": max_tokens,
        "rows": rows,
        "median_decode_tok_s": round(statistics.median(dec), 3) if dec else None,
        "median_e2e_tok_s": round(statistics.median(e2e), 3) if e2e else None,
        "counters_pre": pre,
        "counters_post": post,
    }
    path = f"/tmp/dflash-instrument-{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
