#!/usr/bin/env python3
"""Shareable DFlash public-boundary suite.

This is the suite we can quote without the 256-token thinking-window caveats.
It is not Spec-Bench (480 prompts) and not a no-spec A/B.

Protocol (v1):
  * public prompts only (GSM8K test[0], HumanEval/0, MT-Bench 81)
  * OpenAI streaming chat on the live cell
  * greedy temperature=0
  * generate until stop; max_tokens is a cap, not a target
  * warmup excluded
  * a row is quoteable only if finish=stop AND content is non-empty
  * GSM8K / HumanEval have exact checks on *content*, not reasoning
  * decode tok/s is completion_tokens / (last − first generated chunk); e2e tok/s includes TTFT
  * acceptance is accepted draft tokens / draft runs (vLLM, no bonus)

Usage:
  python3 vllm-dflash-share-suite.py [num_reps] [max_tokens] [out.json]
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROTOCOL = "dflash-share-suite/v1"
BASE = os.environ.get("DFLASH_SHARE_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("DFLASH_SHARE_MODEL", "muse-glimmer-gptq")

PROMPTS = (
    {
        "id": "gsm8k-janet",
        "kind": "math",
        "source": "GSM8K test[0], zero-shot chat",
        "check": "gsm8k",
        "expected": "18",
        "prompt": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with "
            "four. She sells the remainder at the farmers' market daily for "
            "$2 per fresh duck egg. How much in dollars does she make every "
            "day at the farmers' market?"
        ),
    },
    {
        "id": "humaneval-0",
        "kind": "code",
        "source": "HumanEval/0 doctest, chat-wrapped",
        "check": "humaneval0",
        "expected": "doctest",
        "prompt": (
            "Complete the following Python function. Return only the function "
            "implementation, including the docstring.\n\n"
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
    },
    {
        "id": "mtbench-81",
        "kind": "writing",
        "source": "MT-Bench question 81 turn 1",
        "check": "content",
        "expected": "content>=400",
        "prompt": (
            "Compose an engaging travel blog post about a recent trip to "
            "Hawaii, highlighting cultural experiences and must-see attractions."
        ),
    },
)


def chat(prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": MODEL,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{BASE.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_any = None
    first_content = None
    usage = None
    finish = None
    content_parts: list[str] = []
    reason_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
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
            if not ch:
                continue
            choice = ch[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            now = time.perf_counter()
            reason = (
                (delta.get("reasoning") or "")
                + (delta.get("reasoning_content") or "")
            )
            content = delta.get("content") or ""
            if reason:
                if first_any is None:
                    first_any = now
                reason_parts.append(reason)
            if content:
                if first_any is None:
                    first_any = now
                if first_content is None:
                    first_content = now
                content_parts.append(content)
    t1 = time.perf_counter()
    content = "".join(content_parts)
    reasoning = "".join(reason_parts)
    ct = (usage or {}).get("completion_tokens")
    wall = t1 - t0
    decode_s = None if first_any is None else t1 - first_any
    return {
        "ttft_s": None if first_any is None else round(first_any - t0, 4),
        "ttft_content_s": None if first_content is None else round(first_content - t0, 4),
        "wall_s": round(wall, 4),
        "decode_wall_s": None if decode_s is None else round(decode_s, 4),
        "completion_tokens": ct,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "e2e_tok_s": None if not (ct and wall) else round(ct / wall, 3),
        "decode_tok_s": None if not (ct and decode_s) else round(ct / decode_s, 3),
        "finish_reason": finish,
        "content": content,
        "reasoning": reasoning,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
    }


def scrape_spec_counters() -> dict[str, float]:
    want = re.compile(r"^vllm:spec_decode_num_(accepted_tokens|drafts|draft_tokens)_total$")
    totals: dict[str, float] = {}
    with urllib.request.urlopen(f"{BASE.rstrip('/')}/metrics", timeout=10) as resp:
        text = resp.read().decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if not want.fullmatch(name):
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


def counter_deltas(pre: dict[str, float], post: dict[str, float]) -> dict[str, float]:
    names = set(pre) | set(post)
    return {
        name: post.get(name, 0.0) - pre.get(name, 0.0)
        for name in sorted(names)
        if abs(post.get(name, 0.0) - pre.get(name, 0.0)) > 0.5
    }


def acceptance_stats(deltas: dict[str, float]) -> dict[str, float | None]:
    acc = deltas.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    drafts = deltas.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tok = deltas.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    return {
        "accepted_draft_tokens": acc or None,
        "draft_runs": drafts or None,
        "draft_tokens": draft_tok or None,
        "accepted_per_draft_run": (acc / drafts) if drafts else None,
        "accepted_per_draft_token": (acc / draft_tok) if draft_tok else None,
        "emitted_per_verify": ((acc + drafts) / drafts) if drafts else None,
        "draft_tokens_per_run": (draft_tok / drafts) if drafts else None,
    }


def gsm8k_final_number(content: str) -> str | None:
    hashed = re.findall(r"####\s*(-?\d+)", content)
    if hashed:
        return hashed[-1]
    nums = re.findall(r"-?\d+", content.replace(",", ""))
    return nums[-1] if nums else None


def extract_python(content: str) -> str | None:
    fenced = re.findall(r"```(?:python)?\n(.*?)```", content, re.S)
    if fenced:
        return fenced[0]
    if "def has_close_elements" in content:
        return content
    return None


def score_humaneval0(content: str) -> tuple[bool, str]:
    code = extract_python(content)
    if not code:
        return False, "no python in content"
    if "def has_close_elements" not in code:
        return False, "missing has_close_elements"
    script = (
        "from typing import List\n"
        + code
        + "\n"
        "assert has_close_elements([1.0, 2.0, 3.0], 0.5) is False\n"
        "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True\n"
        "print('PASS')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode == 0 and "PASS" in proc.stdout:
        return True, "doctest pass"
    err = (proc.stderr or proc.stdout or "fail").strip().splitlines()
    return False, err[-1][:200] if err else "fail"


def score_row(item: dict, row: dict) -> dict:
    finish_ok = row.get("finish_reason") == "stop"
    content_ok = (row.get("content_chars") or 0) > 0
    check = item["check"]
    detail = None
    passed = False
    if check == "gsm8k":
        got = gsm8k_final_number(row.get("content") or "")
        detail = got
        passed = got == item["expected"]
    elif check == "humaneval0":
        passed, detail = score_humaneval0(row.get("content") or "")
    elif check == "content":
        passed = (row.get("content_chars") or 0) >= 400
        detail = f"content_chars={row.get('content_chars')}"
    quoteable = bool(finish_ok and content_ok and passed)
    return {
        "finish_ok": finish_ok,
        "content_ok": content_ok,
        "check": check,
        "passed": passed,
        "detail": detail,
        "quoteable": quoteable,
    }


def median(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.median(vals), 4) if vals else None


def compact_row(row: dict) -> dict:
    skip = {"content", "reasoning"}
    return {k: v for k, v in row.items() if k not in skip}


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    out_path = sys.argv[3] if len(sys.argv) > 3 else (
        f"/tmp/dflash-share-suite-{int(time.time())}.json"
    )
    timeout = int(os.environ.get("DFLASH_SHARE_TIMEOUT", "180"))
    if max_tokens < 512:
        raise SystemExit("refuse max_tokens < 512; this suite generates until stop")

    prompt_results = []
    print(
        f"== {PROTOCOL} prompts={len(PROMPTS)} reps={reps} "
        f"max_tokens={max_tokens} temp=0 model={MODEL} ==",
        flush=True,
    )
    for item in PROMPTS:
        print(f"-- {item['id']} warmup --", flush=True)
        chat(item["prompt"], max_tokens, timeout)
        pre = scrape_spec_counters()
        print(f"-- {item['id']} measured --", flush=True)
        rows = [chat(item["prompt"], max_tokens, timeout) for _ in range(reps)]
        post = scrape_spec_counters()
        deltas = counter_deltas(pre, post)
        acc = acceptance_stats(deltas)
        scored = [score_row(item, row) for row in rows]
        quoteable_e2e = [
            row["e2e_tok_s"]
            for row, sc in zip(rows, scored)
            if sc["quoteable"]
        ]
        quoteable_decode = [
            row["decode_tok_s"]
            for row, sc in zip(rows, scored)
            if sc["quoteable"]
        ]
        summary = {
            "id": item["id"],
            "kind": item["kind"],
            "source": item["source"],
            "check": item["check"],
            "rows": [
                {**compact_row(row), "score": sc, "content": row["content"],
                 "reasoning": row["reasoning"]}
                for row, sc in zip(rows, scored)
            ],
            "median_e2e_tok_s": median([r["e2e_tok_s"] for r in rows]),
            "median_decode_tok_s": median([r["decode_tok_s"] for r in rows]),
            "median_ttft_content_s": median([r["ttft_content_s"] for r in rows]),
            "median_completion_tokens": median([r["completion_tokens"] for r in rows]),
            "quoteable_n": sum(1 for sc in scored if sc["quoteable"]),
            "quoteable_median_e2e_tok_s": median(quoteable_e2e),
            "quoteable_median_decode_tok_s": median(quoteable_decode),
            "all_quoteable": all(sc["quoteable"] for sc in scored),
            "counter_deltas": deltas,
            "acceptance": acc,
        }
        prompt_results.append(summary)
        acc_run = acc["accepted_per_draft_run"]
        print(
            item["id"],
            "quoteable", f"{summary['quoteable_n']}/{reps}",
            "finish", ",".join(r["finish_reason"] or "?" for r in rows),
            "decode", summary["quoteable_median_decode_tok_s"],
            "e2e", summary["quoteable_median_e2e_tok_s"],
            "ttfc", summary["median_ttft_content_s"],
            "toks", summary["median_completion_tokens"],
            "acc/run", None if acc_run is None else round(acc_run, 3),
            "checks", ",".join(
                "PASS" if sc["passed"] else f"FAIL:{sc['detail']}"
                for sc in scored
            ),
            flush=True,
        )

    quoteable = [p for p in prompt_results if p["all_quoteable"]]
    out = {
        "protocol": PROTOCOL,
        "reps": reps,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "model": MODEL,
        "base": BASE,
        "warmup_excluded": True,
        "concurrency": 1,
        "prompts": prompt_results,
        "quoteable_prompt_n": len(quoteable),
        "across_quoteable_median_decode_tok_s": median(
            [p["quoteable_median_decode_tok_s"] for p in quoteable]
        ),
        "across_quoteable_median_e2e_tok_s": median(
            [p["quoteable_median_e2e_tok_s"] for p in quoteable]
        ),
        "across_quoteable_median_accepted_per_draft_run": median(
            [
                p["acceptance"]["accepted_per_draft_run"]
                for p in quoteable
                if p["acceptance"]["accepted_per_draft_run"] is not None
            ]
        ),
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print("quoteable_prompts", f"{len(quoteable)}/{len(prompt_results)}")
    print("across_quoteable_median_decode_tok_s", out["across_quoteable_median_decode_tok_s"])
    print("across_quoteable_median_e2e_tok_s", out["across_quoteable_median_e2e_tok_s"])
    print(
        "across_quoteable_median_accepted_per_draft_run",
        out["across_quoteable_median_accepted_per_draft_run"],
    )
    print("saved", out_path)
    if len(quoteable) != len(prompt_results):
        raise SystemExit("suite has non-quoteable prompts; do not share a headline number")


if __name__ == "__main__":
    main()
