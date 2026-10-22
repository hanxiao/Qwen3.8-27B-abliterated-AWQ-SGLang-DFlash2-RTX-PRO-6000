#!/usr/bin/env python3
"""Decode speed deep into a context, where agent traffic lives (median ~115k tokens here).

Short-prompt benchmarks miss what changes with depth: every decode step also reads the attention
KV of the whole context, and an attention-kernel change can move deep-context decode while short
prompts look the same. At ~30k and ~100k tokens of random words, the prompt is first prefilled into
the radix cache, then a greedy 600-token answer is timed from it, as bench2 times: (tokens - 1) /
(last token - first token), with the tokens each verify step committed. Two answers per depth.

    SGLANG_API_KEY=... python3 longdecode.py <url>
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accept  # noqa: E402  (its counters(); its benchmark runs only as a script)
import prefill  # noqa: E402  (its words(); its benchmark runs only as a script)

URL, KEY = sys.argv[1], os.environ["SGLANG_API_KEY"]
MODEL = "Qwen3.8-27B-abliterated-AWQ-INT4"
ASK = "\nIgnore the words above. Write a 500-word story about a lighthouse keeper and a storm."


def run(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    first = last = None
    toks = ptok = 0
    with urllib.request.urlopen(r, timeout=3600) as s:
        for line in s:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.time()
                first = first or now
                last = now
            if d.get("usage"):
                toks, ptok = d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]
    return ptok, toks, (toks - 1) / (last - first) if last and last > first else float("nan")


for n in (11000, 37000):
    prompt = prefill.words(n, n) + ASK
    run(prompt, 1)  # into the radix cache, so the timed answers start without a long prefill
    rates = []
    g0, v0 = accept.counters()
    for _ in range(2):
        p, t, rate = run(prompt, 600)
        rates.append(rate)
    g1, v1 = accept.counters()
    print(f"  context {p:7d} tok  decode {sum(rates) / len(rates):6.1f} tok/s  ({', '.join(f'{x:.1f}' for x in rates)})"
          f"  tokens/verify {(g1 - g0) / max(1, v1 - v0):.2f}", flush=True)
