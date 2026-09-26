#!/usr/bin/env python3
"""Prefill: time to first token on prompts of unique random words (the radix cache cannot serve
them) at ~9k, ~31k and ~124k tokens, streamed so the first chunk marks the end of prefill.

    SGLANG_API_KEY=... python3 prefill.py <url>
"""
import json, os, random, sys, time, urllib.request
URL = sys.argv[1]
MODEL, KEY = "Qwen3.8-27B-NVFP4", os.environ["SGLANG_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def ttft(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H)
    t0 = time.time(); first = None; ptok = None
    with urllib.request.urlopen(r, timeout=1800) as s:
        for line in s:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if first is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None:
                first = time.time() - t0
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
    return first, ptok


def words(n, seed):
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    return " ".join("".join(rng.choice(alpha) for _ in range(rng.randint(3, 8))) for _ in range(n))


def prefill():
    ttft("hello")
    for n in (3000, 10000, 40000):  # ~2.7 tokens a word
        t, p = ttft(words(n, time.time_ns()) + "\nReply with OK.")
        print(f"  prefill {p:7d} tok  ttft {t:6.2f} s  {p / t:7.0f} tok/s", flush=True)



prefill()
