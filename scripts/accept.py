#!/usr/bin/env python3
"""Tokens per verification step, the lever speculative decoding pulls, on sampled answers.

Runs bench2's fourteen prompts with thinking on and the model's own sampling, one at a time, and
reads the server's counters before and after: generated tokens over verify calls is the mean
number of tokens one target pass commits. On a server with no other traffic the counters are
exact, where timing carries noise. Decode rate is reported too, as in bench2: (tokens - 1) / (last
token - first token) per answer.

    SGLANG_API_KEY=... python3 accept.py <url> [rounds]
"""
import json
import os
import re
import sys
import urllib.request

sys.argv += [] if len(sys.argv) > 2 else ["1"]
URL, ROUNDS = sys.argv[1], int(sys.argv[2])
KEY = os.environ["SGLANG_API_KEY"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench2  # noqa: E402  (its prompts and stream(); its benchmark runs only as a script)


def counters():
    r = urllib.request.Request(URL + "/metrics", headers={"Authorization": "Bearer " + KEY})
    text = urllib.request.urlopen(r, timeout=30).read().decode()
    got = lambda name: sum(float(v) for v in re.findall(rf"^sglang:{name}{{[^}}]*}} ([0-9.e+]+)$", text, re.M))
    return got("generation_tokens_total"), got("spec_verify_calls_total")


if __name__ == "__main__":  # longdecode.py imports counters()
    bench2.stream("hi", max_tokens=8)
    g0, v0 = counters()
    rates, toks = [], 0
    for _ in range(ROUNDS):
        for ps in bench2.W.values():
            for p in ps:
                rate, n = bench2.stream(p, thinking=True, sampled=True, max_tokens=1200)
                rates.append(rate)
                toks += n
    g1, v1 = counters()
    ok = [r for r in rates if r == r]
    print(f"{URL}  answers={len(rates)} tokens={toks}")
    print(f"  tokens per verify step: {(g1 - g0) / max(1, v1 - v0):.3f}  ({int(g1 - g0)} / {int(v1 - v0)})")
    print(f"  decode, mean of answers: {sum(ok) / len(ok):.1f} tok/s")
