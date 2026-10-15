#!/usr/bin/env python3
"""What a long prompt does to answers already streaming: D streams decode, then a prompt of N
unique random words arrives. Reports each stream's rate before and during the prefill and its
longest gap between chunks, plus the long prompt's time to first token.

    SGLANG_API_KEY=... python3 stall.py <url> [D] [words]     # default D=2, words=40000 (~124k tokens)
"""
import json, os, random, sys, threading, time, urllib.request

URL = sys.argv[1]
D = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
MODEL, KEY = "Qwen3.8-27B-AWQ-INT4", os.environ["SGLANG_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}


def run(prompt, max_tokens, marks):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "ignore_eos": True}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H)
    with urllib.request.urlopen(r, timeout=1800) as s:
        for line in s:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                d = json.loads(line[5:])
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    marks.append(time.time())


rng = random.Random(7)
long_prompt = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8)))
                       for _ in range(N)) + "\n\nReply with one word."
streams = [[] for _ in range(D)]
ts = [threading.Thread(target=run, args=(f"Write a very long essay about topic number {i}: the history of bridges.", 1500, streams[i]))
      for i in range(D)]
for t in ts:
    t.start()
time.sleep(6)
t_long = time.time(); long_marks = []
tl = threading.Thread(target=run, args=(long_prompt, 4, long_marks)); tl.start(); tl.join()
t_first = long_marks[0] if long_marks else float("nan")
for t in ts:
    t.join()
print(f"long prompt: {N} words, TTFT {t_first - t_long:.1f}s with {D} streams decoding")
for i, m in enumerate(streams):
    before = [x for x in m if x < t_long]
    during = [x for x in m if t_long <= x <= t_first]
    gaps = [b - a for a, b in zip(m, m[1:])]
    rate = lambda xs: (len(xs) - 1) / (xs[-1] - xs[0]) if len(xs) > 1 and xs[-1] > xs[0] else float("nan")
    print(f"  stream {i}: chunks/s before {rate(before):5.1f}, during prefill {len(during) / max(t_first - t_long, 1e-9):5.1f}, "
          f"longest gap {max(gaps):.2f}s, after {rate([x for x in m if x > t_first]):5.1f}")
