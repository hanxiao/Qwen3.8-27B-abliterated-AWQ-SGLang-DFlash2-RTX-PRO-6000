#!/usr/bin/env python3
"""Decode under concurrency: bench2's fourteen prompts pushed through C streams at once.

Per stream it is bench2's rate, (tokens - 1) / (last token - first token); the aggregate is all
completion tokens over the wall time of the round, so it includes prefill and queueing. Answers
are greedy with thinking off and are saved, so two configurations can be compared token for token.

    SGLANG_API_KEY=... python3 bench3.py <url> <out.json> [C ...]     # default C = 1 2 4
"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, OUT = sys.argv[1], sys.argv[2]
CS = [int(c) for c in sys.argv[3:]] or [1, 2, 4]
MODEL, KEY = "Qwen3.8-27B-abliterated-AWQ-INT4", os.environ["SGLANG_API_KEY"]
W = {
    "prose": ["Write a detailed technical essay on the history of computing, from Babbage to GPUs.",
              "Write a long, vivid short story about a lighthouse keeper who finds a message in a bottle."],
    "chat": ["Explain to a friend over coffee why vector databases became popular.",
             "My team keeps missing deadlines. Talk me through what I could change, like a mentor would."],
    "translation": ["Translate this paragraph into French, German and Japanese: 'Every morning the baker opens her shop before sunrise. The smell of fresh bread fills the street, and neighbours stop by on their way to work to chat, buy a loaf, and share the news of the day.'",
                    "Translate into Spanish and Chinese: 'The committee reviewed the proposal in detail and agreed to fund the first phase, on the condition that the team reports progress every quarter and publishes its data openly.'"],
    "summary": ["Summarise the architecture of a modern transformer language model in detail.",
                "Summarise the main causes and consequences of the 2008 financial crisis."],
    "code": ["Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test.",
             "Write a TypeScript function that parses a CSV string with quoted fields into an array of objects, with tests."],
    "math": ["Compute 47*83, then 1234/7 to three decimals, showing each step.",
             "A train leaves at 9:40 at 72 km/h; another leaves the same station at 10:15 at 90 km/h on the same track. When and where does the second catch up? Show every step."],
    "json": ["Return a JSON object describing ten fruits with colour, origin and season.",
             "Return a JSON array of eight fictional employees with name, role, team, start date and three skills each."],
}
JOBS = [(name, p) for name, ps in W.items() for p in ps]


def stream(prompt, max_tokens=800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); first = last = None; toks = 0; text = []
    with urllib.request.urlopen(r, timeout=1800) as s:
        for line in s:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            piece = ch and (ch[0].get("delta") or {}).get("content")
            if piece:
                now = time.time(); first = first or now; last = now; text.append(piece)
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
    rate = (toks - 1) / (last - first) if last and last > first else float("nan")
    return {"rate": rate, "toks": toks, "ttft": (first or t0) - t0, "text": "".join(text)}


stream("hi", max_tokens=8)
report = {"url": URL, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "rounds": {}}
for c in CS:
    t0 = time.time()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda j: stream(j[1]), JOBS))
    wall = time.time() - t0
    for (name, p), r in zip(JOBS, res):
        r.update(workload=name, prompt=p)
    per = {}
    for r in res:
        per.setdefault(r["workload"], []).append(r["rate"])
    mean = sum(sum(v) / len(v) for v in per.values()) / len(per)
    agg = sum(r["toks"] for r in res) / wall
    ttft = sorted(r["ttft"] for r in res)[len(res) // 2]
    report["rounds"][str(c)] = {"per_stream_mean": mean, "aggregate": agg, "wall": wall, "median_ttft": ttft,
                                "workloads": {k: sum(v) / len(v) for k, v in per.items()}, "answers": res}
    print(f"C{c}: per-stream {mean:6.1f} tok/s   aggregate {agg:6.1f} tok/s   wall {wall:5.1f}s   "
          f"median TTFT {ttft:4.2f}s   " + " ".join(f"{k} {sum(v) / len(v):.0f}" for k, v in per.items()), flush=True)
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
