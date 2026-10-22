#!/usr/bin/env python3
"""Decode speed by streaming: (tokens - 1) / (last token - first token) for each answer, so
prefill and network are excluded and a short answer is still a fair sample (the two-call
delta in bench.py breaks when a build answers in fewer tokens than its 600 cap).

    SGLANG_API_KEY=... python3 bench2.py <url> [rounds]
"""
import json, os, sys, time, urllib.request
URL, ROUNDS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
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


def stream(prompt, thinking=False, sampled=False, max_tokens=800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if not sampled:
        body["temperature"] = 0
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    first = last = None; toks = 0
    with urllib.request.urlopen(r, timeout=1800) as s:
        for line in s:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                now = time.time(); first = first or now; last = now
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
    return (toks - 1) / (last - first) if last and last > first else float("nan"), toks


if __name__ == "__main__":  # accept.py imports the prompts and stream()
    stream("hi", max_tokens=8)
    res = {}
    for rnd in range(ROUNDS):
        for name, ps in W.items():
            for p in ps:
                res.setdefault(name, []).append(stream(p))
    tot = []
    for name, v in res.items():
        rate = sum(t for t, _ in v) / len(v); tot.append(rate)
        print(f"  {name:12} {rate:6.1f} tok/s   tokens {[n for _, n in v]}", flush=True)
    print(f"  {'mean':12} {sum(tot) / len(tot):6.1f} tok/s", flush=True)
    for name in ("prose", "chat"):
        v = [stream(p, thinking=True, sampled=True, max_tokens=1200) for p in W[name]]
        print(f"  {name:5} thinking+sampled {sum(t for t, _ in v) / len(v):6.1f} tok/s   tokens {[n for _, n in v]}", flush=True)
