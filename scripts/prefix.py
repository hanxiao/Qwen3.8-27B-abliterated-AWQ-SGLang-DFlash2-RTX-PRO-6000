#!/usr/bin/env python3
"""Do answers stay right when a conversation resumes from the prefix cache?

A follow-up turn reuses the cached prefix of the conversation, which for this model means the
recurrent (Gated DeltaNet) state saved at a checkpoint as well as the attention KV. A state saved
or restored wrong shows as a model that no longer finds what the document says. One conversation:
a ~20k-token document of random words with ten facts planted at spread-out depths, then ten turns,
each asking for one fact, greedy, each turn carrying the earlier questions and answers. Reports the
facts found and how much of each turn's prompt the server served from cache.

    SGLANG_API_KEY=... python3 prefix.py <url>
"""
import json
import os
import random
import sys
import urllib.request

URL, KEY, MODEL = sys.argv[1], os.environ["SGLANG_API_KEY"], "Qwen3.8-27B-abliterated-AWQ-INT4"
rnd = random.Random(7)
WORDS = [w for w in "time river stone light paper glass winter garden signal window market harbor "
         "silver orbit ladder candle meadow engine forest marble lantern copper thunder violet "
         "anchor bridge cotton desert ember falcon granite hollow island jacket kettle lemon".split()]
CITIES = ["Lisbon", "Oslo", "Nairobi", "Lima", "Hanoi", "Quito", "Perth", "Tunis", "Riga", "Dakar"]
codes = {c: rnd.randint(1000, 9999) for c in CITIES}
chunks = [" ".join(rnd.choice(WORDS) for _ in range(1600)) for _ in range(10)]
doc = ""
for c, chunk in zip(CITIES, chunks):
    cut = rnd.randint(200, 1400)
    words = chunk.split()
    doc += " ".join(words[:cut]) + f" The vault code for {c} is {codes[c]}. " + " ".join(words[cut:]) + "\n"

msgs = [{"role": "user", "content": "Read this document; I will ask about it.\n\n" + doc + "\n\nReply 'ready'."},
        {"role": "assistant", "content": "ready"}]
found, cached = 0, []
for i, c in enumerate(CITIES):
    msgs.append({"role": "user", "content": f"What is the vault code for {c}? Answer with the number only."})
    body = {"model": MODEL, "messages": msgs, "max_tokens": 600, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    u = d.get("usage") or {}
    hit = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cached.append((hit, u.get("prompt_tokens")))
    ok = str(codes[c]) in ans
    found += ok
    print(f"  turn {i + 1:2d} {c:8s} want {codes[c]}  got {ans[:30]!r:34s} {'ok' if ok else 'WRONG'}  "
          f"cached {hit}/{u.get('prompt_tokens')}")
    msgs.append({"role": "assistant", "content": ans})
print(f"{URL}: {found}/{len(CITIES)} facts; turns 2+ served {sum(h for h, _ in cached[1:])}"
      f"/{sum(p or 0 for _, p in cached[1:])} prompt tokens from cache")
