#!/usr/bin/env python3
"""Quality gate for a kernel change: long-context retrieval and multi-step arithmetic.

    SGLANG_API_KEY=... python3 quality.py <url>

needle: five facts ("The access code for <city> is <n>.") at depths 5-95% of a haystack of
random words, ~30k and ~120k tokens; the model is asked for all five. Scored per fact.
math: 12 generated three-step word problems with exact positive integer answers, greedy, thinking on.
"""
import json, os, random, re, sys, urllib.request
URL, KEY, MODEL = sys.argv[1], os.environ["SGLANG_API_KEY"], "Qwen3.8-27B-abliterated-AWQ-INT4"


def ask(prompt, mt=300, thinking=False):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": mt,
         "chat_template_kwargs": {"enable_thinking": thinking}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())["choices"][0]["message"]["content"] or ""


def needle(nwords, seed):
    rng = random.Random(seed)
    w = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8))) for _ in range(nwords)]
    cities = ["Lisbon", "Osaka", "Nairobi", "Quito", "Tallinn"]
    codes = [rng.randint(10000, 99999) for _ in cities]
    for i, (c, n) in enumerate(zip(cities, codes)):
        w.insert(int(len(w) * (0.05 + 0.225 * i)), f". The access code for {c} is {n}. ")
    out = ask(" ".join(w) + "\n\nList the access code for each of Lisbon, Osaka, Nairobi, Quito and Tallinn.")
    return sum(str(n) in out for n in codes)


def math(seed):
    rng = random.Random(seed); ok = 0
    for _ in range(12):
        a, b = rng.randint(12, 60), rng.randint(12, 60)
        c = rng.randint(3, 9); d = rng.randint(2, (a * b) // (c * 2))
        e = rng.randint(10, 99)
        ans = a * b - c * d + e
        q = (f"A warehouse holds {a} crates with {b} bottles in each. It ships out {c} orders of {d} "
             f"bottles each, then receives a delivery of {e} loose bottles. How many bottles does it hold "
             f"now? End your answer with 'Answer: <number>'.")
        out = ask(q, 3000, thinking=True).replace(",", "")
        m = re.findall(r"Answer:\s*(-?\d+)", out)
        ok += bool(m) and int(m[-1]) == ans
    return ok


print(f"  needle 30k  {needle(9500, 1)}/5, {needle(9500, 2)}/5", flush=True)
print(f"  needle 120k {needle(38000, 3)}/5", flush=True)
print(f"  math        {math(5)}/12", flush=True)
