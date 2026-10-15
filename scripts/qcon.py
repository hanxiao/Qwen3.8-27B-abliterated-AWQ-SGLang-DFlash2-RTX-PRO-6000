#!/usr/bin/env python3
"""Does serving several requests at once change answers? sglang#36548 reports DFlash2 answers
going wrong, and prompts bleeding into other conversations, only under concurrent load.

Probe set: 32 three-step arithmetic problems and 32 four-item ordering puzzles with a unique
answer, greedy, thinking on. Run once alone (C=1) and once at C streams with distractor requests
(essays and code on unrelated topics) interleaved. Reports accuracy both ways, how many probe
answers are byte-identical, and any probe answer that mentions a distractor's topic words.

    SGLANG_API_KEY=... python3 qcon.py <url> <out.json> [C]     # default C=4
"""
import itertools, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, OUT = sys.argv[1], sys.argv[2]
C = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MODEL, KEY = "Qwen3.8-27B-NVFP4", os.environ["SGLANG_API_KEY"]


def chat(prompt, mt, thinking):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": mt,
         "chat_template_kwargs": {"enable_thinking": thinking}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    m = json.loads(urllib.request.urlopen(r, timeout=1800).read())["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n</think>\n" + (m.get("content") or "")


rng = random.Random(11)
probes = []
for _ in range(32):
    a, b = rng.randint(12, 60), rng.randint(12, 60)
    c = rng.randint(3, 9); d = rng.randint(2, (a * b) // (c * 2)); e = rng.randint(10, 99)
    probes.append({"kind": "math", "answer": str(a * b - c * d + e),
                   "q": f"A warehouse holds {a} crates with {b} bottles in each. It ships out {c} orders of {d} bottles "
                        f"each, then receives a delivery of {e} loose bottles. How many bottles does it hold now? "
                        f"End your answer with 'Answer: <number>'."})
NAMES = ["Mira", "Tobias", "Ilse", "Rafael", "Noor", "Pavel", "Greta", "Omar", "Lena", "Kofi"]
while len(probes) < 64:
    ppl = rng.sample(NAMES, 4)
    clauses = []
    for _ in range(12):
        x, y = rng.sample(ppl, 2)
        k = rng.choice(["before", "notfirst", "notlast", "adjacent"])
        clauses.append((k, x, y))
    for n in range(2, 8):
        cs = clauses[:n]
        ok = [o for o in itertools.permutations(ppl) if all(
            (k == "before" and o.index(x) < o.index(y)) or (k == "notfirst" and o[0] != x) or
            (k == "notlast" and o[-1] != x) or (k == "adjacent" and abs(o.index(x) - o.index(y)) == 1) for k, x, y in cs)]
        if len(ok) == 1:
            break
    if len(ok) != 1:
        continue
    text = {"before": "{x} arrives before {y}.", "notfirst": "{x} is not the first to arrive.",
            "notlast": "{x} is not the last to arrive.", "adjacent": "{x} and {y} arrive one right after the other, in some order."}
    rules = " ".join(text[k].format(x=x, y=y) for k, x, y in cs)
    probes.append({"kind": "order", "answer": ", ".join(ok[0]),
                   "q": f"Four people, {', '.join(ppl)}, arrive at a meeting one at a time. {rules} "
                        f"In what order do they arrive? End your answer with 'Answer: ' followed by the four names in order, comma-separated."})
DISTRACT = [("Kant", "Explain Kant's categorical imperative and its critics at length."),
            ("lighthouse", "Write a long story about a lighthouse keeper on a stormy coast."),
            ("LRUCache", "Write a Python class LRUCache with O(1) get and put, with tests."),
            ("photosynthesis", "Explain photosynthesis in detail for a biology student."),
            ("Byzantine", "Describe the fall of the Byzantine Empire in detail."),
            ("Kubernetes", "Explain how Kubernetes schedules pods, in detail.")]


def score(p, out):
    m = re.findall(r"Answer:\s*(.+)", out.split("</think>")[-1])
    if not m:
        return False
    got = m[-1].strip().rstrip(".").replace("*", "")
    if p["kind"] == "math":
        return re.sub(r"[,\s]", "", got).lstrip("$") == p["answer"]
    return [s.strip() for s in got.split(",")] == [s.strip() for s in p["answer"].split(",")]


def leak(out):
    return [w for w, _ in DISTRACT if w.lower() in out.lower()]


report = {"url": URL, "C": C}
t0 = time.time()
serial = [chat(p["q"], 6000, True) for p in probes]
report["serial_secs"] = time.time() - t0
jobs = []
for i, p in enumerate(probes):
    jobs.append(("probe", i))
    if i % 4 == 0:
        jobs.append(("distract", (i // 4) % len(DISTRACT)))
t0 = time.time()
with ThreadPoolExecutor(C) as ex:
    res = list(ex.map(lambda j: chat(probes[j[1]]["q"], 6000, True) if j[0] == "probe"
                      else chat(DISTRACT[j[1]][1], 1500, False), jobs))
report["concurrent_secs"] = time.time() - t0
conc = [None] * len(probes)
for j, out in zip(jobs, res):
    if j[0] == "probe":
        conc[j[1]] = out
rows = []
for p, s, c in zip(probes, serial, conc):
    rows.append({"kind": p["kind"], "answer": p["answer"], "serial_ok": score(p, s), "conc_ok": score(p, c),
                 "identical": s == c, "leak_serial": leak(s), "leak_conc": leak(c), "serial": s, "conc": c})
report["rows"] = rows
json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=1)
for kind in ("math", "order"):
    r = [x for x in rows if x["kind"] == kind]
    print(f"{kind:5}: serial {sum(x['serial_ok'] for x in r)}/{len(r)}, C{C} {sum(x['conc_ok'] for x in r)}/{len(r)}, "
          f"identical {sum(x['identical'] for x in r)}/{len(r)}")
lk = [(i, x["leak_conc"]) for i, x in enumerate(rows) if x["leak_conc"] and not x["leak_serial"]]
print(f"distractor words in concurrent answers only: {lk or 'none'}")
print(f"wall: serial {report['serial_secs']:.0f}s, C{C} {report['concurrent_secs']:.0f}s (incl. {len(jobs) - len(probes)} distractors)")
