#!/usr/bin/env python3
"""Seven-workload net decode and an arithmetic check.

Decode is a two-call delta: the same prompt at 60 and 600 tokens, (c2-c1)/(d2-d1). The
subtraction removes prefill and the network round trip, which a single timed call includes and
which on a remote service are most of a short generation.

    python3 scripts/bench.py https://YOUR-SERVICE.run.app
"""
import json, subprocess, sys, time, urllib.request
URL, MODEL = sys.argv[1], "Qwen3.8-27B-NVFP4"
TOK = subprocess.run(["gcloud","auth","print-identity-token"],capture_output=True,text=True).stdout.strip()
def post(body, timeout=900):
    r = urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type":"application/json","Authorization":"Bearer "+TOK})
    t0=time.time(); d=json.loads(urllib.request.urlopen(r,timeout=timeout).read()); return time.time()-t0, d
def ndec(prompt):
    b=lambda mt:{"model":MODEL,"messages":[{"role":"user","content":prompt}],"temperature":0,
                 "max_tokens":mt,"chat_template_kwargs":{"enable_thinking":False}}
    d1,r1=post(b(60)); d2,r2=post(b(600))
    c1,c2=r1["usage"]["completion_tokens"],r2["usage"]["completion_tokens"]
    return (c2-c1)/(d2-d1)
WORK=[("prose","Write a detailed technical essay on the history of computing, from Babbage to GPUs."),
      ("code","Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test."),
      ("json","Return a JSON object describing ten fruits with colour, origin and season."),
      ("chat","Explain to a friend over coffee why vector databases became popular."),
      ("math","Compute 47*83, then 1234/7 to three decimals, showing each step."),
      ("translation","Translate the following into French, German and Japanese: 'The quick brown fox jumps over the lazy dog, and the dog does not mind.'"),
      ("summary","Summarise the architecture of a modern transformer language model in detail.")]
post({"model":MODEL,"messages":[{"role":"user","content":"hi"}],"max_tokens":8})
rows=[]
for name,p in WORK:
    t=ndec(p); rows.append((name,t)); print(f"  {name:12} {t:6.1f} tok/s", flush=True)
lo=min(r[1] for r in rows); hi=max(r[1] for r in rows)
print(f"  {'mean':12} {sum(r[1] for r in rows)/len(rows):6.1f} tok/s   (min {lo:.1f} {min(rows,key=lambda r:r[1])[0]}, max {hi:.1f} {max(rows,key=lambda r:r[1])[0]})")
# arithmetic quality, greedy
qs=[("17*23","391"),("144/12","12"),("2^10","1024"),("99+101","200"),("7*8*9","504"),
    ("1000-333","667"),("15%4","3"),("sqrt(169)","13"),("12*12","144"),("81/9","9")]
ok=0
for q,a in qs:
    _,d=post({"model":MODEL,"messages":[{"role":"user","content":f"Compute {q}. Reply with only the number."}],
              "temperature":0,"max_tokens":200,"chat_template_kwargs":{"enable_thinking":False}})
    txt=(d["choices"][0]["message"].get("content") or "").strip()
    ok += a in txt.replace(",","")
print(f"  arithmetic   {ok}/{len(qs)}")
