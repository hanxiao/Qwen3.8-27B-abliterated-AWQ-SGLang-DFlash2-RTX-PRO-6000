#!/usr/bin/env python3
# Net-decode tok/s by the two-call delta method from MiaAI-Lab/bench/ndec.py:
# same prompt at 60 and 600 tokens, (c2-c1)/(d2-d1). Cancels prefill AND network latency,
# which my earlier single-call numbers both included.
import json, subprocess, sys, time, urllib.request
URL, MODEL = sys.argv[1], sys.argv[2]
TOK = subprocess.run(["gcloud","auth","print-identity-token"],capture_output=True,text=True).stdout.strip()
def call(prompt, mt, thinking=False):
    body = {"model": MODEL, "messages":[{"role":"user","content":prompt}],
            "temperature":0.0, "max_tokens":mt,
            "chat_template_kwargs":{"enable_thinking":thinking}}
    r = urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type":"application/json","Authorization":"Bearer "+TOK})
    t0=time.time(); d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    return time.time()-t0, d["usage"]["completion_tokens"]
call("Say OK", 16)
probes = [("code","Write a Python class LRUCache with O(1) get and put using OrderedDict, plus a small test."),
          ("prose","Write a detailed technical essay on the history of computing, from Babbage to GPUs.")]
tot=[]
for name,p in probes:
    d1,c1 = call(p,60); d2,c2 = call(p,600)
    tps=(c2-c1)/(d2-d1); tot.append(tps)
    print(f"  {name:6} net decode = {tps:6.1f} tok/s   (c1={c1} d1={d1:.2f} | c2={c2} d2={d2:.2f})")
print(f"  {'mean':6}             {sum(tot)/len(tot):6.1f} tok/s")
