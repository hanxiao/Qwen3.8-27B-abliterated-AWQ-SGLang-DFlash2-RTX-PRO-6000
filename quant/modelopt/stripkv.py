"""Remove the FP8 KV-cache fields from a ModelOpt checkpoint dir so --kv-cache-dtype=bf16 is safe."""
import json, os, sys


def strip(o):
    if isinstance(o, dict):
        for k in ("kv_cache_scheme", "kv_cache_quant_algo"):
            o.pop(k, None)
        for v in o.values():
            strip(v)
    elif isinstance(o, list):
        for v in o:
            strip(v)


for f in ("config.json", "hf_quant_config.json"):
    p = os.path.join(sys.argv[1], f)
    if os.path.exists(p):
        c = json.load(open(p))
        strip(c)
        json.dump(c, open(p, "w"), indent=2)
        print("stripkv", p, flush=True)
