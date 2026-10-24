"""Serve an NVFP4 lm_head as W4A16_NVFP4 (Marlin, BF16 activations): config-only change, same weights."""
import json, os, sys

n = 0
for f in ("config.json", "hf_quant_config.json"):
    p = os.path.join(sys.argv[1], f)
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    q = c.get("quantization_config", c.get("quantization", c))
    ql = q.get("quantized_layers", {})
    if "lm_head" in ql:
        ql["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": ql["lm_head"].get("group_size", 16)}
        n += 1
    json.dump(c, open(p, "w"), indent=2)
print("w4a16head: patched", n, "files", flush=True)
