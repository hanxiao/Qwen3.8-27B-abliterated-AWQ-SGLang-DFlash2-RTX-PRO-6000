"""Own ModelOpt mixed quantization of huihui-ai/Huihui-Qwen3.8-27B-abliterated (BF16).

FP8 (per-tensor W8A8, static) for attention q/k/v/o and GDN in_proj_qkv/in_proj_z/out_proj,
NVFP4 W4A4 for the MLPs, lm_head NVFP4 / FP8 / BF16, FP8 KV scales.
Also: offline proxy KL vs BF16 on the fid corpus (held out from calibration), optional
per-layer sensitivity sweep, export in ModelOpt (hf_quant_config) format, plus a BF16-head twin.

    python mo.py <bf16 dir> <local out dir>
env: NAME ALGO(max|mse|local_hessian|fos) HEAD(nvfp4|fp8|bf16) KV(fp8|none) SENS(0|1)
     EXCL_FP8="3,63" (MLP layers kept FP8) EXCL_W4A16="..." (MLP layers with BF16 activations)
     NCAL SEQ EXPORT(0|1) BH(0|1) CALIB FIDC RES
"""
import copy, fnmatch, gc, json, os, shutil, sys, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer

t0 = time.time()
SRC, OUT = sys.argv[1], Path(sys.argv[2])
NAME = os.environ.get("NAME", "x")
ALGO = os.environ.get("ALGO", "max")
HEAD = os.environ.get("HEAD", "nvfp4")
KV = os.environ.get("KV", "fp8")
SENS = os.environ.get("SENS", "0") == "1"
EXPORT = os.environ.get("EXPORT", "1") == "1"
BH = os.environ.get("BH", "0") == "1"
NCAL = int(os.environ.get("NCAL", "512"))
SEQ = int(os.environ.get("SEQ", "4096"))
EXCL_FP8 = [int(x) for x in os.environ.get("EXCL_FP8", "").split(",") if x.strip()]
EXCL_W4A16 = [int(x) for x in os.environ.get("EXCL_W4A16", "").split(",") if x.strip()]
OQ = os.environ.get("OQ", "/gcs/ownq")
DEV = os.environ.get("DEV", "cuda")
EVLEN = int(os.environ.get("EVLEN", "2048"))
RES = Path(os.environ.get("RES", OQ + "/res"))
RES.mkdir(parents=True, exist_ok=True)
results = {"name": NAME, "algo": ALGO, "head": HEAD, "kv": KV, "excl_fp8": EXCL_FP8, "excl_w4a16": EXCL_W4A16,
           "ncal": NCAL, "seq": SEQ}


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def save_results():
    (RES / f"{NAME}.json").write_text(json.dumps(results, indent=1))


# ---------------------------------------------------------------- model + data
model = AutoModelForImageTextToText.from_pretrained(SRC, dtype=torch.bfloat16, device_map=DEV)
model.eval()
NL = model.config.get_text_config().num_hidden_layers
tok = AutoTokenizer.from_pretrained(SRC)
log("loaded", type(model).__name__, (torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0), "GiB")
if os.environ.get("RMSRC"):
    for f in Path(SRC).glob("*.safetensors"):
        f.unlink()

cal = pd.read_parquet(os.environ["CALIB"])
cal_ids = [tok(t, add_special_tokens=False, max_length=SEQ, truncation=True, return_tensors="pt")["input_ids"]
           for t in cal["text"][:NCAL]]
log("calib", len(cal_ids), "samples", sum(x.shape[1] for x in cal_ids), "tokens")

docs = json.load(open(os.environ["FIDC"]))
EV = []
seen = {}
for d in docs:                               # 2 docs per kind, 2048 tokens each
    if seen.get(d["kind"], 0) >= 2:
        continue
    seen[d["kind"]] = seen.get(d["kind"], 0) + 1
    ids = tok(d["text"], add_special_tokens=False, max_length=EVLEN, truncation=True, return_tensors="pt")["input_ids"]
    EV.append((d["kind"], ids))
SUB = [0, 2, 4, 6]                            # one doc per kind for the per-layer sweep
log("eval docs", [(k, x.shape[1]) for k, x in EV])


@torch.inference_mode()
def logits_of(ids):
    return model(input_ids=ids.to(DEV), use_cache=False).logits[0].float()


REF = []
with torch.inference_mode():
    for kind, ids in EV:
        lp = torch.log_softmax(logits_of(ids), -1)
        v, i = lp.topk(64, -1)
        REF.append((v.cpu(), i.cpu()))
        del lp
torch.cuda.empty_cache()
log("bf16 reference done")


@torch.inference_mode()
def evaluate(which=None):
    idx = range(len(EV)) if which is None else which
    per = {}
    tot_kl = tot_n = agree = 0.0
    for j in idx:
        kind, ids = EV[j]
        rv, ri = REF[j]
        lp = torch.log_softmax(logits_of(ids), -1)
        rv, ri = rv.to(DEV), ri.to(DEV)
        cand = lp.gather(1, ri)
        kl = (rv.exp() * (rv - cand)).sum(-1)
        a = (lp.argmax(-1) == ri[:, 0]).float()
        s = per.setdefault(kind, [0.0, 0, 0.0])
        s[0] += kl.sum().item(); s[1] += kl.numel(); s[2] += a.sum().item()
        tot_kl += kl.sum().item(); tot_n += kl.numel(); agree += a.sum().item()
        del lp
    return {"kl": round(tot_kl / tot_n, 5), "top1": round(agree / tot_n, 4),
            **{"kl_" + k: round(v[0] / v[1], 5) for k, v in per.items()}}


# ---------------------------------------------------------------- quant config
def preset_cfg(preset, name):
    for e in preset["quant_cfg"]:
        e = dict(e)
        if e.get("quantizer_name") == name and e.get("cfg") is not None:
            return copy.deepcopy(e["cfg"])
    raise KeyError(name)


FP8 = {"num_bits": (4, 3), "axis": None}
NVFP4_DYN = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}, "axis": None}
NVFP4_STA = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}, "axis": None}
try:
    NVFP4_DYN = preset_cfg(mtq.NVFP4_DEFAULT_CFG, "*weight_quantizer")
    NVFP4_STA = preset_cfg(mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG, "*weight_quantizer")
    FP8 = preset_cfg(mtq.FP8_DEFAULT_CFG, "*weight_quantizer")
except Exception as e:
    log("preset extraction failed, using literals", repr(e))
NVFP4_ACT = copy.deepcopy(NVFP4_DYN)
KVCFG = copy.deepcopy(FP8)
log("cfgs", FP8, NVFP4_DYN, NVFP4_STA)

if ALGO == "max":
    algorithm, WQ = "max", NVFP4_DYN
elif ALGO == "mse":
    algorithm, WQ = {"method": "mse", "fp8_scale_sweep": True}, NVFP4_STA
elif ALGO == "local_hessian":
    algorithm, WQ = {"method": "local_hessian", "fp8_scale_sweep": True}, NVFP4_STA
elif ALGO in ("mse_hr", "lh_hr", "max_hr"):
    wsa = {"mse_hr": {"method": "mse", "fp8_scale_sweep": True}, "lh_hr": {"method": "local_hessian", "fp8_scale_sweep": True},
           "max_hr": {"method": "max"}}[ALGO]
    algorithm = {"method": "nvfp4_act_headroom", "anchor_percentile": float(os.environ.get("HR_ANCHOR", "1")),
                 "upper_percentile": float(os.environ.get("HR_UPPER", "99.99")), "rho": float(os.environ.get("HR_RHO", "16384")),
                 "weight_scale_algorithm": wsa}
    WQ = NVFP4_DYN if ALGO == "max_hr" else NVFP4_STA
elif ALGO == "fos":
    algorithm = {"method": "mse", "fp8_scale_sweep": False, "start_multiplier": 1.0, "stop_multiplier": 1.5, "step_size": 0.5}
    WQ = preset_cfg(mtq.NVFP4_FOUR_OVER_SIX_CFG, "*weight_quantizer")
else:
    raise SystemExit("bad ALGO " + ALGO)

L = "*language_model.layers"
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ATT = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
qc = [{"quantizer_name": "*", "enable": False}]
for p in MLP:
    qc += [{"quantizer_name": f"{L}.*.{p}.weight_quantizer", "cfg": WQ},
           {"quantizer_name": f"{L}.*.{p}.input_quantizer", "cfg": NVFP4_ACT}]
for p in ATT:
    qc += [{"quantizer_name": f"{L}.*.{p}.weight_quantizer", "cfg": FP8},
           {"quantizer_name": f"{L}.*.{p}.input_quantizer", "cfg": FP8}]
for i in EXCL_FP8:
    for p in MLP:
        qc += [{"quantizer_name": f"{L}.{i}.{p}.weight_quantizer", "cfg": FP8},
               {"quantizer_name": f"{L}.{i}.{p}.input_quantizer", "cfg": FP8}]
for i in EXCL_W4A16:
    qc += [{"quantizer_name": f"{L}.{i}.mlp.*input_quantizer", "enable": False}]
if KV == "fp8":
    qc += [{"quantizer_name": "*[kv]_bmm_quantizer", "cfg": KVCFG}]
for pat in ["*linear_attn.conv1d*", "*linear_attn.in_proj_a*", "*linear_attn.in_proj_b*", "*lm_head*", "*visual*",
            "*vision*", "mtp.*", "*mtp*", "*embed_tokens*"]:
    qc += [{"quantizer_name": pat, "enable": False}]
if HEAD == "nvfp4":
    qc += [{"quantizer_name": "*lm_head.weight_quantizer", "cfg": WQ},
           {"quantizer_name": "*lm_head.input_quantizer", "cfg": NVFP4_ACT}]
elif HEAD == "fp8":
    qc += [{"quantizer_name": "*lm_head.weight_quantizer", "cfg": FP8},
           {"quantizer_name": "*lm_head.input_quantizer", "cfg": FP8}]
config = {"quant_cfg": qc, "algorithm": algorithm}


def fwd(m):
    with torch.inference_mode():
        for j, ids in enumerate(cal_ids):
            m(input_ids=ids.to(DEV), use_cache=False)
            if j % 100 == 0:
                log("calib fwd", j)


head_bf16 = model.lm_head.weight.detach().to("cpu", copy=True)
log("quantize", ALGO)
mtq.quantize(model, config, forward_loop=fwd)
log("quantized")
Q = {n: m for n, m in model.named_modules() if isinstance(m, TensorQuantizer) and m.is_enabled}
kvq = [n for n in Q if n.endswith("k_bmm_quantizer") or n.endswith("v_bmm_quantizer")]
results["n_enabled_quantizers"] = len(Q)
results["n_kv_quantizers"] = len(kvq)
log("enabled quantizers", len(Q), "kv", len(kvq), "examples", list(Q)[:4], kvq[:2])
results["head_quantizers"] = [n for n in Q if "lm_head" in n]


def toggle(names, on):
    for n in names:
        (Q[n].enable if on else Q[n].disable)()


def group(*pats):
    return [n for n in Q if any(fnmatch.fnmatch(n, p) for p in pats)]


toggle(kvq, False)                              # proxy KL with BF16 KV
results["eval"] = evaluate()
log("EVAL", results["eval"])
save_results()

if SENS:
    sens = {}
    groups = {"mlp": group(f"{L}.*.mlp.*"), "mlp_act": group(f"{L}.*.mlp.*input_quantizer"),
              "attn": group(f"{L}.*.self_attn.*_proj.*"), "gdn": group(f"{L}.*.linear_attn.*"),
              "head": group("*lm_head*")}
    for g, names in groups.items():
        if not names:
            continue
        toggle(names, False)
        sens["off_" + g] = evaluate()
        toggle(names, True)
        log("SENS off", g, len(names), sens["off_" + g])
    sens["sub_base"] = evaluate(SUB)
    per = []
    for i in range(NL):
        r = {"layer": i}
        for g, pats in (("mlp", [f"{L}.{i}.mlp.*"]), ("mlp_act", [f"{L}.{i}.mlp.*input_quantizer"]),
                        ("att", [f"{L}.{i}.self_attn.*_proj.*", f"{L}.{i}.linear_attn.*"])):
            names = group(*pats)
            toggle(names, False)
            r[g] = evaluate(SUB)["kl"]
            toggle(names, True)
        per.append(r)
        log("SENS layer", r)
        sens["per_layer"] = per
        results["sens"] = sens
        save_results()
    order = sorted(per, key=lambda r: r["mlp"])
    for k in (2, 4, 6, 8, 12):
        names = group(*[f"{L}.{r['layer']}.mlp.*" for r in order[:k]])
        toggle(names, False)
        sens[f"top{k}_mlp_off"] = {"layers": [r["layer"] for r in order[:k]], **evaluate()}
        toggle(names, True)
        log("SENS top", k, sens[f"top{k}_mlp_off"])
    order_a = sorted(per, key=lambda r: r["mlp_act"])
    for k in (4, 8, 16):
        names = group(*[f"{L}.{r['layer']}.mlp.*input_quantizer" for r in order_a[:k]])
        toggle(names, False)
        sens[f"top{k}_mlpact_off"] = {"layers": [r["layer"] for r in order_a[:k]], **evaluate()}
        toggle(names, True)
        log("SENS top act", k, sens[f"top{k}_mlpact_off"])
    order_t = sorted(per, key=lambda r: r["att"])
    for k in (4, 8):
        names = group(*[p for r in order_t[:k] for p in (f"{L}.{r['layer']}.self_attn.*_proj.*", f"{L}.{r['layer']}.linear_attn.*")])
        toggle(names, False)
        sens[f"top{k}_att_off"] = {"layers": [r["layer"] for r in order_t[:k]], **evaluate()}
        toggle(names, True)
        log("SENS top att", k, sens[f"top{k}_att_off"])
    results["sens"] = sens
    save_results()

if not EXPORT:
    log("NO EXPORT, DONE")
    print("MO DONE", flush=True)
    sys.exit(0)

toggle(kvq, True)
from modelopt.torch.export import export_hf_checkpoint

OUT.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=OUT)
log("exported", sorted(os.listdir(OUT)))
for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
          "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "LICENSE"):
    if (Path(SRC) / f).exists() and not (OUT / f).exists():
        shutil.copy(Path(SRC) / f, OUT / f)
hq = json.load(open(OUT / "hf_quant_config.json"))
q = hq.get("quantization", hq)
results["export"] = {k: v for k, v in q.items() if k != "quantized_layers"}
from collections import Counter
results["export"]["algos"] = dict(Counter(v.get("quant_algo") for v in q.get("quantized_layers", {}).values()))
results["export"]["head"] = q.get("quantized_layers", {}).get("lm_head")
save_results()
log("export summary", results["export"])



def upload(local, name, skip=()):
    import subprocess
    r = subprocess.run(["python3", "/gcs/ownq/job/gcsul.py", str(local), os.environ.get("BUCKET", "BUCKET"), "ownq/" + name, *skip])
    if r.returncode != 0:
        log("gcsul failed, falling back to copytree")
        dst = Path(OQ) / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(local):
            if f not in skip:
                shutil.copy(Path(local) / f, dst / f)

upload(OUT, NAME)
log("uploaded", NAME)


def strip_head(obj):
    """Remove lm_head from quantized-layer maps/targets; add it to ignore lists."""
    if isinstance(obj, dict):
        obj.pop("lm_head", None)
        for k, v in list(obj.items()):
            if k in ("ignore", "exclude_modules") and isinstance(v, list):
                if "lm_head" not in v:
                    v.append("lm_head")
            elif isinstance(v, list):
                obj[k] = [x for x in v if x != "lm_head"]
                for x in obj[k]:
                    strip_head(x)
            else:
                strip_head(v)


if BH and HEAD != "bf16":
    from safetensors.torch import load_file, save_file
    BHD = Path(str(OUT) + "-bh")
    BHD.mkdir(parents=True, exist_ok=True)
    ip = OUT / "model.safetensors.index.json"
    if ip.exists():
        idx = json.load(open(ip))
    else:
        from safetensors import safe_open
        with safe_open(OUT / "model.safetensors", "pt") as sf:
            idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in sf.keys()}}
    wm = idx["weight_map"]
    hk = [k for k in wm if k.startswith("lm_head.")]
    shard = wm["lm_head.weight"]
    sd = load_file(OUT / shard)
    for k in hk:
        sd.pop(k, None)
        wm.pop(k)
    sd["lm_head.weight"] = head_bf16.to(torch.bfloat16).contiguous()
    wm["lm_head.weight"] = shard
    save_file(sd, BHD / shard, metadata={"format": "pt"})
    del sd
    json.dump(idx, open(BHD / "model.safetensors.index.json", "w"), indent=2)
    for f in os.listdir(OUT):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy(OUT / f, BHD / f)
    for cf in ("hf_quant_config.json", "config.json"):
        c = json.load(open(BHD / cf))
        strip_head(c)
        json.dump(c, open(BHD / cf, "w"), indent=2)
    log("bh variant", hk)
    upload(BHD, NAME + "-bh")
    upload(OUT, NAME + "-bh", [f for f in os.listdir(OUT) if not f.endswith(".safetensors") or f == shard])
    log("copied bh")
results["done"] = True
save_results()
print("MO DONE", flush=True)
