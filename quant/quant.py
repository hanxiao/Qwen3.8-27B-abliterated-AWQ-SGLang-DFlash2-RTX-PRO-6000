"""cyankiwi's Qwen3.8-27B-AWQ-INT4 recipe, applied to another checkpoint of the same model.

Read off cyankiwi's config.json (their recipe file is not published): AWQ, int4 weights,
asymmetric, group size 32, mse observer, every Linear except lm_head, the vision tower, the MTP
head, linear_attn.in_proj_a/b, and layer 0's linear_attn.out_proj. Calibration: cyankiwi's public
calibration set (384 agentic and STEM conversations; their card names calibration-medium, which
is private). llm-compressor copies the MTP tensors over unquantized, as cyankiwi has them.

    python quant.py <model id or dir> <out dir> [samples] [seq_len]
"""
import ast, json, os, shutil, sys, time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

SRC, OUT = sys.argv[1], Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 384
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
# cyankiwi's public calibration set unless CALIB names a local copy.
CALIB = os.environ.get("CALIB") or hf_hub_download("cyankiwi/calibration", "calibration_dataset.parquet",
                                                   repo_type="dataset")

IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*", r"re:.*linear_attn\.in_proj_[ab]$",
          r"re:.*language_model\.layers\.0\.linear_attn\.out_proj$"]
SCHEME = {"group_0": {"targets": ["Linear"],
                      "weights": {"num_bits": 4, "type": "int", "symmetric": False, "group_size": 32,
                                  "strategy": "group", "observer": "mse", "dynamic": False}}}

t0 = time.time()
src = SRC if Path(SRC).is_dir() else snapshot_download(SRC)
print(f"weights at {src} ({time.time() - t0:.0f}s)", flush=True)
# On the CPU: llm-compressor's sequential pipeline moves one layer at a time to the GPU. With all
# 55 GB resident on a 96 GB card, layer 3's attention plus AWQ's cached activations ran out of
# memory. SDPA rather than eager attention: eager materializes every attention matrix.
model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="sdpa")
tok = AutoTokenizer.from_pretrained(src)

rows = pd.read_parquet(CALIB)
texts = []
for c in rows["conversations"][:N]:
    msgs = ast.literal_eval(c) if isinstance(c, str) else list(c)
    msgs = [{"role": m.get("role", m.get("from")), "content": m.get("content", m.get("value", ""))} for m in msgs]
    try:
        texts.append(tok.apply_chat_template(msgs, tokenize=False))
    except Exception:
        texts.append("\n\n".join(f"{m['role']}: {m['content']}" for m in msgs))
ds = Dataset.from_dict({"text": texts}).map(
    lambda b: tok(b["text"], max_length=SEQ, truncation=True, add_special_tokens=False),
    batched=True, remove_columns=["text"])
print(f"calibration: {len(ds)} samples, <= {SEQ} tokens, "
      f"{sum(len(x) for x in ds['input_ids'])} tokens total", flush=True)

oneshot(model=model, processor=tok, dataset=ds, recipe=[AWQModifier(config_groups=SCHEME, ignore=IGNORE)],
        max_seq_length=SEQ, num_calibration_samples=len(ds))
print(f"calibrated ({time.time() - t0:.0f}s)", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT, save_compressed=True)
# Tokenizer and processor files as the source has them, byte for byte.
for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json",
          "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "LICENSE"):
    if (Path(src) / f).exists():
        shutil.copy(Path(src) / f, OUT / f)

print(f"saved to {OUT} ({time.time() - t0:.0f}s)", flush=True)
