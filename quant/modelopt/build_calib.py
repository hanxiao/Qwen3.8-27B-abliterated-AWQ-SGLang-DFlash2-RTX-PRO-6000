"""Build the calibration set for our own quant: code + agentic/tool-call + zh/en chat + some math.
Writes calib/ownmix.parquet with columns text (fully rendered with the Qwen3.8 chat template), kind, ntok.
Never uses /tmp/qwenopt/fid_corpus.json (that is the KL eval set).
"""
import glob, json, random, re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(0)
tok = AutoTokenizer.from_pretrained("/tmp/codebench/tok")
MAXTOK = 4096
rows = []


def ntok(t):
    return len(tok(t, add_special_tokens=False)["input_ids"])


def add(text, kind):
    n = ntok(text)
    if n < 200:
        return False
    rows.append({"text": text, "kind": kind, "ntok": min(n, MAXTOK)})
    return True


def render(msgs, tools=None):
    try:
        return tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    except Exception as e:
        return None


def cjk_ratio(s):
    if not s:
        return 0
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff") / len(s)


# 1) agentic: OpenHands SWE trajectories (system + multi-turn with tool observations)
ds = load_dataset("SWE-Gym/OpenHands-SFT-Trajectories", split="train.success.oss", streaming=True)
n = 0
for x in ds:
    msgs = [{"role": m["role"], "content": m["content"]} for m in x["messages"] if m.get("content")]
    t = render(msgs)
    if t and add(t, "agentic_swe"):
        n += 1
    if n >= 90:
        break
print("agentic_swe", n, flush=True)

# 2) tool calls rendered natively (qwen3_coder XML via the chat template tools= path)
ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split="train", streaming=True)
n = 0
for x in ds:
    try:
        tools = json.loads(x["tools"]) if isinstance(x["tools"], str) else x["tools"]
        msgs = []
        for m in x["conversations"]:
            r, v = m["from"], m["value"]
            if r == "system":
                continue
            if r == "human":
                msgs.append({"role": "user", "content": v})
            elif r == "gpt":
                calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", v, re.S)
                if calls:
                    tc = []
                    for c in calls:
                        j = json.loads(c.replace("'", '"')) if c.strip().startswith("{'") else json.loads(c)
                        tc.append({"type": "function", "function": {"name": j["name"], "arguments": j.get("arguments", {})}})
                    msgs.append({"role": "assistant", "content": "", "tool_calls": tc})
                else:
                    msgs.append({"role": "assistant", "content": v})
            elif r == "tool":
                for resp in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", v, re.S) or [v]:
                    msgs.append({"role": "tool", "content": resp})
        t = render(msgs, tools=tools)
    except Exception:
        t = None
    if t and "<tool_call>" in t and add(t, "toolcall"):
        n += 1
    if n >= 70:
        break
print("toolcall", n, flush=True)

# 3) code: real source files (python, rust, ts/js) as code-review / edit requests
files = (glob.glob("/tmp/prefillexp/sgl522/python/sglang/srt/**/*.py", recursive=True)
         + glob.glob("/tmp/prefillexp/sgl522/sgl-model-gateway/**/*.rs", recursive=True)
         + glob.glob("/tmp/prefillexp/sgl522/rust/**/*.rs", recursive=True)
         + glob.glob("/tmp/prefillexp/sgl522/python/sglang/srt/**/*.cu", recursive=True)
         + glob.glob("/tmp/prefillexp/sgl522/**/*.ts", recursive=True))
random.shuffle(files)
asks = ["Review this file and point out bugs or risky edge cases.", "Refactor this code for readability; keep behavior identical.",
        "Explain what this module does and how the main functions interact.", "Add type hints and docstrings where they are missing.",
        "这段代码有什么潜在的问题？请给出修改建议。", "Write unit tests for the most important function in this file."]
n = 0
for f in files:
    try:
        src = open(f, encoding="utf-8").read()
    except Exception:
        continue
    if len(src) < 3000:
        continue
    src = src[:14000]
    ext = f.rsplit(".", 1)[-1]
    lang = {"py": "python", "rs": "rust", "ts": "typescript", "cu": "cuda"}[ext]
    msgs = [{"role": "user", "content": f"{random.choice(asks)}\n\n`{f.split('sgl522/')[-1]}`:\n```{lang}\n{src}\n```"}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if add(t, "code_" + ext):
        n += 1
    if n >= 120:
        break
print("code", n, flush=True)

# 4) zh/en chat: sharegpt_gpt4 (mixed zh/en), keep ~60% Chinese
ds = load_dataset("shibing624/sharegpt_gpt4", split="train", streaming=True)
nzh = nen = 0
for x in ds:
    msgs = []
    for m in x["conversations"]:
        r = {"human": "user", "gpt": "assistant"}.get(m["from"])
        if r:
            msgs.append({"role": r, "content": m["value"]})
    if len(msgs) < 2:
        continue
    zh = cjk_ratio(" ".join(m["content"] for m in msgs)) > 0.15
    if zh and nzh >= 160:
        continue
    if not zh and nen >= 60:
        continue
    t = render(msgs)
    if t and add(t, "chat_zh" if zh else "chat_en"):
        if zh:
            nzh += 1
        else:
            nen += 1
    if nzh >= 160 and nen >= 60:
        break
print("chat zh/en", nzh, nen, flush=True)

# 5) cyankiwi (the AWQ calib set): its zh/en multilingual, agentic SWE and a little math/stem
cy = pd.read_parquet("/tmp/ownquant/calib/cyankiwi.parquet")
take = {"agentic": 32, "code": 20, "math": 15, "stem": 10}
cnt = {}
for _, r in cy.iterrows():
    lt = r["language_type"]
    if r["language"] not in ("en", "zh"):
        continue
    k = lt if lt in take else None
    if k is None or cnt.get(k, 0) >= take[k]:
        continue
    msgs = [{"role": m.get("role", m.get("from")), "content": m.get("content", m.get("value", ""))} for m in list(r["conversations"])]
    t = render(msgs)
    if t and add(t, "cy_" + k + "_" + r["language"]):
        cnt[k] = cnt.get(k, 0) + 1
print("cyankiwi", cnt, flush=True)

# 6) a few long code contexts from our codebench prompts (windows of the 30k context)
ctx = open("/tmp/codebench/ctx_30.txt", encoding="utf-8").read()
for i in range(6):
    s = random.randint(0, max(0, len(ctx) - 15000))
    msgs = [{"role": "user", "content": "Here is part of a repository. Summarize the architecture.\n\n" + ctx[s:s + 15000]}]
    add(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "codebench_ctx")

random.shuffle(rows)
df = pd.DataFrame(rows)
df.to_parquet("/tmp/ownquant/calib/ownmix.parquet")
print(df.groupby("kind").agg(n=("ntok", "size"), tok=("ntok", "sum")))
print("TOTAL", len(df), int(df.ntok.sum()))
