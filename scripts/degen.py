#!/usr/bin/env python3
"""Do long sampled answers stay healthy? A recurrent state that goes wrong mid-answer shows as
repetition or drift long before it shows in a short benchmark.

Twenty-four long-form prompts in English and Chinese, thinking on and returned inline, the model's
own sampling, up to 4,000 tokens, two at a time so both slots run together. Per answer: the zlib
ratio of the text (compressed / raw bytes; a loop compresses far better than prose), whether it
reached the length cap, and whether its last 4,000 characters repeat themselves (the share of
64-character windows seen earlier in that tail). Compare servers on the same prompts.

    SGLANG_API_KEY=... python3 degen.py <url> [out.json]
"""
import json
import os
import sys
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
KEY, MODEL = os.environ["SGLANG_API_KEY"], "Qwen3.8-27B-abliterated-AWQ-INT4"
PROMPTS = [
    "Write a long essay on how the printing press changed European science and religion.",
    "Tell a long story about two rival clockmakers in 18th-century Geneva.",
    "Explain, in depth, how a CPU executes an instruction, from fetch to retire.",
    "Write a detailed travel guide to Kyoto for a first-time visitor in autumn.",
    "Explain the causes of the First World War and argue which one mattered most.",
    "Write a Python module implementing a trie with insert, search, prefix listing and deletion, with tests.",
    "Describe how vaccines train the immune system, at the level of a university course.",
    "Write a long letter from a lighthouse keeper to his daughter, over one stormy winter.",
    "Compare relational and document databases for a growing e-commerce company, in depth.",
    "Explain how transformers work, from tokenization to sampling, with the math.",
    "Write a long, funny story about a cat who becomes mayor of a small town.",
    "Teach me the basics of music theory: scales, intervals, chords and progressions.",
    "写一篇长文，讨论唐朝为什么能成为一个开放的国际化帝国。",
    "写一个长篇故事，讲述一位老裁缝在上海弄堂里度过的五十年。",
    "详细解释量子纠缠是什么，以及它为什么不能用来超光速通信。",
    "写一篇详细的成都美食指南，包括街头小吃和老字号餐馆。",
    "分析宋朝经济为什么发达，从农业、商业、技术三个角度展开。",
    "用 Python 写一个简单的 LRU 缓存和一个 LFU 缓存，并解释两者的区别和测试方法。",
    "写一封长信，假设你是一位在南极科考站过冬的研究员，写给家人。",
    "详细讲解区块链的工作原理，从哈希、默克尔树到共识机制。",
    "写一个长篇科幻故事，讲人类第一次和另一个文明交换数学。",
    "比较中医和西医在慢性病治疗上的思路，尽量客观全面。",
    "详细介绍围棋的基本规则、常见定式和初学者的学习路径。",
    "写一篇长文，讨论人工智能对教育的影响，包括好处和风险。",
]


def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000,
            "chat_template_kwargs": {"enable_thinking": True}, "separate_reasoning": False}
    if os.environ.get("DEGEN_GREEDY") == "1":  # the same prompts at temperature 0
        body["temperature"] = 0
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
    c = d["choices"][0]
    text = c["message"].get("content") or ""
    raw = text.encode()
    tail = text[-4000:]
    wins = [tail[i:i + 64] for i in range(0, max(0, len(tail) - 64), 8)]
    seen, rep = set(), 0
    for w in wins:
        rep += w in seen
        seen.add(w)
    return {"prompt": prompt[:40], "tokens": d["usage"]["completion_tokens"],
            "capped": c.get("finish_reason") == "length",
            "zlib": round(len(zlib.compress(raw)) / max(1, len(raw)), 3),
            "tail_repeat": round(rep / max(1, len(wins)), 3)}


with ThreadPoolExecutor(2) as ex:
    rows = list(ex.map(ask, PROMPTS))
for x in rows:
    flag = "  LOOP?" if x["tail_repeat"] > 0.3 else ""
    print(f"  {x['prompt']:42s} {x['tokens']:5d} tok  zlib {x['zlib']:.3f}  tail-repeat {x['tail_repeat']:.2f}"
          f"{'  capped' if x['capped'] else ''}{flag}")
n = len(rows)
print(f"{URL}: mean zlib {sum(x['zlib'] for x in rows) / n:.3f}, capped {sum(x['capped'] for x in rows)}/{n}, "
      f"tail-repeat > 0.3: {sum(x['tail_repeat'] > 0.3 for x in rows)}/{n}")
if OUT:
    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=1)
