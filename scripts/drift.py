#!/usr/bin/env python3
"""Where do two servers' greedy answers part? A state-handling change that is not bit-exact shows
as answers diverging from the reference; how early, and how often, against how much the reference
diverges from itself, says whether the change is noise or a drift.

Twenty-four prompts, greedy, thinking off, 400 tokens, one request at a time (batch size changes
the arithmetic on this stack, so concurrency would confound it). Saves each server's answers, and
compares two saved runs: the token index of the first difference (or "same"), per prompt.

    SGLANG_API_KEY=... python3 drift.py run <url> out.json
    python3 drift.py cmp a.json b.json
"""
import json
import os
import sys
import urllib.request

PROMPTS = [
    "Explain how a hash map handles collisions.", "Write a short poem about autumn rain.",
    "What causes the seasons on Earth?", "Summarise the plot of Hamlet in one paragraph.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between TCP and UDP.", "Give three tips for writing clear emails.",
    "Describe how photosynthesis works.", "Translate 'The weather is lovely today' into French and German.",
    "What is the Pythagorean theorem? Give an example.", "Explain what a mutex is.",
    "Write a haiku about the sea.", "解释一下什么是机器学习。", "写一首关于月亮的短诗。",
    "简述长城的历史。", "用三句话介绍北京。", "什么是区块链？", "如何提高睡眠质量？",
    "Explain recursion to a ten-year-old.", "List five uses of vinegar around the house.",
    "Why is the sky blue?", "Write a SQL query that finds the top 5 customers by revenue.",
    "Describe the water cycle.", "What are the main ideas of stoicism?",
]


def run(url, out):
    key = os.environ["SGLANG_API_KEY"]
    res = []
    for p in PROMPTS:
        body = {"model": "Qwen3.8-27B-abliterated-AWQ-INT4", "messages": [{"role": "user", "content": p}],
                "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        d = json.loads(urllib.request.urlopen(r, timeout=600).read())
        res.append(d["choices"][0]["message"].get("content") or "")
    json.dump(res, open(out, "w"), ensure_ascii=False)


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same, firsts = 0, []
    for x, y in zip(A, B):
        if x == y:
            same += 1
            continue
        i = next((k for k, (c1, c2) in enumerate(zip(x, y)) if c1 != c2), min(len(x), len(y)))
        firsts.append(i)
    firsts.sort()
    print(f"{a} vs {b}: identical {same}/{len(A)}; first difference at character "
          f"{firsts if firsts else '-'}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3])
