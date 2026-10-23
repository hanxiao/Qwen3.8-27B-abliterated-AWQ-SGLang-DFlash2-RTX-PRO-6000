"""Decode speed and tokens per verify step when the answer copies its prompt: a document quoted
back, a source file rewritten with one rename, passages quoted from a document. Greedy, thinking
off, each prompt twice (the second run hits the prefix cache), counters read from /metrics.

    python3 scripts/copybench.py URL   # SGLANG_API_KEY in the environment

This is where lookup drafting pays (SGLANG_DFLASH_LOOKUP=1); bench2.py and accept.py are the
workloads where it must not cost anything.
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accept  # noqa: E402  (counters())
import bench2  # noqa: E402  (stream())

R = pathlib.Path(__file__).resolve().parents[1]
doc = (R / "README.md").read_text()[:12000]
code = (R / "scripts" / "qcon.py").read_text()
W = {
    "quote": ("Repeat the following text exactly, word for word, then add one final line that "
              "says DONE.\n\n" + doc, 4500),
    "edit": ("Here is a Python file. Return the complete file with every occurrence of the name "
             "`main` renamed to `run`. Output only the code, nothing else.\n\n```python\n" + code
             + "\n```", 4500),
    "passages": ("Quote verbatim every paragraph of the document below that mentions prefill or "
                 "decode. Quote only, no commentary.\n\n" + doc, 3000),
}

if __name__ == "__main__":
    bench2.stream("hi", max_tokens=8)
    for name, (prompt, mx) in W.items():
        for rnd in range(2):
            g0, v0 = accept.counters()
            tps, toks = bench2.stream(prompt, max_tokens=mx)
            g1, v1 = accept.counters()
            print(f"  {name:9s} run {rnd}  {tps:6.1f} tok/s  {toks:5d} tokens  "
                  f"{(g1 - g0) / max(1, v1 - v0):5.2f} tokens per verify", flush=True)
