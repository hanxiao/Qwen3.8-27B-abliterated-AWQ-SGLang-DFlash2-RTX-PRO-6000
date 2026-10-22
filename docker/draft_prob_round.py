"""Accept draft probabilities that round just above 1 in sampled chain verify (sglang PR #45819),
applied to an installed SGLang v0.5.22.

The chain-verify kernel treats a draft token whose probability q is above 1.0 as impossible and
rejects it. An fp32 softmax over a peaked row can return 1.0 plus a few ulp, so exactly the drafts
the drafter is surest of were always rejected whenever sampling is on. DFlash's sampled verify,
with or without block verification, runs this kernel. The fix admits q up to 1 + 1e-3 and clamps
it to 1.0, in the acceptance tests and in both residual passes; NaN, infinities, negatives and
larger values are still rejected.

    python3 draft_prob_round.py   # edits the installed package; fails if v0.5.22's text moved
"""
import pathlib

import sglang

P = pathlib.Path(sglang.__file__).parent / "kernels/ops/speculative/reject_sampling.py"
s = P.read_text()


def sub(old, new, count):
    global s
    assert s.count(old) == count, f"expected {count}x, found {s.count(old)}x: {old!r}"
    s = s.replace(old, new)


sub("import triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n\n_MAX_DRAFT_PROB = tl.constexpr(1.0 + 1e-3)\n", 1)
sub("q_is_prob = (q > 0.0) & (q <= 1.0)\n",
    "q_is_prob = (q > 0.0) & (q <= _MAX_DRAFT_PROB)\n", 2)
# Clamp after each guard, keeping the guard line's own indentation.
lines = s.split("\n")
out = []
for line in lines:
    out.append(line)
    if line.strip() == "q_is_prob = (q > 0.0) & (q <= _MAX_DRAFT_PROB)":
        out.append(line[: len(line) - len(line.lstrip())] + "q = tl.minimum(q, 1.0)")
s = "\n".join(out)
sub("q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)",
    "q_val = tl.where((q_val >= 0.0) & (q_val <= _MAX_DRAFT_PROB), tl.minimum(q_val, 1.0), 0.0)", 2)
P.write_text(s)
print("patched", P)
