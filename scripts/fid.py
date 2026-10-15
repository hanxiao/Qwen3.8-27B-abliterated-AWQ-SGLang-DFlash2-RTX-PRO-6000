#!/usr/bin/env python3
"""Fidelity of one checkpoint on a fixed corpus, teacher-forced: the log-probability of every
actual next token and the top-20 next-token distribution at every position, from SGLang's
native /generate. Two dumps compare with `fid.py cmp`: KL divergence and top-1 agreement
against a reference checkpoint, and each one's perplexity on the corpus text.

    SGLANG_API_KEY=... python3 fid.py dump <url> <corpus.json> <out.json.gz>
    python3 fid.py cmp <ref.json.gz> <cand.json.gz> [...]

KL is taken over the reference's top 20 (P_ref(x) log P_ref(x)/P_cand(x)); a candidate token
outside its own top 20 gets the candidate's 20th log-probability, an upper bound on its mass.
"""
import gzip, json, math, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

K = 20


def dump(url, corpus, out):
    key = os.environ["SGLANG_API_KEY"]
    docs = json.load(open(corpus))

    def one(d):
        body = {"text": d["text"], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": K}
        r = urllib.request.Request(url + "/generate", json.dumps(body).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        m = json.loads(urllib.request.urlopen(r, timeout=1800).read())["meta_info"]
        tok = [(t[1], t[0]) for t in m["input_token_logprobs"]]          # (id, logprob of the actual token)
        top = [[(x[1], x[0]) for x in (p or [])] for p in m["input_top_logprobs"]]
        return {"kind": d["kind"], "tok": tok, "top": top}

    with ThreadPoolExecutor(int(os.environ.get("FID_JOBS", "1"))) as ex:
        res = list(ex.map(one, docs))
    with gzip.open(out, "wt") as f:
        json.dump({"url": url, "docs": res}, f)
    n = sum(len(r["tok"]) for r in res)
    print(f"{out}: {len(res)} docs, {n} positions")


def load(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)["docs"]


def cmp(ref_p, cands):
    ref = load(ref_p)
    rows = [("ref " + os.path.basename(ref_p), ref, None)] + [(os.path.basename(c), load(c), True) for c in cands]
    kinds = sorted({d["kind"] for d in ref})
    print(f"{'checkpoint':34} {'ppl':>7} " + " ".join(f"{'kl ' + k:>11}" for k in kinds) + f" {'kl all':>8} {'top1':>7}")
    for name, docs, is_cand in rows:
        nll = n = 0
        kl = {k: [0.0, 0] for k in kinds}; agree = tot = 0
        for rd, cd in zip(ref, docs):
            assert len(rd["tok"]) == len(cd["tok"]), "different tokenization"
            for i in range(1, len(cd["tok"])):            # position 0 has no logprob
                lp = cd["tok"][i][1]
                if lp is not None:
                    nll -= lp; n += 1
                if not is_cand:
                    continue
                rt, ct = rd["top"][i], cd["top"][i]          # the distribution token i was drawn from
                if not rt or not ct:
                    continue
                cmap = dict((t, l) for t, l in ct); floor = min(l for _, l in ct)
                s = sum(math.exp(l) * (l - cmap.get(t, floor)) for t, l in rt)
                kl[rd["kind"]][0] += s; kl[rd["kind"]][1] += 1
                agree += rt[0][0] == ct[0][0]; tot += 1
        ppl = math.exp(nll / max(n, 1))
        if is_cand:
            allk = sum(v[0] for v in kl.values()) / max(sum(v[1] for v in kl.values()), 1)
            print(f"{name:34} {ppl:7.3f} " + " ".join(f"{kl[k][0] / max(kl[k][1], 1):11.4f}" for k in kinds)
                  + f" {allk:8.4f} {agree / max(tot, 1) * 100:6.2f}%")
        else:
            print(f"{name:34} {ppl:7.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(*sys.argv[2:5])
    else:
        cmp(sys.argv[2], sys.argv[3:])
