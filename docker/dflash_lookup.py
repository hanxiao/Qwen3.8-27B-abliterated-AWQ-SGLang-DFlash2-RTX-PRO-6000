"""Lookup-augmented drafting for DFlash, installed by lookup_drafting.py as
sglang/srt/speculative/dflash_lookup.py. The idea and its thresholds come from HyperQwen's vLLM
patch (github.com/syv-ai/HyperQwen, patches/dflash2-lookup-drafting.patch).

When the tokens just produced already occur earlier in the request's own context, the
continuation is sitting in the context: propose it instead of the drafter's guess. Each request
keeps its token ids in a GPU history (prompt copied at prefill, accepted tokens appended on the
GPU after each verify, so no step waits on the host). Before verification, a grid of programs
finds the most recent occurrence of the longest suffix (NMIN..NMAX tokens) of the history, ending
at the block's anchor token, and the tokens that followed it replace the drafter's positions when
the match is at least NMIN long. A match may overlap the suffix, so a repeating pattern (list
markers, indentation) is proposed from its own period.

The output distribution does not change. Greedy verification compares each drafted token with the
target's argmax. Sampled verification needs the probability q the draft was drawn with: every
position the lookup filled gets a point mass on its token, a legal draft distribution for the
rejection test (accept with probability p(x), residual from p).
"""
import logging
import os
from array import array

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SHIFT = 32  # score = match_len << 32 | end: longest match first, then the most recent


@triton.jit
def _anchor_kernel(hist_ptr, hist_stride, slots_ptr, prefix_ptr, anchor_ptr, best_ptr):
    b = tl.program_id(0)
    slot = tl.load(slots_ptr + b).to(tl.int64)
    p = tl.load(prefix_ptr + b).to(tl.int64)
    tok = tl.load(anchor_ptr + b)
    tl.store(hist_ptr + slot * hist_stride + p, tok.to(tl.int32))
    tl.store(best_ptr + b, -1)


@triton.jit
def _search_kernel(hist_ptr, hist_stride, slots_ptr, prefix_ptr, best_ptr,
                   NMIN: tl.constexpr, NMAX: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    blk = tl.program_id(1)
    slot = tl.load(slots_ptr + b).to(tl.int64)
    t = tl.load(prefix_ptr + b).to(tl.int64)  # the suffix ends at the anchor, position t
    e = (blk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)  # candidate match ends
    alive = (e < t) & (e >= NMIN - 1)
    if tl.max(alive.to(tl.int32), axis=0) == 0:
        return
    base = hist_ptr + slot * hist_stride
    for j in tl.static_range(NMIN):
        s_j = tl.load(base + t - j)
        t_j = tl.load(base + e - j, mask=alive, other=-1)
        alive = alive & (t_j == s_j)
    if tl.max(alive.to(tl.int32), axis=0) == 0:
        return
    match = tl.where(alive, NMIN, 0)
    longer = alive
    for j in tl.static_range(NMIN, NMAX):
        s_j = tl.load(base + t - j, mask=(t - j) >= 0, other=-2)
        ok = longer & ((e - j) >= 0)
        t_j = tl.load(base + e - j, mask=ok, other=-1)
        longer = ok & (t_j == s_j)
        match = tl.where(longer, match + 1, match)
    score = tl.where(alive, (match.to(tl.int64) << 32) + e, -1)
    tl.atomic_max(best_ptr + b, tl.max(score, axis=0))


@triton.jit
def _fuse_kernel(hist_ptr, hist_stride, slots_ptr, prefix_ptr, best_ptr, credit_ptr, taken_ptr,
                 draft_ptr, draft_stride, cand_ptr, cand_s0, cand_s1, q_ptr, q_s0, q_s1,
                 stats_ptr, NMIN: tl.constexpr, NSTRONG: tl.constexpr, AGREE: tl.constexpr,
                 K: tl.constexpr, TOPK: tl.constexpr,
                 HAS_Q: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr):
    b = tl.program_id(0)
    tl.store(taken_ptr + b, -1)
    best = tl.load(best_ptr + b)
    if best < 0:
        return
    match_len = best >> 32
    end = best - (match_len << 32)
    if match_len < NMIN:
        return
    slot = tl.load(slots_ptr + b).to(tl.int64)
    t = tl.load(prefix_ptr + b).to(tl.int64)
    valid = tl.minimum(t - end, K)
    idx = tl.arange(0, BK)
    m = idx < valid
    toks = tl.load(hist_ptr + slot * hist_stride + end + 1 + idx, mask=m, other=0).to(tl.int64)
    # A match shorter than NSTRONG replaces the draft only if the drafter independently proposed
    # the same first AGREE tokens: repeated phrases in JSON or arithmetic match for six tokens
    # and then continue differently, and there the drafter is usually right.
    drafted = tl.load(draft_ptr + b * draft_stride + idx, mask=m, other=-1)
    agree = tl.minimum(tl.min(tl.where((drafted != toks) & m, idx, BK), axis=0), valid)
    if (match_len < NSTRONG) & (agree < AGREE):
        return
    # A request whose recent lookups were rejected by the target stops overriding until its
    # credit recovers (see _commit_kernel).
    if tl.load(credit_ptr + slot) < 0:
        return
    if valid > agree:
        tl.store(taken_ptr + b, agree)
    tl.store(draft_ptr + b * draft_stride + idx, toks, mask=m)
    if HAS_Q:
        kk = tl.arange(0, BT)
        m2 = m[:, None] & (kk[None, :] < TOPK)
        tok2 = toks[:, None] + tl.zeros((BK, BT), dtype=tl.int64)
        tl.store(cand_ptr + b * cand_s0 + idx[:, None] * cand_s1 + kk[None, :], tok2, mask=m2)
        tl.store(q_ptr + b * q_s0 + idx[:, None] * q_s1 + kk[None, :],
                 tl.full((BK, BT), 1.0, tl.float32), mask=m2)
    tl.atomic_add(stats_ptr, 1)
    tl.atomic_add(stats_ptr + 1, valid)


@triton.jit
def _commit_kernel(hist_ptr, hist_stride, slots_ptr, prefix_ptr, out_ptr, out_stride, lens_ptr,
                   credit_ptr, taken_ptr, PENALTY: tl.constexpr, BLK: tl.constexpr):
    b = tl.program_id(0)
    slot = tl.load(slots_ptr + b).to(tl.int64)
    p = tl.load(prefix_ptr + b).to(tl.int64)
    n = tl.load(lens_ptr + b)
    # The lookup earned its override if the target accepted tokens past the prefix the drafter
    # had proposed itself; otherwise it cost the drafter its tokens there.
    taken = tl.load(taken_ptr + b)
    c = tl.load(credit_ptr + slot)
    if taken >= 0:
        c = tl.where(n - 1 > taken, tl.minimum(c + 1, 8), c - PENALTY)
    else:
        c = tl.minimum(c + 1, 0)
    tl.store(credit_ptr + slot, c)
    idx = tl.arange(0, BLK)
    m = idx < n
    tok = tl.load(out_ptr + b * out_stride + idx, mask=m, other=0)
    tl.store(hist_ptr + slot * hist_stride + p + 1 + idx, tok.to(tl.int32), mask=m)


def _as_tensor(ids) -> torch.Tensor:
    if isinstance(ids, array) and ids.typecode == "q":
        return torch.frombuffer(memoryview(ids), dtype=torch.int64) if len(ids) else torch.empty(0, dtype=torch.int64)
    return torch.tensor(list(ids), dtype=torch.int64)


class LookupDrafter:
    """Per-slot token history plus the four kernels. Slots are req_to_token_pool rows."""

    BLOCK = 1024

    def __init__(self, num_slots: int, max_len: int, block_size: int, max_bs: int, device):
        self.nmin = int(os.environ.get("SGLANG_DFLASH_LOOKUP_NMIN", "6"))
        self.nmax = max(self.nmin, int(os.environ.get("SGLANG_DFLASH_LOOKUP_NMAX", "12")))
        # A match of NSTRONG or more is taken on its own; a shorter one needs the drafter to agree
        # on its first AGREE tokens. The default never takes a match without that agreement.
        self.nstrong = int(os.environ.get("SGLANG_DFLASH_LOOKUP_NSTRONG", "1000"))
        self.agree = int(os.environ.get("SGLANG_DFLASH_LOOKUP_AGREE", "1"))
        # Credit lost per rejected override, regained one per step; 0 turns the feedback off.
        self.penalty = int(os.environ.get("SGLANG_DFLASH_LOOKUP_PENALTY", "0"))
        self.block_size = int(block_size)
        self.k = self.block_size - 1
        self.width = int(max_len) + 2 * self.block_size + 1
        self.device = device
        self.hist = torch.zeros((num_slots, self.width), dtype=torch.int32, device=device)
        self.nblk = triton.cdiv(self.width, self.BLOCK)
        self.best = torch.full((max_bs,), -1, dtype=torch.int64, device=device)
        self.slots = torch.zeros((max_bs,), dtype=torch.int64, device=device)
        self.prefix = torch.zeros((max_bs,), dtype=torch.int64, device=device)
        self.stats = torch.zeros((2,), dtype=torch.int64, device=device)  # steps taken, tokens
        self.credit = torch.zeros((num_slots,), dtype=torch.int32, device=device)
        self.taken = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        self.owner = {}
        self.written = {}
        self._self_check()
        logger.info("DFLASH lookup drafting on: nmin=%d nmax=%d nstrong=%d agree=%d penalty=%d "
                    "history %d x %d (%.1f MB)", self.nmin, self.nmax, self.nstrong, self.agree,
                    self.penalty,
                    num_slots, self.width,
                    self.hist.numel() * 4 / 1e6)

    # Prefill: copy the request's tokens [start, extend_end) from its fill ids.
    def on_extend(self, reqs) -> None:
        for req in reqs:
            slot = int(req.kv.req_pool_idx)
            end = int(req.extend_end) if req.extend_end is not None else len(req.full_untruncated_fill_ids)
            end = min(end, self.width - 2 * self.block_size - 1)
            start = 0
            if self.owner.get(slot) == req.rid:
                start = min(self.written.get(slot, 0), int(req.prefix_len))
            if end > start:
                src = _as_tensor(req.full_untruncated_fill_ids)[start:end]
                self.hist[slot, start:end].copy_(src.to(self.device, non_blocking=False).to(torch.int32))
            if self.owner.get(slot) != req.rid:
                self.credit[slot] = 0
            self.owner[slot] = req.rid
            self.written[slot] = end

    # Decode, between the draft and the verify: fuse lookup tokens into draft_next [bs, k].
    def fuse(self, draft_next, anchors, req_pool_indices, prefix_lens, selector_sample) -> None:
        bs = int(draft_next.shape[0])
        if bs == 0:
            return
        assert draft_next.stride(1) == 1 and draft_next.shape[1] == self.k
        self.slots[:bs].copy_(req_pool_indices.view(-1))
        self.prefix[:bs].copy_(prefix_lens.view(-1))
        _anchor_kernel[(bs,)](self.hist, self.hist.stride(0), self.slots, self.prefix,
                              anchors.contiguous(), self.best)
        _search_kernel[(bs, self.nblk)](self.hist, self.hist.stride(0), self.slots, self.prefix,
                                        self.best, NMIN=self.nmin, NMAX=self.nmax,
                                        BLOCK=self.BLOCK, num_warps=4)
        if selector_sample is not None:
            cand, q = selector_sample
            assert cand.stride(2) == 1 and q.stride(2) == 1
            topk = int(cand.shape[2])
            args = (cand, cand.stride(0), cand.stride(1), q, q.stride(0), q.stride(1))
        else:
            topk = 1
            args = (draft_next, 0, 0, draft_next, 0, 0)
        _fuse_kernel[(bs,)](self.hist, self.hist.stride(0), self.slots, self.prefix, self.best,
                            self.credit, self.taken,
                            draft_next, draft_next.stride(0), *args, self.stats,
                            NMIN=self.nmin, NSTRONG=self.nstrong, AGREE=self.agree,
                            K=self.k, TOPK=topk,
                            HAS_Q=selector_sample is not None,
                            BK=triton.next_power_of_2(self.k),
                            BT=triton.next_power_of_2(topk))

    # After the verify: append the committed tokens (accepted drafts + new bonus).
    def on_commit(self, out_tokens, commit_lens) -> None:
        bs = int(out_tokens.shape[0])
        if bs == 0:
            return
        _commit_kernel[(bs,)](self.hist, self.hist.stride(0), self.slots, self.prefix,
                              out_tokens, out_tokens.stride(0), commit_lens,
                              self.credit, self.taken, PENALTY=self.penalty, BLK=triton.next_power_of_2(int(out_tokens.shape[1])))

    def _self_check(self) -> None:
        """Run the search and fuse on a planted history and compare with a Python reference."""
        g = torch.Generator().manual_seed(0)
        n = 5000
        h = torch.randint(10, 1000, (n,), generator=g)
        h[3000:3040] = h[1000:1040]  # a copy: suffix ends at 3039, continuation at 1040..
        t = 3039
        ref_best, ref_end = -1, -1
        hl = h.tolist()
        for e in range(self.nmin - 1, t):
            L = 0
            while L < self.nmax and e - L >= 0 and hl[e - L] == hl[t - L]:
                L += 1
            if L >= self.nmin and (L, e) > (ref_best, ref_end):
                ref_best, ref_end = L, e
        self.hist[0, :n].copy_(h.to(torch.int32))
        want_tokens = hl[ref_end + 1: ref_end + 1 + min(self.k, t - ref_end)]
        # The drafter agrees on the first AGREE tokens and guesses wrong after them.
        draft = torch.zeros((1, self.k), dtype=torch.int64, device=self.device)
        draft[0, :self.agree] = torch.tensor(want_tokens[:self.agree])
        anchors = torch.tensor([hl[t]], dtype=torch.int64, device=self.device)
        self.fuse(draft, anchors, torch.zeros(1, dtype=torch.int64, device=self.device),
                  torch.tensor([t], dtype=torch.int64, device=self.device), None)
        torch.cuda.synchronize()
        best = int(self.best[0])
        got = (best >> _SHIFT, best & 0xFFFFFFFF) if best >= 0 else (-1, -1)
        if got != (ref_best, ref_end) or draft[0, :len(want_tokens)].tolist() != want_tokens:
            raise RuntimeError(f"lookup self-check failed: kernel {got} {draft.tolist()}, "
                               f"reference {(ref_best, ref_end)} {want_tokens}")
        self.hist.zero_()
        self.stats.zero_()
        self.credit.zero_()
