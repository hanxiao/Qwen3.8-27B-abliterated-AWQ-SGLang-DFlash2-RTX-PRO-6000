"""A quantized copy of the target lm_head for DFlash's draft candidates, installed by
draft_head_quant.py as sglang/srt/speculative/dflash_qhead.py.

DFlash 2 scores every draft position against the whole vocabulary through the target's BF16
lm_head, 2.54 GB read once per step on top of the 2.54 GB the verify pass reads. The draft only
proposes: the target still verifies each token with its own BF16 head, and the sampled verify
uses the q the selector drew with. So a lower-precision copy for the draft changes which tokens
get proposed, never what the output distribution is.

int8 keeps one scale per vocabulary row (1.27 GB); int4 one per row and 128 columns (0.64 GB).
The kernel reads the weight once and streams it (evict_first), for the few rows a draft block
has: M = 7 per request. On half an RTX PRO 6000 at M = 7 the BF16 matmul takes 3.35 ms, int8
1.68 ms (both at the card's ~760 GB/s) and int4 1.23 ms; the block sizes below are the fastest
of a sweep there.
"""
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _wq8_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, som,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    om = tl.arange(0, BM)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptr + om[:, None] * sxm + (k0 + ok)[None, :], mask=om[:, None] < M, other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + (k0 + ok)[None, :], mask=on[:, None] < N, other=0,
                    eviction_policy="evict_first")
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + on, mask=on < N, other=0.0)
    acc = acc * s[None, :]
    tl.store(o_ptr + om[:, None] * som + on[None, :], acc.to(tl.bfloat16),
             mask=(om[:, None] < M) & (on[None, :] < N))


@triton.jit
def _wq4_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, ssn, som,
                BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    # w: uint8 [N, K/2], low nibble = even column; s: fp32 [N, K/G]; one group per K step.
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    om = tl.arange(0, BM)
    ok = tl.arange(0, G)
    oh = tl.arange(0, G // 2)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, K // G):
        x = tl.load(x_ptr + om[:, None] * sxm + (g * G + ok)[None, :], mask=om[:, None] < M, other=0.0)
        p = tl.load(w_ptr + on[:, None] * swn + (g * (G // 2) + oh)[None, :], mask=on[:, None] < N,
                    other=0, eviction_policy="evict_first")
        lo = (p & 0xF).to(tl.int8) - 8
        hi = (p >> 4).to(tl.int8) - 8
        w = tl.join(lo, hi).reshape(BN, G)
        part = tl.dot(x, tl.trans(w.to(tl.bfloat16)))
        s = tl.load(s_ptr + on * ssn + g, mask=on < N, other=0.0)
        acc += part * s[None, :]
    tl.store(o_ptr + om[:, None] * som + on[None, :], acc.to(tl.bfloat16),
             mask=(om[:, None] < M) & (on[None, :] < N))


class QuantDraftHead:
    """Callable hidden [M, K] -> logits [M, N] in BF16. A plain object, not an nn.Module, so it
    is not registered under the draft model and nothing iterates its tensors as parameters."""

    def __init__(self, weight: torch.Tensor, num_org: int, mode: str, chunk: int = 8192):
        w = weight[:num_org]
        self.N, self.K = int(w.shape[0]), int(w.shape[1])
        self.mode = mode
        dev = w.device
        cfg = os.environ.get("SGLANG_DFLASH_DRAFT_HEAD_CFG", "").strip()
        if mode == "int8":
            self.q = torch.empty((self.N, self.K), dtype=torch.int8, device=dev)
            self.s = torch.empty((self.N,), dtype=torch.float32, device=dev)
            for a in range(0, self.N, chunk):
                b = min(a + chunk, self.N)
                f = w[a:b].float()
                s = f.abs().amax(1).clamp(min=1e-12) / 127.0
                self.q[a:b] = torch.round(f / s[:, None]).clamp(-127, 127).to(torch.int8)
                self.s[a:b] = s
            self.BN, self.BK, self.warps, self.stages = (
                tuple(int(v) for v in cfg.split(",")) if cfg else (128, 256, 8, 3))
        elif mode == "int4":
            self.G = 128
            assert self.K % self.G == 0
            self.q = torch.empty((self.N, self.K // 2), dtype=torch.uint8, device=dev)
            self.s = torch.empty((self.N, self.K // self.G), dtype=torch.float32, device=dev)
            for a in range(0, self.N, chunk):
                b = min(a + chunk, self.N)
                f = w[a:b].float().view(b - a, self.K // self.G, self.G)
                s = f.abs().amax(-1).clamp(min=1e-12) / 7.0
                q = (torch.round(f / s[..., None]).clamp(-8, 7) + 8).to(torch.uint8).view(b - a, self.K)
                self.q[a:b] = q[:, 0::2] | (q[:, 1::2] << 4)
                self.s[a:b] = s
            self.BN, self.warps, self.stages = (
                tuple(int(v) for v in cfg.split(",")) if cfg else (64, 4, 6))
        else:
            raise ValueError(f"SGLANG_DFLASH_DRAFT_HEAD={mode!r}: expected int8 or int4")

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        M = x.shape[0]
        out = torch.empty((M, self.N), dtype=torch.bfloat16, device=x.device)
        BM = 16 if M <= 16 else 32 if M <= 32 else 64
        if self.mode == "int8":
            _wq8_kernel[(triton.cdiv(self.N, self.BN),)](
                x, self.q, self.s, out, M, self.N, self.K, x.stride(0), self.q.stride(0),
                out.stride(0), BM=BM, BN=self.BN, BK=self.BK, num_warps=self.warps,
                num_stages=self.stages)
        else:
            _wq4_kernel[(triton.cdiv(self.N, self.BN),)](
                x, self.q, self.s, out, M, self.N, self.K, x.stride(0), self.q.stride(0),
                self.s.stride(0), out.stride(0), BM=BM, BN=self.BN, G=self.G,
                num_warps=self.warps, num_stages=self.stages)
        return out


def build_quant_draft_head(lm_head, mode: str, warm_rows=(1, 7, 14)):
    num_org = int(getattr(lm_head, "org_vocab_size", lm_head.weight.shape[0]))
    head = QuantDraftHead(lm_head.weight, num_org, mode)
    # Compile before any CUDA graph capture reaches it, and check it against the BF16 head.
    for m in warm_rows:
        x = torch.randn((m, head.K), dtype=torch.bfloat16, device=lm_head.weight.device)
        got = head(x).float()
        ref = torch.matmul(x, lm_head.weight[:num_org].T).float()
        err = ((got - ref).norm() / ref.norm()).item()
        if not err < (0.05 if mode == "int8" else 0.2):
            raise RuntimeError(f"quantized draft head relative error {err:.4f} at M={m}")
    torch.cuda.synchronize()
    logger.info("DFLASH quantized draft head: %s, %d x %d, %.2f GB, rel err %.4f",
                mode, head.N, head.K, (head.q.numel() * head.q.element_size()
                                       + head.s.numel() * 4) / 1e9, err)
    return head
