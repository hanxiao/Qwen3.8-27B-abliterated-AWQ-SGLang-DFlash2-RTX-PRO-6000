"""Give DFlash's draft candidates a quantized copy of the target lm_head, applied to an installed
SGLang v0.5.22 (see dflash_qhead.py for why this keeps the output distribution).

Off unless SGLANG_DFLASH_DRAFT_HEAD is int8 or int4. The verify pass keeps the BF16 head.

    python3 draft_head_quant.py   # edits the installed package; fails if v0.5.22's text moved
"""
import os
import pathlib
import shutil

root = os.environ.get("SGLANG_PKG")
if not root:
    import sglang

    root = pathlib.Path(sglang.__file__).parent
root = pathlib.Path(root)
here = pathlib.Path(__file__).resolve().parent


def edit(rel, pairs):
    p = root / rel
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, f"{rel}: expected 1x, found {s.count(old)}x: {old!r}"
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", p)


shutil.copy(here / "dflash_qhead.py", root / "srt/speculative/dflash_qhead.py")
print("installed", root / "srt/speculative/dflash_qhead.py")

edit("srt/models/dflash.py", [(
    """        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        ids, vals, _ = candidate_topk(""",
    """        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        qhead = getattr(self, "dflash_quant_head", None)
        if qhead is not None:
            vals, ids = _radix_topk(qhead(hidden), self.candidate_selector.top_k)
            return ids.long(), self._transform_unary_logits(vals)
        ids, vals, _ = candidate_topk(""")])

edit("srt/speculative/dflash_worker_v2.py", [(
    """            self.draft_model.lm_head = lm_head
            if self._target_tp_rank == 0:
                logger.info(
                    "DFLASH selector decode folded into the draft cuda graph \"""",
    """            self.draft_model.lm_head = lm_head
            _qmode = os.environ.get("SGLANG_DFLASH_DRAFT_HEAD", "").strip().lower()
            if _qmode:
                if get_parallel().tp_size != 1:
                    raise RuntimeError("SGLANG_DFLASH_DRAFT_HEAD needs tp_size 1")
                from sglang.srt.speculative.dflash_qhead import build_quant_draft_head

                self.draft_model.dflash_quant_head = build_quant_draft_head(lm_head, _qmode)
            if self._target_tp_rank == 0:
                logger.info(
                    "DFLASH selector decode folded into the draft cuda graph \"""")])
