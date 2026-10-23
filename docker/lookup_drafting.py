"""Hook lookup-augmented drafting (dflash_lookup.py) into SGLang v0.5.22's DFlash worker.

Off unless SGLANG_DFLASH_LOOKUP=1. Three hooks in dflash_worker_v2.py: copy the prompt into the
token history after the prefill, fuse lookup tokens into the drafted block before the verify, and
append the committed tokens after it.

    python3 lookup_drafting.py   # edits the installed package; fails if v0.5.22's text moved
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

shutil.copy(here / "dflash_lookup.py", root / "srt/speculative/dflash_lookup.py")
print("installed", root / "srt/speculative/dflash_lookup.py")

p = root / "srt/speculative/dflash_worker_v2.py"
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, f"expected 1x, found {s.count(old)}x: {old!r}"
    s = s.replace(old, new)


# A lazily built drafter: the history is sized from the req_to_token pool, which exists by the
# first forward. Defined as a method so the worker's __getattr__ delegation is never consulted.
sub("""    def clear_cache_pool(self):""",
    """    def _lookup_drafter(self):
        lk = self.__dict__.get("_lookup_drafter_obj", False)
        if lk is False:
            lk = None
            if os.environ.get("SGLANG_DFLASH_LOOKUP", "") == "1":
                if get_parallel().tp_size != 1:
                    raise RuntimeError("SGLANG_DFLASH_LOOKUP needs tp_size 1")
                from sglang.srt.speculative.dflash_lookup import LookupDrafter

                r2t = self.model_runner.req_to_token_pool.req_to_token
                lk = LookupDrafter(
                    num_slots=int(r2t.shape[0]),
                    max_len=int(r2t.shape[1]),
                    block_size=int(self.block_size),
                    max_bs=max(get_exec().graph.cuda_graph_config.decode.bs) + 8,
                    device=self.device,
                )
            self.__dict__["_lookup_drafter_obj"] = lk
        return lk

    def clear_cache_pool(self):""")

sub("""            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(""",
    """            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            _lk = self._lookup_drafter()
            if _lk is not None:
                _lk.on_extend(batch.reqs)

            batch_output.next_draft_input = self._make_next_draft_input_prefill(""")

sub("""        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
""",
    """        _lk = self._lookup_drafter()
        if _lk is not None:
            _lk.fuse(
                draft_next,
                block_ids[:, 0],
                batch.req_pool_indices,
                prefix_lens,
                self._selector_sample,
            )

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
""")

sub("""            prefix_lens=prefix_lens,
            bs=bs,
        )

        sampling_mask_capture = """,
    """            prefix_lens=prefix_lens,
            bs=bs,
        )
        if _lk is not None:
            _lk.on_commit(out_tokens, commit_lens)

        sampling_mask_capture = """)

p.write_text(s)
print("patched", p)
