"""Count a request's held req_to_token row once at admission (sglang PR #44401, merged after
v0.5.22), applied to an installed SGLang v0.5.22: the main admission gate only.

A continuing chunked prefill is in the batch being built and already holds its row, which
`req_to_token_pool.available_size()` has excluded; the gate counted it again and declared the batch
full with a row still free. With two slots that capped a long prefill's batch at one request, and
the `batch_is_full` it set stayed until a request finished: a second request arriving during a
prefill longer than one chunk (4,096 tokens here) waited for the first to finish decoding.

    python3 sched_admission.py   # edits the installed package; fails if v0.5.22's text moved
"""
import pathlib

import sglang

P = pathlib.Path(sglang.__file__).parent / "srt/managers/scheduler.py"
s = P.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, f"expected once, found {s.count(old)}x: {old[:80]!r}"
    s = s.replace(old, new)


sub("""        running_batch: Optional[ScheduleBatch] = None,
    ) -> int:
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        available = self.req_to_token_pool.available_size()
""", """        running_batch: Optional[ScheduleBatch] = None,
        num_held_rows: int = 0,
    ) -> int:
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        # Requests the caller counts that already hold a row (a continuing chunk) need no free
        # one; give their rows back (sglang #44401).
        available = self.req_to_token_pool.available_size() + num_held_rows
""")
sub("""            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
            ):
                running_batch.batch_is_full = True
""", """            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
                num_held_rows=num_held_rows,
            ):
                running_batch.batch_is_full = True
""")
P.write_text(s)
print("patched", P)
