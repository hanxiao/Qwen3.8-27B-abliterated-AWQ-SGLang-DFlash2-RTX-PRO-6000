"""Let DFlash commit through ReplaySSM on a Gated DeltaNet target (the part of sglang PR #35725
that a fixed-block DFlash draft needs), applied to an installed SGLang v0.5.22.

ReplaySSM keeps compact records of recurrence inputs instead of one intermediate recurrent state
per draft position, and replays the accepted prefix into the checkpoint after verification. In
v0.5.22 a Gated DeltaNet target uses compact replay (KDA uses the fold): verify writes the records
whatever the drafter, and the generic (MTP) commit replays them; but DFlash commits through
HybridLinearAttnBackend.update_mamba_state_after_mtp_verify, which handles only KDA, so a
configurator guard refuses --enable-linear-replayssm-spec for DFlash on any other model. This
gives that commit the generic path's compact replay and lifts the guard for Gated DeltaNet models.
(The fold that PR #35725 routes to is what v0.5.22 runs for KDA only; a first port that called it
fell through to the intermediate-state scatter and crashed on the missing cache.)

    python3 replayssm_dflash.py   # edits the installed package; fails if v0.5.22's text moved
"""
import pathlib

import sglang

ROOT = pathlib.Path(sglang.__file__).parent


def patch(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    assert s.count(old) == 1, f"{rel}: expected text not found once ({s.count(old)} times)"
    p.write_text(s.replace(old, new))
    print("patched", rel)


patch(
    "srt/layers/attention/hybrid_linear_attn_backend.py",
    """        mamba_pool = req_pool.mamba_pool
        if getattr(mamba_pool, "replayssm_is_kda", False):
            from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
                commit_kda_replayssm_after_verify,
            )
""",
    """        mamba_pool = req_pool.mamba_pool
        # ReplaySSM-GDN (sglang #35725, on v0.5.22's compact replay): the commit the generic MTP
        # path makes (spec_utils.commit_mamba_states_after_verify), with DFlash's physical slot
        # ids for the state and conv kernels and the request ids for the replay cursors.
        if (
            getattr(mamba_pool, "replayssm_cache_base", None) is not None
            and not getattr(mamba_pool, "replayssm_is_kda", False)
        ):
            if request_number == 0:
                return
            from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
                commit_gdn_replayssm_circular,
                commit_gdn_replayssm_spec,
            )
            from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
                fused_conv_window_scatter_with_mask,
            )

            assert req_pool_indices is not None, "ReplaySSM-GDN commit needs req_pool_indices"
            accept_lens = last_correct_step_indices + 1
            replay_indices = req_pool_indices[:request_number]
            commit_gdn_replayssm_spec(
                write_pos=mamba_pool.replayssm_spec_write_pos,
                cache_base=mamba_pool.replayssm_cache_base,
                is_flush=mamba_pool.replayssm_is_flush,
                num_accepted=accept_lens,
                replay_indices=replay_indices,
                max_cache_len=mamba_caches.replayssm_d.shape[-2],
                max_spec_len=get_spec().speculative_num_draft_tokens,
                fold_every_commit=mamba_caches.temporal.dtype != torch.float32,
                null_block_id=-1,
            )
            commit_gdn_replayssm_circular(
                checkpoint_state=mamba_caches.temporal,
                d_cache=mamba_caches.replayssm_d,
                k_cache=mamba_caches.replayssm_k,
                g_cache=mamba_caches.replayssm_g,
                d_residual_cache=mamba_caches.replayssm_rawv,
                k_residual_cache=mamba_caches.replayssm_rawk,
                state_batch_indices=state_indices_tensor,
                replay_indices=replay_indices,
                write_pos=mamba_pool.replayssm_spec_write_pos,
                cache_base=mamba_pool.replayssm_cache_base,
                is_flush=mamba_pool.replayssm_is_flush,
                accept_lens=accept_lens,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                null_block_id=-1,
            )
            fused_conv_window_scatter_with_mask(
                mamba_caches.conv[0],
                mamba_caches.intermediate_conv_window[0],
                state_indices_tensor,
                last_correct_step_indices,
                src_indices_raw,
            )
            if mamba_track_indices is not None:
                fused_conv_window_scatter_with_mask(
                    mamba_caches.conv[0],
                    mamba_caches.intermediate_conv_window[0],
                    mamba_track_indices,
                    mamba_steps_to_track,
                    src_indices_raw,
                )
            return
        if getattr(mamba_pool, "replayssm_is_kda", False):
            from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
                commit_kda_replayssm_after_verify,
            )
""",
)

patch(
    "srt/mem_cache/kv_cache_configurator.py",
    """            and _algo in ("DSPARK", "DFLASH")
            and self.hybrid_kda_config is None
        ):
            raise ValueError(
                "--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA "
                "model; got a non-KDA model."
            )""",
    """            and _algo in ("DSPARK", "DFLASH")
            and self.hybrid_kda_config is None
            and self.hybrid_gdn_config is None
        ):
            raise ValueError(
                "--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA "
                "or GDN model."
            )""",
)
