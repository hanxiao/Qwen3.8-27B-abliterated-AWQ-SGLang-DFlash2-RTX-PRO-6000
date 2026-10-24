# Stock SGLang v0.5.22 with five changes, each a script that fails the build if v0.5.22's text
# moved (see the README), and Granian for --enable-http2:
#   qwen3_coder_detector.py  streaming tool-call arguments (sglang PR #25246, ported), and closes
#                            the arguments at </tool_call> when the model omits </function>
#                            (issue #43150, fix of PR #43467 plus the held-terminator case), or
#                            at the next <tool_call>/<function= when it omits both closing tags
#   replayssm_dflash.py      DFlash commits through ReplaySSM on Gated DeltaNet (PR #35725, ported)
#   draft_prob_round.py      sampled verify accepts draft probabilities that round above 1 (PR #45819)
#   sched_admission.py       admission counts a prefilling request's row once (PR #44401)
#   draft_head_quant.py      draft candidates from an int8 copy of lm_head (SGLANG_DFLASH_DRAFT_HEAD=int8)
#
# The parser step stops unless the stock parser fails test_stream.py (else the test proves
# nothing), the patched parser passes it, and the patched parser passes SGLang's own Qwen3 Coder
# tests. The last step prints the patched files' hashes, to match a build against a tested image.
FROM lmsysorg/sglang:v0.5.22
COPY qwen3_coder_detector.py test_stream.py replayssm_dflash.py draft_prob_round.py sched_admission.py \
     dflash_qhead.py draft_head_quant.py anthropic_reasoning_history.py /tmp/patch/
RUN set -e; \
    T="$(python3 -c 'import sglang.srt.function_call.qwen3_coder_detector as m; print(m.__file__)')"; \
    S="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).parents[2])')"; \
    cp "$T" /tmp/patch/qwen3_coder_detector.orig.py; \
    if python3 /tmp/patch/test_stream.py /tmp/patch/qwen3_coder_detector.orig.py; then \
        echo "the stock parser passed test_stream.py: it proves nothing on this version"; exit 1; fi; \
    cp /tmp/patch/qwen3_coder_detector.py "$T"; \
    python3 /tmp/patch/test_stream.py; \
    cd "$S/test/registered/unit/function_call"; \
    python3 -m unittest -v test_function_call_parser.TestQwen3CoderDetector; \
    python3 /tmp/patch/replayssm_dflash.py; \
    python3 /tmp/patch/draft_prob_round.py; \
    python3 /tmp/patch/sched_admission.py; \
    python3 /tmp/patch/draft_head_quant.py; \
    python3 /tmp/patch/anthropic_reasoning_history.py; \
    rm -rf /tmp/patch; \
    P="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).parent)')"; \
    cd "$P" && sha256sum srt/function_call/qwen3_coder_detector.py \
        srt/layers/attention/hybrid_linear_attn_backend.py srt/mem_cache/kv_cache_configurator.py \
        kernels/ops/speculative/reject_sampling.py srt/managers/scheduler.py \
        srt/models/dflash.py srt/speculative/dflash_worker_v2.py srt/speculative/dflash_qhead.py \
        && pip install --no-cache-dir granian==2.8.4 && python3 -c 'import granian; print("granian", granian.__version__)'
