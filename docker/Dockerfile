# Stock SGLang v0.5.20 with one file replaced: the qwen3_coder tool-call parser, which in v0.5.20
# holds every parameter until its </parameter> tag. qwen3_coder_detector.py is sglang PR #25246
# ported to v0.5.20 (see the README, "Tool calls"): string parameters stream as they are
# generated, everything else is converted as before.
#
# The build stops unless the stock parser fails test_stream.py (else the test proves nothing),
# the patched parser passes it, and the patched parser passes SGLang's own Qwen3 Coder tests.
FROM lmsysorg/sglang:v0.5.20
COPY qwen3_coder_detector.py test_stream.py /tmp/patch/
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
    rm -rf /tmp/patch; \
    echo "patched $T"
