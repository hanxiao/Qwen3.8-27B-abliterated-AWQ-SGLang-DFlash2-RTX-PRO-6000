"""Streaming of Qwen3 Coder tool-call arguments, checked inside the serving image.

    python3 test_stream.py [path/to/qwen3_coder_detector.py]

Exit 0 when string arguments stream before their closing tag and every streamed call equals
the non-streaming parse of the same text; exit 1 otherwise.
"""
import importlib.util
import json
import sys

from sglang.srt.entrypoints.openai.protocol import Function, Tool

if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("qcd_under_test", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    import sglang.srt.function_call.qwen3_coder_detector as mod
Detector = mod.Qwen3CoderDetector

TOOLS = [
    Tool(type="function", function=Function(name="write", parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                       "overwrite": {"type": "boolean"}, "mode": {"type": "integer"},
                       "meta": {"type": "object"}},
        "required": ["path", "content"]})),
    Tool(type="function", function=Function(name="bash", parameters={
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})),
    Tool(type="function", function=Function(name="edit", parameters={
        "type": "object",
        "properties": {"note": {"type": ["string", "null"]}, "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                       "free": {"description": "no type"}, "count": {"type": ["integer", "null"]}}})),
]
FAILED = 0


def check(name, ok, detail=""):
    global FAILED
    FAILED += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {detail}"), flush=True)


def stream(text, chunk):
    d = Detector()
    per_tool, names = {}, {}
    for i in range(0, len(text), chunk):
        piece = text[i:i + chunk]
        r = d.parse_streaming_increment(piece, TOOLS)
        for c in r.calls:
            if c.name:
                names[c.tool_index] = c.name
            if c.parameters:
                per_tool[c.tool_index] = per_tool.get(c.tool_index, "") + c.parameters
    return names, per_tool


def whole(text):
    r = Detector().detect_and_parse(text, TOOLS)
    return [(c.name, json.loads(c.parameters)) for c in r.calls]


BODY = 'line "one" \\ back\n<div class="x">中文 é</div>\n  indented </par not a close\n' * 40
CASES = {
    "write with long content": f"<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.html\n</parameter>\n<parameter=content>\n{BODY}\n</parameter>\n</function>\n</tool_call>",
    "typed params kept typed": f"<tool_call>\n<function=write>\n<parameter=path>\nx.txt\n</parameter>\n<parameter=mode>\n420\n</parameter>\n<parameter=overwrite>\ntrue\n</parameter>\n<parameter=meta>\n{{\"a\": [1, 2]}}\n</parameter>\n<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>",
    "two calls in a row": "<tool_call>\n<function=bash>\n<parameter=command>\nls -la | grep \"x\"\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write>\n<parameter=path>\nb.txt\n</parameter>\n<parameter=content>\nsecond\n</parameter>\n</function>\n</tool_call>",
    "empty string value": "<tool_call>\n<function=write>\n<parameter=path>\nc.txt\n</parameter>\n<parameter=content>\n\n</parameter>\n</function>\n</tool_call>",
    "null string values": "<tool_call>\n<function=edit>\n<parameter=note>\nnull\n</parameter>\n<parameter=tag>\nNULL\n</parameter>\n<parameter=free>\nnullable\n</parameter>\n<parameter=count>\nnull\n</parameter>\n</function>\n</tool_call>",
    "type lists, anyOf, untyped, unknown": "<tool_call>\n<function=edit>\n<parameter=note>\na \"b\"\n</parameter>\n<parameter=tag>\nt1\n</parameter>\n<parameter=free>\n{\"x\": 1}\n</parameter>\n<parameter=count>\n7\n</parameter>\n<parameter=extra>\nnot in schema\n</parameter>\n</function>\n</tool_call>",
    "blank lines around the value": "<tool_call>\n<function=write>\n<parameter=path>\nd.txt\n</parameter>\n<parameter=content>\n\n\nbody\n\n\n</parameter>\n</function>\n</tool_call>",
    "tool not in the list": "<tool_call>\n<function=unknown_tool>\n<parameter=q>\nhello\n</parameter>\n</function>\n</tool_call>",
    # sglang#43150: the model closes </tool_call> but omits </function> (and sometimes </parameter>)
    "no </function>, typed last": "<tool_call>\n<function=write>\n<parameter=path>\ne.txt\n</parameter>\n<parameter=mode>\n20\n</parameter>\n</tool_call>",
    "no </function>, string last": "<tool_call>\n<function=write>\n<parameter=path>\nf.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</tool_call>",
    "no </function> or </parameter>, typed": "<tool_call>\n<function=write>\n<parameter=path>\ng.txt\n</parameter>\n<parameter=mode>\n20\n</tool_call>",
    "no </function> or </parameter>, string": "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</tool_call>",
    # seen live on Qwen3.8-27B: neither </function> nor </tool_call>, the next call starts right away
    "no </function> or </tool_call>, then <tool_call>": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<tool_call>\n<function=write>\n<parameter=path>\ni.txt\n</parameter>\n<parameter=mode>\n20\n</parameter>\n</function>\n</tool_call>",
    "no </function> or </tool_call>, then <function=": "<tool_call>\n<function=write>\n<parameter=path>\nj.txt\n</parameter>\n<parameter=mode>\n20\n</parameter>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>",
    "no </function> then a second call": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</tool_call>\n<tool_call>\n<function=write>\n<parameter=path>\nh.txt\n</parameter>\n<parameter=content>\nsecond\n</parameter>\n</function>\n</tool_call>",
}

print("the content streams before its closing tag")
text = CASES["write with long content"]
d = Detector()
got = ""
close_at = text.index("</parameter>", text.index("<parameter=content>"))
for i in range(0, close_at, 7):
    for c in d.parse_streaming_increment(text[i:min(i + 7, close_at)], TOOLS).calls:
        got += c.parameters or ""
check("content fragments emitted before </parameter>", '"content": "' in got and len(got) > 2000,
      f"{len(got)} chars emitted before the close")

print("streamed equals non-streamed, at several chunk sizes")
for name, text in CASES.items():
    ref = whole(text)
    check(f"{name}, whole parse is not empty", bool(ref), ref)
    for chunk in (1, 2, 3, 5, 17, 4096):
        names, per_tool = stream(text, chunk)
        try:
            out = [(names[i], json.loads(per_tool[i])) for i in sorted(per_tool)]
        except Exception as e:  # noqa: BLE001
            out = f"invalid JSON: {e}"
        check(f"{name}, chunk {chunk}", out == ref, f"\n    streamed={str(out)[:300]}\n    whole={str(ref)[:300]}")

print("no whitespace-only text rides with call items (on /v1/messages it would close the tool_use block)")
for name in ("two calls in a row", "no </function> then a second call"):
    text = CASES[name]
    for chunk in (1, 3, 17, 4096):
        d = Detector()
        mixed = []
        for i in range(0, len(text), chunk):
            r = d.parse_streaming_increment(text[i:i + chunk], TOOLS)
            if r.calls and r.normal_text and not r.normal_text.strip():
                mixed.append(r.normal_text)
        check(f"{name}, chunk {chunk}", not mixed, mixed)

print(f"\n  {FAILED} failure(s)")
sys.exit(1 if FAILED else 0)
