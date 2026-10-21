#!/usr/bin/env python3
"""Do a tool call's arguments arrive while they are generated, or all at once at the end?

Asks for one `write` call whose content is the integers 1 to N, streamed, on the OpenAI and the
Anthropic endpoint. Reports how many pieces the arguments arrived in, the longest silence inside
the call, the time of the call, and whether the assembled arguments parse to exactly that content.
A client with an idle timeout shorter than the longest silence drops the request.

    SGLANG_API_KEY=... python3 toolstream.py <url> [N]     # default N=3000
"""
import json, os, sys, time, urllib.request

URL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
KEY, MODEL = os.environ["SGLANG_API_KEY"], "Qwen3.8-27B-abliterated-AWQ-INT4"
ASK = (f"Call the write tool once: path numbers.txt, content = the integers 1 to {N}, each on its "
       "own line, every one written out. No other text.")
SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
          "required": ["path", "content"]}
WANT = "\n".join(str(i) for i in range(1, N + 1))


def stream(path, body, headers):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY, **headers})
    t0, pieces, args = time.time(), [], ""
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            got = ""
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "input_json_delta":
                got = ev["delta"].get("partial_json") or ""
            for ch in ev.get("choices") or []:
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    got += (tc.get("function") or {}).get("arguments") or ""
            if got:
                pieces.append(time.time() - t0)
                args += got
    return pieces, args


def report(name, pieces, args):
    try:
        content = json.loads(args).get("content", "").rstrip("\n")
        ok = "exact" if content == WANT else f"differs ({len(content)} vs {len(WANT)} chars)"
    except ValueError:
        ok = "arguments are not valid JSON"
    gap = max((b - a for a, b in zip(pieces, pieces[1:])), default=0.0)
    print(f"  {name:28s} {len(pieces):5d} pieces   longest silence {gap:6.1f} s   "
          f"call {pieces[-1] - pieces[0] if pieces else 0:6.1f} s   content {ok}")


print(f"{URL}  N={N}")
report("anthropic /v1/messages", *stream("/v1/messages", {
    "model": MODEL, "max_tokens": 32000, "stream": True, "thinking": {"type": "disabled"},
    "tools": [{"name": "write", "description": "Write a file", "input_schema": SCHEMA}],
    "messages": [{"role": "user", "content": ASK}]}, {"anthropic-version": "2023-06-01"}))
report("openai /v1/chat/completions", *stream("/v1/chat/completions", {
    "model": MODEL, "max_tokens": 32000, "stream": True, "chat_template_kwargs": {"enable_thinking": False},
    "tools": [{"type": "function", "function": {"name": "write", "description": "Write a file", "parameters": SCHEMA}}],
    "messages": [{"role": "user", "content": ASK}]}, {}))
