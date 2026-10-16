#!/usr/bin/env python3
"""API surface a deployment must survive, each followed by a health check: structured output and
forced tool calls (both killed v0.5.20 + DFlash2 with a CUDA error), ordinary tool calls on the
OpenAI and Anthropic endpoints, the cache report a client reads hits from, and a 1-megapixel image.

    SGLANG_API_KEY=... python3 apitest.py <url> [image.png]
"""
import base64, json, os, sys, time, urllib.error, urllib.request

URL = sys.argv[1]
IMG = sys.argv[2] if len(sys.argv) > 2 else None
KEY, MODEL = os.environ["SGLANG_API_KEY"], "Qwen3.8-27B-abliterated-AWQ-INT4"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "anthropic-version": "2023-06-01"}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), H)
    try:
        return 200, json.loads(urllib.request.urlopen(r, timeout=600).read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def healthy():
    time.sleep(3)
    try:
        return urllib.request.urlopen(urllib.request.Request(URL + "/health"), timeout=15).status == 200
    except Exception:
        return False


def chat(extra, content="What is the weather in Paris?"):
    b = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 400,
         "chat_template_kwargs": {"enable_thinking": False}, **extra}
    return post("/v1/chat/completions", b)


rows = []
code, r = chat({"response_format": {"type": "json_schema", "json_schema": {"name": "o", "strict": True, "schema": {
    "type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}, "required": ["colors"]}}}},
    "List three colours as JSON.")
rows.append(("response_format json_schema", code, (r["choices"][0]["message"]["content"] or "")[:80] if code == 200 else r, healthy()))
code, r = chat({"tools": [TOOL], "tool_choice": "required"})
rows.append(("tool_choice required", code, (r["choices"][0]["message"].get("tool_calls") or "no tool call") if code == 200 else r, healthy()))
code, r = chat({"tools": [TOOL]})
tc = r["choices"][0]["message"].get("tool_calls") if code == 200 else None
rows.append(("tool call, auto (openai)", code, tc[0]["function"] if tc else r, healthy()))
code, r = post("/v1/messages", {"model": MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                                "tools": [{"name": "get_weather", "description": "Get the weather for a city",
                                           "input_schema": TOOL["function"]["parameters"]}]})
tu = [c for c in r.get("content", []) if c.get("type") == "tool_use"] if code == 200 else None
rows.append(("tool call, auto (anthropic)", code, tu[0]["input"] if tu else r, healthy()))
sysmsg = "You are terse. " + "Context line about bridges and rivers. " * 150
for i in range(2):
    code, r = post("/v1/messages", {"model": MODEL, "max_tokens": 5, "system": sysmsg, "messages": [{"role": "user", "content": "Say ok."}]})
rows.append(("cache report (anthropic, 2nd call)", code, r.get("usage") if code == 200 else r, healthy()))
if IMG:
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    code, r = chat({}, [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                        {"type": "text", "text": "Describe this image in one sentence."}])
    rows.append(("1MP image", code, (r["choices"][0]["message"]["content"] or "")[:80] if code == 200 else r, healthy()))
for name, code, what, ok in rows:
    print(f"  {name:36} HTTP {code}  healthy after: {ok}  {str(what)[:110]}")
