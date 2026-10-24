"""Pass prior-turn thinking on /v1/messages to the chat template as reasoning_content,
applied to an installed SGLang v0.5.22.

Stock v0.5.22 passes assistant thinking blocks natively only for custom chat encoders (dsv4, kimi,
...). For a Jinja template it splices them into the content as "<think>...\n</think>", so the
Qwen3.8 template renders an empty think block followed by the old reasoning inside the visible
content. The re-rendered history then differs from the tokens the model generated right after the
prompt, the prefix cache stops at the end of the previous prompt, and every agent turn re-prefills
the whole previous answer. The Qwen3.8 template reads message.reasoning_content itself, so we hand
it over natively (SGLANG_ANTHROPIC_NATIVE_REASONING_HISTORY=0 restores the stock behaviour).

    python3 anthropic_reasoning_history.py   # edits the installed package; fails if the text moved
"""
import pathlib

import sglang

P = pathlib.Path(sglang.__file__).parent / "srt/entrypoints/anthropic/serving.py"
s = P.read_text()
old = "            if self.openai_serving_chat.supports_native_reasoning_history():\n                return reasoning_text, None\n"
new = ("            if self.openai_serving_chat.supports_native_reasoning_history() or (\n"
       "                __import__(\"os\").environ.get(\"SGLANG_ANTHROPIC_NATIVE_REASONING_HISTORY\", \"1\") == \"1\"\n"
       "            ):\n                return reasoning_text, None\n")
assert s.count(old) == 1, "anthropic serving.py text moved"
s = s.replace(old, new)
P.write_text(s)
print("patched", P)
