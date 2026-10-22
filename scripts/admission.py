#!/usr/bin/env python3
"""Does a second request get its slot while the first is still prefilling a long prompt?

With two slots, a request that arrives during another's multi-chunk prefill should start as soon
as there is a free slot. sglang #44401: admission counted the prefilling request twice, set
batch_is_full, and nothing cleared it until a request finished - so the newcomer waited for the
first request's whole answer. Here: a ~40k-token uncached prompt with a 400-token answer, and two
seconds later a one-line question. Reports when each finished; with the bug the short answer comes
after the long one.

    SGLANG_API_KEY=... python3 admission.py <url>
"""
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill  # noqa: E402  (its words(); its benchmark runs only as a script)

URL, KEY = sys.argv[1], os.environ["SGLANG_API_KEY"]
MODEL = "Qwen3.8-27B-abliterated-AWQ-INT4"
done = {}


def ask(name, prompt, max_tokens, t0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    urllib.request.urlopen(r, timeout=1800).read()
    done[name] = time.time() - t0


t0 = time.time()
long_prompt = prefill.words(15000, time.time_ns()) + "\nIgnore the words above and write a 400-word story."
a = threading.Thread(target=ask, args=("long", long_prompt, 400, t0))
a.start()
time.sleep(2)
b = threading.Thread(target=ask, args=("short", "Reply with one word: hello.", 8, t0))
b.start()
a.join()
b.join()
print(f"{URL}: long request done at {done['long']:.1f} s, short request done at {done['short']:.1f} s"
      f" ({'waited for the long answer' if done['short'] > done['long'] - 1 else 'served alongside'})")
