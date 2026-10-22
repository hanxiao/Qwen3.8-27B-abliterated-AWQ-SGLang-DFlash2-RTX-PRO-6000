"""Does a client that hangs up free its slot? Streams a long answer, closes the connection after
a few seconds, and polls the server's running-request gauge until it reaches 0.

    python3 scripts/disconnect.py URL   # SGLANG_API_KEY in the environment

Behind Cloud Run over HTTP/1.1 the request kept running until it would have finished anyway;
an end-to-end HTTP/2 connection (Cloud Run use-http2 + SGLang --enable-http2) carries the
client's reset to the server as a stream cancel.
"""
import http.client, json, os, re, sys, time, urllib.parse, urllib.request

U = sys.argv[1].rstrip("/")
KEY = os.environ["SGLANG_API_KEY"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def running():
    m = urllib.request.urlopen(urllib.request.Request(U + "/metrics", headers=H), timeout=15).read().decode()
    return sum(float(v) for v in re.findall(r"^sglang:num_running_reqs\{[^}]*\} ([0-9.e+-]+)$", m, re.M))


u = urllib.parse.urlparse(U)
c = http.client.HTTPSConnection(u.hostname, timeout=60)
body = {"model": "x", "stream": True, "max_tokens": 6000, "temperature": 0,
        "messages": [{"role": "user", "content": "Count from 1 to 3000, one number per line."}],
        "chat_template_kwargs": {"enable_thinking": False}}
c.request("POST", "/v1/chat/completions", json.dumps(body), H)
r = c.getresponse()
t0 = time.time(); n = 0
while time.time() - t0 < 3:
    if r.readline():
        n += 1
print(f"read {n} chunks in 3 s; running before hang-up: {running():.0f}")
c.sock.close(); c.close()
t1 = time.time()
for _ in range(120):
    v = running()
    if v == 0:
        print(f"slot freed {time.time() - t1:.1f} s after the hang-up")
        break
    time.sleep(0.5)
else:
    print(f"still running {time.time() - t1:.0f} s after the hang-up")
