"""Fetch model directories from a bucket to local disk before SGLang starts, with the storage
client's parallel ranged downloads: through the gcsfuse mount, mmap loading took 849 s for 21 GB
and a 32-thread copy 400 s, and the drafter from Hugging Face 56-363 s depending on throttling.

    gcsdl.py <bucket> <prefix>=<local dir> [...]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from google.cloud.storage import transfer_manager


def fetch(bucket, b, dst, attempts=3):
    """One blob, with retries: a chunked transfer runs in a process pool, and a single worker
    dying (`BrokenProcessPool`) used to take the whole download, and with it the container, down.
    The last attempt falls back to a plain single-stream copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(attempts):
        try:
            if b.size >= 256 << 20:
                transfer_manager.download_chunks_concurrently(
                    b, dst, chunk_size=64 << 20, max_workers=32)
            else:
                b.download_to_filename(dst)
            if os.path.getsize(dst) == b.size:
                return
        except Exception as e:
            print(f"gcsdl: {b.name} attempt {i + 1}/{attempts} failed: {e!r}", flush=True)
            time.sleep(5)
    b.download_to_filename(dst)
    if os.path.getsize(dst) != b.size:
        raise RuntimeError(f"gcsdl: {b.name} short read")

t0 = time.time()
bucket = storage.Client().bucket(sys.argv[1])
total = 0
for spec in sys.argv[2:]:
    prefix, dst = spec.split("=", 1)
    prefix = prefix.rstrip("/") + "/"
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith("/")]
    small = [b for b in blobs if b.size < 256 << 20]
    big = [b for b in blobs if b.size >= 256 << 20]

    def path(b):
        p = os.path.join(dst, b.name[len(prefix):])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda b: fetch(bucket, b, path(b)), small))
    for b in big:
        fetch(bucket, b, path(b))
    total += sum(b.size for b in blobs)
print(f"gcsdl: {total / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
