"""Fetch model directories from a bucket to local disk before SGLang starts, with the storage
client's parallel ranged downloads: through the gcsfuse mount, mmap loading took 849 s for 21 GB
and a 32-thread copy 400 s, and the drafter from Hugging Face 56-363 s depending on throttling.

    gcsdl.py <bucket> <prefix>=<local dir> [...]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from google.cloud.storage import transfer_manager

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
        list(ex.map(lambda b: b.download_to_filename(path(b)), small))
    for b in big:
        transfer_manager.download_chunks_concurrently(b, path(b), chunk_size=64 << 20, max_workers=32)
    total += sum(b.size for b in blobs)
print(f"gcsdl: {total / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
