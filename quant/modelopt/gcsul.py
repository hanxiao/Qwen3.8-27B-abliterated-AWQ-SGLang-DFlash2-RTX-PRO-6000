"""Upload a local directory to gs://<bucket>/<prefix>/ with parallel chunked uploads.

    python3 gcsul.py <local dir> <bucket> <prefix> [skip-file ...]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from google.cloud.storage import transfer_manager

src, bname, prefix = sys.argv[1], sys.argv[2], sys.argv[3].rstrip("/")
skip = set(sys.argv[4:])
t0 = time.time()
bucket = storage.Client().bucket(bname)
files = [f for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f)) and f not in skip]
small = [f for f in files if os.path.getsize(os.path.join(src, f)) < 256 << 20]
big = [f for f in files if f not in small]


def up_small(f):
    bucket.blob(f"{prefix}/{f}").upload_from_filename(os.path.join(src, f))


with ThreadPoolExecutor(8) as ex:
    list(ex.map(up_small, small))
for f in big:
    for i in range(3):
        try:
            transfer_manager.upload_chunks_concurrently(os.path.join(src, f), bucket.blob(f"{prefix}/{f}"),
                                                        chunk_size=64 << 20, max_workers=16,
                                                        worker_type=transfer_manager.THREAD)
            break
        except Exception as e:
            print(f"gcsul: {f} attempt {i + 1} failed: {e!r}", flush=True)
            if i == 2:
                raise
tot = sum(os.path.getsize(os.path.join(src, f)) for f in files)
print(f"gcsul: {len(files)} files {tot / 1e9:.1f} GB in {time.time() - t0:.0f}s -> gs://{bname}/{prefix}", flush=True)
