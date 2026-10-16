#!/bin/bash
# Cloud Run entry: a fresh venv with llm-compressor; huihui's checkpoint downloaded once into the
# bucket as plain files (the container's disk is RAM, and the model itself needs that RAM);
# quant.py writes straight to the bucket.
set -euxo pipefail
J=/gcs/huihui-awq   # the bucket, mounted at /gcs
pip install -q uv
uv venv -q /venv --python 3.12
uv pip install -q --python /venv/bin/python torch "llmcompressor==0.14.0" "transformers>=5" pandas pyarrow datasets
/venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
export HF_HUB_DISABLE_XET=1  # Xet writes at random offsets, which the bucket mount refuses (EIO)
/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('huihui-ai/Huihui-Qwen3.8-27B-abliterated', local_dir='$J/src', max_workers=4)"
# snapshot_download returns a partial directory when the Hub rate-limits it (anonymous Cloud Run
# egress hit 429 on 2026-10-16); every shard the index names must be here, at full size.
/venv/bin/python -c "
import json, os, sys
from huggingface_hub import HfApi
src = '$J/src'
want = {s.rfilename: s.size for s in HfApi().model_info('huihui-ai/Huihui-Qwen3.8-27B-abliterated', files_metadata=True).siblings}
bad = [f for f, n in want.items() if f.endswith('.safetensors') and (not os.path.exists(f'{src}/{f}') or os.path.getsize(f'{src}/{f}') != n)]
print('missing or short:', bad); sys.exit(1 if bad or not os.path.exists(f'{src}/model.safetensors.index.json') else 0)"
export CALIB=$J/job/calib.parquet
/venv/bin/python $J/job/quant.py $J/src $J/out ${SAMPLES:-384} ${SEQ:-2048}
echo JOB DONE
