#!/bin/bash
# Cloud Run GPU entry for the own ModelOpt quant (mo.py). Bucket mounted at /gcs.
# VARIANTS="NAME=a ALGO=max ...;NAME=b ..." run one after another on one download; afterwards the
# container polls /gcs/ownq/queue/$K_SERVICE/*.v (one variant line per file) so new variants reuse
# the local BF16 copy.
set -uxo pipefail
pip install -q uv
uv venv -q /venv --python 3.12
uv pip install -q --python /venv/bin/python torch "transformers>=5" "nvidia-modelopt==0.47.0" accelerate pandas pyarrow safetensors requests
/venv/bin/python -c "import torch,transformers,modelopt,pandas,safetensors;import modelopt.torch.quantization as mtq;from modelopt.torch.export import export_hf_checkpoint;print('IMPORTS OK',torch.__version__, transformers.__version__, modelopt.__version__, torch.cuda.get_device_name(0))" || { echo "IMPORT FAIL"; exit 3; }
uv pip install -q --python /venv/bin/python flash-linear-attention || true
/venv/bin/python -c "
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B,T,H,K=1,256,4,128
q=torch.randn(B,T,H,K,device='cuda',dtype=torch.bfloat16);k=torch.nn.functional.normalize(torch.randn(B,T,H,K,device='cuda',dtype=torch.bfloat16),dim=-1);v=torch.randn(B,T,H,K,device='cuda',dtype=torch.bfloat16)
g=-torch.rand(B,T,H,device='cuda',dtype=torch.float32);b=torch.rand(B,T,H,device='cuda',dtype=torch.bfloat16)
o,_=chunk_gated_delta_rule(q,k,v,g,b);torch.cuda.synchronize();print('FLA OK',o.shape)" || uv pip uninstall -q --python /venv/bin/python flash-linear-attention fla-core || true
pip install -q 'google-cloud-storage==3.17.1' && python3 /gcs/gcsdl.py ${BUCKET:?set BUCKET} pf/bf16=/tmp/src || exit 4
export CALIB=/gcs/ownq/job/${CALIBF:-ownmix.parquet} FIDC=/gcs/ownq/job/fid_corpus.json
runv() {
  echo "=== VARIANT $1"
  env $1 /venv/bin/python -u /gcs/ownq/job/mo.py /tmp/src /tmp/out
  echo "=== VARIANT RC $? $1"
  rm -rf /tmp/out /tmp/out-bh
}
IFS=';' read -ra VS <<< "${VARIANTS:-}"
for v in "${VS[@]}"; do runv "$v"; done
QD=/gcs/ownq/queue/${K_SERVICE:-none}
mkdir -p $QD
echo "QUEUE $QD"
while true; do
  for f in $(ls $QD/*.v 2>/dev/null | sort); do
    v=$(cat $f); rm -f $f
    runv "$v"
  done
  sleep 20
done
