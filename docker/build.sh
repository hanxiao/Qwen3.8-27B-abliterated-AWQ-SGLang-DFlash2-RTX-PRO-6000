#!/usr/bin/env bash
# Bake the DFlash 2 backport into the SGLang image and push it to Artifact Registry.
#
# DFlash 2 landed in SGLang on 2026-08-19 (PR #35371, PR #35496). The cookbook-pinned image
# lmsysorg/sglang:qwen38-27b predates it and ships only the DFlash 1 model class, so DFlash 2
# needs eleven modules overlaid onto it. The reference recipe bind-mounts them at `docker run`;
# Cloud Run has no bind mounts, so they are copied into an image here.
#
# The overlay is fetched from MiaAI-Lab's repo rather than vendored into this one: it is their
# backport of upstream SGLang, and a second copy is a second thing to keep current.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-sglang}"
TAG="${TAG:-qwen38-dflash2:v1}"
UPSTREAM="${UPSTREAM:-https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark.git}"

stage="$(mktemp -d)"; trap 'rm -rf "$stage"' EXIT
git clone -q --depth 1 "$UPSTREAM" "$stage/upstream"
[ -d "$stage/upstream/patch/sglang" ] || { echo "overlay not found in $UPSTREAM"; exit 1; }
cp -r "$stage/upstream/patch/sglang" "$stage/overlay"

cat > "$stage/Dockerfile" <<'DOCKER'
FROM lmsysorg/sglang:qwen38-27b
ARG SGL=/sgl-workspace/sglang/python/sglang
COPY overlay/srt/models/dflash.py                   ${SGL}/srt/models/dflash.py
COPY overlay/kernels/ops/speculative/dflash.py      ${SGL}/kernels/ops/speculative/dflash.py
COPY overlay/srt/speculative/dflash_utils.py        ${SGL}/srt/speculative/dflash_utils.py
COPY overlay/srt/speculative/dflash_worker_v2.py    ${SGL}/srt/speculative/dflash_worker_v2.py
COPY overlay/srt/speculative/dflash_info.py         ${SGL}/srt/speculative/dflash_info.py
COPY overlay/srt/speculative/dflash_info_v2.py      ${SGL}/srt/speculative/dflash_info_v2.py
COPY overlay/srt/speculative/draft_worker_common.py ${SGL}/srt/speculative/draft_worker_common.py
COPY overlay/srt/speculative/spec_utils.py          ${SGL}/srt/speculative/spec_utils.py
COPY overlay/srt/mem_cache/allocation_sizing.py     ${SGL}/srt/mem_cache/allocation_sizing.py
COPY overlay/srt/layers/moe/utils.py                ${SGL}/srt/layers/moe/utils.py
COPY overlay/srt/layers/logprob_processor.py        ${SGL}/srt/layers/logprob_processor.py
DOCKER

gcloud artifacts repositories create "$REPO" --repository-format=docker \
  --location="$REGION" --project="$PROJECT" 2>/dev/null || true

gcloud builds submit "$stage" --project="$PROJECT" --region="$REGION" \
  --tag="$REGION-docker.pkg.dev/$PROJECT/$REPO/$TAG" --timeout=3600s

echo "built $REGION-docker.pkg.dev/$PROJECT/$REPO/$TAG"
echo "set it as the image in cloudrun/service.yaml"
