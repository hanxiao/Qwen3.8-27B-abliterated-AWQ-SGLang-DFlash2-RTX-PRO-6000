# Qwen3.8-27B · NVFP4 · SGLang + DFlash 2 · ½ NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | Qwen3.8-27B, [`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) |
| Weights | NVFP4 W4A4 with a dense BF16 `lm_head`, **23.13 GiB resident** |
| KV cache | `fp8_e4m3`, a 150,904-token pool; **42 KiB/token** (32 target + 10 drafter) |
| Context | 262,144 native; **150,898 usable** per request, verified against the server's own refusal |
| Decode, seven workloads | **110.9 tok/s** (prose 76.2, chat 76.2, translation 76.2, summary 98.8, code 135.7, math 148.8, JSON 164.3) |
| Prefill | 5,016 tok/s at a 27k prompt, 2,862 at 108k, both uncached |
| Mean accepted length | **6.53** of a possible 8; draft acceptance 0.789 |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), BF16 3.8 GB, block drafting at 8 |
| Engine | SGLang, `lmsysorg/sglang:qwen38-27b` with the DFlash 2 backport baked in |
| Attention | FlashInfer for target and draft |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell — 47.64 GiB of 96, SM120 |
| Batching | one slot, `--max-running-requests 1`, `--max-mamba-cache-size 5` |
| Quality | 10/10 on the arithmetic set |
| Vision | native, preprocessed on the GPU; input bounded to 1,024 image tokens |
| Cold start | 620 s to first token; 0 s in practice, held warm by `minScale: '1'` |

Decode is the seven-workload benchmark measured by a two-call delta — the same prompt at 60 and
600 tokens, `(c2−c1)/(d2−d1)` — which cancels prefill and network from the figure. Prefill is
time to first token over prompts of unique random words, so the radix cache cannot serve them.
Accepted length is tokens emitted per target forward pass, read from SGLang's `spec_accept_length`
after a sustained decode.

The table above is the shipped configuration. The sections below give what fixes each value in
it.

```bash
bash docker/build.sh                 # bake the DFlash 2 backport into the SGLang image
gcloud alpha run services replace cloudrun/service.yaml
```

## Why this GPU is half a card

The fractional RTX PRO 6000 is a Cloud Run preview, and that is a different product from Compute
Engine's G4 machine types. The two carry separate quotas on separate services: `run.googleapis.com`
grants milliGPUs, where a half-GPU instance consumes 500, while `compute.googleapis.com` grants
`NVIDIA_RTX_PRO_6000` whole cards. A project can hold thousands of the former and none of the
latter, and G4 shapes were stocked out in every US zone while the Cloud Run grant was live and
idle.

Only 0.5 and 1 are accepted for `nvidia.com/gpu`; 0.25 is refused. The half card presents 47.64 GiB
and takes 10–16 vCPU with 40–64 GiB of system memory.

## The drafter is the configuration

Qwen3.8-27B ships an MTP head, and using it is the obvious choice. It is also the slower one. On
this card, raising MTP's draft depth buys accepted length and does not buy throughput:

| MTP draft tokens | Accepted length | Decode |
| --- | --- | --- |
| 2 | 2.58 | 61.2 tok/s |
| 3 | 3.22 | 66.1 |
| 5 | 3.78 | 68.0 |
| 7 | 4.07 | 67.5 |

Accepted length rises the whole way and decode stops at five. MTP drafts autoregressively, so a
depth of seven costs seven sequential passes of the draft head, and past five those passes cost
what the extra acceptance earns. The plateau is drafting time, not acceptance.

DFlash 2 removes exactly that cost. It is a block-diffusion drafter: it predicts a whole block of
eight tokens in one pass, keeps the top candidates at every position, and traces one path through
them with a lightweight selector. Two-tap dynamic convolutions in its backbone keep the draft from
decaying toward the end of the block. At 6.53 accepted tokens per target pass it emits more than
MTP does at any depth, and it pays one draft pass rather than eight.

Decoding is lossless. The target verifies every drafted token, so greedy output matches the
unspeculated model exactly and sampling preserves its distribution; the arithmetic set is 10/10
either way.

## The head must be dense

DFlash 2's selector runs a top-k over logits, which makes the precision of `lm_head` structural
rather than a rounding detail. The packed-FP4 checkpoint is 1.7 GB smaller and SGLang refuses it:

```
DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head.
```

`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` is the same NVFP4 body with that head left dense, and it
is what the numbers above were measured on. Upstream carries a patch that runs a quantized head in
place through `lm_head.quant_method.apply`; it exists because the alternative — dequantizing once
into a dense copy — allocates several gigabytes during draft-graph capture.

## The image carries a backport

DFlash 2 landed in SGLang on 2026-08-19 (PR #35371, PR #35496). The cookbook-pinned image
`lmsysorg/sglang:qwen38-27b` predates it and ships only the DFlash 1 model class, so serving DFlash 2
means overlaying eleven modules from upstream. The reference recipe bind-mounts them, which Cloud
Run cannot do; [`docker/build.sh`](docker/build.sh) bakes them into an image instead.

vLLM is not an alternative today. Its released image has only `DFlashDraftModel`, and the nightly
that carries the merge of PR #52816 has a `DFlash2Qwen3Model` whose decoder layer has no
`attention_conv`, so the checkpoint refuses to load:

```
ValueError: There is no module or parameter named 'layers.0.attention_conv' in DFlash2Qwen3Model.
```

## Flags that are not preferences

`--attention-backend flashinfer` is required on SM120; `trtllm_mha` is SM100-only. `--kv-cache-dtype
fp8_e4m3` is the best supported precision for this model, because SGLang's 4-bit KV paths are not
built for a hybrid-GDN architecture. `--tool-call-parser qwen3_coder` decodes the
`<tool_call><function=…>` payload this template emits; `hermes` parses none of it, and a model whose
tool calls are silently dropped answers nothing at all. `--speculative-draft-model-quantization
modelopt_fp4` is required for the NVFP4 drafter - SGLang does not requantize a draft model on the
fly, and the flag is what makes it read the ModelOpt scales - and
`--speculative-draft-attention-backend flashinfer` gives the draft the same kernels as the target.

DFlash 2 is a pure-attention drafter of five sliding-window layers, so it needs no recurrent state
pool of its own. The `--mamba-*` flags and `--linear-attn-verify-backend` belong to the DSpark
recipe and are absent here; the target model keeps its own small state pool, sized by SGLang.

## Memory

Every step below is a logged number, not an estimate. The two Prometheus gauges are **target-only**
- `weight_memory_usage_gb` and `kv_cache_memory_usage_gb` exclude the drafter entirely and the
Mamba pool has no gauge at all - so metrics alone leave 11.9 GiB unaccounted. The startup log
closes it:

| Step | GiB | Avail after | Source |
| --- | ---: | ---: | --- |
| Card visible | 47.64 | | ½ of 96 |
| CUDA context + driver reserve | −2.40 | **45.24** | `Load weight begin. avail mem=45.24` |
| Weights, target | −23.13 | **22.12** | `Load weight end … mem usage=23.13` |
| Weights, drafter | −1.59 | **20.49** | `type=DFlash2DraftModel, quant=modelopt_fp4, mem usage=1.59` |
| Mamba/GDN state cache | −3.13 | 17.36 | `ssm 0.84 + intermediate_ssm 2.25 + conv 0.04` |
| KV pool, target (16 full layers) | −6.30 | **11.04** | `K size: 3.15 GB, V size: 3.15 GB` |
| KV pool, drafter (5 layers) | −1.97 | **9.07** | `K size: 0.98 GB, V size: 0.98 GB` |
| CUDA graphs | −2.50 | 6.57 | prefill 2.26 + target verify 0.11 + draft verify 0.13 |
| FlashInfer autotune + fragmentation | −0.67 | **5.90** | `available_gpu_mem=5.90` |

The line that matters for sizing context is the pair of KV pools. Geometry is `head_dim` **256**
with 4 KV heads, so one token of fp8 KV costs `2 × 4 × 256 = 2 KiB` per layer: **32 KiB** over the
target's 16 full-attention layers, plus **10 KiB** over the drafter's 5. A token therefore costs
**42 KiB**, and context on a speculative deployment is ~30% more expensive than on a plain one.
That is the price of 118 tok/s, and it is worth paying, but it should be paid knowingly.

`--max-mamba-cache-size 5` is on the floor and deliberately so. The hybrid state cache is sized in
units of `mamba_ratio`, which is 5 here, so slots must be at least 5 per request; SGLang's
auto-sizing picks 8, and 4 fails at startup with `resulting max_num_reqs=0`. Both 5 and 8 admit
exactly one request - `floor(size/5)` - so the three extra slots buy no concurrency, only radix
buffer. Dropping them is worth **+10,738 tokens** and costs nothing measurable: prefix reuse stays
at 13,504 of 13,552 tokens (99.6%) and decode is 113.2 tok/s against 110.9 - both measured on
the earlier BF16-drafter build, so read them against each other rather than against the
117.9 above. **Raising
`--max-running-requests` requires raising this in step**, or the server refuses to start.

The saving is smaller than the slot count suggests because only `ssm_state` scales with slots
(1.27 → 0.84 GiB). `intermediate_ssm_state_cache` is a fixed **2.25 GiB** - the largest single
piece of the Mamba pool and immovable. That is why 8 → 5 returns 0.43 GiB rather than 1.34.

At `--mem-fraction-static 0.80` the drafter's weights and KV are the difference between this and
an MTP deployment on the same card. The five points given up against 0.85 cost 45,824 tokens of
pool and no decode at all - 110.9 tok/s against 111.0, which is noise. Pool size and decode rate
are close to independent here, because one slot decoding one sequence touches a fixed working set
whatever the pool's ceiling is.

This model is a VLM, and vision cost GPU memory in a way the static budget does not see. The
"fast" Qwen2-VL image processor is the torchvision one: it allocates on the device, in the API
process, outside the engine's budget, and the allocation grows with the image. A 1024x1024 image
failed at `0.90` needing 352 MiB, and failed again at `0.85` needing 392 MiB with 154 MiB free.
Shrinking the fraction is the wrong instrument for this - the fraction is a constant and the
allocation is a function of the input, so it can always be outgrown.

Bounding the input fixes it at the source. Qwen2-VL's processor resizes rather than rejects:
aspect ratio is preserved and dimensions round to patch multiples, so a cap is a ceiling on cost
and not a rejection of large images. This checkpoint has patch_size 16 and merge_size 2, so a
token covers 32x32 = 1024 pixels and 1,048,576 pixels is exactly 1,024 image tokens. Preprocessing
then runs on the GPU, which is both what it is for and faster - a 3000x2000 image takes 0.52 s on
the device against 0.94 s on the CPU, because the resize happens where the pixels already are.

The kwarg is `size`, not `max_pixels`, and this is worth stating because the wrong one fails
silently. This processor's `preprocessor_config.json` carries `size: {longest_edge: 16777216,
shortest_edge: 65536}` and no `min_pixels`/`max_pixels` at all. `--mm-process-config
'{"image":{"max_pixels":1048576}}'` is accepted, echoed back by `/get_server_info` verbatim, and
ignored: a 3000x2000 image still produced 5,828 image tokens. `longest_edge` is an area in pixels
here rather than an edge length, which is why the untouched default is a 16,384-token ceiling.

| Image | Default | `size.longest_edge` 1048576 | Latency |
| --- | --- | --- | --- |
| 512x512 | 256 tok | 256 | 0.55 s |
| 1024x1024 | 1,024 | 1,024 | 0.65 s |
| 2048x2048 | 4,096 | **1,024** | 0.63 s |
| 3000x2000 | 5,828 | **1,014** | 0.52 s |

Below the cap nothing changes; above it, cost is flat. `0.80` is then headroom for a bounded
allocation rather than an attempt to outrun an unbounded one.

A 128x128 test image passes under every one of these configurations, which is how the first round
of vision testing missed all of it. Test at the size the clients actually send.

Two numbers describe the window and the smaller one binds: `context_length` is 262,144, but
`max_req_input_len` is what the pool will admit - **206,492** here. Size clients from the latter,
and re-read it after any change to `--mem-fraction-static` or `--max-mamba-cache-size`, both of
which move it, and the drafter's precision moves it most. Verified by bisection rather than
trusted on the earlier BF16-drafter build, where the pool was 150,904: 150,732 tokens was served
and 151,388 refused with *exceeds the maximum allowed length (150898 tokens)*.

## Quantizing the drafter buys context and speed, and is WITHDRAWN

**Do not use this configuration.** It is documented here because the measurements below are real
and the failure it caused is not visible in any of them.

In use, this drafter produced stretches of degenerate text - long runs of loosely associated words
with no syntax - most often inside thinking blocks. The deployment has been reverted to the BF16
drafter (`z-lab/Qwen3.8-27B-DFlash2`, `--speculative-draft-model-quantization unquant`), which
restores 150,898 usable context and the 110.9 tok/s figures at the top of this file.

The cause is not established. What can be said is that the reasoning I used to rule the risk out
was wrong: I argued that the target verifies every drafted token, so a cheaper drafter could not
change what is emitted, only how fast it is proposed. That holds for speculative decoding with
rejection sampling. DFlash 2 drafts a whole block through its own `candidate_selector`, and if
that path accepts on a looser criterion, the drafter's distribution reaches the output - which
would show up first in the flattest distribution the model produces, which is exactly where it
did show up.

One number in the measurements now reads as a warning rather than a result: mean accepted length
came out at **4.19 of 8** here, against 3.60 reported by the checkpoint's author on their target
and 3.71 for the BF16 drafter. A quantized drafter whose blocks are accepted *more* often than
the unquantized one is not self-evidently a better drafter.

Nothing in the benchmark below would have caught this. Seven throughput workloads and a 10/10
arithmetic check all pass while prose comes apart, and `s/it` improves when long blocks of
nonsense are accepted. Judging a drafter needs generated text read as text.

## What the measurements were

The drafter was BF16 at 3.72 GiB resident. In NVFP4 it is **1.59 GiB**, and nearly all of the
2.13 GiB that frees turns into KV pool: at 42 KiB/token the window goes **150,898 -> 206,492
usable, +36.8%**. That is the main result. A speculative deployment pays for its drafter twice -
once in weights, once in a second KV pool sized at the same token count - so shrinking the weights
compounds.

It is also faster, which was not obvious in advance. Quantizing a drafter risks the only thing a
drafter is for: if acceptance falls, the target verifies more rejected tokens and throughput drops.
The uncalibrated round-to-nearest version of this same checkpoint demonstrates exactly that on its
author's card - identical VRAM, 3.26 accepted of 8, and *slower* than BF16. The calibrated one
recovers it, and the recovery transfers to this target:

| workload | BF16 drafter | NVFP4 calibrated | |
| --- | ---: | ---: | ---: |
| prose | 76.2 | **84.5** | +10.9% |
| chat | 76.2 | **80.0** | +5.0% |
| translation | 76.2 | **77.8** | +2.1% |
| summary | 98.8 | **102.5** | +3.7% |
| code | 135.7 | **151.6** | +11.7% |
| math | 148.8 | **160.6** | +7.9% |
| JSON | 164.3 | **168.0** | +2.3% |
| **mean** | **110.9** | **117.9** | **+6.3%** |

Both columns are `scripts/bench.py` against this deployment, two runs each, spread under 0.5%.
Arithmetic stays 10/10: the target verifies every drafted token, so a cheaper drafter cannot change
what is emitted, only how fast it is proposed. Mean accepted length measured 4.19 of 8 across these
workloads.

The two flags are the whole change - `--speculative-draft-model-path` and
`--speculative-draft-model-quantization modelopt_fp4`. The base weights are byte-identical to
`z-lab/Qwen3.8-27B-DFlash2` (same 3,848,817,896 bytes, same LFS sha256), so this is a requantization
of the drafter already in use rather than a different model.

## Serving

`cloudrun/service.yaml` is the deployed configuration. Three things about it are Cloud Run's rather
than SGLang's: the container must listen on 8080, `gpu-zonal-redundancy-disabled` must be `'true'`
or the deploy hangs, and `minScale: '1'` is load-bearing for the reason the next section measures.

`--max-model-len` has no counterpart here, but the equivalent trap does: SGLang validates
`input + completion ≤ context` and returns 400, where llama.cpp clamps. A client that asks for the
whole 262,144-token window beside a 53-token prompt is refused, and a harness that reads that 400
as context overflow will compact, retry and fail. Cap the completion.

## Cold start

A cold start is 620 s wall clock — 10m18s from `services replace` to serving. Phase boundaries are
read off the startup log's own timestamps, `22:52:11` to `23:01:21`:

```mermaid
gantt
    title Cold start to first token - 620 s measured
    dateFormat X
    axisFormat %M:%S
    section Cloud Run
    Schedule + image pull 70s     :done, sched, 0, 70s
    section Load
    Import + init 21s             :done, init, after sched, 21s
    Target weights from HF 406s   :crit, tw, after init, 406s
    Drafter weights 56s           :crit, dw, after tw, 56s
    section Allocate
    Mamba + KV pools 2s           :done, pools, after dw, 2s
    FlashInfer autotune 15s       :done, tune, after pools, 15s
    section Compile
    Prefill CUDA graph 33s        :active, g1, after tune, 33s
    Target verify graph 1s        :active, g2, after g1, 1s
    Draft verify graph 6s         :active, g3, after g2, 6s
    section Serve
    Warmup to first token 10s     :done, warm, after g3, 10s
```

**462 s of the 620 — three quarters — is pulling weights over the network**, 23 GB at 57 MB/s from
HuggingFace plus the drafter's 3.7 GB. Nothing else is close: graph capture is 40 s and every
allocation together is 17 s.

Two levers, and only one of them is shipped:

| Lever | Effect | Status |
| --- | --- | --- |
| `minScale: '1'` | Users pay **0 s**. The instance is never reaped, so no request ever meets a cold start. | **Shipped**, load-bearing |
| Bake weights into the image | Replaces a 57 MB/s HuggingFace pull with an in-region Artifact Registry layer pull. The 70 s already spent pulling this image is the evidence it is the faster path. | Not built — **saving unmeasured** |

`minScale: '1'` turns cold start from a user-facing cost into a deploy-time one: the 620 s is only
ever paid when a revision is replaced. It is not a tuning preference — with `maxScale: '1'` and no
warm instance, every request during those ten minutes is answered with *no available instance*.

Baking is the obvious remaining win, and the honest position is that its size is not yet known:
building a ~25 GB image and timing it is the only way to find out, and this repo has not done it.
`docker/build.sh` is where it would go.

## Reproducing

```bash
python3 scripts/bench.py https://YOUR-SERVICE.run.app     # seven workloads + arithmetic
python3 scripts/ndec.py  https://YOUR-SERVICE.run.app Qwen3.8-27B-NVFP4
```

## Credit

The serving recipe follows [`MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
which worked out the flags and the memory budget for the full card and carries the DFlash 2
backport overlay this build fetches. The two-call delta measurement is its `bench/ndec.py`. DFlash 2
is [z-lab/dflash](https://github.com/z-lab/dflash); the drafter card is
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The NVFP4
quantization of that drafter, and the activation calibration that makes it worth using, are
[`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal). The L4 companion to
this deployment is [`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
