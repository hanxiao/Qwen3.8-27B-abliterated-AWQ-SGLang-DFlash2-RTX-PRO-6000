# Qwen3.8-27B · NVFP4 · SGLang + DFlash 2 · ½ NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | Qwen3.8-27B, [`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) |
| Weights | NVFP4 W4A4 with a dense BF16 `lm_head`, 23.6 GB; 22.13 GiB resident |
| KV cache | `fp8_e4m3`, a 185,990-token pool across the 16 full-attention layers |
| Context | 262,144 native; **185,984 usable** per request, which is what the KV pool admits |
| Decode, seven workloads | **111.0 tok/s** (translation 73.7, prose 76.6, chat 79.2, summary 85.4, math 139.4, code 150.0, JSON 172.7) |
| Prefill | 5,016 tok/s at a 27k prompt, 2,862 at 108k, both uncached |
| Mean accepted length | **6.53** of a possible 8; draft acceptance 0.789 |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), BF16 3.8 GB, block drafting at 8 |
| Engine | SGLang, `lmsysorg/sglang:qwen38-27b` with the DFlash 2 backport baked in |
| Attention | FlashInfer for target and draft |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell — 47.64 GiB of 96, SM120, 43.6 GiB used |
| Batching | one slot, `--max-running-requests 1` |
| Quality | 10/10 on the arithmetic set |

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
unquant` keeps the BF16 drafter unquantized, and `--speculative-draft-attention-backend flashinfer`
gives the draft the same kernels as the target.

DFlash 2 is a pure-attention drafter of five sliding-window layers, so it needs no recurrent state
pool of its own. The `--mamba-*` flags and `--linear-attn-verify-backend` belong to the DSpark
recipe and are absent here; the target model keeps its own small state pool, sized by SGLang.

## Memory

| | |
| --- | --- |
| Visible | 47.64 GiB |
| Weights, target | 22.13 GiB |
| Weights, drafter | 3.72 GiB |
| KV pool, `fp8_e4m3` | 235,393 tokens |
| Free after capture | 3.62 GiB |

At `--mem-fraction-static 0.85` the budget is 40.5 GiB, and the drafter's 3.7 GiB is the difference
between this and an MTP deployment on the same card.

The fraction is 0.85 rather than the full card recipe's 0.90 because this model is a VLM and the
Qwen2-VL image preprocessor runs on the GPU in the API process, outside the engine's static
budget. At 0.90 the engine took 42.88 of 47.64 GiB and left 390 MiB, and a 1024x1024 image died
in `preprocess` with *Tried to allocate 352.00 MiB* - while a 128x128 test image had passed, which
is how it survived the first round of testing.

Two numbers describe the window and the smaller one binds: `context_length` is 262,144, but
`max_req_input_len` is what the pool will admit. Size clients from the latter.

## Serving

`cloudrun/service.yaml` is the deployed configuration. Three things about it are Cloud Run's rather
than SGLang's: the container must listen on 8080, `gpu-zonal-redundancy-disabled` must be `'true'`
or the deploy hangs, and `minScale: '1'` is load-bearing — with `maxScale: '1'` and no warm
instance, every request during the ten-minute cold start is answered with *no available instance*.

A cold start is ~10 minutes: 400 s to pull 23 GB of weights, 94 s of `torch.compile`, 44 s of
warmup and FlashInfer autotuning. Baking the weights into the image removes the first and largest
of those.

`--max-model-len` has no counterpart here, but the equivalent trap does: SGLang validates
`input + completion ≤ context` and returns 400, where llama.cpp clamps. A client that asks for the
whole 262,144-token window beside a 53-token prompt is refused, and a harness that reads that 400
as context overflow will compact, retry and fail. Cap the completion.

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
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The L4 companion to
this deployment is [`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
