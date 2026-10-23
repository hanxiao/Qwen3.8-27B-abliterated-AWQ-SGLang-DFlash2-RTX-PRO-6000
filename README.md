# Qwen3.8-27B abliterated · AWQ-INT4 · SGLang + DFlash 2 · ½ NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), quantized here with the AWQ-INT4 recipe of [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) (`quant/`), served as `Qwen3.8-27B-abliterated-AWQ-INT4` |
| Weights | INT4 W4A16 (AWQ, group 32, asymmetric) with a dense BF16 `lm_head`, 19.12 GiB resident |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), quantized to FP8 at load (2.53 GiB), blocks of 8 tokens |
| Engine | SGLang v0.5.22 (`lmsysorg/sglang:v0.5.22`) with five patches in `docker/`, FlashInfer attention and GDN prefill kernels |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell (47.64 GiB, SM120) on Cloud Run |
| Slots | 4 concurrent requests, fp32 recurrent state |
| Context | **262,144** tokens per request; one 266,083-token FP8 KV pool shared by the four slots |
| Decode, greedy, thinking off | **140.5 tok/s** (prose 89.2, chat 98.1, translation 99.3, summary 118.0, code 193.4, math 180.8, JSON 204.6) |
| Decode, greedy, thinking off, four streams | 120.7 tok/s each, **348.3 tok/s** in total; 217.4 with the former two slots |
| Decode, sampled, thinking on | 126.2 tok/s, 3.60 tokens per verification |
| Decode, greedy, thinking off, long context | 75.5 tok/s at 34k tokens, 65.8 at 115k |
| Prefill | 2,783 tok/s at 9k tokens, 2,568 at 31k, 1,793 at 124k |
| Refusals | 0 of 6 borderline prompts; the stock model refuses 5 |
| Quality | needle retrieval 5/5 at 30k and 120k, multi-step math 12/12, 64 of 64 checked answers at four concurrent streams; greedy text byte-identical to the two-slot configuration on 104 of 104 prompts |
| Vision | native, preprocessed on the GPU, at most 1,024 tokens per image |
| Tool calls | arguments stream as they are generated, at most 0.1 s apart |
| Structured output | not supported: `response_format` and `tool_choice: "required"` return 400 |
| Disconnects | a client that disconnects frees its slot within a second |
| Cold start | 14 min; one instance stays warm |

We measure decode with `scripts/bench2.py`, which streams each answer and divides the tokens after
the first by the time from the first to the last token. Each of seven workloads runs two prompts of
up to 800 tokens, greedy with thinking off unless stated. Sampled decode comes from
`scripts/accept.py`, fourteen prompts with thinking on, averaged over three runs, and
`scripts/bench3.py` sends the same fourteen prompts over 1, 2, 4 or 8 connections at once. Its
per-stream figure is the bench2 rate, and its total is every completion token over the wall time of
the round, so prefill and queueing count. Prefill is the time to first token on prompts of unique
random words, so that no prefix is cached.

The single-stream decode above was measured on the production revision on 2026-10-23. The
four-stream, sampled and quality figures come from a test service with the same image, weights and
flags at `--mem-fraction-static 0.80` (production runs 0.79, which changes the pool size and the free memory, not the computation), as
medians of five runs; see [the tuning study](#tuning-study-2026-10-23). Long-context decode,
prefill, refusals and the tables in the sections on the checkpoint and speculative decoding were
measured on the earlier two-slot configuration and not repeated at four slots.

```bash
bash docker/build.sh                                  # lmsysorg/sglang:v0.5.22 + five patches, into Artifact Registry
gcloud alpha run services replace quant/service.yaml  # once: quantize into the bucket (~75 min)
gcloud alpha run services replace cloudrun/service.yaml
```

## GPU

The fractional RTX PRO 6000 is a Cloud Run offering, separate from the G4 machine types of Compute
Engine and with its own quota: `run.googleapis.com` grants milliGPUs, of which a half card uses
500, and a project can hold this grant without any Compute Engine GPU quota. Cloud Run accepts 0.5
or 1 for `nvidia.com/gpu`, and the half card comes with 10 to 16 vCPU and 40 to 64 GiB of system
memory.

The half card is a MIG `2g.48gb` partition with 94 of the 188 SMs. We measure 765 GB/s of read
bandwidth (1,524 GB/s on the whole card) and 177 dense BF16 TFLOPS (412). Decode is bound by
bandwidth, since a step reads the weights of the target and the drafter once for all the tokens it
verifies, so it becomes faster only by committing more tokens per step or by reading fewer bytes.
Prefill is bound by compute and runs at about 145 TFLOP/s at 9k tokens, which is 82% of the BF16
peak.

## Checkpoint

We serve the abliterated Qwen3.8-27B of huihui-ai, in which the residual-stream direction that
mediates refusals is projected out of the weights. Of six borderline prompts in
`scripts/refusal.py`, among them a profane roast, lock picking and a paywall question, the stock
model refuses five and the abliterated model none.

DFlash 2 ranks its draft candidates with a top-k over the target logits, so the target `lm_head`
stays dense. To choose the 4-bit format of the body, we compare three checkpoints of the stock
model against the official FP8 checkpoint, whose median per-position KL to BF16 on text is 0.0117
([discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)). `scripts/fid.py`
feeds the same text through each server, teacher-forced, and records the top-20 distribution at
each of 91,769 positions from 55 private documents of the traffic this deployment serves: agent
answers, code, technical documents and Chinese. The following table reports perplexity, KL on each
slice and top-1 agreement.

| Checkpoint | PPL | KL answers | KL code | KL docs | KL Chinese | KL all | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.8-27B-FP8` (reference) | 7.874 | | | | | | |
| **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** | 7.979 | **0.0137** | **0.0126** | **0.0181** | **0.0935** | **0.0466** | **91.74%** |
| `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` | 8.062 | 0.0289 | 0.0274 | 0.0381 | 0.1330 | 0.0724 | 89.26% |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | **7.902** | 0.0383 | 0.0328 | 0.0492 | 0.1526 | 0.0853 | 88.33% |

AWQ is closest to the reference on every slice, with about half the divergence of the RadixArk
NVFP4 checkpoint everywhere except Chinese, while QUASAR has the lowest perplexity and the largest
divergence. On the same card with the same flags and one stream, AWQ is also the smallest and the
fastest to decode:

| | NVFP4 (RadixArk) | AWQ-INT4 | NVFP4 (QUASAR) |
| --- | ---: | ---: | ---: |
| Weights resident | 21.84 GiB | **19.12 GiB** | |
| KV pool, one slot | 180,573 | 248,597 | **256,218** |
| Decode, mean of seven | 107.9 | **119.2** | 116.4 |
| Prefill at 9k / 31k / 124k | **6,045 / 5,247 / 2,659** | 2,751 / 2,622 / 1,784 | |

W4A16 multiplies on BF16 tensor cores (Marlin) and NVFP4 on FP4 tensor cores, so prefill runs at
about half the NVFP4 rate, while decode gains from the smaller weights. Over 143 agent runs on this
deployment, a turn adds about 2,300 prompt tokens and generates 1,223, so decode takes over 90% of
GPU time and its gain outweighs the prefill cost.

The two published 4-bit builds of the abliterated model give up either context or vision. We
measure them with a BF16 drafter:

| | Format | Weights | KV pool, two slots | Decode, mean of seven | Vision |
| --- | --- | ---: | ---: | ---: | --- |
| `TelperionAI/Huihui-Qwen3.8-27B-abliterated-INT4-AWQ-GPTQ` | INT4 MLP, INT8 attention and last 8 MLPs | 25.1 GB | 110,391 | 105.1 | yes |
| `hotdogs/Qwen3.8-27B-abliterated-AWQ-INT4` | INT4 symmetric, group 128 | 17.6 GB | | | no |
| ours | INT4 asymmetric, group 32 | 21.0 GB | **202,065** | **118.0** | yes |

We therefore quantize the BF16 abliterated checkpoint with the cyankiwi recipe, which we read from
the `config.json` of their checkpoint: AWQ, INT4, asymmetric, group 32 and the `mse` observer. It
covers every `Linear` layer except `lm_head`, the vision tower, the MTP head,
`linear_attn.in_proj_a/b` and the `linear_attn.out_proj` of layer 0. Calibration uses their public
`calibration` set of 384 agentic and STEM conversations in ten languages, truncated to 2,048 tokens
(380,221 tokens). The result has the same 2,396 tensors, names and formats as the cyankiwi
checkpoint.

`quant/quant.py` runs llm-compressor 0.14 with its hybrid-attention AWQ mappings on a whole RTX
PRO 6000 (`quant/service.yaml`) in 71 minutes, 64 of them for calibration. The 55 GB model stays in
CPU memory and moves to the GPU one layer at a time, since the model and the calibration
activations do not fit on the 96 GB card together. The run is a Cloud Run service, because Cloud
Run limits GPU jobs to one hour. Against the FP8 reference of the stock model, the result has KL
0.0544 overall and 0.1067 on Chinese, against 0.0466 and 0.0935 for the stock AWQ checkpoint, and
perplexity 7.966 against 7.979. These numbers include both the abliteration and the quantization.

## Speculative decoding

### Drafter

Qwen3.8-27B ships an MTP head, which drafts one token per pass. Measured with vLLM on an NVFP4
checkpoint of the same model, a deeper MTP draft raises the accepted length but stops raising
throughput at five tokens. Each drafted token costs one more sequential pass of the head:

| MTP draft tokens | Accepted length | Decode |
| --- | --- | --- |
| 2 | 2.58 | 61.2 tok/s |
| 3 | 3.22 | 66.1 |
| 5 | 3.78 | 68.0 |
| 7 | 4.07 | 67.5 |

DFlash 2 is a block-diffusion drafter that predicts eight tokens in a single pass. It keeps the top
candidates at every position, and a lightweight selector traces one path through them; two-tap
dynamic convolutions in its backbone keep the draft from degrading toward the end of the block. The
target verifies every drafted token, so the output follows the target distribution, greedy or
sampled.

We quantize the drafter to FP8 at load (`--speculative-draft-model-quantization fp8`). This shrinks
the weights it reads at every step from 3.72 to 2.53 GiB and returns the difference to the KV pool.
Two services measured in the same hour compare the two formats:

| | BF16 drafter | FP8 drafter |
| --- | ---: | ---: |
| Decode, mean of seven, greedy | 116.8 | **124.1** |
| Decode, thinking on, sampled (prose, chat) | 64.7, 79.6 | **71.6, 80.0** |

Long sampled answers show no added repetition. Over 24 prompts in English and Chinese with thinking
on and up to 4,000 tokens, the mean zlib compression ratio is 0.391 with the BF16 drafter and 0.399
with FP8, and no answer falls into a loop.

### Verification

Three changes to verification raise decode and keep the output distribution. Token-wise
verification stops at the first rejected draft token. Block verification
(`--speculative-use-block-verification`, [Sun et al., 2024](https://arxiv.org/abs/2403.10444)) can
accept a longer prefix with a correction that preserves the target distribution, and it applies to
sampled requests.

The sampled verification kernel rejects a draft token whose probability exceeds 1.0. An fp32
softmax over a peaked row returns 1.0 plus a few units in the last place, so the kernel rejects
exactly the drafts the drafter is most confident about. `docker/draft_prob_round.py` (sglang PR
#45819) accepts probabilities up to 1 + 10⁻³ and clamps them to 1.

To roll back a rejected draft, verification stores the recurrent state of all 48 GDN layers at each
of the 8 drafted positions, 1.2 GB per request and step. ReplaySSM
(`--enable-linear-replayssm-spec`, [arXiv 2610.13778](https://arxiv.org/abs/2610.13778)) stores
compact records of the recurrence inputs and replays the accepted prefix into the checkpoint. This
frees a 3.38 GB intermediate-state cache and the state writes of every step. SGLang applies
ReplaySSM to MTP drafts, and `docker/replayssm_dflash.py` (from sglang PR #35725) routes the DFlash
commit through it.

We measure each change against the same deployment without it, on two services in the same hour.
Block verification and the probability fix act only on sampled requests, and the int8 draft head is
described below.

| Change | Tokens per verification, sampled | Decode, sampled | Decode, greedy |
| --- | ---: | ---: | ---: |
| Block verification | 3.47 → 3.56 | 109.7 → 112.8 | |
| ReplaySSM | 3.61 → 3.56 | 113.2 → 120.9 | 124.6 → 133.9 |
| Probabilities above 1 | 3.54 → 3.70 | 116.0 → 125.1 | |
| int8 draft head | 3.54 → 3.60 | 119.3 → 126.5 | 133.6 → 141.2 |

The quality checks hold with all four changes. Needle retrieval scores 5/5 at 30k and 120k and math
12/12. All 64 checked answers are right at two streams, and ten facts are recalled across a
conversation served from the prefix cache (`scripts/prefix.py`). Twenty-four long sampled answers
have a mean zlib ratio of 0.414, and none falls into a loop (`scripts/degen.py`). Greedy text can
still differ in rounding. Of 24 greedy answers (`scripts/drift.py`), a second run on the same server
reproduces all 24, the int8 draft head changes 2, and ReplaySSM, whose replayed states round
differently from stored ones, changes 14.

The drafter commits fewer tokens per step deep in a context. With 34k and 115k tokens of random
words before the question (`scripts/longdecode.py`), greedy decode runs at 75.5 and 65.8 tok/s with
2.6 and 2.5 tokens per verification. Agent traffic, whose context is real text, commits 3.94 tokens
per verification at a median context of 115k tokens.

### Draft head

A torch profile of 30 decode steps (`/start_profile`), taken with a BF16 draft head, divides a 35.4
ms step as follows:

| | ms per step |
| --- | ---: |
| Target layers, INT4 (Marlin) | 19.8 |
| `lm_head`, BF16, read twice: draft candidates and verification | 6.7 |
| Drafter layers, FP8 | 2.7 |
| GDN replay and convolution | 1.4 |
| Other kernels | 2.6 |
| GPU idle between launches | 2.2 |

The drafter uses `lm_head` only to rank its candidates, and the target verifies every token with
its own BF16 head, so a cheaper head for the draft changes which tokens are proposed but not the
output distribution. `docker/dflash_qhead.py` gives the draft an int8 copy of `lm_head` with one
scale per vocabulary row (`SGLANG_DFLASH_DRAFT_HEAD=int8`). A Triton kernel reads it in 1.68 ms,
against 3.35 ms for the BF16 head, both at about 760 GB/s. The copy takes 1.27 GB from the memory
left after the pools, so the pools keep their size.

## Prefill

48 of the 64 layers are gated delta net (GDN) layers. `--linear-attn-prefill-backend flashinfer`
selects the FlashInfer GDN prefill kernel, which SGLang selects by default only on SM100. On an
NVFP4 checkpoint of this model, the kernel raises prefill at 9k and 31k tokens from 5,195 and 4,923
tok/s to 5,900 and 5,180. With it, a turn served from the radix cache and the same turn computed
from scratch produce identical greedy text. The kernel needs CUDA 13 and prefill chunks of at most
8,192 tokens.

## Concurrency

Every running request holds its own recurrent states, which share memory with the KV pool. With
ReplaySSM a request needs five fp32 recurrent states, three of its own and two for prefix-cache
tracking. `--max-mamba-cache-size` counts these states and must be at least five times
`--max-running-requests`. We serve four slots and set 22 = 5 × 4 + 2, keeping two in reserve, since
the engine stops when every state is taken and locked (sglang#43204). The 22 states take 3.34 GiB.
The pool itself is shared: one request can use all of it while the other slots are idle, and when
long requests compete, SGLang delays one or evicts the cached prefix of another. Agent runs on this
deployment reach 115k tokens of context at the median and 145k at the 90th percentile, so the pool
holds a long run beside shorter requests while a single request keeps the full 262k context.

We serve four slots because the clients of this deployment send at most four requests at once, and
because four slots fit with fp32 recurrent state and the full context. Eight slots with fp32 state
leave a 203,338-token pool at `--mem-fraction-static 0.80`, below the 262,144-token
context. Eight slots fit only with bfloat16 state, which changes greedy text (see
[the tuning study](#tuning-study-2026-10-23)). With fp32 state the four-slot configuration returns
the same greedy text as the two-slot one on 104 of 104 prompts, so the slot count, the batch-size-4
CUDA graphs and the retiled draft head are bit-neutral at one stream. `containerConcurrency` is 8,
two requests per slot as before, so that requests five to eight queue inside SGLang and Cloud Run
returns 429 only beyond eight.

The extra slots cost nothing at one or two streams and raise the total once more than two requests
run. `scripts/bench3.py`,
medians of five runs on test services (two slots: seven runs):

| Streams | Two slots, per stream | Two slots, total | Four slots, per stream | Four slots, total |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 140.8 tok/s | 121.7 tok/s | 140.6 tok/s | 121.4 tok/s |
| 2 | 133.3 | 210.1 | 133.3 | 209.6 |
| 4 | 132.3 (two run, two wait) | 217.4 | 120.7 | **348.3** |
| 8 | HTTP 429 (`containerConcurrency` 4) | | 120.3 (four run, four wait) | 360.3 |

The total at one stream is below the per-stream rate because it includes the time to first token.
On production, four concurrent 800-token prose essays with thinking off took 13.8 to 13.9 s on the
four-slot revision, 230.1 and 232.1 tok/s in total over two rounds, against 22 to 24 s, 133.3 and
143.0 tok/s, on the two-slot revision.

DFlash 2 has an open report of wrong answers, and of prompts appearing in other conversations, under
concurrent load (sglang#36548). `scripts/qcon.py` runs 64 problems with checkable answers,
three-step arithmetic and four-person ordering puzzles, greedy with thinking on, once alone and once
at two streams interleaved with unrelated essay and code requests. All 64 answers are right in both
runs, and none contains a word from the other requests. Of the 64 answers, 29 are byte-identical
across the two runs, because a verify batch of a different size rounds differently. On the
four-slot configuration at four streams, all 64 answers are again right, none contains a word from
the other requests, and 21 are byte-identical to the serial run.

SGLang schedules a waiting prefill ahead of decode, so a long prompt that arrives while the other
slot is answering stalls that answer. `--prefill-decode-interval N` runs decode steps after each
prefill chunk. The following table measures a 124k-token prompt that arrives beside a running
answer:

| | Long prompt, time to first token | Other stream, longest gap |
| --- | ---: | ---: |
| no interval | 70.0 s | 67.96 s |
| `--prefill-decode-interval 16` | 83.5 s | 2.87 s |
| **`--prefill-decode-interval 40`** | 84.3 s | **1.97 s** |

Interval 40 fits about a dozen decode steps between two 4,096-token chunks. The long prompt then
takes about 20% longer to its first token, and prefill on an idle server is unchanged.

## Memory

The Prometheus gauges `weight_memory_usage_gb` and `kv_cache_memory_usage_gb` count only the
target, and the recurrent state has no gauge, so we account for memory from the startup log of the
production revision:

| Step | GiB | Available after | Log line |
| --- | ---: | ---: | --- |
| Card | 47.64 | | ½ of 96 |
| CUDA context and driver reserve | -2.40 | **45.24** | `Load weight begin. avail mem=45.24` |
| Target weights | -19.12 | **26.13** | `Load weight end ... mem usage=19.12` |
| Drafter weights, FP8 | -2.53 | **23.56** | `type=DFlash2DraftModel, avail mem=23.56, mem usage=2.53` |
| Recurrent state, 22 states, fp32 | -3.34 | | `ssm_state size: 3.23GB`, `conv_state size: 0.06GB`, `intermediate_conv_window_cache size: 0.05GB` |
| KV pool, target (16 full-attention layers) | -8.12 | **12.23** | `K size: 4.06 GB, V size: 4.06 GB` |
| KV pool, drafter (5 layers) | -2.54 | **9.69** | `K size: 1.27 GB, V size: 1.27 GB` |
| Autotune and prefill CUDA graph | -2.74 | 6.95 | `Capture target prefill CUDA graph end ... avail mem=6.95` |
| Target verify CUDA graphs, batch 1 to 4 | -0.04 | 6.91 | `Capture target verify CUDA graph end. bs=[1, 2, 3, 4] ... avail mem=6.91` |
| Draft head, int8 | -1.19 | 5.72 | `DFLASH quantized draft head: int8, 248320 x 5120, 1.27 GB` |
| Draft verify CUDA graphs, batch 1 to 4 | -0.08 | 5.64 | `Capture draft verify CUDA graph end ... avail mem=5.64` |
| Rest of startup | -0.15 | **5.49** | `available_gpu_mem=5.49` |

The recurrent state and the target pool together take 11.33 GiB of the 23.56, 0.13 less than their
logged sizes add up to.

The model has `head_dim` 256 and 4 KV heads, so a token of FP8 KV takes `2 × 4 × 256 = 2 KiB` per
layer: 32 KiB over the 16 full-attention layers of the target and 10 KiB over the 5 layers of the
drafter. A token of context therefore costs 42 KiB, about 30% more than without speculation, and
the pools match this exactly: 266,083 tokens at 32 KiB is 8.12 GiB, and at 10 KiB 2.54 GiB.

The two-slot configuration had a 314,253-token pool: twelve recurrent states took 1.90 GiB, and
`--mem-fraction-static 0.80` claimed 0.48 GiB more. We lowered the fraction to 0.79 with the move to
more slots. At 0.80 the four-slot configuration left 5.00 GB after startup, and an eight-slot
configuration with bfloat16 state 4.67 GB. On both, one `/generate` request with full-prompt top-20 input logprobs
(`scripts/fid.py`) asked for 1.89 GiB, hit CUDA OOM and killed the server, while the two-slot
configuration, with 5.21 GB left, survived it. At 0.79 the eight-slot configuration left 5.17 GB
and survived the same request. The four-slot production revision now leaves 5.49 GiB; we did not repeat the
request on it.

The usable context is `max_req_input_len`, 262,138 here, since the pool is larger than the
262,144-token context of the model. Clients take their context window from it and read it again
after any change to `--mem-fraction-static`, `--max-running-requests`, `--max-mamba-cache-size`,
the checkpoint, the drafter or the SGLang version.

## Vision

The fast Qwen2-VL image processor runs in the API process and allocates GPU memory outside the
static budget of the engine, in proportion to the image. We leave room for it with
`--mem-fraction-static 0.79` and cap the image size, so that the allocation stays bounded. The
processor resizes larger images to the cap and keeps their aspect ratio. With patch size 16 and
merge size 2, a token covers 32x32 pixels, so a cap of 1,048,576 pixels is exactly 1,024 image
tokens. Preprocessing on the GPU takes 0.52 s for a 3000x2000 image, against 0.94 s on the CPU.

The cap is set through `size`, where `longest_edge` is an area in pixels:

    --mm-process-config '{"image":{"size":{"longest_edge":1048576,"shortest_edge":65536}}}'

The `preprocessor_config.json` of the model has no `max_pixels` key, and SGLang ignores a
`max_pixels` setting although `/get_server_info` echoes it. The cap leaves smaller images
unchanged:

| Image | Default | Capped | Latency |
| --- | --- | --- | --- |
| 512x512 | 256 tok | 256 | 0.55 s |
| 1024x1024 | 1,024 | 1,024 | 0.65 s |
| 2048x2048 | 4,096 | **1,024** | 0.63 s |
| 3000x2000 | 5,828 | **1,014** | 0.52 s |

`scripts/apitest.py` tests with a 1-megapixel image, since small images pass under any setting.

The move to four slots leaves image answers unchanged: four real photos are recognized alone and
eight of eight at four concurrent requests, and on production both revisions give the same answer
to the same test image. That image reads "ZEBRA 7341" and both revisions answer "ZEBRA 741", an
error of the model that the change neither introduced nor fixed.

## Tool calls

The `qwen3_coder` parser of SGLang converts every parameter to the type in the schema of the tool
and therefore holds each parameter until its `</parameter>` tag. A call that writes a file sends
its name and then nothing until the last line of the file is generated. Agent clients put an idle
timeout on the stream, five minutes by default in pi, and drop such a call to retry it from the
start.

We serve the parser of [sglang PR #25246](https://github.com/sgl-project/sglang/pull/25246), ported
to this SGLang version in `docker/`. It streams string parameters as they are generated and converts
the others when they close. `scripts/toolstream.py` asks for one `write` of the integers 1 to 3,000,
a call with 16,931 characters of arguments. The following table compares the two parsers on it:

| Parser | Pieces | Longest silence | Call |
| --- | ---: | ---: | ---: |
| stock | 4 | 64.2 s | 64.2 s |
| **streaming** | **1,744** | **0.1 s** | 64.0 s |

The numbers hold on both the Anthropic and the OpenAI endpoint, and on both the assembled arguments
equal the requested content. The port decides what to stream with the type inference of the
installed parser, so that a typed `anyOf` parameter is converted and not streamed as a string. It
also keeps two behaviours of the non-streaming path: `null` becomes JSON null for every type, and
only the first newline after the tag is dropped, however the value is chunked. `docker/test_stream.py`
checks that content streams before its closing tag and that every streamed call equals the
non-streaming parse of the same text, for eight calls at chunk sizes from 1 to 4,096 characters.
The image build runs it on the stock parser, which must fail, and on the port, together with the
Qwen3 Coder tests of SGLang.

## Structured output

With DFlash 2, a request that carries `response_format` (JSON schema) or `tool_choice: "required"`
crashes the scheduler with *NCCL error ... unhandled cuda error*, the failure class of sglang#24283
for EAGLE and NEXTN. A client that retries on a 500 then crashes every new instance as soon as it
is ready. `--grammar-backend none` answers such a request with a 400, *Grammar-based generation
(json_schema, regex, ebnf, structural_tag) is not supported*, and the server stays up. Ordinary tool
calls do not use the grammar, and clients that need JSON ask for it in the prompt and validate the
answer.

## Flags

`--attention-backend flashinfer` is required on SM120, which has no `trtllm_mha`.
`--kv-cache-dtype fp8_e4m3` is the lowest KV precision SGLang supports for a hybrid GDN model.
`--tool-call-parser qwen3_coder` parses the `<tool_call><function=…>` format of this chat template,
which the `hermes` parser drops. `--speculative-draft-attention-backend flashinfer` gives the
drafter the attention kernels of the target. `--enable-cache-report` reports cache hits to clients,
as `cache_read_input_tokens` on `/v1/messages` and `prompt_tokens_details` on chat completions.
`--cuda-graph-max-bs-decode 4` captures the decode and verify graphs for batch sizes 1 to 4, one
per slot count. `SGLANG_DFLASH_DRAFT_HEAD_CFG=128,128,8,2` sets the tile of the int8 draft-head
kernel (BN, BK, warps, stages): above 32 draft rows per verify batch, slots times block, the kernel
tiles 64 rows, and its default tile `128,256,8,3` then needs 131,072 bytes of shared memory against
the 101,376 of this partition, so the server failed at startup with Triton `OutOfResources`. The
retiled kernel needs 98,304 bytes. Four slots of block 8 make 32 rows, so this revision does not
need it; it keeps the setting of the configuration it was measured with.
DFlash 2 is a pure-attention drafter of five sliding-window
layers without recurrent state, so the `--mamba-*` and `--linear-attn-verify-backend` flags of the
DSpark recipe do not apply to it.

## Serving

`cloudrun/service.yaml` is the deployed configuration. Cloud Run requires the container to listen on
8080 and `gpu-zonal-redundancy-disabled: 'true'` for the deployment to complete. With `minScale:
'1'` and `maxScale: '1'`, one instance stays warm, and a cold start occurs only when a revision is
replaced. `containerConcurrency: 8` admits two requests per slot, so that the excess queues inside
SGLang and Cloud Run returns 429 only beyond eight. `timeoutSeconds: 3600`, the maximum, lets an
agent turn that writes a long file in one tool call run for up to an hour. The service declares the
`ALPHA` launch stage, without which Cloud Run refuses the 64 GiB memory limit.

The production service is `qwen38-exp-e` in us-central1. Since 2026-10-23 all its traffic goes to
the four-slot revision `qwen38-exp-e-s4b8-fp32`, which this file describes. The previous revision,
`qwen38-exp-e-qhead-h2`, is the two-slot configuration of commit `ed0cb7e`, with
`--max-running-requests 2`, `--max-mamba-cache-size 12`, `--mem-fraction-static 0.80`,
`containerConcurrency: 4` and an unpinned `google-cloud-storage`. Both run the same image, weights
and drafter.

The weights come from a bucket in the same region (`BUCKET` in both yaml files).
`quant/service.yaml` writes the quantized checkpoint to `huihui-awq/out/`, the drafter is copied
once to `dflash2/`, and `cloudrun/gcsdl.py` sits at the bucket root. It retries a failed blob and
falls back to a single-stream copy, since one dead worker of a chunked transfer used to take the
download and the container down. The container installs `google-cloud-storage==3.17.1` before
running it; an unpinned install crash-looped a container on 2026-10-23. The service accounts of both
services need object access on the bucket. The quantization run needs a Hugging Face token, since
anonymous downloads from Cloud Run are rate-limited and `snapshot_download` then returns a partial
directory without an error; `quant/job.sh` checks the shard sizes.

Cloud Run speaks HTTP/2 to the container (port name `h2c` and `--enable-http2`, served by Granian),
whatever protocol the client uses. A client that disconnects reaches SGLang as a stream reset, and
SGLang aborts the request within a second (`scripts/disconnect.py`). Over HTTP/1.1 the front end
keeps the backend request alive until the answer completes, so a stopped agent run would hold its
slot; decode speed is the same under both servers. SGLang returns 400 when the prompt plus
`max_tokens` exceeds the context, so clients cap the completion length below the full window.

## Cold start

From instance start to a ready server takes 824 s on the production revision, according to the
timestamps of its startup log on 2026-10-23:

| Phase | Seconds |
| --- | ---: |
| Instance start to download start (`pip install google-cloud-storage==3.17.1`) | 13 |
| Target and drafter weights from the bucket, 24.9 GB (`cloudrun/gcsdl.py`) | 471 |
| SGLang start to weight loading | 93 |
| Target weights into the GPU | 16 |
| Drafter weights into the GPU | 5 |
| Recurrent state and KV pools, autotune, CUDA graphs, warmup | 225 |

`gcsdl.py` copies both models to local disk with parallel ranged downloads of the storage client,
after which SGLang loads the target in 16 seconds. Loading the target through the gcsfuse mount
takes 849 s, because safetensors reads through the mount in small pieces.

## Tuning study, 2026-10-23

We measured five configurations on test services with the same image digest, weights and drafter as
production, in the same region, from one client over the internet. Each got one warm-up round, then
five runs of `bench2.py` and five of `bench3.py`; the eight-slot configuration at 0.79 and 2 × 16
with torch-compile got three of each. The table gives medians. Token-weighted is every token after
the first over all decode time in the one-stream round of `bench3.py`, which a gain on short answers
alone cannot move. Totals are as in [Concurrency](#concurrency), and the last column counts greedy
answers byte-identical to the two-slot configuration out of 104 prompts of 256 tokens.

| Slots × block, state | bench2 | Token-weighted | Total, 2 streams | Total, 4 streams | Total, 8 streams | KV pool | Same greedy text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 × 8, fp32, mem 0.80 (former) | 140.8 | 124.7 | 210.1 | 217.4 | HTTP 429 | 314,367 | 104/104 (rerun) |
| **4 × 8, fp32, mem 0.80 (deployed at 0.79)** | **140.8** | 124.6 | 209.6 | **348.3** | 360.3 | 277,281 | **104/104** |
| 8 × 8, bf16, mem 0.79 (option) | 141.4 | 125.1 | 219.4 | 367.6 | 584.6 | 267,509 | 41/104 |
| 2 × 16, fp32, mem 0.80 | 153.4 | 124.5 | 208.5 | 211.8 | 210.6 (queued) | 314,253 | |
| 2 × 16, fp32, torch-compile | 153.1 | 125.3 | 206.5 | 209.1 | 216.0 (queued) | | |

On the production revision, `bench2.py` gives 140.5 tok/s (prose 89.2, chat 98.1, translation
99.3, summary 118.0, code 193.4, math 180.8, JSON 204.6), and 80.7 tok/s for prose with thinking on
and sampling. The server reports `max_running_requests` 4, `mamba_ssm_dtype` float32 and context
262,144, and parallel tool calls work. Sampled with thinking on (`accept.py`, 42 answers), the
configurations reach 124.2 tok/s at 3.63 tokens per verification (two slots), 126.2 at 3.60 (four
slots), 130.1 at 3.68 (eight slots, bf16, at 0.80), 133.9 at 3.75 (2 × 16) and 134.4 at 3.76
(2 × 16 with torch-compile).

**Four slots, fp32 (deployed).** One stream is unchanged and the total at four streams is 60% higher.
Greedy text equals the two-slot configuration on 104 of 104 prompts. All 64 `qcon.py` answers are
right at four streams, with no words from the other requests, and four photos are recognized alone
and eight of eight at four concurrent requests.

**Eight slots, bfloat16 state (option, not deployed).** Eight slots of block 8 with
`--mamba-ssm-dtype bfloat16`, `--max-mamba-cache-size 42`, `--cuda-graph-max-bs-decode 8`, the
retiled draft head and `--mem-fraction-static 0.79` reach 584.6 tok/s in total at eight streams and
keep the full context (a 261,074-token prompt is served, a 269,179-token one is refused with 400, as
before). Every quality check equals the two-slot configuration: needle 5/5 at 30k and 120k, math
12/12, refusals 0/6, `qcon.py` 64/64 at eight streams, GSM8K 193 of the first 200 test problems,
no loops in eight greedy and 24 sampled 4,000-token answers, vision 16/16 at eight concurrent
requests, and `apitest.py`. Greedy text changes, though: 41 of 104 answers are byte-identical, where
the two-slot configuration reproduces itself 104 of 104 times. Teacher-forced over 191,301 positions
of 46 documents, the KL to fp32 state is 0.0015 (Chinese 0.0056, code 0.0000), top-1 agreement is
99.69% and perplexity is equal, 3.722; the AWQ quantization itself costs KL 0.047. The divergence
comes from the bfloat16 state alone: eight slots with fp32 state give 104/104. We did not deploy
this option because it changes output and the clients of this deployment send at most four requests
at once, where four fp32 slots reach 348.3 against 367.6.

**2 × 16.** Block 16 raises `bench2.py` by 9%, but this is an artefact of its workload mix: code,
math and JSON decode 15 to 30% faster, prose, chat and summary 5 to 7% slower, and token-weighted
decode is equal (124.5 against 124.7). It helps sampled thinking-on decode and gives up all
multi-slot throughput. torch-compile adds nothing at two slots, and at eight slots its CUDA-graph
capture took about six minutes per batch size, beyond the 24-minute startup probe.

The same study found the shared-memory failure that kept every configuration with more than 32
draft rows from starting (see [Flags](#flags)) and the out-of-memory failure behind
`--mem-fraction-static 0.79` (see [Memory](#memory)). Dead ends, measured: two-batch overlap and an
int4 draft head crash with DFLASH at startup; the DP path selector lowers acceptance from 4.06 to
2.48 tokens per verification; 24 or 32 slots of block 2; fine-tuned drafters of block 8 and 16; and
the BF16 drafter.

An earlier revision of this README (commits `ed655c3` to `444a42a`) stated 840 to 898 tok/s at eight
streams and an eight-slot deployment. The figure was the single-stream rate times eight, not a
measurement, and eight slots were never deployed. The measured total at eight streams is 584.6
tok/s for the eight-slot option and 360.3 for the deployed four slots.
[`REPORT-2026-10-23.md`](REPORT-2026-10-23.md) lists every run of the study.

## Reproduction

Each script takes the service URL and reads the API key from `SGLANG_API_KEY`:

```bash
export SGLANG_API_KEY=...
U=https://YOUR-SERVICE.run.app
python3 scripts/bench2.py  $U                # seven workloads, streamed decode, one stream
python3 scripts/bench3.py  $U out.json 1 2 4 8   # the same prompts at 1, 2, 4 and 8 concurrent streams
python3 scripts/accept.py  $U 2              # tokens per verification and decode, sampled, thinking on
python3 scripts/longdecode.py $U             # decode at ~34k and ~115k tokens of context
python3 scripts/prefill.py $U                # time to first token at ~9k, ~31k, ~124k
python3 scripts/stall.py   $U 1 40000        # one stream decoding while a ~124k prompt arrives
python3 scripts/admission.py $U              # a short request beside a long prefill
python3 scripts/quality.py $U                # needle at 30k/120k, 12 math problems
python3 scripts/qcon.py    $U out.json 4     # 64 checked answers alone and at 4 streams
python3 scripts/prefix.py  $U                # facts recalled across a conversation served from cache
python3 scripts/degen.py   $U                # 24 long sampled answers checked for loops
python3 scripts/drift.py run $U a.json       # greedy answers, to compare two builds (drift.py cmp)
python3 scripts/refusal.py $U [$U2]          # six borderline prompts, refusals per server
python3 scripts/apitest.py $U image.png      # structured output, tool calls, cache report, 1 MP image
python3 scripts/toolstream.py $U 3000        # streaming of the arguments of a long tool call
python3 scripts/disconnect.py $U             # slot release after a client disconnects
python3 scripts/fid.py dump $U corpus.json out.json.gz   # then: fid.py cmp ref.json.gz out.json.gz
```

## Credit

The serving recipe follows
[`MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
which established the flags and the memory budget for the full card. DFlash 2 is
[z-lab/dflash](https://github.com/z-lab/dflash), with the drafter
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The weights are
[`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
quantized with the recipe and calibration set of
[`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) through
[llm-compressor](https://github.com/vllm-project/llm-compressor).
[Discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192) compares the published
4-bit checkpoints. The L4 companion of this deployment is
[`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
