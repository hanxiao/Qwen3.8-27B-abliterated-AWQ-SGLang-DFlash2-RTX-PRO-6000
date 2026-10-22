# Qwen3.8-27B abliterated · AWQ-INT4 · SGLang + DFlash 2 · ½ NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), quantized here with the AWQ-INT4 recipe of [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) (`quant/`), served as `Qwen3.8-27B-abliterated-AWQ-INT4` |
| Weights | INT4 W4A16 (AWQ, group 32, asymmetric) with a dense BF16 `lm_head`, 19.12 GiB resident |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), quantized to FP8 at load (2.53 GiB), blocks of 8 tokens |
| Engine | SGLang v0.5.22, the `lmsysorg/sglang:v0.5.22` image with four patches (`docker/`), FlashInfer attention and GDN prefill kernels |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell (47.64 GiB, SM120) on Cloud Run |
| Slots | 2 concurrent requests |
| Context | **262,144** native and usable per request; one 314,365-token FP8 KV pool shared by both slots |
| Decode, one stream | **133.9 tok/s** (prose 85.4, chat 92.6, translation 94.7, summary 112.0, code 184.0, math 173.7, JSON 194.7) |
| Decode, two streams | 127.2 tok/s each, 202.8 tok/s in total |
| Decode, thinking on, sampled | 125.1 tok/s over fourteen prompts, 3.70 tokens per verification |
| Decode deep in a context | 68.8 tok/s at 34k tokens, 61.3 at 115k |
| Prefill | 2,830 tok/s at 9k tokens, 2,622 at 31k, 1,794 at 124k |
| Refusals | 0 of 6 borderline prompts, against 5 of 6 for the stock model |
| Quality | needle retrieval 5/5 at 30k and 120k, multi-step math 12/12, 64/64 checked answers at two concurrent streams |
| Vision | native, preprocessed on the GPU, at most 1,024 tokens per image |
| Structured output | disabled: `response_format` and `tool_choice: "required"` return 400 |
| Tool calls | arguments stream as they are generated, at most 0.1 s apart |
| Cold start | 12 min; the service keeps one instance warm |

We time decode with `scripts/bench2.py`, which streams each answer and divides its tokens after
the first by the time from first to last token. Each of the seven workloads runs two prompts of up
to 800 tokens, greedy with thinking off unless stated. `scripts/bench3.py` sends the same fourteen
prompts over two connections at once. Prefill is time to first token on prompts of unique random
words, so no prefix is cached.

```bash
bash docker/build.sh                                  # lmsysorg/sglang:v0.5.22 + four patches, into Artifact Registry
gcloud alpha run services replace quant/service.yaml  # once: quantize into the bucket (~75 min)
gcloud alpha run services replace cloudrun/service.yaml
```

## GPU

The fractional RTX PRO 6000 is a Cloud Run offering, separate from the G4 machine types of Compute
Engine. Each has its own quota: `run.googleapis.com` grants milliGPUs, of which a half card uses
500, and `compute.googleapis.com` grants whole `NVIDIA_RTX_PRO_6000` cards. A project can hold a
Cloud Run grant with no Compute Engine quota at all. Cloud Run accepts 0.5 or 1 for
`nvidia.com/gpu`. The half card presents 47.64 GiB and takes 10 to 16 vCPU with 40 to 64 GiB of
system memory.

The half card is a MIG `2g.48gb` partition: 94 of the 188 SMs, 765 GB/s of measured read bandwidth
(1,524 GB/s on the whole card) and 177 dense BF16 TFLOPS (412). A decode step reads about 23 GB of
weights, the target's 20.5 and the drafter's 2.7, so bandwidth alone sets 30 ms per step. Decode
therefore runs close to its limit, and gains come from committing more tokens per step or reading
fewer bytes per step. Prefill at 9k tokens computes about 145 TFLOP/s, 82% of the BF16 peak.

## Checkpoint

The deployment serves the abliterated Qwen3.8-27B of huihui-ai, in which the residual-stream
direction that mediates refusals is projected out of the weights. `scripts/refusal.py` sends six
borderline prompts (a profane roast, lock picking, a paywall question and similar) with thinking
off. The stock model refuses five and the abliterated one none.

DFlash 2 selects its draft candidates by a top-k over the target logits, so the `lm_head` of the
target must stay dense; SGLang rejects a packed FP4 head. To choose the format of the body, we
served three 4-bit checkpoints of the stock model with a BF16 head and compared each against the
official FP8 checkpoint, whose own median per-position KL to BF16 on text is 0.0117
([discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)). `scripts/fid.py`
feeds the same text through each server, teacher-forced, and records the top-20 distribution at
every position. The text is 91,769 positions from 55 documents of the traffic this deployment
serves: agent answers, code, technical documents and Chinese. It is private, and the script takes
any corpus.

| Checkpoint | PPL | KL answers | KL code | KL docs | KL Chinese | KL all | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.8-27B-FP8` (reference) | 7.874 | | | | | | |
| **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** | 7.979 | **0.0137** | **0.0126** | **0.0181** | **0.0935** | **0.0466** | **91.74%** |
| `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` | 8.062 | 0.0289 | 0.0274 | 0.0381 | 0.1330 | 0.0724 | 89.26% |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | **7.902** | 0.0383 | 0.0328 | 0.0492 | 0.1526 | 0.0853 | 88.33% |

The AWQ checkpoint is closest to the reference on every slice, with about half the divergence of
the RadixArk NVFP4 checkpoint everywhere except Chinese. QUASAR predicts this text well, with the
lowest perplexity, but its distribution is the furthest from the reference. AWQ is also smaller
and faster to decode, measured on the same card with the same flags and one stream:

| | NVFP4 (RadixArk) | AWQ-INT4 | NVFP4 (QUASAR) |
| --- | ---: | ---: | ---: |
| Weights resident | 21.84 GiB | **19.12 GiB** | |
| KV pool, one slot | 180,573 | 248,597 | **256,218** |
| Decode, mean of seven | 107.9 | **119.2** | 116.4 |
| Prefill at 9k / 31k / 124k | **6,045 / 5,247 / 2,659** | 2,751 / 2,622 / 1,784 | |

W4A16 runs its matrix multiplications on BF16 tensor cores (Marlin) and NVFP4 on FP4 tensor cores,
so compute-bound prefill is about half as fast. Decode reads the weights, and AWQ has fewer bytes
to read. Over 143 agent runs on this deployment, a turn adds about 2,300 new prompt tokens and
generates 1,223, so decode takes over 90% of GPU time and the decode gain outweighs the prefill
cost.

The two published 4-bit builds of the abliterated model have either less context or no vision
tower, measured here with a BF16 drafter:

| | Format | Weights | KV pool, two slots | Decode, mean of seven | Vision |
| --- | --- | ---: | ---: | ---: | --- |
| `TelperionAI/Huihui-Qwen3.8-27B-abliterated-INT4-AWQ-GPTQ` | INT4 MLP, INT8 attention and last 8 MLPs | 25.1 GB | 110,391 | 105.1 | yes |
| `hotdogs/Qwen3.8-27B-abliterated-AWQ-INT4` | INT4 symmetric, group 128 | 17.6 GB | | | no |
| ours | INT4 asymmetric, group 32 | 21.0 GB | **202,065** | **118.0** | yes |

We therefore quantize the BF16 abliterated checkpoint with the
cyankiwi recipe, which we read from the `config.json` of their checkpoint: AWQ, INT4, asymmetric,
group 32, the `mse` observer, and every `Linear` layer except `lm_head`, the vision tower, the MTP
head, `linear_attn.in_proj_a/b` and the `linear_attn.out_proj` of layer 0. Calibration uses their
public `calibration` set, 384 agentic and STEM conversations in ten languages truncated to 2,048
tokens, 380,221 tokens in total. The output has the same 2,396 tensors as the cyankiwi checkpoint,
with the same names and formats.

`quant/quant.py` runs the quantization with llm-compressor 0.14 and its hybrid-attention AWQ
mappings on a whole RTX PRO 6000 (`quant/service.yaml`). Calibration takes 64 minutes and 71 with
the save. The 55 GB model stays in CPU memory and llm-compressor moves one layer at a time to the
GPU, since the whole model and the calibration activations do not fit on the 96 GB card together.
The run is a Cloud Run service, because Cloud Run limits GPU jobs to one hour.

The fidelity cost of abliteration is small. Against the FP8 reference of the stock model, the
abliterated checkpoint has KL 0.0544 overall and 0.1067 on Chinese, against 0.0466 and 0.0935 for
the stock AWQ checkpoint, and a perplexity of 7.966 against 7.979. These numbers include the
abliteration itself as well as the quantization error.

## Speculative decoding

Qwen3.8-27B ships an MTP head, which drafts one token per pass. Measured with vLLM on an NVFP4
checkpoint of the same model, a deeper MTP draft raises the accepted length and stops raising
throughput at five tokens:

| MTP draft tokens | Accepted length | Decode |
| --- | --- | --- |
| 2 | 2.58 | 61.2 tok/s |
| 3 | 3.22 | 66.1 |
| 5 | 3.78 | 68.0 |
| 7 | 4.07 | 67.5 |

Each MTP draft token costs one sequential pass of the draft head, so past five tokens the extra
passes cost as much time as the extra acceptance saves. DFlash 2 is a block-diffusion drafter that
predicts a block of eight tokens in a single pass. It keeps the top candidates at every position,
and a lightweight selector traces one path through them. Two-tap dynamic convolutions in its
backbone keep the quality of the draft from decaying toward the end of the block. The target
verifies every drafted token, so the output follows the target model and sampling keeps its
distribution.

The drafter is quantized to FP8 when it loads (`--speculative-draft-model-quantization fp8`). Its
weights shrink from 3.72 to 2.53 GiB, which the drafter reads on every pass, and the freed memory
goes to the KV pool. We measured both drafters with the abliterated target, on two services that
ran in the same hour:

| | BF16 drafter | FP8 drafter |
| --- | ---: | ---: |
| Decode, mean of seven, greedy | 116.8 | **124.1 / 123.7** |
| Decode, thinking on, sampled (prose, chat) | 64.7, 79.6 | **71.6, 80.0 / 74.2, 84.3** |
| KV pool, two slots | 202,065 | **237,271** |

The two FP8 values in each row come from two consecutive runs. With the FP8 drafter, the abliterated model decodes
faster than the stock AWQ checkpoint with the BF16 drafter, which reaches 120.0 tok/s greedy,
68.3 and 77.8 sampled, and 178.5 tok/s in total at two streams.

We also checked long sampled answers of the FP8 drafter for repetition. We sent 24 prompts in
English and Chinese to both services with thinking on, the thinking returned inline, sampling and
up to 4,000 tokens. The mean zlib compression ratio of the answers is 0.391 with the BF16 drafter and 0.399 with FP8, the same
13 answers reach the length limit on both, and no answer falls into a loop.

### Verification

Three changes to how drafts are verified raise decode without changing the distribution the model
samples from.

**Block verification** (`--speculative-use-block-verification`, [Sun et al.,
2024](https://arxiv.org/abs/2403.10444)). Token-wise verification stops at the first rejected draft
token; block verification can still accept a longer prefix, with a correction that keeps the
target distribution. It acts on sampled requests only.

**Draft probabilities that round above 1** (`docker/draft_prob_round.py`, sglang PR #45819). The
sampled verification kernel treats a draft token whose probability exceeds 1.0 as impossible and
rejects it. An fp32 softmax over a peaked row returns 1.0 plus a few units in the last place, so the
drafts the drafter was surest of were always rejected. The kernel now accepts probabilities up to
1 + 10⁻³ and clamps them to 1.

**ReplaySSM** (`--enable-linear-replayssm-spec`, [arXiv 2610.13778](https://arxiv.org/abs/2610.13778)).
To roll back a rejected draft, verification stores the recurrent state of all 48 GDN layers at
each of the 8 drafted positions, 1.2 GB per request written on every step. ReplaySSM stores compact
records of the recurrence inputs instead and replays the accepted prefix into the checkpoint.
SGLang v0.5.22 runs it for MTP drafts; `docker/replayssm_dflash.py` routes the DFlash commit through
it (from sglang PR #35725). It frees the 3.38 GB intermediate-state cache and the state writes of
every step.

Each column adds to the one before it. Sampled decode is fourteen prompts with thinking on
(`scripts/accept.py`), whose tokens per verification come from the server's counters; greedy is
`scripts/bench2.py`. Sampled decode varies by a few percent between runs: the ReplaySSM column is
the mean of 120.9 and 116.0, while the counters agreed to 3.555 and 3.542. The rounding fix was
measured in the same hour as the 116.0 run.

| | v0.5.20 | v0.5.22, block verification | + ReplaySSM | + rounding fix, 12 state slots |
| --- | ---: | ---: | ---: | ---: |
| Tokens per verification, sampled | 3.39 | 3.56 | 3.55 | **3.70** |
| Decode, sampled | 108.3 | 112.8 | 118.5 | **125.1** |
| Decode, greedy, mean of seven | 123.7 | 124.6 | 133.9 | **133.9** |
| KV pool, two slots | 237,271 | 237,384 | 321,526 | 314,365 |

ReplaySSM rounds the replayed states differently from stored ones. Greedy answers to 24 prompts
(`scripts/drift.py`) are identical when one server answers them twice, 24 of 24, and 10 of 24 match
between verification with and without ReplaySSM; SGLang v0.5.20 and v0.5.22 match on 9. The quality
checks do not change: needle retrieval 5/5 at 30k and 120k, math 12/12, 64/64 checked answers at two
streams, ten facts recalled across a conversation served from the prefix cache (`scripts/prefix.py`),
and 24 long sampled answers with a mean zlib ratio of 0.417 and no loop (`scripts/degen.py`).

Deep in a context the drafter commits fewer tokens per step. With 34k and 115k tokens of random
words before the question (`scripts/longdecode.py`), greedy decode is 66.4 and 57.0 tok/s on v0.5.20
at 2.53 and 2.39 tokens per verification, and 68.8 and 61.3 tok/s with all three changes. Agent
traffic, with real text in its context, commits 3.94 tokens per verification at a median of 115k
tokens.

## Concurrency

Every running request needs its own recurrent states, which come out of the memory that would
otherwise hold the KV pool. With ReplaySSM there are no per-position intermediate states, so a slot
costs five fp32 recurrent states, 0.76 GiB, which equals about 19,000 tokens of pool.
The pool is shared between the slots. A single request can use all of it while the other slot is
idle, and two long requests compete for it, in which case SGLang delays one or evicts the cached
prefix of the other. Agent runs on this deployment reach 115k tokens of context at the median and
145k at the 90th percentile, so two slots let one long run proceed beside shorter requests while a
single request keeps its full 262k tokens of context.

| Streams | Per stream | Total |
| ---: | ---: | ---: |
| 1 | 133.9 tok/s | 133.9 tok/s |
| 2 | 127.2 tok/s | 202.8 tok/s |

DFlash 2 has an open report of wrong answers, and of prompts appearing in other conversations,
under concurrent load (sglang#36548). `scripts/qcon.py` runs 64 problems with checkable answers,
three-step arithmetic and four-person ordering puzzles, greedy with thinking on. It runs them once
alone and once at two streams, interleaved with unrelated essay and code requests. All 64 answers
are right in both runs, and no answer contains a word from the interleaved requests. The answer
text differs between the runs, with 31 of 64 byte-identical, because a verify batch of a different
size rounds differently.

SGLang schedules a waiting prefill ahead of decode, so a long prompt that arrives while the other
slot is answering stalls that answer. `--prefill-decode-interval N` runs decode steps after each
prefill chunk:

| | Long prompt, time to first token | Other stream, longest gap |
| --- | ---: | ---: |
| no interval | 70.0 s | 67.96 s |
| `--prefill-decode-interval 16` | 83.5 s | 2.87 s |
| **`--prefill-decode-interval 40`** | 84.3 s | **1.97 s** |

The long prompt here has 124k tokens. On an idle server, interval 40 prefills at 2,696, 2,621 and
1,784 tok/s at 9k, 31k and 124k tokens, against 2,751, 2,622 and 1,784 without it. Both intervals
fit about a dozen decode steps between two 4,096-token chunks, so a long prompt that arrives beside
a running answer takes about 20% longer to its first token.

`--max-mamba-cache-size` counts recurrent states, five per request on this model (three, plus two for
prefix-cache tracking), and must be at least five times `--max-running-requests` for the server to
start. Exactly ten leaves none spare, and sglang#43204 stops the engine when every state is taken
and locked, so the service runs twelve.

## Structured output

With DFlash 2, a request carrying `response_format` (JSON schema) or `tool_choice: "required"`
crashes SGLang v0.5.20 within six seconds. The scheduler fails in
`dflash_worker_v2.forward_batch_generation` with *NCCL error ... unhandled cuda error* and the
process exits, the same failure class as sglang#24283 for EAGLE and NEXTN. A client that retries
the same request on a 500 crashes every new instance as soon as it is ready.
`--grammar-backend none` turns such a request into a 400, *Grammar-based generation (json_schema,
regex, ebnf, structural_tag) is not supported*, and the server stays up. Ordinary tool calls do not
use the grammar and parse on both the OpenAI and the Anthropic endpoints. Clients that need JSON
ask for it in the prompt and validate the answer.

## Tool calls

The `qwen3_coder` parser of SGLang (v0.5.20 and v0.5.22 alike) converts every parameter to the type in the tool's
schema, and so holds each parameter until its `</parameter>` tag. A call that writes a file sends
its name and then nothing until the last line of the file is generated. Agent clients put an idle
timeout on the stream, five minutes by default in pi, and drop a longer write, then retry it from
the start with the same result.

We serve the parser of [sglang PR #25246](https://github.com/sgl-project/sglang/pull/25246),
ported to v0.5.20 (`docker/`); the parser file is the same in v0.5.22. It streams string parameters as they are generated and converts
the others when they close, as before. `scripts/toolstream.py` asks for one `write` of the
integers 1 to 3,000, 16,931 characters of arguments:

| Parser | Pieces | Longest silence | Call |
| --- | ---: | ---: | ---: |
| v0.5.20 | 4 | 64.2 s | 64.2 s |
| **streaming** | **1,744** | **0.1 s** | 64.0 s |

The OpenAI endpoint gives the same numbers within one piece and 0.1 s, and on both the assembled
arguments equal the requested content exactly. Generation is as fast as before.

The PR predates v0.5.20 and does not run on it as written. It adds a second `_get_param_type`,
which replaces the one v0.5.20 uses for the final conversion, so every non-streaming parse fails.
The port decides what to stream with v0.5.20's own type inference, so that a typed `anyOf`
parameter is converted rather than streamed as a string. It also keeps two behaviours of the
non-streaming path that the PR changes: `null` becomes JSON null for every type, and only the
first newline after the tag is dropped, however the value is chunked. `docker/test_stream.py`
checks that the content streams before its closing tag and that every streamed call equals the
non-streaming parse of the same text, for eight calls at chunk sizes from 1 to 4,096 characters.
The image build runs it on the stock parser, which must fail, and on the port, together with
SGLang's own Qwen3 Coder tests.

## Prefill

48 of the 64 layers are gated delta net (GDN) layers. SGLang selects the FlashInfer GDN prefill
kernel by default only on SM100, and `--linear-attn-prefill-backend flashinfer` selects it on
SM120. On an NVFP4 checkpoint of the same model it raised prefill at 9k and 31k tokens from 5,195
and 4,923 tok/s to 5,900 and 5,180. With this kernel, a follow-up turn served from the radix cache
and the same turn recomputed from scratch produce identical greedy text. The kernel needs CUDA 13
and prefill chunks of at most 8,192 tokens.

SGLang v0.5.20 includes the fix for sglang#37817. Without it, the DFlash worker does not save the
GDN state at radix-cache checkpoints, and a request that reuses a cached prefix continues from the
wrong recurrent state. Every multi-turn conversation reuses its prefix, so this deployment needs
v0.5.20 or later.

## Flags

`--attention-backend flashinfer` is required on SM120, where `trtllm_mha` is unavailable.
`--kv-cache-dtype fp8_e4m3` is the lowest KV precision SGLang supports for a hybrid GDN model.
`--tool-call-parser qwen3_coder` parses the `<tool_call><function=…>` format of this chat
template; the `hermes` parser drops these calls, and the model then answers nothing.
`--speculative-draft-attention-backend flashinfer` gives the drafter the same attention kernels as
the target. `--enable-cache-report` reports cache hits as `cache_read_input_tokens` on
`/v1/messages` and `prompt_tokens_details` on chat completions; without it, clients see every turn
as uncached. DFlash 2 is a pure-attention drafter of five sliding-window layers and has no
recurrent state, so the remaining `--mamba-*` flags and `--linear-attn-verify-backend` of the
DSpark recipe do not apply.

## Memory

The Prometheus gauges `weight_memory_usage_gb` and `kv_cache_memory_usage_gb` count the target
only, and the recurrent state has no gauge, so we account for memory from the startup log:

| Step | GiB | Available after | Log line |
| --- | ---: | ---: | --- |
| Card | 47.64 | | ½ of 96 |
| CUDA context and driver reserve | -2.40 | **45.24** | `Load weight begin. avail mem=45.24` |
| Target weights | -19.12 | **26.13** | `Load weight end ... mem usage=19.12` |
| Drafter weights, FP8 | -2.53 | **23.56** | `type=DFlash2DraftModel, avail mem=23.56, mem usage=2.53` |
| Recurrent state, twelve slots | -1.90 | | `ssm_state 1.83 + conv_state 0.04 + intermediate_conv_window 0.03` |
| KV pool, target (16 full-attention layers) | -9.60 | **12.06** | `K size: 4.80 GB, V size: 4.80 GB` |
| KV pool, drafter (5 layers) | -3.00 | **9.06** | `K size: 1.50 GB, V size: 1.50 GB` |
| Autotune and prefill CUDA graph | -2.54 | 6.52 | `Capture target prefill CUDA graph end ... avail mem=6.52` |
| Verify CUDA graphs | -0.12 | **6.40** | `available_gpu_mem=6.40` |

The model has `head_dim` 256 and 4 KV heads, so a token of FP8 KV takes `2 × 4 × 256 = 2 KiB` per
layer: 32 KiB over the 16 full-attention layers of the target and 10 KiB over the 5 layers of the
drafter. Each token of context therefore costs 42 KiB, about 30% more than without speculation.
The pools match this exactly: 314,365 tokens at 32 KiB is 9.59 GiB, and at 10 KiB 3.00 GiB.

The usable context is `max_req_input_len`, 262,138 here: the pool is larger than the
`context_length` of 262,144, so the model's own limit applies. Clients should be sized from it and
re-read it after any change to
`--mem-fraction-static`, `--max-running-requests`, `--max-mamba-cache-size`, the checkpoint, the
drafter or the SGLang version.

## Vision

The fast Qwen2-VL image processor runs in the API process and allocates GPU memory outside the
static budget of the engine, in proportion to the image. `--mem-fraction-static 0.80` leaves room
for it, and the image size is capped so that the allocation stays bounded: at 0.90 and 0.85 a
1024x1024 image ran out of memory. The processor resizes large images to the cap and keeps their
aspect ratio. With patch size 16 and merge size 2, a token covers 32x32 pixels, so a cap of
1,048,576 pixels is exactly 1,024 image tokens. Preprocessing on the GPU takes 0.52 s for a
3000x2000 image, against 0.94 s on the CPU.

The cap is set through `size`:

    --mm-process-config '{"image":{"size":{"longest_edge":1048576,"shortest_edge":65536}}}'

The `preprocessor_config.json` of this model carries `size: {longest_edge: 16777216,
shortest_edge: 65536}`, where `longest_edge` is an area in pixels, and has no `max_pixels` key.
SGLang accepts `{"image":{"max_pixels":1048576}}`, echoes it in `/get_server_info`, and ignores it.

| Image | Default | Capped | Latency |
| --- | --- | --- | --- |
| 512x512 | 256 tok | 256 | 0.55 s |
| 1024x1024 | 1,024 | 1,024 | 0.65 s |
| 2048x2048 | 4,096 | **1,024** | 0.63 s |
| 3000x2000 | 5,828 | **1,014** | 0.52 s |

`scripts/apitest.py` tests with a 1-megapixel image, since a small image passes under every
configuration above.

## Serving

`cloudrun/service.yaml` is the deployed configuration. Cloud Run requires the container to listen
on 8080 and `gpu-zonal-redundancy-disabled: 'true'` for the deploy to complete. The service runs
with `minScale: '1'` and `maxScale: '1'`, so one instance stays warm, no request waits for a cold
start, and the cold start is paid only when a revision is replaced. `timeoutSeconds: 3600`, the
maximum, lets one request run for an hour: an agent turn that writes a long file in a single tool
call takes many minutes, and the default of 600 s ended such turns.

The weights come from a bucket in the same region (`BUCKET` in both yaml files).
`quant/service.yaml` writes the quantized checkpoint to `huihui-awq/out/`, the drafter is copied
once to `dflash2/`, and `cloudrun/gcsdl.py` sits at the bucket root. The service accounts of both
services need object access on the bucket. The quantization run needs a Hugging Face token, because
anonymous downloads from Cloud Run are rate-limited and `snapshot_download` then returns a partial
directory without an error; `quant/job.sh` checks the shard sizes.

SGLang returns 400 when the prompt plus `max_tokens` exceeds the context. A client that requests
the whole 262,144-token window as completion is refused, so clients cap the completion length.

## Cold start

From instance start to the first token takes 720 s, from the timestamps of the startup log:

| Phase | Seconds |
| --- | ---: |
| Instance start to download start (`pip install google-cloud-storage`) | 10 |
| Target and drafter weights from the bucket, 24.9 GB (`cloudrun/gcsdl.py`) | 471 |
| Target weights into the GPU | 10 |
| Drafter weights into the GPU | 0.5 |
| Recurrent state and KV pools, autotune, CUDA graphs, warmup | 190 |

`gcsdl.py` copies both models to local disk with parallel ranged downloads of the storage client,
after which SGLang loads the target in ten seconds. Loading the target directly from the gcsfuse
mount takes 849 s, because safetensors reads through the mount in small pieces.

## Reproducing

```bash
export SGLANG_API_KEY=...
U=https://YOUR-SERVICE.run.app
python3 scripts/bench2.py  $U                # seven workloads, streamed decode, one stream
python3 scripts/bench3.py  $U out.json 1 2   # the same prompts at 1 and 2 concurrent streams
python3 scripts/prefill.py $U                # time to first token at ~9k, ~31k, ~124k
python3 scripts/quality.py $U                # needle at 30k/120k, 12 math problems
python3 scripts/qcon.py    $U out.json 2     # 64 checked answers alone and at 2 streams
python3 scripts/stall.py   $U 1 40000        # one stream decoding while a ~124k prompt arrives
python3 scripts/apitest.py $U image.png      # structured output, tool calls, cache report, 1 MP image
python3 scripts/accept.py  $U 2              # tokens per verification and decode, sampled, thinking on
python3 scripts/longdecode.py $U             # decode at ~34k and ~115k tokens of context
python3 scripts/prefix.py  $U                # facts recalled across a conversation served from cache
python3 scripts/degen.py   $U                # 24 long sampled answers checked for loops
python3 scripts/drift.py run $U a.json       # greedy answers, to compare two builds (drift.py cmp)
python3 scripts/admission.py $U              # a short request beside a long prefill
python3 scripts/toolstream.py $U 3000        # do a long tool call's arguments stream while generated
python3 scripts/refusal.py $U [$U2]          # six borderline prompts, refusals per server
python3 scripts/fid.py dump $U corpus.json out.json.gz   # then: fid.py cmp ref.json.gz out.json.gz
```

## Credit

The serving recipe follows
[`MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
which worked out the flags and the memory budget for the full card. DFlash 2 is
[z-lab/dflash](https://github.com/z-lab/dflash), with the drafter
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The weights are
[`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
quantized with the recipe and calibration set of
[`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) through
[llm-compressor](https://github.com/vllm-project/llm-compressor).
[Discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192) compares the published
4-bit checkpoints. The L4 companion of this deployment is
[`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
