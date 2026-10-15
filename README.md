# Qwen3.8-27B · AWQ-INT4 · SGLang + DFlash 2 · ½ NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | Qwen3.8-27B, [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4), served under the name `Qwen3.8-27B-NVFP4` so existing clients keep working |
| Weights | INT4 W4A16 (AWQ, group 32), dense BF16 `lm_head`; **19.12 GiB resident** |
| Fidelity | on this deployment's own traffic, KL 0.0466 to the FP8 reference and 91.7% top-1 agreement, against 0.0724 and 89.3% for the NVFP4 build this replaced |
| Slots | **2** (`--max-running-requests 2`, `--max-mamba-cache-size 10`) |
| KV cache | `fp8_e4m3`, one 202,065-token pool shared by both slots; **42 KiB/token** (32 target + 10 drafter) |
| Context | 262,144 native; **202,059 usable** per request (`max_req_input_len`) |
| Decode, one stream | **122.3 tok/s** streamed (prose 78.2, chat 81.4, translation 85.3, summary 94.2, code 180.9, math 165.7, JSON 170.6) |
| Decode, two streams | 112.3 tok/s each, 178.5 tok/s together |
| Decode, thinking on, sampled | prose 69.0, chat 78.7 tok/s |
| Prefill | 2,866 tok/s at 9k, 2,594 at 31k, 1,783 at 124k, all uncached |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), BF16 3.72 GiB, block drafting at 8 |
| Engine | SGLang v0.5.20, the stock `lmsysorg/sglang:v0.5.20` image |
| Attention | FlashInfer for target and draft; FlashInfer GDN kernels for prefill |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell: 47.64 GiB of 96, SM120 |
| Structured output | off (`--grammar-backend none`): `response_format` and forced tool calls get a 400, because with it on either one kills the server |
| Quality | needle retrieval 5/5 at 30k and 120k, 12/12 multi-step math with thinking; 64/64 checked answers still right at two and three concurrent streams |
| Vision | native, preprocessed on the GPU; input bounded to 1,024 image tokens |
| Cold start | ~11 min to first token; 0 s in practice, held warm by `minScale: '1'` |

The repository name is from the first deployment, which served NVFP4. [The checkpoint](#the-checkpoint)
explains the switch.

Decode is `scripts/bench2.py`: each answer is streamed and timed from its first token to its last,
`(tokens - 1) / (t_last - t_first)`, two prompts per workload at up to 800 tokens, greedy with
thinking off unless stated. Two streams is `scripts/bench3.py`, the same fourteen prompts pushed
through two connections at once. Prefill is time to first token over prompts of unique random
words, so the radix cache cannot serve them. Quality is `scripts/quality.py`, fidelity
`scripts/fid.py`, concurrency `scripts/qcon.py`.

```bash
bash docker/build.sh                 # mirror lmsysorg/sglang:v0.5.20 into Artifact Registry
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
decaying toward the end of the block. In a sustained decode on the NVFP4 build it emitted 6.53 tokens
per target pass, more than MTP at any depth, and it pays one draft pass rather than seven.

The target verifies every drafted token, so speculation does not change what the model emits, up
to the rounding differences between the verify and decode kernels, and sampling keeps its
distribution.

## The checkpoint

DFlash 2's selector runs a top-k over the target's logits, so the target's `lm_head` must be dense:
SGLang refuses a packed-FP4 head with *DFlash2 selector requires a dense FP16/BF16/FP32 target
lm_head*. That leaves the body's format open, and NVFP4 is not automatically the best 4-bit choice.
Three 4-bit checkpoints with a BF16 head were served on this card and compared against the official
FP8 checkpoint, whose own distance to BF16 is published as a median per-position KL of 0.0117 on text
([Qwen3.8-27B discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)).

`scripts/fid.py` feeds the same text through each server, teacher-forced, and keeps the
probability of every actual next token and the top-20 distribution at every position. The corpus
is 55 documents, 91,769 positions, of the traffic this deployment serves: agent answers, code,
technical documents and Chinese text. It is private; the script takes any corpus.

| Checkpoint | PPL | KL answers | KL code | KL docs | KL Chinese | KL all | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.8-27B-FP8` (reference) | 7.874 | | | | | | |
| **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** | 7.979 | **0.0137** | **0.0126** | **0.0181** | **0.0935** | **0.0466** | **91.74%** |
| `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (previous) | 8.062 | 0.0289 | 0.0274 | 0.0381 | 0.1330 | 0.0724 | 89.26% |
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | 7.902 | 0.0383 | 0.0328 | 0.0492 | 0.1526 | 0.0853 | 88.33% |

The AWQ checkpoint is closer to the reference on every slice, about half the divergence of the
NVFP4 build on everything but Chinese. QUASAR has the lowest perplexity and the largest divergence:
it predicts this text well, but not the way the reference does. The published
comparison in discussion #192 (vLLM, top-128, wikitext and code against BF16) ranks the AWQ
checkpoint first of the 4-bit ones on text and last on code; on this traffic it is first on both.

It is also faster to decode and smaller, and slower to prefill. Same day, same card, same flags,
one stream:

| | NVFP4 (previous) | AWQ-INT4 | QUASAR NVFP4 |
| --- | ---: | ---: | ---: |
| Weights resident | 21.84 GiB | **19.12 GiB** | |
| KV pool at one slot | 180,573 | **248,597** | 256,218 |
| Decode, mean of seven | 107.9 | **119.2** | 116.4 |
| Prefill at 9k / 31k / 124k | **6,045 / 5,247 / 2,659** | 2,751 / 2,622 / 1,784 | |

W4A16 runs its matrix multiplies on BF16 tensor cores (Marlin) and NVFP4 runs them on FP4 ones, so
prefill, which is compute-bound, halves. Decode reads weights, and the AWQ checkpoint has fewer to
read. Over 143 agent runs on this deployment a turn adds about 2,300 new prompt tokens (estimated
from the growth of its context) and generates 1,223 on average, which makes decode over 90% of GPU
time on either checkpoint, so the decode gain outweighs the prefill loss. Where the loss shows is
a long first prompt: a 31k-token document waits about 12 s for its first token instead of 6.

## Two slots

Every slot running at once needs its own recurrent state, and with speculation, the intermediate
states of every drafted position so a rejected draft can be rolled back. Both come out of the
same memory as the KV pool, so slots are bought with context. From the startup logs:

| Slots | AWQ-INT4 pool | NVFP4 pool |
| ---: | ---: | ---: |
| 1 | 248,597 | 180,573 |
| **2** | **202,065** | |
| 3 | 155,532 | |
| 4 | | 40,975 |

Each slot after the first costs 46,532 tokens: five recurrent states (0.70 GiB) and the
speculative intermediate states (1.13 GiB). The pool is shared, not split. One request can use all
of it while the other slot is idle; what must fit is the sum of what is running, plus the cached
prefixes worth keeping. Agent runs on this deployment peak at 115k tokens of context at the median
and 145k at the 90th percentile, so two long runs at once contend: SGLang makes one wait, or evicts
the other's cached prefix, which then has to be prefilled again. Two slots is the setting where one
long run and shorter requests run side by side and a single request still gets more context than
the one-slot NVFP4 build did.

| Streams | AWQ, 2 slots | AWQ, 3 slots | NVFP4, 4 slots |
| ---: | ---: | ---: | ---: |
| 1 | 121.8 | 117.4 | 107.9 |
| 2 | 111.4 each, 187.7 total | 107.0 each, 185.2 total | 102.0 each, 171.0 total |
| 3 | | 103.1 each, 248.0 total | |
| 4 | | | 90.1 each, 291.2 total |

Concurrency is where DFlash 2 has an open correctness report: answers going wrong, and prompts
appearing in the wrong conversation, only under load (sglang#36548). `scripts/qcon.py` runs 64
problems with checkable answers (three-step arithmetic and four-person ordering puzzles, greedy,
thinking on) once alone and once at C streams with unrelated essay and code requests interleaved.
At two and three streams on the AWQ checkpoint and four on NVFP4, all 64 answers were right both
ways and no answer contained a word from the interleaved requests. The text itself differs between
the two runs - 4 to 25 of the 64 are byte-identical - because a verify batch of a different size
rounds differently; the answers do not move.

**A long prefill must not freeze the other stream.** SGLang schedules a waiting prefill ahead of
decode, so a 124k-token prompt arriving while the other slot is answering stalled that answer for
68 s. `--prefill-decode-interval N` holds back the next prefill chunk after each one, so decode
runs in between:

| | Long prompt, time to first token | Other stream, longest gap |
| --- | ---: | ---: |
| no interval | 70.0 s | 67.96 s |
| `--prefill-decode-interval 16` | 83.5 s | 2.87 s |
| **`--prefill-decode-interval 40`** | 84.3 s | **1.97 s** |

With nothing else running, a prompt prefills at the same speed with interval 40: 2,696, 2,621 and
1,784 tok/s at 9k, 31k and 124k against 2,751, 2,622 and 1,784 without. Both settings fit about a
dozen decode steps between 4,096-token chunks, so a long prompt that arrives beside a running answer
pays about 20%.

`--max-mamba-cache-size` is in units of the recurrent state, five per request here
(`mamba_ratio`); it must be at least 5 × `--max-running-requests` or the server refuses to start.

The speculative intermediate states are most of a slot's cost, and they are what
`--enable-linear-replayssm-spec` replaces with a small cache of recent inputs. SGLang v0.5.20 and
v0.5.21 refuse it for DFlash on a GDN target until sglang#35725 lands; the patch does not apply
cleanly to either release.

## Structured output is off

A request with `response_format` (JSON schema) or `tool_choice: "required"` crashes SGLang v0.5.20
and v0.5.21 with DFlash 2 within six seconds: the scheduler dies in
`dflash_worker_v2.forward_batch_generation` → `sampler._sync_token_ids_across_tp` with
*NCCL error ... unhandled cuda error*, and the process exits. Cloud Run restarts it in about ten
minutes, and a client that treats the 500 as transient and resends the same request kills each
new instance about a second after it comes up; that took this deployment down for over three
hours in one evening. It is the same failure class as sglang#24283 for EAGLE and NEXTN.

`--grammar-backend none` turns it into a clean 400, *Grammar-based generation (json_schema, regex,
ebnf, structural_tag) is not supported*, and the server stays up. Ordinary tool calls do not use
the grammar and still parse, on the OpenAI and the Anthropic endpoints. A client that needs JSON
asks for it in the prompt and validates.

`--enable-cache-report` is what makes cache hits visible. Without it SGLang sends no
`cache_read_input_tokens` on `/v1/messages` and no `prompt_tokens_details` on chat completions, and
a client reports every turn as uncached.

## Prefill runs on FlashInfer's GDN kernel

48 of the 64 layers are gated delta net layers, and SGLang only selects FlashInfer's GDN prefill
kernel by default on SM100; SM120 gets the Triton one. `--linear-attn-prefill-backend flashinfer`
picks it here. On the NVFP4 build it took prefill at 9k and 31k from 5,195 and 4,923 tok/s to
5,900 and 5,180. It needs CUDA 13 and a prefill chunk of at most 8,192.

It also made prefix caching consistent. With the Triton kernel, a follow-up turn served from the
radix cache and the same turn recomputed after `/flush_cache` gave different greedy text within
the first sentence. With the FlashInfer kernel the two are identical.

SGLang v0.5.20 carries the fix for sglang#37817: before it, the DFlash worker
never saved the GDN state at radix-cache checkpoints, and a request reusing such a prefix continued
from the wrong recurrent state. Every multi-turn conversation reuses its prefix, so older images
are not an option.

## Flags that are not preferences

`--attention-backend flashinfer` is required on SM120; `trtllm_mha` is SM100-only. `--kv-cache-dtype
fp8_e4m3` is the best supported precision for this model, because SGLang's 4-bit KV paths are not
built for a hybrid-GDN architecture. `--tool-call-parser qwen3_coder` decodes the
`<tool_call><function=…>` payload this template emits; `hermes` parses none of it, and a model whose
tool calls are silently dropped answers nothing at all. `--speculative-draft-model-quantization
unquant` keeps the drafter in BF16, and `--speculative-draft-attention-backend flashinfer` gives
the draft the same kernels as the target. `--grammar-backend none`, `--enable-cache-report` and
`--prefill-decode-interval 40` are explained above.

DFlash 2 is a pure-attention drafter of five sliding-window layers, so it needs no recurrent state
pool of its own. The `--mamba-*` flags other than the cache size, and `--linear-attn-verify-backend`,
belong to the DSpark recipe and are absent here.

## Memory

Every step below is a logged number. The two Prometheus gauges are **target-only** -
`weight_memory_usage_gb` and `kv_cache_memory_usage_gb` exclude the drafter entirely and the
recurrent state pool has no gauge at all - so the startup log is the account:

| Step | GiB | Avail after | Source |
| --- | ---: | ---: | --- |
| Card visible | 47.64 | | ½ of 96 |
| CUDA context + driver reserve | -2.40 | **45.24** | `Load weight begin. avail mem=45.24` |
| Weights, target | -19.12 | **26.13** | `Load weight end ... mem usage=19.12` |
| Weights, drafter | -3.72 | **22.37** | `type=DFlash2DraftModel, avail mem=22.37, mem usage=3.72` |
| Recurrent state, two slots | -4.99 | | `ssm 1.55 + intermediate_ssm 3.38 + conv 0.03 + 0.03` |
| KV pool, target (16 full layers) | -6.16 | **11.10** | `K size: 3.08 GB, V size: 3.08 GB` |
| KV pool, drafter (5 layers) | -1.92 | **9.17** | `K size: 0.96 GB, V size: 0.96 GB` |
| Autotune, prefill CUDA graph | -2.74 | 6.43 | `Capture target prefill CUDA graph end ... mem usage=2.22, avail mem=6.43` |
| Verify CUDA graphs | -0.12 | **6.31** | `available_gpu_mem=6.31` |

The line that matters for sizing context is the pair of KV pools. Geometry is `head_dim` **256**
with 4 KV heads, so one token of fp8 KV costs `2 × 4 × 256 = 2 KiB` per layer: **32 KiB** over the
target's 16 full-attention layers, plus **10 KiB** over the drafter's 5. A token therefore costs
**42 KiB**, and context on a speculative deployment is ~30% more expensive than on a plain one.
The pools match it exactly: 202,065 tokens at 32 KiB is 6.16 GiB, and at 10 KiB 1.92.

`--mem-fraction-static 0.80` leaves room for what the static budget does not see. This model is a
VLM, and the "fast" Qwen2-VL image processor is the torchvision one: it allocates on the device,
in the API process, outside the engine's budget, and the allocation grows with the image. A
1024x1024 image failed at `0.90` needing 352 MiB, and at `0.85` needing 392 MiB with 154 MiB free.
Shrinking the fraction is the wrong instrument for this - the fraction is a constant and the
allocation is a function of the input, so it can always be outgrown.

Bounding the input fixes it at the source. Qwen2-VL's processor resizes rather than rejects:
aspect ratio is preserved and dimensions round to patch multiples, so a cap is a ceiling on cost
and not a rejection of large images. This checkpoint has patch_size 16 and merge_size 2, so a
token covers 32x32 = 1024 pixels and 1,048,576 pixels is exactly 1,024 image tokens. Preprocessing
then runs on the GPU, which is both what it is for and faster - a 3000x2000 image takes 0.52 s on
the device against 0.94 s on the CPU.

The kwarg is `size`, not `max_pixels`, and the wrong one fails silently. This processor's
`preprocessor_config.json` carries `size: {longest_edge: 16777216, shortest_edge: 65536}` and no
`min_pixels`/`max_pixels` at all. `--mm-process-config '{"image":{"max_pixels":1048576}}'` is
accepted, echoed back by `/get_server_info` verbatim, and ignored: a 3000x2000 image still produced
5,828 image tokens. `longest_edge` is an area in pixels here rather than an edge length, which is
why the untouched default is a 16,384-token ceiling.

| Image | Default | `size.longest_edge` 1048576 | Latency |
| --- | --- | --- | --- |
| 512x512 | 256 tok | 256 | 0.55 s |
| 1024x1024 | 1,024 | 1,024 | 0.65 s |
| 2048x2048 | 4,096 | **1,024** | 0.63 s |
| 3000x2000 | 5,828 | **1,014** | 0.52 s |

A 128x128 test image passes under every one of these configurations, which is how the first round
of vision testing missed all of it. Test at the size the clients actually send; `scripts/apitest.py`
sends a 1-megapixel one.

Two numbers describe the window and the smaller one binds: `context_length` is 262,144, but
`max_req_input_len` is what the pool will admit - **202,059** here. Size clients from the latter,
and re-read it after any change to `--mem-fraction-static`, `--max-running-requests`,
`--max-mamba-cache-size`, the checkpoint, the drafter or the SGLang version, all of which move it.

## Measured and dropped

Each of these was deployed as its own service on the same half card and measured against the
shipped configuration.

| Change | Result |
| --- | --- |
| SGLang v0.5.21 | Decode unchanged (106.4 tok/s against 107.9 on the NVFP4 build), and the structured-output crash is still there. |
| NVFP4 drafter (`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`) | +6.3% decode and +37% context, and stretches of degenerate text inside thinking blocks in use, which no benchmark caught. The cause is not established; DFlash 2 drafts through its own candidate selector, and a looser acceptance there would let the drafter's distribution reach the output. Withdrawn. |
| QUASAR-QAT NVFP4 target | +7.9% decode over the NVFP4 build, and the largest divergence from FP8 of the three (above). |
| Four slots on the NVFP4 build | 291 tok/s at four streams, and a 40,975-token pool. |
| `--enable-mixed-chunk` | Disabled by SGLang for DFlash; mixed steps would also drop running requests to one-token decode. |
| `--enable-linear-replayssm-spec` | Refused for DFlash on a GDN target until sglang#35725. |
| `--enable-torch-compile` | Decode CUDA graph capture fails at startup, `bmm_fp8_run_with_algo failed`, on both boots tried. |
| FlashInfer b12x NVFP4 GEMM (sglang#38170, patched in) | Active and autotuned, and decode unchanged on the NVFP4 build: prose 71.3 against 71.3, chat 75.6 against 75.8. |
| `--chunked-prefill-size 8192` | Prefill -1% to +3% against 4096, and more activation memory. |
| `--speculative-draft-window-size 2048` | The drafter attends to 2,048 tokens anyway, and its KV pool stays sized to the full context, so nothing is freed. |
| Draft block 12 or 16 with the block-8 drafter | Best case up, median flat, prose slower (helix.ml's measurement on the same model). |

## Serving

`cloudrun/service.yaml` is the deployed configuration. Three things about it are Cloud Run's rather
than SGLang's: the container must listen on 8080, `gpu-zonal-redundancy-disabled` must be `'true'`
or the deploy hangs, and `minScale: '1'` is load-bearing for the reason the next section measures.

`--max-model-len` has no counterpart here, but the equivalent trap does: SGLang validates
`input + completion ≤ context` and returns 400, where llama.cpp clamps. A client that asks for the
whole 262,144-token window beside a 53-token prompt is refused, and a harness that reads that 400
as context overflow will compact, retry and fail. Cap the completion.

## Cold start

From the instance starting to the first token is 666 s, read off the startup log's own
timestamps:

| Phase | Seconds |
| --- | ---: |
| Schedule, image pull, import | 45 |
| Target weights from Hugging Face | 374 |
| Drafter weights | 56 |
| State and KV pools | 2 |
| Autotune and prefill CUDA graph | 169 |
| Verify CUDA graphs | 10 |
| Warmup | 8 |

Downloading and loading weights is two thirds of it. The prefill graph takes 169 s here against
27 s on the NVFP4 build.

`minScale: '1'` turns cold start from a user-facing cost into a deploy-time one: the instance is
never reaped, so no request meets it, and it is only paid when a revision is replaced. It is not a
tuning preference - with `maxScale: '1'` and no warm instance, every request during those eleven
minutes is answered with *no available instance*. Baking the weights into the image would replace
the Hugging Face pull with an in-region Artifact Registry pull; its saving is unmeasured.

## Reproducing

```bash
export SGLANG_API_KEY=...
U=https://YOUR-SERVICE.run.app
python3 scripts/bench2.py  $U            # seven workloads, streamed decode, one stream
python3 scripts/bench3.py  $U out.json 1 2   # the same prompts at 1 and 2 concurrent streams
python3 scripts/prefill.py $U            # time to first token at ~9k, ~31k, ~124k
python3 scripts/quality.py $U            # needle at 30k/120k, 12 math problems
python3 scripts/qcon.py    $U out.json 2 # 64 checked answers alone and at 2 streams
python3 scripts/stall.py   $U 1 40000    # one stream decoding while a ~124k prompt arrives
python3 scripts/apitest.py $U image.png  # structured output, tool calls, cache report, 1 MP image
python3 scripts/fid.py dump $U corpus.json out.json.gz   # then: fid.py cmp ref.json.gz out.json.gz
```

## Credit

The serving recipe follows [`MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
which worked out the flags and the memory budget for the full card. DFlash 2 is
[z-lab/dflash](https://github.com/z-lab/dflash); the drafter card is
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The checkpoint is
[`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4); the
shootout in [discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192) is where it
came up as a candidate. The L4 companion to this deployment is
[`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
