# Qwen3.8-27B abliterated · own NVFP4/FP8 quant · SGLang + DFlash 2 · NVIDIA RTX PRO 6000

The repository name still says AWQ. Production served an AWQ-INT4 build until 2026-10-24 and now
serves our own mixed NVFP4/FP8 quantization of the same abliterated weights. The repository keeps
its name, and the served model name stays `Qwen3.8-27B-abliterated-AWQ-INT4` so that clients need
no change. [AWQ build, 2026-10-23](#awq-build-2026-10-23-previous-production) keeps the figures of
the previous production.

| | |
| --- | --- |
| Model | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), quantized here to mixed NVFP4/FP8 with NVIDIA ModelOpt 0.47 (`quant/modelopt/`), served as `Qwen3.8-27B-abliterated-AWQ-INT4` |
| Weights | FP8 W8A8 attention and GDN projections, NVFP4 W4A4 MLPs, NVFP4 `lm_head` served as W4A16; 20.18 GB resident |
| Accuracy | KL 0.0497 to BF16 and 92.5% top-1 agreement on our private corpus; the AWQ build had 0.0346 and 94.5% |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), quantized to FP8 at load (2.51 GB), 8 draft tokens per block |
| Engine | SGLang v0.5.22 (`lmsysorg/sglang:v0.5.22`) with six patches in `docker/`, FlashInfer attention and GDN prefill kernels, FlashInfer CuTe-DSL NVFP4 GEMM |
| GPU | one whole NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) on Cloud Run, with 20 vCPU and 80 GiB |
| Slots | 8 concurrent requests, fp32 recurrent state (16 until the afternoon of 2026-10-24, see [Eight slots](#eight-slots-2026-10-24)) |
| Context | **262,144** tokens per request; one 566,383-token BF16 KV pool shared by the 8 slots (492,440 with 16) |
| Prefill | **12,003 tok/s** at 9k tokens, 11,223 at 31k, 6,728 at 124k, which is 2.2, 2.2 and 1.9 times the AWQ build |
| Time to first token, coding agent | **2.5 s** at 30k tokens of code context, 6.2 s at 60k, 17.2 s at 120k; a follow-up turn 0.39 s at 30k, 0.70 s at 60k, 1.01 s at 120k |
| Decode, greedy, thinking off | **235.4 tok/s** (prose 153.9, chat 176.1, translation 153.7, summary 198.4, code 320.4, math 313.5, JSON 332.0) |
| Decode, sampled, thinking on, coding agent | **220 tok/s** at 30k tokens of code context, 199 at 60k, 192 at 120k; per stream at two and three streams, see [Production, 2026-10-24](#production-2026-10-24) |
| Refusals | not measured again on this build; on the AWQ build, 0 of 6 borderline prompts, against 5 for the stock model |
| Quality | needle retrieval 5/5 at 30k and 120k tokens, multi-step math 12/12, `qcon.py` 63 of 64 alone and at three streams (a different ordering puzzle missed each time), ten of ten facts across a conversation served from cache, 16 of 16 parallel tool-call cases |
| Vision | native, preprocessed on the GPU, at most 1,024 tokens per image |
| Tool calls | arguments stream as they are generated, at most 0.1 s apart; earlier thinking reaches the template natively, so each tool round hits the prefix cache |
| Structured output | not supported: `response_format` and `tool_choice: "required"` return 400 |
| Disconnects | a client that disconnects frees its slot within a second |
| Cold start | 14 min 26 s from instance start to ready; one instance stays warm |

These figures come from different measurements:

- **Prefill, KL and HumanEval:** test services on 2026-10-24 that ran the production flags without
  block verification, ReplaySSM, `--prefill-decode-interval` and HTTP/2 (see
  [Quantization](#quantization)).
- **Everything else in the table:** the production revision itself, `SERVICE-ownq8`. We
  measured it through a tagged URL with no traffic, before the cutover.

How the benchmarks work:

- **Decode:** `scripts/bench2.py` streams each answer and divides the tokens after the first by the
  time from the first to the last token. Each of seven workloads runs two prompts of up to 800
  tokens, greedy with thinking off unless stated. The table gives the median of three runs.
- **Prefill:** the time to first token on prompts of unique random words, so that no prefix is
  cached.
- **Coding agent:** a separate benchmark streams `/v1/messages` with three tools and thinking on, at
  the default effort of the template and the default sampling of the model. The contexts are 29k,
  59k and 119k tokens of SGLang source. A follow-up turn is one more user message after the answer,
  with the earlier thinking in the history.

The other sections were mostly measured on the AWQ build: [Checkpoint](#checkpoint-awq-build-production-until-2026-10-24)
to [Vision](#vision) on the half card it used until 2026-10-23, and
[Full card, 2026-10-23](#full-card-2026-10-23) on the whole card. Where a section says nothing
else, read its figures as AWQ figures.

```bash
bash docker/build.sh                                            # lmsysorg/sglang:v0.5.22 + six patches, into Artifact Registry
gcloud alpha run services replace quant/modelopt/service.yaml   # once: quantize into the bucket (ModelOpt, mixed NVFP4/FP8)
gcloud alpha run services replace cloudrun/service.yaml
```

## GPU

Production runs on one whole RTX PRO 6000 (`nvidia.com/gpu: 1`) with 20 vCPU and 80 GiB of system
memory. Until 2026-10-23 it ran on half a card, and most of the studies below were done there.

The fractional RTX PRO 6000 is a Cloud Run offering, separate from the G4 machine types of Compute
Engine and with its own quota: `run.googleapis.com` grants milliGPUs, of which a half card uses
500, and a project can hold this grant without any Compute Engine GPU quota. Cloud Run accepts 0.5
or 1 for `nvidia.com/gpu`, and the half card comes with 10 to 16 vCPU and 40 to 64 GiB of system
memory.

The half card is a MIG `2g.48gb` partition with 94 of the 188 SMs. We measure 765 GB/s of read
bandwidth (1,524 GB/s on the whole card) and 177 dense BF16 TFLOPS (412). Decode is bound by
bandwidth, since a step reads the weights of the target and the drafter once for all the tokens it
verifies, so it becomes faster only by committing more tokens per step or by reading fewer bytes.
Prefill is bound by compute and runs at about 145 TFLOP/s at 9k tokens on the half card, which is
82% of the BF16 peak. The whole card has twice the bandwidth and the compute, and decodes one stream
1.65 times as fast (231.7 against 140.5 tok/s in bench2) and prefills 9k tokens 1.9 times as fast
(5,417 against 2,783 tok/s).

## Quantization

Production serves our own mixed NVFP4/FP8 quantization of the same BF16 abliterated checkpoint,
`huihui-ai/Huihui-Qwen3.8-27B-abliterated`, that the earlier AWQ build came from. No third-party
checkpoint is involved. `quant/modelopt/` has the scripts.

- **Tool:** NVIDIA ModelOpt 0.47.0 (`quant/modelopt/mo.py`). It runs as a Cloud Run service on one
  whole RTX PRO 6000 (`quant/modelopt/job.sh`), because Cloud Run caps GPU jobs at one hour.
- **Precision layout:**
  - FP8 per-tensor W8A8 with static scales: the attention projections (`self_attn.q/k/v/o_proj`)
    and the GDN projections (`linear_attn.in_proj_qkv`, `in_proj_z` and `out_proj`).
  - NVFP4 W4A4 with group 16: every MLP (`gate_proj`, `up_proj` and `down_proj`).
  - BF16: GDN `in_proj_a/b`, `conv1d`, the vision tower and the embeddings.
  - The FP8 and NVFP4 GEMMs run on the FP8 and FP4 tensor cores of SM120. AWQ W4A16 runs on BF16
    tensor cores, so prefill is about twice as fast.
- **Weight scales:** ModelOpt `local_hessian` with the FP8 scale sweep for the NVFP4 block scales.
  ModelOpt defaults to `max`, which gives plain round-to-nearest scales.
- **Calibration:** 583 samples and 1.34M tokens, each at most 4,096 tokens long, all rendered with
  the chat template of the model (`quant/modelopt/build_calib.py`).
  - 46% agentic and tool-call data: SWE trajectories from OpenHands/SWE-Gym and Nemotron-SWE, plus
    70 tool-call conversations rendered through the native `tools=` path of the template.
  - 26% code: real files from the SGLang repository, framed as review, refactor and test requests
    in English and Chinese, plus six windows of the coding-agent benchmark context.
  - 21% Chinese chat and 6% English chat (sharegpt_gpt4).
  - 1% math and STEM from the calibration set of the AWQ build.
  - The private documents of the KL evaluation (`scripts/fid.py`) are held out.
- **`lm_head`:** the head is exported as NVFP4 but served as `W4A16_NVFP4`, meaning NVFP4 weights
  with BF16 activations on Marlin (`quant/modelopt/w4a16head.py`).
  - This edits the config only; the head is not quantized again.
  - It reads the same 0.64 GB per verify step as NVFP4 W4A4, but it no longer quantizes its input
    to FP4.
  - Against the plain NVFP4 head it lowers KL from 0.0532 to 0.0497 at the same decode.
  - DFlash ranks draft candidates with the same quantized head, so the int8 draft head
    (`SGLANG_DFLASH_DRAFT_HEAD`) is unset. It works only with a BF16 head.
- **KV cache:** ModelOpt also calibrates FP8 KV scales. Production serves a BF16 KV cache, so
  `quant/modelopt/stripkv.py` removes `kv_cache_scheme` and `kv_cache_quant_algo` from
  `config.json` and `hf_quant_config.json`.
  - With those fields left in, a BF16 KV cache produces garbage output.
  - The bucket keeps both variants: `ownq/molh-w4h-bkv` with the fields stripped, for
    `--kv-cache-dtype bf16`, and `ownq/molh-w4h` with them kept, for `fp8_e4m3`.

KL comes from `scripts/fid.py` against the BF16 checkpoint, on the private corpus of agent answers,
code, documents and Chinese text. Every row was measured on the whole card with image `b4802ba` and
8 draft tokens. The first three rows come from an earlier study the same day. Our rows use the
production flags without block verification, ReplaySSM, the prefill-decode interval and HTTP/2
(see [Production, 2026-10-24](#production-2026-10-24) for the deployed flags).

| Build (body / `lm_head` / KV) | Prefill at 9k / 31k / 124k, tok/s | TTFT at 30k / 60k / 120k | bench2 | KL to BF16 (code / Chinese), top-1 | GSM8K of 200 | HumanEval 41×2 | Parallel tool calls |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| AWQ W4A16 (previous production), BF16 head, FP8 KV | 5,401 / 5,105 / 3,563 | 5.6 / 12.8 / 32.6 s | 232.9 | 0.0346 (0.008 / 0.103), 94.5% | 191 | 80/82 | 16/16 |
| third-party mixed NVFP4/FP8 (edp1096, ModelOpt `max`) | 11,970 / 10,808 / 6,741 | 2.5 / 6.2 / 17.2 | 236.5 | 0.0606 (0.018 / 0.149), 91.0% | | | |
| own GPTQ (llm-compressor), BF16 head | 11,830 / 11,324 / 6,825 | 2.5 / 6.1 / 17.1 | 220.0 | 0.0553 (0.013 / 0.142), 92.6% | 193 | | |
| **lh-wb: local-Hessian / W4A16-NVFP4 / BF16 (deployed)** | **12,003 / 11,223 / 6,728** | **2.6 / 6.3 / 17.3** | **236.1** | **0.0497 (0.0134 / 0.1315), 92.5%** | 192 | 79/82 | 16/16 |
| lh-wf: local-Hessian / W4A16-NVFP4 / FP8 | 11,839 / 10,084 / 5,475 | 2.7 / 7.3 / 21.1 | 225.5 | 0.0512 (0.0135 / 0.1358), 92.5% | 192 | | |
| lh-nb: local-Hessian / NVFP4 / BF16 | 11,241 / 11,215 / 6,736 | 2.5 / 6.2 / 17.2 | 237.3 | 0.0532 (0.0149 / 0.1362), 91.9% | 189 | 80/82 | 16/16 |
| lh-nf: local-Hessian / NVFP4 / FP8 | 11,489 / 10,280 / 5,495 | 2.8 / 7.2 / 21.0 | 225.1 | 0.0546 (0.0151 / 0.1404), 91.7% | 193 | | 16/16 |
| lh-fb: local-Hessian / FP8 / BF16 | 12,049 / 10,832 / 6,640 | 2.5 / 6.3 / 17.5 | 224.1 | 0.0471 (0.0124 / 0.1282), 93.0% | 192 | 80/82 | 16/16 |
| lh-ff: local-Hessian / FP8 / FP8 | 11,574 / 10,084 / 5,466 | 2.7 / 7.2 / 21.1 | 224.8 | 0.0485 (0.0126 / 0.1326), 93.0% | 191 | 80/82 | 16/16 |
| lh-bb: local-Hessian / BF16 + int8 draft head / BF16 | 11,250 / 11,165 / 6,727 | 2.5 / 6.3 / 17.3 | 220.0 | **0.0457** (0.0119 / 0.1264), 93.3% | 190 | 80/82 | 16/16 |

How to read the table:

- Every build scored needle retrieval 5/5 at 30k and 120k tokens and math 12/12.
- HumanEval runs 41 problems, two samples each, with thinking on. HumanEval/145 fails on every
  build, AWQ included, so 80/82 is the ceiling. The one extra miss of lh-wb is one sample of
  HumanEval/125.
- The parallel tool-call check sends 8 prompts twice over `/v1/messages`. A case passes only with
  the exact number of calls, the names, the arguments and `stop_reason`.

lh-wb is on the Pareto frontier of KL against decode, with lh-bb (KL 0.046, 220 tok/s) and lh-fb
(0.047, 224).

- **Against AWQ:** 2.2 times the prefill at 9k and 31k tokens and 1.9 times at 124k, the same
  decode, and KL 0.050 against 0.035.
- **Against the third-party build:** the same speed and 18% lower KL.

FP8 KV would double the pool, but it costs 8 to 19% of prefill at 31k to 124k tokens, about 5% of
decode and about 0.0015 of KL.

The offline proxy, fake-quantized KL over the BF16 top-64 on held-out documents, shows where the
gain came from. These are cumulative steps, all with our calibration set:

| Step | Proxy KL | Served KL |
| --- | ---: | ---: |
| ModelOpt `max` (the edp1096 recipe on our weights), NVFP4 head | 0.0681 | |
| + MSE FP8-scale sweep for the NVFP4 weight scales | 0.0558 | |
| + local-Hessian weighting instead | 0.0549 | 0.0532 |
| + head served as W4A16 (same weights) | | **0.0497** |
| + FP8 head instead | 0.0494 | 0.0471 |
| + BF16 head | | 0.0457 |

We put each group back in BF16 on its own, starting from the `max` build:

- All MLPs: KL 0.068 → 0.032. MLP activations alone: 0.050.
- The head: 0.059.
- The GDN projections: 0.064.
- Attention: no change.

The error is spread over the layers. Putting the 12 most sensitive MLP layers back in BF16 reaches
only 0.058, which the scale search gives for free, so no layer is excluded. We did not try:

- ModelOpt GPTQ: its layerwise mode cannot find the nested layers of the VLM.
- AWQ-lite: its scale would have to be folded into the `(1+w)` RMSNorm.
- SVDQuant: SGLang's mixed-precision path does not load the LoRA branch.

Downloading the 55.6 GB BF16 checkpoint takes 17.5 minutes, since
these services reach the bucket at about 55 MB/s, so `job.sh` keeps the BF16 copy and
polls a queue directory for more variants.

## Checkpoint (AWQ build, production until 2026-10-24)

This section describes the AWQ-INT4 build that production served until 2026-10-24, and how it was
chosen. [Quantization](#quantization) describes the build that replaced it. The tables in this
section were measured on the half card.

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

The DFlash measurements in this section were taken on the half card. On the whole card, blocks of
12 and 16 tokens again lose to blocks of 8 (see [Full card, 2026-10-23](#full-card-2026-10-23)),
and the int8 draft head is still worth about 4.5%: with eight slots, bench2 gives 230.4 tok/s with
it and 221.5 without it and without block verification, which acts only on sampled requests.

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
8,192 tokens. The half card ran 4,096-token chunks; the whole card runs 8,192.

AWQ W4A16 prefills on BF16 tensor cores. Our mixed NVFP4/FP8 checkpoint (see
[Quantization](#quantization)) multiplies on FP4 and FP8 tensor cores. On test services it
prefilled 12,003 tok/s at 9k tokens, 11,223 at 31k and 6,728 at 124k, against 5,401, 5,105 and
3,563 for the AWQ checkpoint in the same study. This is the main reason production moved to it on
2026-10-24. The FP8 KV cache costs a further 8 to 19% of prefill at 31k to 124k tokens, which is
why production now runs a BF16 KV cache.

## Concurrency

Production has served 8 slots since 2026-10-24 (see [Eight slots, 2026-10-24](#eight-slots-2026-10-24)),
with `--max-mamba-cache-size 42` = 5 × 8 + 2, `--cuda-graph-max-bs-decode 8` and
`containerConcurrency: 16`. The rest of this section describes the 16-slot configuration that ran
before it, from 2026-10-23 to 2026-10-24.

Every running request holds its own recurrent states, which share memory with the KV pool. With
ReplaySSM a request needs five fp32 recurrent states, three of its own and two for prefix-cache
tracking. `--max-mamba-cache-size` counts these states and must be at least five times
`--max-running-requests`. We serve 16 slots and set 82 = 5 × 16 + 2, keeping two in reserve, since
the engine stops when every state is taken and locked (sglang#43204). The pool itself is shared:
one request can use all of it while the other slots are idle, and when long requests compete,
SGLang delays one or evicts the cached prefix of another. Agent runs on this deployment reach 115k
tokens of context at the median and 145k at the 90th percentile, so the 1,011,164-token pool holds
several long runs at once while a single request keeps the full 262k context.

On the whole card the slot count no longer competes with the context: with the four-slot flags the
pool has 1,232,992 tokens, with 16 slots 1,011,164. The state stays fp32, and 16 slots return the
same greedy text as the four-slot flags on the same card on 24 of 24 `drift.py` prompts and on the
14 of 14 one-stream answers of `bench3.py`. `containerConcurrency` is 32, two requests per slot, so
that requests 17 to 32 queue inside SGLang and Cloud Run returns 429 only beyond 32. `bench3.py` on
the whole card, medians of three runs on test services:

| Streams | Four slots, per stream | Four slots, total | 16 slots, per stream | 16 slots, total |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 232.5 tok/s | 195.7 tok/s | 231.8 tok/s | 196.3 tok/s |
| 2 | 221.6 | 348.5 | 221.6 | 349.0 |
| 4 | 198.5 | 563.7 | 197.7 | 562.5 |
| 8 | 197.9 (four run, four wait) | 573.1 | 170.8 | **863.3** |
| 16 | | | about 84 | about 962 |

The 16-stream row ran two eight-stream rounds of `bench3.py` at the same time, 28 requests in all,
and is an estimate rather than a median of the standard round. Eight slots reach the same total at
eight streams (863.1). On the coding-agent benchmark, four slots and 16 decode one stream at the
same rate, within noise. All 64 `qcon.py` answers (described below) are right at 16 streams, none contains a word
from the other requests, and 26 are byte-identical to the serial run.

### Half card, four slots

The rest of this section describes the half card, as measured on 2026-10-23. We served four slots
there because the clients of this deployment send at most four requests at once, and
because four slots fit with fp32 recurrent state and the full context. Eight slots with fp32 state
leave a 203,338-token pool at `--mem-fraction-static 0.80`, below the 262,144-token
context. Eight slots fit only with bfloat16 state, which changes greedy text (see
[the tuning study](#tuning-study-half-card-2026-10-23)). With fp32 state the four-slot configuration returns
the same greedy text as the two-slot one on 104 of 104 prompts, so the slot count, the batch-size-4
CUDA graphs and the retiled draft head are bit-neutral at one stream. `containerConcurrency` was 8,
two requests per slot as before, so that requests five to eight queued inside SGLang and Cloud Run
returned 429 only beyond eight.

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

SGLang schedules a waiting prefill ahead of decode, so a long prompt that arrives while another
slot is answering stalls that answer. `--prefill-decode-interval N` runs decode steps after each
prefill chunk. The following table measures a 124k-token prompt that arrives beside a running
answer on the half card, with 4,096-token chunks:

| | Long prompt, time to first token | Other stream, longest gap |
| --- | ---: | ---: |
| no interval | 70.0 s | 67.96 s |
| `--prefill-decode-interval 16` | 83.5 s | 2.87 s |
| **`--prefill-decode-interval 40`** | 84.3 s | **1.97 s** |

Interval 40 fits about a dozen decode steps between two 4,096-token chunks. The long prompt then
takes about 20% longer to its first token, and prefill on an idle server is unchanged. Production
keeps interval 40 with 8,192-token chunks on the whole card; we did not repeat this table there.

## Memory

The production revision since 2026-10-24 serves the mixed checkpoint with a BF16 KV cache and 8
slots. On it, `/get_server_info` reports:

- 20.18 GB of target weights;
- a 34.57 GB target KV pool of 566,383 tokens;
- 14.72 GB free after startup;
- `max_req_input_len` 262,138, so a single request still gets the full context.

With 16 slots (revision `SERVICE-ownq8`) the pool was 30.06 GB and 492,440 tokens, with
12.90 GB free after startup. Fewer slots need fewer recurrent states and smaller CUDA graphs, and
the pool takes the difference.

A BF16 token takes twice the bytes of an FP8 one, so the pool holds about half the tokens. Agent
runs here reach 115k tokens of context at the median, so the pool holds about four such runs at
once. Three concurrent 120k-token sessions ran in the coding-agent benchmark (see
[Production, 2026-10-24](#production-2026-10-24)).

On the AWQ build, whole card with 16 slots, `/get_server_info` reports 19.11 GB of target weights, a 30.86 GB
target KV pool of 1,011,164 tokens and 13.16 GB free after startup, at `--mem-fraction-static
0.79`. The usable context is still 262,138 tokens, and a 259,728-token prompt was served (106.7 s to
the first token). The rest of this section is the accounting of the half card.

The Prometheus gauges `weight_memory_usage_gb` and `kv_cache_memory_usage_gb` count only the
target, and the recurrent state has no gauge, so we accounted for memory on the half card from the
startup log of the four-slot production revision:

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
the pools match this exactly: 266,083 tokens at 32 KiB is 8.12 GiB, and at 10 KiB 2.54 GiB. On the
whole card, 1,011,164 tokens at 32 KiB is the 30.86 GB the server reports for the target pool.

The two-slot configuration had a 314,253-token pool: twelve recurrent states took 1.90 GiB, and
`--mem-fraction-static 0.80` claimed 0.48 GiB more. We lowered the fraction to 0.79 with the move to
more slots. At 0.80 the four-slot configuration left 5.00 GB after startup, and an eight-slot
configuration with bfloat16 state 4.67 GB. On both, one `/generate` request with full-prompt top-20 input logprobs
(`scripts/fid.py`) asked for 1.89 GiB, hit CUDA OOM and killed the server, while the two-slot
configuration, with 5.21 GB left, survived it. At 0.79 the eight-slot configuration left 5.17 GB
and survived the same request. The four-slot half-card revision left 5.49 GiB; we did not repeat the
request on it, nor on the whole card, which leaves 13.16 GB.

The usable context is `max_req_input_len`, 262,138 on both cards, since the pool is larger than the
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

On the half card, the move to four slots left image answers unchanged: four real photos are recognized alone and
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
a call with 16,931 characters of arguments. The following table compares the two parsers on it, on
the half card:

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

On `/v1/messages`, parallel tool calls lost the closing `}` of a call, and the client received
`{"query": "CEO announcement", "num": 20`. It rejected that as invalid JSON, and the model usually
retried the same call. The main cause is
[sglang#42260](https://github.com/sgl-project/sglang/issues/42260). Under speculative decoding, one
increment can carry `</function>\n</tool_call>\n`. The parser returns the `}` as a call item and the
newline as text, and `serving_chat` sends the text first. The Anthropic adapter closes the open
`tool_use` block on that text, then drops the `}` and logs *Dropping tool_call argument delta with no
open tool_use block*. With a prompt that asks for two parallel calls, 8 of 48 calls on the previous
revision came back invalid. The port drops whitespace-only text that arrives in the same increment
as call items, and 48 of 48 now come back valid. The OpenAI endpoint accumulates arguments by index
and was never affected.

The port also closes calls the model leaves open: `</tool_call>` without `</function>`
([sglang#43150](https://github.com/sgl-project/sglang/issues/43150), the change of
[PR #43467](https://github.com/sgl-project/sglang/pull/43467)), or neither closing tag before the next
`<tool_call>`. In the second case, the non-streaming parse also merged two calls into one, so it now
splits calls at `<function=`. `docker/test_stream.py` covers all of these shapes. On the whole card,
prompts that ask for two parallel calls came back as two valid calls in 6 of 6 trials on the
16-slot test service and in 8 of 8 with the reasoning-history patch.

One case remains open: when the arguments of a call contain the literal text `<tool_call>`,
`<function=` or `<parameter=`, as in code that handles this very format, the `qwen3_coder` parser
takes it for markup and splits one `write` into dozens of broken calls (50 to 90 in the coding-agent
benchmark).

### Thinking across turns

With thinking on, an agent client sends the earlier thinking back as `thinking` blocks of the
assistant messages. The Qwen3.8 chat template reads it from `reasoning_content` and, since
`preserve_thinking` defaults to true, renders it in every assistant turn of the history, not only
in the turns after the last user message. Stock v0.5.22 passes thinking to the template natively
only for its custom chat encoders. For a Jinja template, the Anthropic adapter splices it into the
visible content as `<think>...</think>` text, and the template renders an empty think block followed
by that text. The re-rendered assistant message then differs from the tokens the model generated,
and the prefix cache stops at the end of the previous prompt. Before the patch, every request that
followed an assistant turn with thinking re-prefilled that whole assistant message: each tool-call
round inside one user turn, and each new user message.

`docker/anthropic_reasoning_history.py` passes the thinking to the template as `reasoning_content`,
so the history renders to the tokens the model generated and the cache covers it. On the
coding-agent benchmark, a follow-up turn reaches its first token in 0.34 s at 30k tokens and 0.46 s
at 60k, against 1.05 and 1.34 s without the patch. `SGLANG_ANTHROPIC_NATIVE_REASONING_HISTORY=0`
restores the stock behaviour.

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
`--kv-cache-dtype bf16` is what production has served since 2026-10-24. FP8 KV doubles the pool,
but on the mixed checkpoint it costs 8 to 19% of prefill at 31k to 124k tokens and about 5% of
decode (see [Quantization](#quantization)). The AWQ build ran `fp8_e4m3`, the lowest KV precision
SGLang supports for a hybrid GDN model. A ModelOpt checkpoint with FP8 KV scales must not be served
with a BF16 KV cache, as its output is then garbage. `quant/modelopt/stripkv.py` removes the scales.
`--fp4-gemm-backend flashinfer_cutedsl` selects the NVFP4 GEMM of the MLPs.
`--tool-call-parser qwen3_coder` parses the `<tool_call><function=…>` format of this chat template,
which the `hermes` parser drops. `--speculative-draft-attention-backend flashinfer` gives the
drafter the attention kernels of the target. `--enable-cache-report` reports cache hits to clients,
as `cache_read_input_tokens` on `/v1/messages` and `prompt_tokens_details` on chat completions.
`--cuda-graph-max-bs-decode 8` captures the decode and verify graphs for batch sizes 1 to 8, one
per slot count (16 with 16 slots until 2026-10-24). `--chunked-prefill-size 8192` and `--max-prefill-tokens 8192` set the prefill chunk,
the largest the FlashInfer GDN kernel takes; 16,384-token chunks with the Triton GDN kernel did not
shorten the time to first token. The two `SGLANG_DFLASH_DRAFT_HEAD*` variables below belong to the
AWQ build, whose head is BF16. The mixed checkpoint has a quantized head, and the int8 draft head
works only with a BF16 head, so production leaves both unset since 2026-10-24.
`SGLANG_DFLASH_DRAFT_HEAD_CFG=128,128,8,2` sets the tile of the
int8 draft-head kernel (BN, BK, warps, stages): above 32 draft rows per verify batch, slots times
block, the kernel tiles 64 rows, and on the half card its default tile `128,256,8,3` then needed
131,072 bytes of shared memory against the 101,376 of the partition, so the server failed at
startup with Triton `OutOfResources`. The retiled kernel needs 98,304 bytes. Sixteen slots of block
8 make 128 rows, so production runs with the retiled kernel; we did not try the default tile on
the whole card.
DFlash 2 is a pure-attention drafter of five sliding-window
layers without recurrent state, so the `--mamba-*` and `--linear-attn-verify-backend` flags of the
DSpark recipe do not apply to it.

## Serving

`cloudrun/service.yaml` is the deployed configuration. Cloud Run requires the container to listen on
8080 and `gpu-zonal-redundancy-disabled: 'true'` for the deployment to complete. With `minScale:
'1'` and `maxScale: '1'`, one instance stays warm, and a cold start occurs only when a revision is
replaced. `containerConcurrency: 16` admits two requests per slot, so that the excess queues inside
SGLang and Cloud Run returns 429 only beyond 16 (32 with 16 slots until 2026-10-24). `timeoutSeconds: 3600`, the maximum, lets an
agent turn that writes a long file in one tool call run for up to an hour. The service declares the
`ALPHA` launch stage, without which Cloud Run refused the 64 GiB memory limit of the half card; we
kept it for the 80 GiB limit of the whole card and did not try without it.

The production service is `SERVICE` in `REGION`. Since 2026-10-24 all its traffic goes to
`SERVICE-ownq8-c8`, the revision that serves the mixed checkpoint with 8 slots, which
`cloudrun/service.yaml` describes (see [Production, 2026-10-24](#production-2026-10-24) and
[Eight slots, 2026-10-24](#eight-slots-2026-10-24)). `SERVICE-ownq8`, the same with 16 slots,
stays on the service as the first rollback target. Its image, flags and Cloud Run settings are those
of the previous revision, with these differences:

- the weights are `ownq/molh-w4h-bkv/` instead of `huihui-awq/out/`;
- `--kv-cache-dtype bf16` replaces `fp8_e4m3`, and `--fp4-gemm-backend flashinfer_cutedsl` is added;
- both `SGLANG_DFLASH_DRAFT_HEAD*` variables are removed.

From 2026-10-23 to 2026-10-24 all traffic went to the whole-card AWQ revision
`SERVICE-full-arh-m1`, which stays on the service as the rollback target: image
`qwen38-sglang:v0.5.22-patched6-arh` (pinned by digest),
16 slots and `minScale: '1'`. On the night of the switch, two whole-card revisions ran with
`minScale: '0'`: `SERVICE-full-e`, with the same flags and the image before the
reasoning-history patch, and `SERVICE-full-arh`, the same as production. The last half-card
revision is `SERVICE-s4b8-fp32-tc3`, with four slots, `--max-mamba-cache-size 22`,
`--cuda-graph-max-bs-decode 4`, 4,096-token prefill chunks, `containerConcurrency: 8`, 16 vCPU,
64 GiB and `nvidia.com/gpu: 0.5`. Before it, `SERVICE-qhead-h2` was the two-slot configuration
of commit `ed0cb7e`, with `--max-running-requests 2`, `--max-mamba-cache-size 12`,
`--mem-fraction-static 0.80`, `containerConcurrency: 4` and an unpinned `google-cloud-storage`. All
run the same weights and drafter.

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

The production revision since 2026-10-24, `SERVICE-ownq8`, took 14 min 26 s from instance
start to ready on its first start (16:26:36 to 16:41:02 UTC, from its log):

- about 9 minutes to download the target and the drafter and start SGLang (`server_args` at
  16:35:40);
- 18 s to load the target weights;
- about 5 minutes for the pools, autotune, CUDA graphs and warmup.

The figures below are for the AWQ build.

On the whole card, the 16-slot test service took 13 min 24 s from instance start to a ready server
on 2026-10-23. Production keeps one instance warm (`minScale: '1'`), so a cold start occurs only
when a revision is replaced. While production ran with `minScale: '0'`, we measured part of a start
from zero: the weights downloaded in 465 s (54 MB/s) one time, and in 887 s another time, when
`gcsdl.py` failed once and retried. We never measured the time from zero to the first token, and
production went back to `minScale: '1'`.

On the half card, from instance start to a ready server took 824 s on the four-slot revision,
according to the timestamps of its startup log on 2026-10-23:

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

## Production, 2026-10-24

On 2026-10-24 production moved from the AWQ build to our mixed NVFP4/FP8 checkpoint (see
[Quantization](#quantization)). We added two revisions to the production service, each with 0%
traffic and its own tag, while all traffic stayed on the AWQ revision `SERVICE-full-arh-m1`.
They had the image and flags of production, with these differences:

- the weights `ownq/molh-w4h-bkv`, a BF16 KV cache and the CuTe-DSL NVFP4 GEMM;
- no int8 draft head;
- `SERVICE-ownq8` kept 8 draft tokens per block, and `SERVICE-ownq16` used
  `--speculative-num-draft-tokens 16`.

We ran every check against the tag URLs, then moved all traffic to `SERVICE-ownq8`. The AWQ
column comes from the test services of [Full card, 2026-10-23](#full-card-2026-10-23), with the same
flags and the same benchmark. Its one-stream coding-agent row is the deployed revision with the
reasoning-history patch, three runs. Its two- and three-stream rows come from the same flags without
the patch.

| | AWQ, 8 draft tokens (previous) | **mixed, 8 draft tokens (deployed)** | mixed, 16 draft tokens |
| --- | ---: | ---: | ---: |
| Needle 30k / 120k, math | 5/5, 5/5, 12/12 | 5/5, 5/5, 12/12 | 5/5, 5/5, 12/12 |
| `qcon.py`, alone / three streams | – / 64/64 | 63/64 / 63/64 | 64/64 / 64/64 |
| Parallel tool calls, `/v1/messages` (8 cases × 2; non-streaming + streaming) | 16/16; 8 of 8 | 16/16; 8/8 + 8/8 | 16/16; 8/8 + 8/8 |
| `toolstream.py`, 1,500 integers: longest silence, call | 0.1 s (3,000 integers, half card) | 0.1 s, 15.5 s | 0.1 s, 10.7 s |
| `prefix.py`, facts / prompt served from cache | 10/10 | 10/10, 99.6% | 10/10, 99.6% |
| `apitest.py` (400 for grammar, tool calls, cache report, 1 MP image) | pass | pass | pass |
| bench2, greedy, mean of seven | 231.7 | **235.4** | 259.5 |
| bench2, prose / chat, greedy | 145.3 / 164.2 | 153.9 / 176.1 | 133.3 / 145.3 |
| bench2, code / math / JSON | 320.1 / 296.0 / 342.5 | 320.4 / 313.5 / 332.0 | 392.9 / 369.0 / 439.8 |
| Coding agent, one stream, at 30k / 60k / 120k | 207 / 207 / 189 (n3) | **220 / 199 / 192** (n8) | 221 / 170 / 170 (n8) |
| Coding agent, per stream at two streams | 175 / 148 / 131 | 168 / 152 / 134 | 165 / 165 / 124 |
| Coding agent, per stream at three streams | 156 / 126 / 80 | 156 / 125 / 99 | 167 / 128 / 105 |
| Coding agent, time to first token | 5.5 / 12.8 / 32.6 s | **2.5 / 6.2 / 17.2 s** | 2.5 / 6.2 / 17.2 s |
| Coding agent, follow-up time to first token | 0.34 / 0.46 / 1.54 s | 0.39 / 0.70 / 1.01 s | 0.38 / 0.69 / 1.22 s |
| Tokens per verification, one stream | 4.31 / 4.53 / 4.39 | 4.51 / 4.34 / 4.40 | 5.01 / 4.33 / 4.78 |
| KV pool | 1,011,164 FP8 | 492,440 BF16 | 492,440 BF16 |

Notes on the table:

- **Coding-agent rows:** medians over all runs. The two-stream rows have four runs each and the
  three-stream rows six.
- **Follow-up turn:** it carries the earlier thinking in the history. The server reported 94–99%
  of its prompt as served from cache.
- **Three streams at 120k tokens:** three concurrent sessions with 118k-token prompts ran with no
  request errors.
- **Missed `qcon.py` answers on ownq8:**
  - Alone, one ordering puzzle got a wrong order.
  - At three streams, a different ordering puzzle fell into a greedy repetition loop until
    `max_tokens`.
  - No answer contained a word from the other requests. This is the only check where the mixed
    build scored below AWQ.

**8 against 16 draft tokens.** We decided on the one-stream coding-agent decode beforehand: 16 had
to be equal or better there, and no more than about 5% worse at two and three streams.

- At one stream, 16 tied at 30k tokens (221 against 220) and lost 14% at 60k and 12% at 120k
  (medians of eight runs each).
- At two and three streams the difference ran from −7% to +8% across sizes, within the noise of
  four to six runs.
- bench2 rises 10% with 16, but only because code, math and JSON get 15–33% faster. Prose and chat
  get 13–18% slower.
- So production keeps 8. This repeats what we saw with blocks of 12 and 16 on the AWQ build.

**Cutover.** We moved all traffic with:

```bash
gcloud run services update-traffic SERVICE --region REGION --to-revisions SERVICE-ownq8=100
```

Then we removed the `ownq8` and `ownq16` tags. A Cloud Run revision without traffic and without a
tag keeps no instance, whatever its `minScale`, so the AWQ and 16-token revisions released their
GPUs. The logs show `SIGTERM` within seconds of the change:

- 16-token revision: tag removed at 17:31 UTC, `SIGTERM` at 17:32:00.
- AWQ revision: `SIGTERM` at 17:43:15, three seconds after the traffic change, with no request in flight.

The AWQ revision `SERVICE-full-arh-m1` stays on the service as the rollback target:

```bash
gcloud run services update-traffic SERVICE --region REGION --to-revisions SERVICE-full-arh-m1=100
```

It has no warm instance, so a rollback waits through a cold start of about 14 minutes. During that
time Cloud Run keeps serving from the current revision until the AWQ instance is ready.

What we learned from the rollout:

- Tagging two new GPU revisions in a single `services replace` made the route wait for both. When
  its deadline passed, the first revision lost its instance and needed a second cold start.
- Add tagged GPU revisions one at a time.

### Eight slots, 2026-10-24

Later on 2026-10-24 Han chose 8 concurrent requests for production instead of 16. The new revision
`SERVICE-ownq8-c8` has the flags of `SERVICE-ownq8` with these differences only:

- `--max-running-requests 8` instead of 16;
- `--max-mamba-cache-size 42` (5 × 8 + 2) instead of 82;
- `--cuda-graph-max-bs-decode 8` instead of 16;
- `containerConcurrency: 16` instead of 32.

It was added with 0% traffic and the tag `con8`; it was ready after a cold start of 13 min 5 s. On
the tag URL:

| | 16 slots (`ownq8`) | **8 slots (`ownq8-c8`, deployed)** |
| --- | ---: | ---: |
| `/v1/loads` `max_running_requests` | 16 | 8 |
| KV pool, BF16 | 492,440 tokens, 30.06 GB | **566,383 tokens, 34.57 GB** |
| Free after startup | 12.90 GB | 14.72 GB |
| Needle 30k / 120k, math | 5/5, 5/5, 12/12 | 5/5 + 5/5, 5/5, 12/12 |
| Parallel tool calls, `/v1/messages` (non-streaming + streaming) | 8/8 + 8/8 | 8/8 + 8/8 |
| `toolstream.py`, 1,500 integers: longest silence, call | 0.1 s, 15.5 s | 0.1 s, 15.6 s |
| `/v1/messages` with thinking | – | pass |
| Coding agent, one stream at 30k, decode | 220 (n8) | 211 (n2: 220.1, 202.6) |
| Coding agent, time to first token at 30k / follow-up | 2.5 / 0.39 s | 2.55 / 0.54 s |

The one-stream decode is within the ±10% per-request noise of the coding-agent benchmark. We then
moved all traffic and removed the tag in one command:

```bash
gcloud run services update-traffic SERVICE --region REGION --to-revisions SERVICE-ownq8-c8=100 --remove-tags con8
```

The 16-slot revision got `SIGTERM` at 21:46:10 UTC, six seconds later, with no request in flight. It
stays on the service as the rollback target:

```bash
gcloud run services update-traffic SERVICE --region REGION --to-revisions SERVICE-ownq8=100
```

## AWQ build, 2026-10-23 (previous production)

From 2026-10-23 to 2026-10-24, production served the AWQ-INT4 build (`quant/`, described in
[Checkpoint](#checkpoint-awq-build-production-until-2026-10-24)) from revision
`SERVICE-full-arh-m1`, which stays on the service as the rollback target. This was the headline
table of this README while it served:

| | |
| --- | --- |
| Model | [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), quantized here with the AWQ-INT4 recipe of [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) (`quant/`), served as `Qwen3.8-27B-abliterated-AWQ-INT4` |
| Weights | INT4 W4A16 (AWQ, group 32, asymmetric) with a dense BF16 `lm_head`, 19.12 GiB resident |
| Drafter | [DFlash 2](https://inco.ai/blog/dflash2/) [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), quantized to FP8 at load (2.53 GiB), blocks of 8 tokens |
| Engine | SGLang v0.5.22 (`lmsysorg/sglang:v0.5.22`) with six patches in `docker/`, FlashInfer attention and GDN prefill kernels |
| GPU | one whole NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) on Cloud Run, with 20 vCPU and 80 GiB |
| Slots | 16 concurrent requests, fp32 recurrent state |
| Context | **262,144** tokens per request; one 1,011,164-token FP8 KV pool shared by the 16 slots |
| Decode, greedy, thinking off | **231.7 tok/s** (prose 145.3, chat 164.2, translation 166.2, summary 187.7, code 320.1, math 296.0, JSON 342.5) |
| Decode, greedy, thinking off, concurrent streams | 197.7 tok/s each and **562.5** in total at four streams, 170.8 each and **863.3** in total at eight, about 962 in total at 16 |
| Decode, sampled, thinking on, coding agent | 209 tok/s at 30k tokens of code context, 219 at 60k, 192 at 120k |
| Prefill | 5,417 tok/s at 9k tokens, 4,958 at 31k, 3,570 at 124k |
| Time to first token, coding agent | 5.5 s at 30k, 12.8 s at 60k, 32.5 s at 120k; a follow-up turn 0.34 s at 30k and 0.46 s at 60k |
| Refusals | 0 of 6 borderline prompts; the stock model refuses 5 |
| Quality | needle retrieval 5/5 at 30k and 120k, multi-step math 12/12, 64 of 64 checked answers at 16 concurrent streams; greedy text byte-identical to the four-slot flags on the same card on 24 of 24 and 14 of 14 prompts |
| Vision | native, preprocessed on the GPU, at most 1,024 tokens per image |
| Tool calls | arguments stream as they are generated, at most 0.1 s apart; earlier thinking reaches the template natively, so each tool round hits the prefix cache |
| Structured output | not supported: `response_format` and `tool_choice: "required"` return 400 |
| Disconnects | a client that disconnects frees its slot within a second |
| Cold start | 13 min 24 s from instance start to ready; one instance stays warm |

We measure decode with `scripts/bench2.py`, which streams each answer and divides the tokens after
the first by the time from the first to the last token. Each of seven workloads runs two prompts of
up to 800 tokens, greedy with thinking off unless stated. Sampled decode comes from
`scripts/accept.py`, fourteen prompts with thinking on, averaged over three runs, and
`scripts/bench3.py` sends the same fourteen prompts over 1, 2, 4 or 8 connections at once. Its
per-stream figure is the bench2 rate, and its total is every completion token over the wall time of
the round, so prefill and queueing count. Prefill is the time to first token on prompts of unique
random words, so that no prefix is cached. The coding-agent figures come from a separate benchmark
that streams `/v1/messages` with three tools, thinking on at the default effort of the template and
the default sampling of the model, over 29k, 59k and 119k tokens of SGLang source; a follow-up turn
is one more user message after the answer.

Production has served from a whole card since 2026-10-23, revision `SERVICE-full-arh-m1`. The
greedy decode, concurrency, prefill and quality figures above were measured that day on a test
service with the flags of production and the image before the reasoning-history patch, which
changes only how `/v1/messages` renders earlier thinking; bench2 and bench3 use chat completions.
They are medians of three runs. The coding-agent figures come from a second set of test services the
same night, and the follow-up time to first token was measured with the patch. Refusals, vision,
the 0.1 s tool-call streaming figure and disconnects were measured on the half card; refusals
depend only on the weights. Unless a section says otherwise, the sections
from [Checkpoint](#checkpoint-awq-build-production-until-2026-10-24) to [Vision](#vision) report measurements on the half card that
production used until 2026-10-23, and [Half card, 2026-10-23](#half-card-2026-10-23) keeps its
headline figures.


## Full card, 2026-10-23

Production moved to a whole RTX PRO 6000 (`nvidia.com/gpu: 1`, 20 vCPU, 80 GiB) with 16 slots,
`--max-mamba-cache-size 82`, `--cuda-graph-max-bs-decode 16`, 8,192-token prefill chunks and
`containerConcurrency: 32`; block 8, fp32 state, the int8 draft head and block verification stay.
We measured six configurations on test services with the image digest of the last half-card
revision, the same weights and drafter, in the same region and from one client over the internet.
Each got one warm-up round, then three runs of `bench2.py`, three of `bench3.py` and two of
`prefill.py`; the table gives medians. Token-weighted is as in the
[tuning study](#tuning-study-half-card-2026-10-23).

| | A: four slots × 8 | B: 8 × 8 | C: 8 × 12 | D: 8 × 16 | **E: 16 × 8, chunk 8192 (deployed)** | F: B without int8 head or block verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bench2 | 231.7 | 230.4 | 241.2 | 247.6 | **231.7** | 221.5 |
| Token-weighted, one stream | 205.1 | 203.6 | 199.0 | 202.8 | 204.3 | 194.7 |
| Total, 4 streams | 563.7 | 562.6 | 548.5 | 532.5 | 562.5 | 548.0 |
| Total, 8 streams | 573.1 (four run, four wait) | 863.1 | 655.1 | 540.2 | **863.3** | 837.0 |
| Prefill at 9k / 31k / 124k | 5,386 / 4,975 / 3,481 | 5,356 / 4,976 / 3,487 | 5,133 / 4,901 / 3,487 (one run) | | 5,417 / 4,958 / 3,570 | |
| KV pool | 1,232,992 | 1,159,049 | 1,159,049 | 1,159,049 | 1,011,164 | 1,159,049 |
| Same greedy text as A | reference | 24/24, 14/14 | 10/24, 4/14 | 9/24, 2/14 | **24/24, 14/14** | 16/24, 7/14 |
| Cold start | 13m28s | 12m56s | 12m50s | 16m36s | 13m24s | 14m32s |

The last column but one counts greedy answers byte-identical to A, of the 24 prompts of `drift.py`
and the 14 one-stream answers of `bench3.py`. Blocks of 12 and 16 repeat the pattern of the half
card: bench2 rises only because code, math and JSON get faster while prose, chat and translation
get slower, token-weighted decode does not rise, the total at eight streams falls and greedy text
changes. E runs 16 slots of block 8 with fp32 state and returns the same greedy text as A. Its
other checks: needle 5/5 at 30k and 120k, math 12/12, `qcon.py` 64/64 at 16 streams, and a
259,728-token prompt served. With thinking on and sampling, bench2 gives 139.9 tok/s for prose and
153.7 for chat.

A second set of test services measured a coding agent on E with thinking on at the default effort
of the template, `xhigh`, which is its highest; the template accepts only `xhigh`, `medium` and
`low`, and a client that sends a thinking budget without an effort gets `xhigh`, since SGLang does
not pass the budget to the template. One stream, medians
of nine runs for E, four slots and no block verification, three for the others:

| | Decode at 30k / 60k / 120k | Time to first token | Follow-up, time to first token at 30k / 60k |
| --- | ---: | ---: | ---: |
| **E** | 209 / 219 / 192 tok/s | 5.5 / 12.8 / 32.5 s | 1.05 / 1.34 s |
| block 16 | 190 / 213 / 171 | 5.5 / 12.8 / 32.6 | 1.24 / 1.54 |
| block 12 | 206 / 210 / 179 | 5.6 / 12.8 / 32.7 | 1.24 / 1.23 |
| four slots | 211 / 211 / 196 | 5.5 / 12.8 / 32.5 | 1.12 / 1.32 |
| no block verification | 206 / 209 / 196 | 5.5 / 12.8 / 32.5 | 1.21 / 1.39 |
| 16,384-token chunks, Triton GDN | 217 / 215 / 179 | 5.7 / 12.9 / 32.8 | 1.08 / 1.50 |
| **E + reasoning-history patch (deployed)** | 207 / 207 / 189 | 5.5 / 12.8 / 32.6 | **0.34 / 0.46** |

No flag beat E beyond the noise of sampled decode. The one gain was the reasoning-history patch,
described in [Thinking across turns](#thinking-across-turns), which production has served since the
same night. With it, needle retrieval is 5/5, math 12/12, `qcon.py` 64/64 at three streams, and
parallel tool calls come back valid 8 of 8 times.

## Half card, 2026-10-23

Until 2026-10-23, production served from half a card (`nvidia.com/gpu: 0.5`, 16 vCPU, 64 GiB) with
four slots, revision `SERVICE-s4b8-fp32-tc3`. Its headline figures, measured as described at
the top of this README:

| | |
| --- | --- |
| GPU | ½ NVIDIA RTX PRO 6000 Blackwell (47.64 GiB, SM120), MIG `2g.48gb` |
| Slots | 4 concurrent requests, fp32 recurrent state |
| Context | 262,144 tokens per request; one 266,083-token FP8 KV pool shared by the four slots |
| Decode, greedy, thinking off | 140.5 tok/s (prose 89.2, chat 98.1, translation 99.3, summary 118.0, code 193.4, math 180.8, JSON 204.6) |
| Decode, greedy, thinking off, four streams | 120.7 tok/s each, 348.3 tok/s in total; 217.4 with the former two slots |
| Decode, sampled, thinking on | 126.2 tok/s, 3.60 tokens per verification |
| Decode, greedy, thinking off, long context | 75.5 tok/s at 34k tokens, 65.8 at 115k |
| Prefill | 2,783 tok/s at 9k tokens, 2,568 at 31k, 1,793 at 124k |
| Quality | needle retrieval 5/5 at 30k and 120k, multi-step math 12/12, 64 of 64 checked answers at four concurrent streams; greedy text byte-identical to the two-slot configuration on 104 of 104 prompts |
| Cold start | 14 min |

The single-stream decode was measured on the four-slot production revision. The four-stream,
sampled and quality figures come from a test service with the same image, weights and flags at
`--mem-fraction-static 0.80` (production ran 0.79, which changes the pool size and the free memory,
not the computation), as medians of five runs. Long-context decode and prefill were measured on the
earlier two-slot configuration and not repeated at four slots.

## Tuning study, half card, 2026-10-23

We measured five configurations on the half card on test services with the same image digest, weights and drafter as
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

On the four-slot half-card production revision, `bench2.py` gives 140.5 tok/s (prose 89.2, chat 98.1, translation
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
tok/s for the eight-slot option and 360.3 for the then deployed four slots.
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
quantized for production since 2026-10-24 with
[NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (`quant/modelopt/`, the precision
layout of the edp1096 mixed checkpoint with our own calibration and scale search), and before that
with the recipe and calibration set of
[`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) through
[llm-compressor](https://github.com/vllm-project/llm-compressor).
[Discussion #192](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192) compares the published
4-bit checkpoints. The L4 companion of this deployment is
[`Qwen3.8-27B-UD-Q4_K_XL-L4`](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4).
