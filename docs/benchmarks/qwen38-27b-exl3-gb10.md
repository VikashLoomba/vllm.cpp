# `qwen38-27b-exl3-gb10` — Qwen3.8-27B EXL3 3.5bpw, with and without its DFlash2 draft

The published EXL3 pair runs on GB10. This file records what was measured, on
what, and — at least as importantly — the three ways the workload differs from
the number it is naturally compared against.

## Disposition

**Measured, NOT comparable to the published figure.** The speed is real and
reproducible. It is not a like-for-like reproduction of the upstream README's
47.5 tok/s, for the reasons under [Limitations](#limitations). Read the ratio
as "same regime on a favourable prompt", never as a win.

## Subject

| | |
|---|---|
| target | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac874c4018295da848e250f23511361cda4` |
| draft | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f0436269bca761b071f05319e8e04a87cc633f9` |
| device | NVIDIA GB10, compute capability 12.1, driver 580.173.02, nvcc 13.0, `sm_121a` |
| tree | `origin/main` `5649e07d2120df4c5d33fd1d245336490c790e2b`, pinned inside the job by tarball sha256 |
| binary | one `vllm-cli`, md5 `3bc87f47b5325468ce575d30114d7928`, serving **both** arms |

Both shard sha256 values were **recomputed on the device** and matched the
download-host pins, so the bytes measured are provably the pinned artifact:
`7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6` and
`411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2`.

## Method

Both arms ran **interleaved in one process, one boot, one binary** — target,
draft, target, draft — because a sequential A/B measures drift as well as the
arm, and this repository has a recorded case of one unchanged binary reading
36.8 and 78.9 tok/s in the same session. Run 1 of every leg is a cold run and is
discarded, which is the harness' own convention on every arm.

```sh
VT_DFLASH_PAGED=0 vllm-cli --model <target> --device cuda \
  --prompt 'The capital of France is' --max-tokens 64 --temperature 0 \
  --seed 0 --repeat 5 --max-num-seqs 1 \
  [--speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":7}']
```

## Results

| arm | warm tok/s, runs 2-5 of two interleaved legs | spread |
|---|---|---|
| target only | 16.706 16.758 16.796 16.729 / 16.737 16.769 16.670 16.701 | 0.75% |
| + DFlash2 draft, k=7 | 48.970 49.079 48.944 48.469 / 48.751 48.672 48.677 48.446 | 1.3% |

**2.91x from speculation.** The two target legs are separated by a draft leg and
agree to 0.75%, so drift does not account for the difference.

## The MATCHED workload: real HumanEval at T = 0.6

The section above is a favourable prompt and says so. This section removes two of the
three differences from the upstream README's 47.5 tok/s: the real 164-problem
HumanEval set (`openai/human-eval` `data/HumanEval.jsonl.gz`, sha256
`1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`, ShareGPT-shaped
for the harness) and **T = 0.6**, their temperature. 128 output tokens, concurrency
1, seed 0, two interleaved legs on one binary (md5 `6ae2949042e256b707c2e75d0a547d9b`).

Counted **decode only**: `Mean per-stream decode rate` is `1 / mean_tpot` where
`tpot = (latency - ttft) / (output_len - 1)`, so prefill is excluded.

| arm | leg A | leg B |
|---|---|---|
| target only | 16.74 | 16.64 |
| + DFlash2 draft, k = 7 | **59.59** | **59.48** |

**Acceptance, measured rather than inferred**: 30,863 draft tokens proposed, 17,841
accepted, rate 0.58 -> **4.06 accepted per step** (5.06 emitted). The upstream README
states 4.43. So this is not a case of an easier drafting task: our acceptance is at or
below theirs and the throughput is higher anyway.

Every request produced exactly 128 tokens (`duration_s` 1355.86 x `output_throughput`
15.48 = 20,992 = 164 x 128), so no leg was skewed by early EOS.

**The counting convention is the open question, and it is theirs, not ours.** Their
README says "decode tok/s" and defines it nowhere. The same run of ours reads **59.5**
counted decode-only and **45.1** counted over whole-run wall time. Against the first
convention we are 1.25x; against the second, 0.95x. Their page pins no engine revision
either, so this cannot be settled from published material. Both numbers are given here
for that reason.

## The draft budget is a real lever, and our knee is not theirs

`pangoleen/qwen3.8-27b-dgx-spark-dflash2` serves the SAME DFlash2 drafter architecture
-- its `dflash_config` is byte-identical to ours, `block_size` 8, taps
`[5,19,33,47,61]`, `selector_rank` 256 -- at a verify budget of 16, and states: "the
drafter's block_size of 8 is not a cap on the verify budget: 8 -> 16 is +69.3%
edit-heavy and +10.5% fresh. 24 is past the knee."

Swept on our engine, real HumanEval at T = 0.6, **32 prompts** (a hotter subset than
the 164 above: acceptance 0.69 against 0.58 at the same k = 7), one sample per rung:

| k | decode tok/s | acceptance rate | accepted per pass | proposed |
|---|---|---|---|---|
| 0 (no draft) | 16.65 | - | - | - |
| 7 | 68.67 | 0.69 | 4.83 | 5,243 |
| **12** | **76.28** | 0.52 | 6.24 | 7,536 |
| 16 | 45.21 | 0.39 | 6.24 | 10,048 |
| 24 | 40.70 | 0.25 | 6.00 | 15,504 |

Each rung's engine-reported `block` equalled `k + 1`, so every budget genuinely took
rather than being clamped back to 8.

**Accepted tokens per pass saturates at ~6.0-6.24 from k = 12 onward**, so past 12 the
arm buys strictly more draft work for no more accepted tokens and throughput falls
monotonically. **Our knee is between 12 and 16; theirs is between 16 and 24.** The
reason is visible in the acceptance: at budget 16 they report 6.8-8.1 accepted per
pass where we saturate at 6.24. Their drafter extracts more from the same budget --
theirs is NVFP4 served by SGLang, ours the EXL3 quant, and our drafter's `block_size`
of 8 is being run at a block of 17.

**These rungs are one sample each and are indicative, not gated.** This repository has
a recorded case of one unchanged binary reading 36.82 and 78.86 tok/s in the same
session. The 41% fall at k = 16 is far too large to be noise, so the knee is real; the
+11.1% from k = 7 to k = 12 is exactly the size where a single sample is weak evidence.
A repeated k = 7 vs k = 12 comparison on the full 164-problem set is owed and is what
should be quoted.

## Correctness, which is the result that gates the other one

**The two arms emit token-identical output.** Both continue `The capital of
France is` into the same list of European capitals, ending mid-phrase at the
same token. Speculative decoding must not change greedy output, and here it
does not. The speed figure is only admissible because this passed.

The target arm alone was also checked from cold in a separate job: greedy, 16
tokens, `rc=0`, ` Paris. The capital of Germany is Berlin. The capital of Italy
is`. A wrong codebook on this format yields a correctly distributed and entirely
wrong weight, so coherent, factually correct continuation is meaningful evidence
and not merely a smoke test.

## Limitations

**Three differences from the upstream README's 47.5 tok/s, and all three favour
this measurement.** Do not quote the two numbers side by side without them.

1. **Workload.** The published figure is HumanEval-style at **T = 0.6 with
   acceptance 4.43**. This is **greedy, T = 0**, on `The capital of France is`,
   which continues into a list of capitals — close to the easiest possible text
   for a drafter to predict, so acceptance sits near its ceiling. The engine did
   not report an acceptance rate for this run, so the gap cannot even be
   quantified from here.
2. **Context and KV cache.** The published recipe is `-cs 262144` with an NVFP4
   KV cache. Here auto-fit reduced `max_model_len` from 262144 to 8192 and
   `max_num_seqs` from 32 to 1 on the KV budget, and no NVFP4 KV cache was used.
3. **The paged draft route was disabled.** `VT_DFLASH_PAGED=0` was required:
   with the paged route on, run 1 of 5 completes and the engine dies on run 2
   with `cudaMemcpyAsync: an illegal memory access`, resurfacing at `cudaFree`.
   That is [#2274](https://github.com/mudler/vllm.cpp/issues/2274), a
   pre-existing fault, here reproduced on a checkpoint it had never been seen on
   — which shows it is not specific to the bf16 drafter it was found with. Both
   arms above ran with it off, so the comparison is internally consistent, but
   neither is the shipped default configuration.

**One prompt, one length, one device, one boot.** 64 tokens, five runs per leg,
two legs per arm. No multi-request batching, no long context, no second box.

## Owed

- ~~A HumanEval-style prompt set at T = 0.6 with the acceptance rate reported~~ —
  **done**, see the matched-workload section above: 59.5 tok/s at acceptance 4.06
  per step against their 47.5 at 4.43.
- A repeated `k = 7` vs `k = 12` comparison on the **full 164-problem set**. The
  budget sweep above ran on a 32-prompt subset that drafts hotter than the full
  set, so its 76.28 is not transferable and must not be scaled onto the 59.5.
- The remaining unmatched axes against the upstream recipe: context (8192 here
  against their `-cs 262144`) and KV dtype (bf16 here against their `-cq nvfp4`,
  which this engine refuses by name — [#2620](https://github.com/mudler/vllm.cpp/issues/2620),
  whose named owner row did not exist). `vllm-bench` also cannot select a KV
  dtype at all ([#2619](https://github.com/mudler/vllm.cpp/issues/2619)), so no
  number on this page states the KV dtype it was measured on.
- [#2274](https://github.com/mudler/vllm.cpp/issues/2274), so the shipped paged
  route can be measured rather than routed around.
- [#2570](https://github.com/mudler/vllm.cpp/issues/2570): our `m <= 8` EXL3
  GEMV instantiates `(3,1)` only, and this checkpoint contains **zero** `(3,1)`
  tensors while upstream's GEMV takes 407 of its 409. That is the named,
  still-unmeasured hypothesis for any gap that a matched workload reveals.
