# `qwen4_exp` on `--device cuda`: the decode identifiers, before and after #2496

One artifact, one prompt, one device, three server runs. This file records what
was measured; the cause and the fix are argued in
[`.agents/specs/qwen4-exp-flash-next.md`](../../.agents/specs/qwen4-exp-flash-next.md)
under `## DECODEDIV`.

## What was run

| | |
|---|---|
| device | `thor:gpu0`, NVIDIA Thor, compute_cap 11.0, driver 595.78, nvcc 13.0.88, inside an `rc` lease |
| artifact | `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, shard 1 `Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`, sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, staged on worker-local disk |
| entry point | `examples/vllm-server`, `POST /v1/completions`, `--block-size 16 --num-blocks 128 --max-model-len 256` |
| request | prompt `The capital of France is`, `max_tokens=8`, `temperature=0`, `logprobs=1` |
| env | `VT_CPU_QUANT_REPACK=0 VT_LOAD_STATS=1 VT_Q4EXP_STATE_FP=1`. **No `CUDA_LAUNCH_BLOCKING`** — this is the production configuration |
| before | tree `f10b3953b6bac7252a0fb3095a5c9de3907877ff` |
| after | tree `1ef7885ec28b4edde88c665a9c1ff00115b63e4a`, server sha256 `082451c96aa20ec8ee34f1ed429972882cf0f8768294914ae2ca6d0691686663` |

Both runs answered HTTP 200 with `completion_tokens=8`, 11 completed steps, and
no `illegal memory access` line.

## The tokens

| arm | token ids | text |
|---|---|---|
| `--device cpu`, both trees | `11751 13 15767 411 2029 11 1092 369` | ` Paris. Given this fact, what is` |
| `--device cuda`, BEFORE | `11751 271 271 271 271 271 0 0` | ` Paris\n\n\n\n\n\n\n\n\n\n!!` |
| `--device cuda`, AFTER | `11751 13 15767 411 1928 11 628 567` | ` Paris. Given this information, can we` |

**The CPU control was re-taken on each tree rather than inherited.** It is
byte-identical both times.

## The tensor that named the cause

`VT_Q4EXP_STATE_FP=1` prints one summary line per persistent state per step. The
PLE n-gram history is int64 TOKEN IDS, so its disagreement can never be a
rounding difference, and it is where the two arms first part:

| step | `--device cpu` | `--device cuda`, BEFORE | `--device cuda`, AFTER |
|---|---|---|---|
| 0 (prefill, T=5) | `[9338, 369]` | `[9338, 369]` | `[9338, 369]` |
| 1 | `[369, 11751]` | `[369, 0]` | `[369, 11751]` |
| 2 | `[11751, 13]` | `[0, 0]` | `[11751, 13]` |
| 3 | `[13, 15767]` | `[0, 0]` | `[13, 15767]` |
| 4 | `[15767, 411]` | `[0, 0]` | `[15767, 411]` |

Before the fix the FIFO rolled correctly and what it PUSHED was `0`: the forward
was handed token id 0 at every decode step, because it read the host `token_ids`
the asynchronous runner deliberately leaves stale for decode rows. After it, the
history carries the sampled ids on both arms.

## What is still open, and it is NOT #2496

The CUDA arm is fluent and agrees on five of eight tokens. It is not
token-exact, and the residual is present at PREFILL, before any decode state is
read:

```
step=0 hidden  cpu   n=12800 nonfinite=0 maxabs=192 sumabs=28054.1 v=0.353516,-0.171875,3.42188,0.486328
step=0 hidden  cuda  n=12800 nonfinite=0 maxabs=192 sumabs=27964.7 v=0.326172,-0.132812,3.42188,0.470703
```

About 0.3% on the aggregate, on the first forward, compounding until a near-tie
flips at token 4. That is [#2547](https://github.com/mudler/vllm.cpp/issues/2547)
and it needs a per-layer tap to attribute. **Token 0 agreed all along only
because ` Paris` after this prompt is not a near-tie**, which is why "the prefill
is right" had to be checked with the fingerprint rather than inferred from one
argmax.

Op-level arm-against-arm gates on the same binary: `test_qwen4_exp_cuda`
351/351, `test_qwen4_exp_cuda_reductions` 160/160.
