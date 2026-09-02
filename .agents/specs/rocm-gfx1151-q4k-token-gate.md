# Spec — the Q4_K_M token gate on gfx1151

Row `BACKEND-ROCM`. Issue
[#2546](https://github.com/mudler/vllm.cpp/issues/2546).

Sibling records: [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the
quant-matched decode number this gate gates),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the SDMA fault that
made this measurement impossible until today),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (the CPU-tier numerics
defect this measurement is compared against), and
[#821](https://github.com/mudler/vllm.cpp/issues/821) (the Q4_K_M arm).

## Now

`ACTIVE` — the measurement is staged and has not yet run.

This spec is committed before the job that produces the number, and it declares
the verdict shape, the identity assertions and the stop conditions in advance,
so that no threshold can move after the fact.

## Scope

In scope: running the declared Qwen3.8-27B Q4_K_M token gate on `strix:gpu0`
(`gfx1151`, Radeon 8060S, ROCm 7.2.4) against pinned llama.cpp `b10451`, and
reporting `TOKEN_GATE=PASS` or `TOKEN_GATE=FAIL` with per-prompt divergence
indices, token ids and logit margins.

In scope: the comparison against the CPU tier's recorded 5-of-6 result on the
same artifact and the same prompts, which decides whether the two tiers carry
one shared term or two.

Out of scope, deliberately:

- **Every speed, latency and memory number.** `AGENTS.md` §Gates admits a
  performance result from an arm only after that arm's declared token gate
  passes. #2497 already had one measurement retracted for quoting a number from
  this arm. No figure of that kind is produced, recorded or quoted here, even
  as a by-product.
- **Fixing the numerics.** #2534 owns the CPU-tier defect and its fix is likely
  shared. This row measures and reports.
- **The SDMA cause.** #2511 owns it. `HSA_ENABLE_SDMA=0` is used here as a
  workaround with that provenance stated, never as a finding.

## The gate

Mirrors `docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`, which is
the CPU tier's run of the same declared gate
(`.agents/specs/qwen38-27b-quant-arms.md` §Gates). Reusing its method is the
point: a different method would not be comparable, and the comparison is the
most valuable output.

- Six raw completion prompts, no chat template, byte-identical to the file the
  2026-08-23 run used.
- 48 tokens each, greedy, concurrency 1, `ignore_eos`, MTP off on both sides.
- One artifact: `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, staged to
  worker-local `/tmp` and re-hashed there, because `/workspace` is CIFS and a
  CIFS mmap is not a run surface.
- Verdict: `TOKEN_GATE=PASS` only when the tokenizer matches on 6 of 6 and the
  generation matches on 6 of 6. Anything else is `FAIL`. The tolerance is exact
  and no band is reached for.

## Pinning the oracle's EXECUTED PATH, not only its revision

`.agents/oracles/llama-cpp.md` records, from a measurement on `thor:gpu0`, that
`b10451`'s greedy decode is **not deterministic across its own supported kernel
paths**: the stock `-nr/--no-repack` flag changes its own greedy tokens on 1 of
these 6 prompts, and its per-step self-perturbation of the final logits is min
0.2020, median 0.3790, max 1.3657. A gate that pins only the revision is
therefore under-specified.

This run records, for every oracle leg:

- `llama_print_system_info()` verbatim, which names the compiled feature set;
- `use_extra_bufts`, set explicitly rather than defaulted, and printed;
- `n_gpu_layers`, so the HIP and CPU legs are distinguishable in the record;
- the host architecture and the enumerated ggml devices;
- the source identity of the pinned tree, as a content manifest hash rather
  than a tarball hash, because gzip framing is not content.

Two oracle legs run, in one lease, from one build:

| leg | `n_gpu_layers` | what it pins |
|---|---|---|
| `hip` | 99 | the board's own kernel path, the denominator for an on-board gate |
| `cpu` | 0 | the x86-64 CPU path on the same host, the bridge to the CPU tier |

The second leg exists because the CPU tier's recorded oracle ids came from
**aarch64** `thor`. Comparing our gfx1151 ids straight to those would confound
three differences at once. An x86-64 CPU oracle leg on the same host, from the
same build, isolates the kernel-path term from the host term.

## The harness adaptation, and why it is not a second oracle

`llama-completion` at `b10451` prints token pieces and never token ids, so the
2026-08-23 run wrote `oracle_tokens.cpp`, which links the stock libllama built
at the pin, calls the public `llama.h` API only, and mirrors `completion.cpp`'s
own tokenize and detokenize choices. `AGENTS.md` admits this as an unavoidable
adaptation of the harness. This run reuses that source with three additions,
all of them recording rather than deciding: `n_gpu_layers`, `use_extra_bufts`
and the `system_info` line.

Chain of custody is re-established on this host rather than inherited: the
harness's detokenized prompt-plus-generation for prompt 0 is compared byte for
byte against a stock `llama-cli` run on the same recipe. A `DIVERGE` there
voids the oracle side and the job says so instead of reporting a gate result.

## Confirming the legs, not assuming them

Without `HSA_ENABLE_SDMA=0` this board faults on roughly 8 of 10 plain Q4_K
decode legs, and a fault is a `GPU Hang` or a `Memory access fault` that kills
the process. A single clean leg is therefore not evidence that the arm ran.

Every one of our legs sets `HSA_ENABLE_SDMA=0`. The gate leg is repeated N
times independently, each with its own process and its own model load, and the
record carries `failures / N` for our arm. A verdict is reported only from legs
that reached their own completion marker. Where two clean legs exist, their id
sequences are compared to each other, because an arm that does not reproduce
its own ids has no business being compared to anything else.

## Risks

- **The board faults anyway.** Contradicts a 10-leg result and is important
  news. Stop condition below.
- **A silent CPU fallback.** `gfx1151` is integrated, so the portable reference
  tier is reachable and a fallback would be invisible in the ids. The run sets
  `VT_OP_PROVIDER_STATS=1` and records the reference-tier hit count; a non-zero
  count means the leg measured the CPU tier wearing a ROCm label.
- **A stale binary.** `vllm-cli` is a ~26 KB thin client and the kernels are in
  `libvllm.so`. Both are hashed, and the source revision is printed by the build
  step rather than assumed from a directory name.
- **A previous run's log read as this run's verdict.** Logs are deleted before
  submission and every wait keys on the `rc` job UUID.

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-20260902.md`,
following the conventions of the 2026-08-23 CPU-tier document: what ran, the
measured identity chain, the `blk.64` asymmetry and how it was handled, the
oracle's chain of custody, the gate, and what is not admissible.

Raw logs on the share under `/mnt/nas_share/rc/rocm-tokgate-strix/`.

## Stop conditions

- The board faults with `HSA_ENABLE_SDMA=0` set: stop, report it, produce no
  gate verdict from a partial set of legs.
- The oracle's chain of custody is not `EXACT` or `PREFIX`: stop, the oracle
  side is not bound to the stock binary.
- A reference-tier hit count above zero on our arm: stop, the leg did not
  measure the ROCm tier.
- Anything that would need a speed number to interpret: stop, and say that the
  question is not answerable before the gate passes.

## Owed

- The fix for whatever divergence this measures. If the ROCm divergences match
  the CPU tier's indices and ids, #2534 owns it. If they do not, ROCm carries
  its own term and that term gets its own issue, filed against this row.
- #2497's quant-matched decode number stays blocked until this gate passes.
