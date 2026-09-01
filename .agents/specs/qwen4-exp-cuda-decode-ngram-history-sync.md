# `qwen4_exp` CUDA decode: the n-gram history is read before its copy lands

**Row:** `QWEN4EXP-TOKENDIV-2496`
**Issue:** [#2496](https://github.com/mudler/vllm.cpp/issues/2496)
**Sibling:** [#2476](https://github.com/mudler/vllm.cpp/issues/2476) — the illegal
memory access that `CUDA_LAUNCH_BLOCKING=1` suppresses. This spec says what it
does and does not claim about that one.
**State:** `ACTIVE`

## Scope

`qwen4_exp` on `--device cuda` produces a token 0 that matches the CPU control
exactly and then degenerates. This spec covers the PLE block's per-step n-gram
history transfer, which is the one piece of decode state this tree moves with a
raw `Backend::Copy` and then reads on the host without draining the queue.

Out of scope, each named under `## Owed`: the illegal access of #2476 (this
spec's fix is a candidate for it and is not asserted to be it), the `q_f32`
query-path width (#2488, #2502), and any throughput claim.

## The defect

`src/vllm/model_executor/models/qwen4_exp_ple_block.cpp`, one site, two hazards.
Both are live only when the n-gram history cache is NOT host-resident, which on
the production by-name path is exactly the CUDA arm (`qwen4_exp_registry.cpp`
publishes `g.states[3]` on `input.queue.device`).

**H1 — the READ is not drained, and it is DECODE-ONLY.**

```cpp
} else if (tokens_on_host) {
  std::memcpy(hist, tokens_row, hist_bytes);
} else {
  d.b.Copy(d.q, hist, tokens_row, hist_bytes);   // enqueued, not completed
}
```

`hist` is `stage.data()`, a zero-initialised host `std::vector<int64_t>`. Thirty
lines later the HOST reads it:

```cpp
hash_state.tokens.assign(hist, hist + ctx);
```

with no `Synchronize` between the two. `Backend::Copy` on CUDA is
`cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`
(`src/vt/cuda/cuda_backend.cu:116-118`) — it ENQUEUES. **`CUDA_LAUNCH_BLOCKING=1`
does not make `cudaMemcpyAsync` synchronous**; it serialises kernel launches.
So this hazard survives the serialisation that #2496 was measured under, which
is what makes #2496 deterministic where #2476 is not.

The branch is taken only when `past_len != 0`. At `past_len == 0` the history is
seeded with EOS on the host and no device read happens at all. **Prefill
therefore cannot see this defect and every decode step can**, which is precisely
the measured shape: index 0 agrees with the control and indices 1-7 do not.

When the host reads `stage` early it reads zeros, so the n-gram hash runs over
token id 0 repeated. The file's own comment at the seeding branch states the
consequence: "A port that trusted the zero-filled cache would hash token 0 into
the first `ngram_size - 1` positions of every sequence and get a fluent wrong
answer." A constant history gives a constant PLE contribution on every step,
which is a repeated token.

**H2 — the WRITE-BACK is enqueued out of a buffer that then dies.**

```cpp
if (tokens_on_host) {
  std::memcpy(tokens_row, hist, hist_bytes);
} else {
  d.b.Copy(d.q, tokens_row, hist, hist_bytes);   // reads `stage` later
}
```

`stage` is a function-local `std::vector` destroyed when `RunQwen4ExpPleBlock`
returns. Nothing keeps it alive and nothing drains the copy. A DMA still
scheduled against freed host memory is a use-after-free.

## Why this is a port defect and not a judgement call

The sibling translation unit already states the contract this one violates.
`qwen4_exp_qsa_block.cpp` moves words the same way and says so at the definition:

> `StageHostWords` only ENQUEUES the read; nothing is readable until the caller
> synchronises the queue once for every range it staged.

and its `HostWords` wrapper ends in `d.b.Synchronize(d.q)` with the comment
"the device arm cannot forget it". The PLE block forgot it. This is one rule,
already written down in this row, applied at the one site that does not follow
it — not a new policy.

`Backend::Synchronize` is a no-op on the base class, so the CPU arm pays nothing
and its bytes are unchanged.

## Upstream anchor

vLLM `e126687a9a` `[Model] Support Qwen3.8-Flash-Next (#53896)`, a FORWARD
reference: it is 1,565 commits ahead of the pin in `.agents/upstream-sync.md`
and is cited for what decode state must persist, not as a mirror source.
`vllm/models/qwen4_exp/nvidia/model_state.py:65-92` rebuilds `ngram_context`
from `all_token_ids` every step and pads left with EOS only where the position
is negative; a decode step's window is `tokens[P-(N-1) … P-1]` and must contain
the previously generated ids. A window stuck at the EOS/zero seed is the
degenerate case, and it is what H1 produces on a device queue.

## Design

Drain the queue on both sides of the staged row, mirroring `HostWords`:

1. after the device READ, before any host use of `hist`;
2. after the device WRITE-BACK, before `stage` leaves scope.

Both calls are unconditional in the device branch and absent from the host
branch, so the CPU arm is byte-identical.

## Risks

- **A synchronise can CURE a race rather than fix it.** #2476 is timing
  dependent and a drain could hide it. This spec therefore claims the drain as
  the fix for #2496 (deterministic, reproduces under serialisation) and lists
  #2476 as a candidate consequence to be MEASURED, never asserted.
- Two extra stream synchronises per PLE layer per step is a decode cost. It is
  recorded under `## Owed` rather than hidden; correctness first.

## Tests

- A red-first hermetic case driving the block over a backend whose `Copy`
  DEFERS until `Synchronize`, which is what a CUDA stream does and what the CPU
  backend cannot express. Red before the change for the intended reason (the
  step-1 history reads zeros), green after.
- The end-to-end arm: the released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S
  artifact through `examples/server` on `thor:gpu0`, `--device cuda`, greedy,
  `max_tokens=8`, against the CPU control taken in the same lease.

## Gates

The completion condition is the GPU emitting the CPU control sequence
`11751 13 15767 411 2029 11 1092 369`, not merely emitting tokens.

## Owed

- The measurement that says whether H2 is #2476's illegal access. Not claimed.
- The `q_f32` query-path width (`qwen4_exp_qsa_block.cpp:694`). vLLM keeps the
  query at the model dtype; this tree widens it to f32 and hands it to
  `vt::RmsNorm` beside a bf16 gamma. It is a real memory-format parity debt
  under AGENTS.md "Inherit vLLM defaults" and a token gate cannot see it.
  **It is NOT this defect**: the buffer is allocated unconditionally with no
  device branch and no prefill/decode fork, so it was equally present on the CPU
  run that produced the correct control sequence, and a path that runs
  identically in prefill and decode cannot yield a correct prefill token and
  wrong decode tokens. Owned by #2488 and draft PR #2502.
- The decode cost of the two added synchronises, unmeasured.

## Now

`ACTIVE`.
