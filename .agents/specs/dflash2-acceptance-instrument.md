# SPEC-DFLASH2 — record acceptance on OUR arm, so the 0.928x gap has a cause

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#2832](https://github.com/mudler/vllm.cpp/issues/2832).
**Parent row spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Kind:** instrument spec, committed before the implementation in the same pull
request (the one-pull-request shape recorded for this row).

## Now

`ACTIVE`. The instrument and its tests are written and gated on fixtures. The
gate itself has not been re-run: it needs `dgx:gpu0`, and this change deliberately
takes no lease.

## The gap, and why nothing can attribute it

`.agents/benchmark-record.md` records ours **14.951** tok/s against vLLM
**16.111** (0.9280x). A same-workload SGLang run folds by the gate's own
predicate to **16.034** (0.9325x). Two independent oracles agree within 0.5% of
each other, so the deficit is an engine property rather than a denominator
artefact — and nothing in the record says WHICH property.

Both oracles record acceptance. Our arm records none:

| engine | tokens per verify step | accept rate | where it is read |
|---|---:|---:|---|
| vLLM | 4.23 | 46.1% | `metrics."vllm:spec_decode_num_accepted_tokens"` = 1000, `..._num_draft_tokens` = 2170, `..._num_drafts` = 310 |
| SGLang | 4.67 | — | `accept_length`, i.e. `completion_tokens / spec_verify_ct` |
| ours | NOT RECORDED | NOT RECORDED | `our-arm.json` legs carry `completion_tokens, finish_reason, generate_start_unix, generate_end_unix, prompt_tokens, run, secs, tok_s` and nothing else |

The two oracles reach the same throughput by different balances — SGLang accepts
more per step than vLLM and lands within 0.5% of it — so throughput alone does
not identify the lever. If we accept about 4.2 tokens per step the deficit is
execution speed and the `A2` async chain is the right ladder; if we accept
materially less it is drafter quality or acceptance collapse and that chain
closes nothing. Choosing between those with no measurement is guessing, and this
row has twice seen acceptance move under configuration: acceptance collapsing to
zero for 40% of a concurrent run, and an MMA kernel arm moving it from 8.7% to
41.6%.

## Scope

An INSTRUMENT. The decode path, the drafter, the veto and every product
behaviour are unchanged. The one product edit is an ABI accessor that READS
three counters the engine already increments; it computes nothing and changes no
decision.

## The counters already exist, and they are not reachable

`src/vllm/v1/worker/gpu/runner.cpp`, in the write-back that follows each
`execute_model`, already keeps exactly the three quantities the oracles publish:

- `spec_drafts_proposed_ += kr` — draft tokens the target VERIFIED.
- `spec_drafts_accepted_ += (ns > 1 ? ns - 1 : 0)` — the drafts the rejection
  sampler ACCEPTED. The `- 1` is the bonus/replacement token the verify step
  always emits, so this count EXCLUDES it.
- `spec_drafts_proposed_by_depth_[0] += 1`, once per row whose `kr > 0` — the
  number of verify steps that carried at least one draft.

`examples/bench/bench_core.h` reads the first two through `loaded->runner()` and
reports "Acceptance rate (accepted/proposed)". That path is closed to the gate:
the harness drives `examples/cli` (`vllm-cli`), which is a pure client of
`include/vllm.h` and includes no engine header, and the C ABI carries no
speculative telemetry at all. §Shared seams requires a shipped capability to be
reachable through the ABI, so the ABI is where it goes.

`spec_drafts_proposed_by_depth_[0]` is used rather than a fourth counter because
it is the same already-computed value: the loop increments index 0 exactly once
for every row that carried a draft, which is the definition of a verify step
with drafts. Adding a counter would be product code that computes something new.

WHAT PINS THAT IDENTITY TODAY, exactly and no more.
`tests/vllm/v1/spec_decode/test_mtp_depth.cpp` asserts
`sum(by_depth) == spec_drafts_proposed()` and, on its k=3 fixture,
`by_depth[0] == by_depth[1] == by_depth[2]`. Together those force
`by_depth[0] == proposed / k` on that fixture, which IS the drafted-row count
there. That is an indirect pin over one fixture and not a direct assertion of
the identity in general; the new C-ABI cases hold the accessor's null and
zeroing contract and cannot reach it, because they build no engine. `## Owed` O3
carries the direct pin.

## Units, and the one-token error the two oracles invite

vLLM's `num_accepted_tokens` EXCLUDES the bonus token. SGLang's `accept_length`
INCLUDES it. Reporting one number under both names is a one-token-per-step
error, which at k=7 is a quarter of the whole quantity. Both are therefore
emitted, each named for its convention:

| field | meaning | bonus token | comparable to |
|---|---|---|---|
| `drafts_proposed` | draft tokens verified | n/a | `vllm:spec_decode_num_draft_tokens` |
| `drafts_accepted` | draft tokens accepted | EXCLUDED | `vllm:spec_decode_num_accepted_tokens` |
| `verify_steps` | verify steps carrying ≥1 draft | n/a | `vllm:spec_decode_num_drafts`, SGLang `spec_verify_ct` |
| `accept_rate` | `drafts_accepted / drafts_proposed` | EXCLUDED | vLLM's 46.1% |
| `tokens_per_verify_step` | `(drafts_accepted + verify_steps) / verify_steps` | INCLUDED | vLLM's 4.23, SGLang's 4.67 |

Nothing is re-derived in Python from token counts. `completion_tokens / secs`
cannot tell an accepted draft from the bonus token, and a harness that guessed
would produce exactly the conflation this table exists to stop.

## Design

1. **ABI v25.** `vllm_spec_acceptance` (three `int64_t`) and
   `vllm_engine_spec_acceptance(engine, out)`, reading the three counters off
   `LoadedEngine::runner()`. Cumulative over the handle's life; a caller that
   wants a per-leg figure subtracts two reads. Zero on an engine that never
   speculated. Refuses a NULL argument or a non-text handle like every other
   text entry point.
2. **`examples/cli`** snapshots the counters either side of each
   `vllm_complete` and prints the DELTA on its own stderr line, keyed by run,
   beside the two lines it already prints. A third line rather than a widened
   one, for the reason #1671 gave: the timing line is parsed by evidence and
   readers that predate this change, so its bytes do not move.
3. **`tools/bench/dflash2_our_arm.py`** parses that line positionally and
   attaches the three counts to the leg. A leg with a timing line and no
   acceptance line is a REFUSAL naming `examples/cli/main.cpp`, exactly as a leg
   with no span marker already is: a binary built before the instrument cannot
   drive this arm, and the alternative is a null field nobody notices.
4. **`tools/bench/dflash2_speed_harness.py`** folds it. `fold_acceptance` runs
   over the legs `is_warm_leg` selects — the SAME predicate the throughput
   median uses — and `fold_legs` calls it, so the two axes cannot come to
   describe different leg populations. Pooled totals, because a rate's estimator
   is a ratio of sums and not a median of ratios, PLUS the per-leg values, so
   the collapse shape stays visible.
5. **`acceptance_reasons`** refuses a run whose legs disagree about whether they
   carry the counters, whose proposals are zero under a declared drafter, or
   whose accepted count exceeds its proposed count.
6. **`build_speed_result`** renders the three arms side by side under
   `acceptance`, with each side's leg population named. Ours is warm legs only;
   vLLM's `vllm:spec_decode_*` counters are cumulative over the whole run,
   COLD LEGS INCLUDED, and that difference is recorded rather than hidden.

No axis, floor or verdict is added. This change measures; it does not gate.

## Reachability

The ABI function's production call site is `examples/cli/main.cpp`, on the
default `--repeat` path the gate drives — not a test and not an example
internal.

MEASURED, not asserted. Deleting the `if (spec_ok) { std::fprintf(...) }` block
in a scratch copy (680 bytes) and rerunning
`tests.tools.test_dflash2_speed_harness.CliMarkerRuntimeTest` gives rc 1 and 3
errors, all of them the harness refusing a leg with no acceptance marker. The
tree was restored byte-for-byte from a copy taken before the mutation and the
same three cases return rc 0. That case links the real `examples/cli/main.cpp`
against a stub `libvllm`, runs it, and reads what the process printed, so it
measures a capability rather than a class.

## Tests

`tests/tools/test_dflash2_speed_harness.py`:

- the acceptance line the CLI prints is the one the harness parses, read
  positionally out of `examples/cli/main.cpp`'s own format string;
- a leg with no acceptance line is a refusal naming the binary;
- the counts reach the leg and the fold is over the WARM legs only;
- `accept_rate` excludes the bonus token and `tokens_per_verify_step` includes
  it, on a fixture whose arithmetic distinguishes them;
- per-leg values survive the fold, and a leg that accepted nothing is visible in
  them rather than averaged away;
- partial, zero and impossible counts are refusals;
- the linked production binary reports a per-leg DELTA and not a cumulative
  total.

`tests/capi/test_capi.cpp`: the accessor's contract, its NULL refusals, and the
`>= 25` floor.

## Gates

CPU-only. `python3 -m unittest tests.tools.test_dflash2_speed_harness` is the
focused gate and it runs the production `examples/cli/main.cpp`: that suite
compiles it against a stub `libvllm` and reads what the process printed, so the
CLI half is executed and not only inspected.

`src/capi/vllm_c.cpp` and `tests/capi/test_capi.cpp` are compile-verified
(`g++ -fsyntax-only` against the real headers) and NOT linked or run. See
`## Owed` O2. The speed gate itself is not re-run here: it needs `dgx:gpu0`.

## Risks

- **R1.** `spec_drafts_proposed_by_depth_[0]` is an indirect reading of the
  verify-step count, and it is pinned only indirectly today -- see
  `## The counters already exist` and `## Owed` O3. What bounds the risk is that
  both counters are written inside the SAME `kr > 0 && !chunked_prefilling`
  guard, so a change that decoupled them would have to move one of two adjacent
  lines. That is a bound, not a gate.
- **R2.** The vLLM arm's acceptance is cumulative over cold legs while ours is
  warm-only, so the two are not exactly the same population. Recorded in the
  comparison block rather than corrected, because correcting it needs per-leg
  `llm.get_metrics()` deltas on the oracle arm and that is a separate change.

## Owed

- **O3.** A direct pin on `spec_drafts_proposed_by_depth()[0] == the number of
  request-steps that carried drafts`, asserted rather than inferred from a
  k=3 fixture's equal-depth profile. It belongs beside the existing per-depth
  assertions in `tests/vllm/v1/spec_decode/test_mtp_depth.cpp`, where an engine
  is already built. Owned by `SPEC-DFLASH2`, tracked by
  [#2832](https://github.com/mudler/vllm.cpp/issues/2832).
- **O2.** The capi suite is UNRUN on this change. `vllm_engine_spec_acceptance`
  and its two refusal cases are compile-verified only, because linking them
  needs the whole 559-translation-unit library and the authoring host was at
  load average 11 with 41 GiB free — the two conditions under which this
  repository has previously OOM-rebooted a box and hit ENOSPC. The function's
  body is four reads and two null guards, so the risk it carries is a wrong
  counter rather than a crash, and the harness suite's linked CLI already proves
  the ABI declaration is well-formed. Owned by `SPEC-DFLASH2`, tracked by
  [#2832](https://github.com/mudler/vllm.cpp/issues/2832).
- **O1.** Per-leg acceptance on the ORACLE arm, folded by the same warm
  predicate, so the comparison is between two identical leg populations. Owned
  by `SPEC-DFLASH2`, tracked by
  [#2832](https://github.com/mudler/vllm.cpp/issues/2832).
