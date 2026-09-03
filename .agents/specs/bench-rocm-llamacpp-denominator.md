# Spec — the llama.cpp denominator for Qwen3.8-27B Q4_K_M on gfx1151

Row `BACKEND-GATE-ROCM-LLAMACPP`. Issue
[#2497](https://github.com/mudler/vllm.cpp/issues/2497), which owns the
quant-matched decode number on `strix:gpu0` and already carries one retraction
for taking a cross-engine number ahead of the correctness gate.

Sibling records: [#2381](https://github.com/mudler/vllm.cpp/issues/2381) (AMD
clock recording, which no in-tree harness does), and
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) (the allocator defect
whose fix made this board finish a leg at all).

Predecessor evidence:
[`rocm-strix-qwen38-q4km-20260901.md`](../../docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md),
whose speed axis is marked INADMISSIBLE and quotable as nothing.

## Now

The oracle side is measured and recorded. Our arm's side is staged and
deliberately unrun.

**The denominator is `12.233 tok/s`**, the median of 6 legs, spread `0.303%` of
the median, 6 of 6 legs `rc=0`, taken on 2026-09-02 in `rc` job
`ff18a029-cd10-42d1-a5f7-9129c1c8af09`. It reproduces the 2026-09-01 lease's
three legs on a different lease. The executed path is pinned by the source
content manifest and by the sha256 of every binary and shared object the run
linked, because this oracle's greedy decode is not deterministic across its own
kernel paths and the revision alone under-specifies it. The job checked that
manifest only against a constant in its own script, which fixes the tree but not
its provenance; the link to upstream was closed separately by reproducing the
manifest from `git archive` of `10bf611e…aebd8` out of `ggml-org/llama.cpp`,
where the tag `b10451` resolves to that commit and the file count matches at
3,425. `build_commit` reading `unknown` is a consequence of staging from a
tarball with no `.git`, not an unpinned build.

The staged arm refuses on `STRIX_ARM_SPEED_RATIFIED_BY`, and the refusal is
proven rather than asserted: **14 of 14 mutations of the guard are detected** by
`tests/scripts/test_rocm_strix_ourarm_staged.py`. Three of those mutations read
NOT DETECTED on the first pass and were repaired rather than recorded as passes
— the length floor and the issue-reference term were being tested as one
condition, and the reference-tier assertion was a substring test that
`VT_OP_PROVIDER_STATS_DISABLED=1` walked straight through. A fourth clause, a
bare `-z` test, was **deleted**: no mutation could break it, because the length
floor already decided the empty and the unset case, and an assertion nothing can
falsify is not a guarantee.

The staged arm's reference-tier assertion was itself defective when it was
written, and `tests/scripts/test_ltx25_ab_memwatch.py` caught it before it
landed. It read the count as `grep -c … || echo 0`, and `grep -c` prints its own
`0` and exits 1 on no match, so the fallback ran on top of the count and produced
the two-line value `0\n0`; an absent stderr capture also read as `0`, which is
exactly the clean reference-tier result the assertion exists to refuse. It now
tests the file is readable first and reports `UNREAD` when it is not, so a
missing capture can no longer pass for a clean one.

The `BACKEND-GATE-ROCM-LLAMACPP` matrix row stays `INVENTORIED`. Nothing here
makes the ROCm floor gateable: a floor is a comparison, and this row has only
one side of one. What the row's record gains is a measured denominator and the
withdrawal of a ratio it was still quoting after that ratio was retracted.

## What this spec is for, and what it refuses to do

`AGENTS.md` §Gates admits a performance result from an arm only after that arm's
declared token-exact gate passes. Our ROCm `gfx1151` arm's gate is
`TOKEN_GATE=FAIL`, 3 of 6 prompts
([`qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)),
so **no throughput, latency or memory figure for vllm.cpp may be produced,
recorded or quoted by this row**, and no ratio may be computed. #2497's first
comment did exactly that and had to be withdrawn.

What is admissible without our arm's gate is the **oracle alone**. llama.cpp's
own decode speed on this board is a single-engine fact. `AGENTS.md` §Gates and
§"When vLLM has no implementation" both require an oracle to demonstrably build
and run the model before it is trusted as a denominator, and
`.agents/benchmarking.md` says the same in one line: "Prove the oracle actually
*runs* the model before trusting it as a denominator." That proof is this row's
deliverable, and it is worth taking now because it is the half of #2497 that our
correctness state does not block.

The second deliverable is the other half, **staged and refusing to run**. When
the token gate is ratified, the paired measurement should take one lease and not
a design session. Staging it now, in the tree, under review, is cheaper than
staging it under time pressure later — and a staged script that can be launched
by accident is how #2497 got its retraction, so this one refuses to start.

## Scope

Two artifacts, one measured and one inert.

1. **Measured — the oracle.** llama.cpp at the recorded pin `b10451` =
   `10bf611e533d81f739128304991c5e133c6aebd8`, built for `gfx1151` in the lease,
   decoding the plain `Qwen3.8-27B-Q4_K_M.gguf` on `strix:gpu0`. Repeated warm
   legs, order recorded, executed kernel path pinned, AMD clock state sampled per
   leg by an ad-hoc sampler that says it is ad-hoc.
2. **Staged, not run — our arm.** `.agents/scripts/rocm-strix-ourarm-staged.sh`,
   the order-alternated paired job that becomes admissible when the token gate
   passes. It refuses to start unless `STRIX_ARM_SPEED_RATIFIED_BY` names the
   decision that ratified it.

### Artifact

`unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10`,
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, sha256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. The sha256
is verified **on the worker**, against the staged worker-local copy, before any
timing runs, and the job aborts on a mismatch rather than reporting one.

The `UD` family is not used. Our loader could not read it when #2497 was opened
([#2510](https://github.com/mudler/vllm.cpp/issues/2510)) and the published
`UD-Q4_K_XL` bytes have moved in place under an unchanged name, so it is not a
pin. The 12.616 tok/s harness-sanity figure the predecessor took on
`UD-Q4_K_M` is a **different artifact** and is context, not this row's number.

### Excluded

- Running our engine for timing, on any leg, for any duration.
- Computing, writing or implying a ratio between the two engines.
- `HSA_ENABLE_SDMA`. It is retired (#2511) and is not set anywhere.
- Any change to engine code.

## Pinning the oracle's EXECUTED PATH, not only its revision

`.agents/oracles/llama-cpp.md` records that `b10451`'s decode is not
deterministic across its own supported kernel paths, and
`rocm-gfx1151-q4k-token-gate-v2.md` §"Pinning the oracle's EXECUTED PATH"
measured a second instance of that on this very board. A denominator that pins
only the revision is under-specified, so the run records:

| Term | How it is obtained |
|---|---|
| Source identity | content manifest of the staged tree, `LC_ALL=C` collation printed beside the value, compared to the recorded `56c26d15…` |
| Binary identity | sha256 of `llama-bench` and `llama-cli`, plus every `libllama`/`libggml` shared object |
| `n_gpu_layers` | the `-ngl` argument, echoed back out of `llama-bench`'s own JSON |
| `use_extra_bufts` | source-anchored: `src/llama-model.cpp:2489` defaults it `true` and `tools/llama-bench/llama-bench.cpp:1218-1267` sets no override, so this run's value is `true`; confirmed at runtime by the `load_tensors:` buffer-type lines |
| `system_info` | `llama-cli -v`, which reaches `common.cpp:417`'s trace-level emission of `common_params_get_system_info` |
| host arch, thread count | `uname -m`, `nproc`, and `llama-bench`'s own `n_threads`, `cpu_info`, `gpu_info` and `backends` JSON fields |
| enumerated devices | `llama-bench --list-devices` |

`llama-bench` exposes no switch for `use_extra_bufts`, which is why that row is
answered from the source rather than from a flag. With `-ngl 99` every loaded
tensor sits in the ROCm buffer, so the CPU repack buffer types the flag admits
hold nothing and the flag cannot move this measurement. The run prints the
buffer lines rather than asserting that.

`llama-cli` is never invoked without `-no-cnv` and never with an open stdin. One
such run wrote 24.9 GB to the share. `llama-bench` and `llama-cli` link
`*-impl.so` beside them, so `build-llamacpp/bin` is on `LD_LIBRARY_PATH` or they
exit 127.

## Design of the measured run

- One `rc` lease on `strix:gpu0`, submitted detached, one job at a time, and the
  fleet's contention state recorded from `rc devices` at submission.
- The GGUF is staged from CIFS `/workspace` to worker-local `/tmp` and its
  sha256 verified there. `/workspace` is never a run surface.
- **6 legs**, each one `llama-bench -m … -p 0 -n 64 -ngl 99 -r 3 -o json`. A leg
  is one process: it loads the model, does `llama-bench`'s own warmup, and times
  3 generations of 64 tokens. `N = 6` comes from this design and not from
  counting log lines; the job log tees and a grep tally would double it.
- Legs run in sequence 1..6 and the sequence index is recorded with each. There
  is one engine, so there is nothing to alternate; the order is recorded so a
  monotonic drift is visible as one.
- The per-leg figure is `avg_ts` from that leg's own JSON. The reported figure is
  the **median over the 6 legs**, with the min-max spread stated.
- A leg is discarded only for a named cause, and the cause is printed with it. No
  leg is discarded for being the first one unless it differs, in which case the
  named cause is the cold page cache and it is stated as such.
- AMD clocks are sampled at 4 Hz per leg from
  `/sys/class/drm/card*/device/{gpu_busy_percent,pp_dpm_sclk}` to worker-local
  disk, because a 4 Hz flush at CIFS stalls the sampler and distorts the sample
  spacing the window is judged on.

### The clock, and why two windows are reported

No in-tree harness samples AMD clocks. `tools/bench/gpu_clock_state.py` reads
NVIDIA fields and does not run on this board; #2381 owns the gap. So the sampler
here is **ad-hoc**, carried beside the job as the predecessor's was, and every
figure it produces is labelled ad-hoc rather than presented as a gate reading.

Two windows are reported per leg and they answer different questions. The whole
window includes the 17 GiB model load, which on this APU dominates the raw
spread and says nothing about decode. The **>=90%-busy compute window** is the
part where the GPU is actually generating. Both are printed, with sample counts,
so a reader can see which one a figure came from.

`.agents/benchmarking.md`'s 5% within-run SM-clock spread ceiling is **not
applied**. It was calibrated on a datacenter part with persistence mode and its
own power budget; this APU shares one package power budget with 32 CPU cores and
the predecessor measured 8.2–12.0% on both engines under sustained load. An AMD
rule needs its own calibration and cannot inherit the NVIDIA number. That
observation is recorded for #2381 and is not repaired here.

The cross-arm clause of that section — that two arms' clock medians and means
must agree, because the offset is the term that lands in the ratio — has no
subject in this run, because there is one arm and there is no ratio.

## The staged run, and why it refuses

`.agents/scripts/rocm-strix-ourarm-staged.sh` carries the paired design in full:
order-alternated rounds, the leg count, the clock window per leg,
`VT_OP_PROVIDER_STATS=1` asserted to zero CPU reference-tier hits, the artifact
sha verified on the worker, and the same ad-hoc clock sampler.

It refuses to start unless `STRIX_ARM_SPEED_RATIFIED_BY` is set to a value
naming the decision that ratified the arm's speed axis — a value carrying an
issue reference, so `=1` is refused. The refusal is the first thing the script
does, before it reads a path, stages a byte or touches a device, and it names the
gate that is failing and the evidence that records it.

The script computes **no ratio**, and says why in its own output: the ratio is
arithmetic a reader can do, but the decision that it is admissible is not, and
that decision is exactly what the env var is asserting. Printing the two medians
without dividing them keeps the assertion where a human made it.

## Risks

- **The board faults.** #2511 landed and the token gate then completed 6 legs of
  6. A fault rate here would contradict that and is reported as a finding rather
  than retried into silence.
- **The pod restarted and dropped the build.** The job verifies the binaries and
  rebuilds from the pinned tarball when they are absent; it never assumes them.
- **A leg measures a shared box.** The fleet state is recorded at submission and
  the job runs inside the lease, one job at a time.
- **The sampler retains nothing.** An empty or entirely idle window is reported
  as such and the clock figure is withheld; it does not silently become a
  narrower one.

## Tests

The staged script's refusal is the one behavior in this row that a test can
falsify, and `tests/scripts/test_rocm_strix_ourarm_staged.py` does that:
unset, empty, and a value with no issue reference each refuse with a non-zero
status and a message naming the token gate; a well-formed value gets past the
guard. The test is red before the script exists and each assertion is mutated to
prove it detects the defect.

The measured run's own assertions are inside the job — the artifact sha, the
source manifest, the binary presence, the leg count — and each aborts rather than
reporting.

## Gates

```sh
python3 -m pytest tests/scripts/test_rocm_strix_ourarm_staged.py -q
scripts/agent-preflight.sh
```

Neither gate measures the GPU. The measured figure is evidence, not a gate; a
number this row produces about the oracle cannot be re-derived by CI.

## Evidence

[`docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md`](../../docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md).

## Stop conditions

- Any instruction, temptation or convenience that produces a vllm.cpp throughput
  number on this board stops the work. The gate is failing and the number is
  inadmissible.
- A ratio. If a ratio is about to be written, the row has left its scope.
- A single-leg figure. A denominator reported from one leg is an anecdote.
- The board faulting, which is reported at its measured rate.

## Owed

- The paired measurement itself, once the token gate passes.
  [#2497](https://github.com/mudler/vllm.cpp/issues/2497) owns it and the staged
  script is what it runs.
- An AMD clock helper with its own calibrated spread rule.
  [#2381](https://github.com/mudler/vllm.cpp/issues/2381) owns it; every clock
  figure this row produces is ad-hoc until then.
