# SPEC — `LTX25-RENDER-CONFIRM`: turn the connector repair's projected render into a wall clock, by advancing the pin rather than removing it

Issue: [#2457](https://github.com/mudler/vllm.cpp/issues/2457).
Owner row: `LTX25-RENDER-CONFIRM`.
Predecessors: [#2296](https://github.com/mudler/vllm.cpp/issues/2296)
(`.agents/specs/ltx25-render-speed-parity.md`, the 516.751 s baseline and its
method), [#2434](https://github.com/mudler/vllm.cpp/issues/2434)
(`.agents/specs/ltx25-connector-repair.md`, the component reading and the
projection), [#1854](https://github.com/mudler/vllm.cpp/issues/1854)
(`.agents/specs/ltx25-oracle-absolute.md`, the correctness gate this row
re-takes), and [#1864](https://github.com/mudler/vllm.cpp/issues/1864) (the
reference render the verdict is taken against).

## The gap

`LTX25-CONNECTOR-REPAIR` measured **one `Ltx2ConnectorForward` call** on GB10 at
**128.808 s -> 50.34 s**, a **2.56x**, with `vt::AttentionCross` **85.704 ->
8.47 s** (10.12x) and the GEMM unmoved at **31.906 -> 32.863 s**. From that it
projected the render: connector compute **224.882 -> ~87.9 s**, the render
**516.751 -> ~380 s**, the oracle gap **5.51x -> ~4.05x**.

**Its own `## Owed` says no render was run**, and the reason is a guard rather
than an oversight. `scripts/ltx25-render-speed-repeat.sh` asserts the binary's
`sha256` against `rc` job `4b0666ee`'s -- the run that returned `VERDICT PASS`
against #1864's reference -- and exits **51** on any other binary. That guard is
what stops a speed number being quoted for a binary whose correctness nobody
established. **A component speedup is not a system speedup**, and the only thing
that makes the distinction checkable is that the wall is timed on a binary with
a verdict behind it.

So the gap is not "run the harness". It is: **produce a binary at the current
head that has itself taken #1864's verdict, and time THAT.**

## Scope

IN scope:

- **W1** — this spec.
- **W2** — one committed harness, `scripts/ltx25-render-confirm.sh`, that in ONE
  lease and on ONE binary: builds the head, runs the CUDA unit gate, renders,
  takes #1864's blockiness verdict against both reference forms, and only on a
  PASS goes on to time `N >= 3` renders and fold their phase tables.
- **W3** — the measurement, the phase table, the verdict, and the pin in
  `scripts/ltx25-render-speed-repeat.sh` advanced to the binary that earned it.

OUT of scope, declared rather than approximated:

- **Any repair.** This row measures. If the wall does not reach ~380 s, the
  result is the number and the next traceable hypothesis, not a fix.
- **Weakening the pin by any route other than re-taking the verdict.** The three
  `WANT_*_SHA` values in `ltx25-render-speed-repeat.sh` move only after a PASS
  on the binary they will name, and they move to that binary's digests. A
  `WANT_*_SHA` made overridable-by-default, a comparison skipped, or a guard
  deleted is `NEEDS_DECISION`, not a workaround.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing: this is one
  request at one geometry, and `ltx25-render-speed-parity.md` already owns the
  public statement of the gap.
- **The oracle side.** The denominator is the pinned oracle's own recorded
  `render_seconds = 93.8` from
  `tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json`, `n = 1`, and its
  missing spread is stated as the limit it is rather than filled in.

## Why one harness and not two

`ltx25-oracle-absolute-render.sh` builds and judges the PICTURE, renders once,
and exits on the comparison's verdict. `ltx25-render-speed-repeat.sh` times the
WALL `N` times and refuses to build. Running them back to back in two leases
would time a binary built twice, and this repository has the standing note that
an A/B reusing one build directory measures one binary twice -- the mirror of
the same defect.

**The two obligations have to close over the same bytes**, so this row's harness
takes the artefacts once, hashes them once, and both the verdict and every
timing quote that one digest. That is the whole reason the file exists, and it
is why it is committed rather than typed into a lease: `oracle-ltx-2-pin.md`
records four earlier attempts whose manifest described a script that no longer
existed.

**It reuses the neighbours' hard-won parts verbatim** rather than re-deriving
them: the `soname_ok` CUDA-toolkit check that #2220 paid 21 minutes for, the
`MemAvailable` start floor #1709 paid four leases for, the `exec` in the clock
sampler subshell that #2305 paid three orphaned windows for, the all-or-nothing
build cache, the `.part`-rename staging, the `steps_observed` denominator read
back from the render's own lines, and the `< 1%` phase-coverage refusal.

## The order is correctness, then timing, and the harness enforces it

1. CUDA unit gate (`test_ltx2_device`) before any render.
2. Render 1.
3. #1864's comparison, against the 25 PPM frames upstream's own decode wrote AND
   against the committed `upstream-render.mp4`. **A non-zero verdict exits the
   job**, before renders 2..N exist. The blockiness gate failing is the headline
   and the speed number is void; a harness that collected the wall anyway would
   invite quoting it.
4. Renders 2..N, then the fold.

Render 1 is timed like every other render and is not discarded for having been
the one compared. The comparison runs after its wall is recorded.

## What the gate cannot see, and what is therefore printed beside it

`ltx25-oracle-absolute.md` records that our render is already **less blocky,
less sharp and less clipped** than upstream's, and that `blockiness_grid8` /
`blockiness_grid32` are **one-sided ceilings**: they are blind to further
smoothing. `sharpness_mean`, `clipped_fraction` and the audio RMS stay
`reported_statistics` in the tool's own JSON for exactly that reason.

**So this row prints the reported-only panel beside the verdict**, both arms'
values, and reports the direction each moved. A PASS on a one-sided ceiling
plus an unremarked collapse in sharpness would be a gate reporting green on a
render that got worse.

## Reachability

Nothing new lands in `src/` or `include/`. The subject is
`ModelRegistry::Forward`'s own LTX-2.5 path exercised through the shipped
`ltx2-gen` binary on its default configuration -- the production entry point --
at the manifest's request. `scripts/ltx25-render-confirm.sh` is a measurement
harness and is reached by being run in the lease this spec records.

## Tests to port

There is no upstream test. Upstream emits no phase table and has no blockiness
gate; #1864's reference render IS the ported artefact, and
`scripts/ltx25-render-compare.py` and its committed suite already hold the
criterion. This row adds no assertion to that tool and changes none of its
thresholds.

## Gates

| ID | Gate | Refusal |
|---|---|---|
| G1 | all four checkpoint `sha256` equal the manifest's, recomputed inside the lease | 23 |
| G2 | a CUDA toolkit whose `libcudart` / `libcublasLt` SONAME links resolve | 38 |
| G3 | `test_ltx2_device` green, case and assertion counts recorded | 44 / 45 |
| G4 | every render writes exactly 25 frames, a non-empty wav and a `phase-log.json` | 48 |
| G5 | every render observes `steps_observed={8}` | 53 |
| G6 | every phase table covers `>= 99%` of its own wall | 52 |
| G7 | **#1864's comparison against the PPM frames exits 0**, on the binary that produced the timing | the comparison's own status |
| G8 | the same comparison against the committed mp4, cross-checked, disagreement reported | reported |
| G9 | `N >= 3` renders completed | 49 |
| G10 | `scripts/check-tree-compiles.py` and `scripts/agent-preflight.sh` | — |

## Risks/decisions

- **A component ratio does not transfer to a system.** The projection is
  explicitly a ratio applied to a different quantity: the probe's connector
  total uses synthetic weights, a different valid-token count, and times two
  streams in one process. This row's deliverable is the wall, and the phase
  table is what says whether the wall moved for the predicted reason.
- **THE WALL CAN MOVE FOR THE WRONG REASON.** `load.dit` had a **49.70%**
  spread in the baseline because its cost is page cache. A render that comes in
  at 380 s because the DiT was warm is not this repair. The claim is checked at
  the LEAF: `conditioning.connector` (0.44% spread) and the connector half of
  `generate.guiders` (1.12%) must fall, while `denoise`, `decode.video` and
  `decode.audio` must not. If the wall moves and those leaves do not, the
  projection was right by accident and this spec says so.
- **`n = 1` is an anecdote.** `N >= 3` and the spread is quoted per phase.
- **Contention reweights rather than adding noise.** The render is 87-88%
  GPU-idle and host-bound, so `loadavg` and `MemAvailable` are the axes that
  transfer. Both are sampled before and after every render, and the SM clock is
  folded through `tools.bench.gpu_clock_state` anyway because a number is
  quotable only with the clock it was taken at.
- **The oracle denominator is `n = 1`.** 93.8 s has no spread of its own. Every
  ratio here inherits that limit and says so.
- **ENOSPC.** A full CMake build writes 9.4 GiB and named targets are used for
  that reason; the checkpoints are 70 GB and are staged to `/root`, which is
  local, with the free-space check before the copy.

## Stop conditions

Stop and report, do not work around:

- **the blockiness gate failing** -- the speed number is not reportable and the
  failure is the finding;
- a checkpoint digest that is not the manifest's;
- an unreachable or unhealthy fleet device, or one held by somebody else;
- a wall that does not reach ~380 s -- that is a RESULT, reported with the phase
  table and the next traceable hypothesis, never a ceiling;
- any pressure to weaken the sha256 pin other than by re-taking the verdict.

## Work breakdown

- **W1** — this spec.
- **W2** — `scripts/ltx25-render-confirm.sh`.
- **W3** — the lease, the measurement, `## Outcome`, and the advanced pin.

## Now

`ACTIVE`. W1 committed.

## Owed

- **The oracle side has no spread.** `render_seconds = 93.8` is `n = 1` at the
  pin. Re-running upstream `n >= 3` at the same request would give the
  denominator its own error bar. Owner: unowned; sizing is one lease.
