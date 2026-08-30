# SPEC — `LTX25-TEXT-COND-DEVICE`: split `conditioning.connector`, then repair what the split blames

Issue: [#2354](https://github.com/mudler/vllm.cpp/issues/2354).
[#2296](https://github.com/mudler/vllm.cpp/issues/2296) is the measurement this
row acts on; [#1269](https://github.com/mudler/vllm.cpp/issues/1269) owns the
CPU-pinned text tower and does not carry the connector.
Owner row: `LTX25-TEXT-COND-DEVICE`.

## Scope

`LTX25-RENDER-SPEED-PARITY` measured, at `n = 3` on `dgx:gpu0`
(`rc` job `0baa109c-43ff-475e-abf1-7a50152ffd5d`, evidence
`/mnt/nas_share/rc/ltx25-render-speed/run/20260829T195851Z`), that an LTX-2.5
render of the oracle's own request is **5.53x** the pinned oracle and that the
21B DiT is **2.92%** of it. `generate.guiders` + `conditioning.connector` are
**312.4 s, 60.3% of the wall, and 3.33x the oracle's entire process.** An
independent instrument — `nvidia-smi` sampling, sharing no code with the phase
log — found the GPU idle for **87 to 88%** of the render at median utilization 0%.

That row measures and repairs nothing, and its `## Owed` says so by name:

> `conditioning.connector` and `generate.guiders` are measured and unowned by any
> row that could move them.

IN scope:

- **W1, the split.** `conditioning.connector` decomposed into WEIGHT
  MATERIALIZATION and CONNECTOR ARITHMETIC, and `generate.guiders` decomposed
  into its tower half and its connector half. Both are sub-leaves of leaves that
  already exist, in the instrument that already ships.
- **W2, the repair the split blames**, together with the red-first test and the
  mutation that says the repair is reached.
- **W3, the same-shape re-measurement**: the same harness, the same request, the
  same geometry, `n >= 3` per arm, two separate build directories.
- **W4, correctness**: the committed absolute-quality gate re-run on the changed
  arm, plus a same-arm pixel comparison against a pre-change render.

OUT of scope, declared rather than approximated:

- **A device arm for the Gemma-4 tower.** #1269 records that the tower is host
  resident BY TYPE — `Gemma4Weights` holds `OwnedTensor` over `OwnedBytes`, which
  carries no device field — so moving it is a weight-arm port and not a queue
  swap. This row does not attempt it and does not pretend the queue at
  `ltx2_video.cpp:2344` is what is in the way.
- **A published benchmark ID.** One request, one geometry, one seed. #2296's own
  reason applies unchanged.
- **The oracle side.** It is `n = 1` at 93.8 s and this row does not re-run it.
  Every ratio quoted here inherits that limit.

## Why the split has to come first, stated as two predictions

#2296's `## Owed` names a hypothesis and is careful to call it one:

> `conditioning.connector` is dominated by loading and widening the connector out
> of the 42 GB DiT file rather than by the connector's arithmetic, and the render
> pays for it four times.

Its corroboration is a MEMORY step, not a TIME one: host peak rises 8.59 GiB
across the tower->connector boundary against 8.06 GB predicted for widening
2.016 B parameters to f32. That is evidence the widen HAPPENS. It is not evidence
the widen is what the 122.388 s went to, and this repository has a name for
promoting the first into the second.

**The arithmetic points the other way, and stating it here is what makes this
row's own result falsifiable rather than confirmatory.** The connector is a
transformer over `rows = 1024` (#1269 records the constant) at
`inner_dim = 4096` video / 2048 audio, 8 layers, 12 `dim^2` parameters a layer:
about 4.2 TFLOP of f32 GEMM per `RunConnector` call. At 122.388 s that is
**34 GFLOP/s**, which is a believable-but-poor rate for `vt::MatmulBT`'s threaded
CPU arm on 20 cores — and it is also within an order of magnitude of what an
8 GB bf16->f32 widen plus 4 GB of first-touch page faults costs. **Neither
estimate excludes the other, and an order of magnitude is not a decision.**

So this row measures the boundary rather than reasoning about it, and it does so
BEFORE choosing between two repairs that are not the same repair: caching weights
and moving arithmetic. The prior row said one sub-phase split settles it. This row
takes that split.

## W1 — the instrument

`RunConnector` (`src/vllm/multimodal/ltx2_video.cpp`) grows a `phase_prefix`
argument and emits two NESTED leaves, `<prefix>.weights` around the two
`Ltx2LoadConnectorWeights` calls and `<prefix>.compute` around
`Ltx2ConnectorCreateEmbeddings`. The negative pass's tower gets `guiders.tower`.

**Nested is load bearing.** `render_phase_log.h` marks a leaf opened inside
another leaf `nested` and EXCLUDES it from `sum_leaf_seconds`, so these five new
records cannot move `unaccounted_seconds` and cannot change what the harness's
own coverage gate (H5, `unaccounted / wall < 1%`) reads. They decompose two
leaves; they do not join the table. That property is what makes the before/after
comparable to #2296's table rather than a different table.

**An empty prefix declines the leaves, and the load-time callers pass one.**
`RunConnector` has five call sites. Two are the load-time `prompt_embeds_path`
path, which the measured render does not take; at load time no leaf is
necessarily open, so a sub-leaf there would be a TOP-LEVEL leaf that joins the
sum and shifts the residue. Declining is therefore correctness, not tidiness.

## W2 — the repair

Named after W1 runs, not before it. The two candidates, and what each would cost:

- **If the split blames WEIGHTS:** materialize `Ltx2VaeWeights` once per render
  and share it between the positive and the negative conditioning pass, instead
  of twice. **Bit-exact by construction** — identical weights and identical
  inputs produce identical outputs, so the pixel comparison should be
  byte-identical rather than merely within tolerance, and that is a checkable
  claim rather than a tolerance to argue about. Peak host bytes do not rise: the
  8 GB is HELD across the two passes instead of being allocated, freed and
  allocated again, so the maximum is the same allocation it always was, now
  live for longer. `RunConnector`'s own header states the opposite policy
  ("THE WEIGHTS LIVE AND DIE INSIDE THIS CALL") and that policy is about the
  ENGINE's lifetime on a 119 GB box; a window inside one render is not the case
  it argues against, and this spec says so rather than quietly contradicting it.
- **If the split blames COMPUTE:** the repair is a device arm for the connector,
  which is not a queue swap either — `Ltx2Attention` interleaves host
  `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs, and
  `Ltx2ConnectorForward` reads weights as host `std::vector<float>`. That is a
  port with its own numerics gate and it is a row of its own. **This row would
  then land the split, report the measured negative, and name it.**

## Tests to port

There is no upstream test for either half: upstream renders, and its connector
is a `torch.nn.Module` whose weights `from_pretrained` already holds. The tests
are therefore this tree's own, and each is an executable observable rather than a
comment:

| ID | Assertion | Red before |
|---|---|---|
| T1 | a prompted one_stage render with CFG on writes `conditioning.connector.weights`, `conditioning.connector.compute`, `guiders.tower`, `guiders.connector.weights` and `guiders.connector.compute`, and every one of them is `nested` | the records do not exist |
| T2 | the same render's phase table holds exactly ONE `*.connector.weights` record | it holds two |

T1's `nested` clause is the part that matters: a sub-leaf that landed
NON-nested would silently enter `sum_leaf_seconds` and change every residue this
campaign has published, and asserting only that the record exists would not see
it.

T2 is W2's guarantee written as a number the shipped instrument already emits, so
it is not a test-only hook. The fixture is
`tests/vllm/multimodal/ltx2_video_fixture.h`'s reduced checkpoint set, reached
through the `one_stage` recipe at `cfg_scale = 3.0` with a negative prompt — the
one shape in this suite that runs `ltx2_video.cpp:3073-3094`.

## Gates

1. `ninja -C <build> test_ltx2_video test_render_phase_log` and both suites green,
   with case and assertion counts recorded rather than the exit status alone.
2. The mutation: delete the production call site's prefix argument in a scratch
   copy and rerun T1/T2; a gate that stays green measured a class, not a
   capability.
3. **The measurement**, under one `rc` lease on `dgx:gpu0`: two CLEAN build
   directories, `n >= 3` renders each, the manifest's exact request.
4. **Correctness, before any speed result is accepted**: `ltx25-render-compare.py`
   against `tests/parity/goldens/ltx2_oracle/`'s committed #1864 reference on the
   changed arm, PLUS a same-arm pixel comparison of arm B's frames against arm
   A's. The blockiness gate is one-sided and our render is already smoother than
   upstream's, so a PASS on it is necessary and not sufficient; the same-arm
   comparison is the sharper instrument and for a bit-exact repair it should be
   byte equality.
5. `scripts/agent-preflight.sh`.

## Dependencies

- `dgx:gpu0` through `rc`. Never `ssh`.
- The four BF16 checkpoints, digests in `ltx2_oracle_manifest.json`.
- CUTLASS and a CUDA 13 toolkit inside the lease, staged the way
  `scripts/ltx25-oracle-absolute-render.sh` stages them (#2220's SONAME check).

## Risks/decisions

- **Two arms means two builds, and they are two DIRECTORIES.** An A/B that reuses
  one build directory measures one binary twice. Both binaries' sha256 are
  recorded and asserted to DIFFER, which is the executable form of that rule.
- **`n = 3` per arm bounds the spread; it does not establish a distribution.**
  #2296 measured `conditioning.connector` at 0.44% spread and `generate.guiders`
  at 1.12% — the two most stable rows in its table — so the phases this row acts
  on are the ones where a small `n` is defensible. `load.dit` at 49.70% is not,
  and nothing here is attributed to it.
- **The SM clock cannot be pinned inside a lease** (`LGC_RC=4`). It is sampled.
  It also matters less here than usual, for the measured reason that 60% of this
  render is host-side: HOST load is the axis that transfers, so `loadavg` and
  `MemAvailable` are recorded per render.
- **The lease is the mutex and nothing else is.** No `flock` beside it.

## Evidence

- The `rc` job id, lease wall-clock, queue depth and contention state.
- Both arms' binary and library sha256, and the two source SHAs.
- The four checkpoint sha256 against the manifest.
- The `weights` / `compute` split, with the sentence that says where the boundary
  is in the source.
- Before/after phase tables, `n >= 3` each, spread per phase.
- The blockiness verdict on the changed arm and the same-arm pixel comparison.

## Stop conditions

Stop and report, do not work around:

- a checkpoint sha256 that does not match the manifest;
- two arms whose binaries hash the same, which means one build was measured twice;
- correctness that cannot be preserved — report rather than trade it;
- an unhealthy or unreachable fleet device.

## Work breakdown

- **W1** — this spec, the instrument, T1.
- **W2** — the repair the split blames, T2.
- **W3** — the lease, both arms, the tables.
- **W4** — correctness, and `## Outcome`.

## Owed

- **The oracle's 93.8 s is still undecomposed.** Inherited from #2296 unchanged.
- **`decode.audio.mel` at 47 s is still unattributed.** Not this row.
- **The 206.0 s that remains after `guiders` and `connector` is still open.**
  Removing both entirely leaves 2.20x. Naming it here is what stops this row's
  result being read as the whole answer.

## Now

`ACTIVE`. W1 is in this change.
