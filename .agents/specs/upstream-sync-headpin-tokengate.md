# Sync cycle `e126687a9a`, wave TOKENGATE

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2794](https://github.com/mudler/vllm.cpp/issues/2794).
Predecessors: [#2593](https://github.com/mudler/vllm.cpp/issues/2593) wave
HEADPIN, [#2611](https://github.com/mudler/vllm.cpp/issues/2611) wave RUNHALF,
[#2764](https://github.com/mudler/vllm.cpp/issues/2764) wave PORTQ-RECONCILE,
which named three blockers, and [#2771](https://github.com/mudler/vllm.cpp/issues/2771)
wave STEP6, which worked the second. This wave is blocker 1.
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).

## Now

**The gate does not yet exist, and what prevents it is a queue, not a
refusal.** This wave establishes what the gate must compare, makes that
comparison executable, proves the instrument on four mutations, stages every
input, and queues the one job that can produce the number. At the time of
writing that job is `efc30c74-005e-4e80-bc28-bd34f5b76b77`, position 12 in the
`dgx:gpu0` queue.

**Two findings stand on their own, and neither needed the job.**

1. **The token path is not pin-gated.** STEP6 established that the online-serving
   harness structurally refuses to measure at any revision but the pinned one.
   That refusal does not reach the token path: `scripts/opt-oracle-capture.py`
   and `scripts/opt-dgx-gate.sh` read no pin constant, and neither does
   `tests/vllm/models/test_opt_paged_engine.cpp`. §2.1 records how that was
   checked, because a zero from a grep is not an absence.
2. **The OPT strict golden was never re-validated at the ACTIVE pin.** It was
   captured once, at `b8358a5b9`, against vLLM 0.25.0 / `e24d1b24`. The
   `5559679229` advance's W3b table put OPT in the "already RATIFIED
   near-tie-robust" row and skipped its re-capture on the grounds that a
   distributional gate absorbs drift. OPT's gate is not distributional. §2.4.

The pin does **not** advance and nothing here is a reason to move it. The active
parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**The question.** `.agents/oracles/vllm.md` lists four obligations a pin advance
to `e126687a9a828d513c01a07cd69f025f27d63280` owes. The second is "a declared
token-exact gate." Does one exist at the target, and if not, what exactly
prevents it?

**In scope.** The OPT-125m greedy gate and its committed goldens
(`tests/parity/goldens/opt_greedy/`), its capture script
(`scripts/opt-oracle-capture.py`), its gate source
(`tests/vllm/models/test_opt_paged_engine.cpp`), the comparison between a
candidate-revision capture and the committed golden, and the record of what the
`5559679229` advance did and did not re-validate.

**Out of scope.** Advancing the pin. The PORT-NOW queue. The FlashInfer step-6
re-measurement. Porting anything. The other model goldens the pin advance's W3b
step covers (27B, 32B, 35B, Coder); those were re-captured at `5559679229` and
their re-capture at the target is a separate, larger job that this wave does not
attempt and does not discharge.

**Not a substitute.** Nothing measured on `thor:gpu0` closes this. §2.3.

## 2. Design

### 2.1 The token path is not pin-gated, and how that was checked

STEP6's refusal is real and it is narrow. `tools/bench/online_gate.py` reads
`VLLM_ORACLE_VERSION`, `VLLM_DISTRIBUTION_VERSION` and `FLASHINFER_VERSION` from
`serve_low_common.py`, which reads `.agents/upstream-sync.md`'s ` ```parity-pin `
block, so that harness cannot measure at a revision the block does not name.
Generalising that across paths is exactly the error the predecessor wave's
discharge withdrawal punished, so it was checked rather than assumed.

The check is a **full read** of both harness files, not a grep, because a grep's
zero has eight recorded ways of being wrong in this tree. `opt-oracle-capture.py`
is 156 lines and `opt-dgx-gate.sh` is 55; both were read end to end. Neither
imports anything from `tools/bench/`, neither opens `.agents/upstream-sync.md`,
and neither takes a revision as input at all: the capture script's only
oracle-side input is whichever `vllm` is importable in the venv on `PATH`.

The grep is recorded as corroboration with its positive control in the same
probe form, so that a zero is legible:

```console
PAT[serve_low_common]     -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[read_parity_pin]      -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[assert_oracle_commit] -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[FLASHINFER_VERSION]   -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[VLLM_ORACLE_VERSION]  -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[parity-pin]           -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[def main]             -> opt-dgx-gate.sh:0  opt-oracle-capture.py:1   <- control
PAT[PROMPTS]              -> opt-dgx-gate.sh:0  opt-oracle-capture.py:5   <- control
PAT[greedy_ids]           -> opt-dgx-gate.sh:0  opt-oracle-capture.py:4   <- control
```

**Scope of that claim.** It is about these two files and the gate source they
feed. It is not a claim about `tools/bench/`, which does read the block, and it
is not a claim that any other gate in the tree is revision-portable.

### 2.2 What the gate compares, and why #2628's run was not one

The declared token-exact gate for OPT is `test_opt_paged_engine.cpp`: **our**
paged engine's greedy tokens against **vLLM's**, the latter committed as
`greedy_ids.npy`. To hold "at `e126687a9a`" the golden must be the answer vLLM
gives at `e126687a9a`. So the gate at the target is the pin advance's own W3
step, which `.agents/specs/pin-advance.md` states as: regenerate the golden at
the target, diff it against the committed one, keep it if unchanged, commit the
new one and re-pass the gate against it if changed.

#2628 laid a target-revision run beside the committed golden and got 96/96. It
is not that step, for three reasons, and the third was not named there.

| | #2628's run | What the gate needs |
|---|---|---|
| Device | Thor, `sm_110` | GB10, `sm_121a` — the device the golden was captured on |
| Checkpoint | raw `facebook/opt-125m` fp16, rounded by vLLM at load | the bf16-materialized artifact `scripts/opt-materialize-checkpoint.py` writes, decisions D1/D2 of [`sweep-opt-125m.md`](sweep-opt-125m.md) |
| K | one run per leg | `--runs 5`, because K **selects the gate** |

The first two are confounds: revision moved together with silicon, and with the
rounding path. An agreement across two moved variables is not attributable, and
neither would a disagreement have been.

The third is not a confound but a missing measurement, and it is the one that
makes this a gate rather than a diff. `test_opt_paged_engine.cpp` uses a STRICT
token-exact bar **with no near-tie band at all**, and the thing that licenses
that bar is `greedy_dist.npy`: K=5 oracle runs with zero multi-valued
(prompt,pos) cells, re-asserted at the top of the test. #2628 ran
`.agents/scripts/runhalf-e126687-gen.py`, a copy of the capture script's prompt
battery with no K loop and no dist output, so **the candidate oracle's own
self-determinism at the target is unmeasured**. The committed `greedy_dist.npy`
cannot stand in for it: it is the previous oracle's measurement of itself. If
the candidate has become non-deterministic on this battery, the strict bar must
be re-derived per [[near-tie-distributional-gate]] and not silently kept — and
no diff of `greedy_ids.npy` alone can see that.

**So the construction is:** on GB10, against the same bf16-materialized
checkpoint, with `scripts/opt-oracle-capture.py --runs 5` verbatim, capture
`greedy_ids.npy`, `greedy_dist.npy` and the six `p{i}_prompt.i32`; then diff all
three against the committed set. The revision is then the only thing that moved.

### 2.3 Why `thor:gpu0` cannot substitute

Thor is `sm_110` and already carries a built candidate wheel, so it is the cheap
option and it is the wrong one. A capture there moves the device again, and the
result would be a different measurement wearing the label of the one that is
owed — the reason STEP6 declined a free Thor for its own re-measurement.

The asymmetry is worth stating because it is the only thing Thor could have
contributed. A **non**-deterministic K=5 result on Thor would be a positive
finding: it would show the candidate can diverge on this battery at all. A
deterministic result there transfers nothing to GB10. Since only one of the two
outcomes is usable and the useful one is the unlikely one, no Thor lease was
spent, and no number from Thor appears in this wave.

There is a second, independent reason. The `sm_110` wheel is built with
`TORCH_CUDA_ARCH_LIST=11.0`, which emits `sm_110` cubins and no PTX, so it
cannot run on GB10 at all. The GB10 arm is a build, not a copy.

### 2.4 The OPT golden was never re-validated at the active pin

`git log --follow` over `tests/parity/goldens/opt_greedy/greedy_ids.npy` and
`greedy_dist.npy` returns exactly one commit, `b8358a5b9`, the OPT W0-W4 landing,
whose subject records the oracle as vLLM 0.25.0. `b8358a5b9` is an ancestor of
`bc415a3e4`, the `e24d1b24` -> `5559679229` pin flip, so the golden predates the
advance and no commit has touched it since.

The advance's W3 step is explicit that every SACRED gate's golden is regenerated
at the new oracle and diffed. Its W3b table discharges OPT in a row covering
"~30 model-matrix `*_greedy` rows (llama/opt/phi/mistral/internlm/minicpm/yi/
olmo2/deepseek-v2/glm4-moe-lite/…)", with the method column reading "already
RATIFIED near-tie-robust gates (`greedy_dist.npy`, `kNearTieMnats=500`)" and the
result column "no re-capture: the distributional gates absorb near-tie drift by
construction."

**OPT is not in that class.** `test_opt_paged_engine.cpp` says so in its own
header — "Where vLLM is self-consistent the honest bar is exact agreement — so
no near-tie band is used here at all" — and the body carries no `kNearTieMnats`.
The row's premise is false for this member, so the discharge does not cover it.

The consequence is bounded and should not be overstated. The active pin does have
declared token-exact gates: 27B W4A4, 32B-NVFP4A16, 35B and Coder were each
re-captured or re-measured at `5559679229` and recorded bit-identical or
byte-stable. What is false is the narrower claim that *every* strict golden was
carried across that advance. OPT's was not, and the same capture this wave
queues answers both questions at once, since a candidate capture that reproduces
the committed bytes reproduces them across both advances.

### 2.5 The instrument

`scripts/opt-oracle-capture.py` writes a golden and prints a determinism report.
Nothing in the tree ever laid two captures beside each other; the pin advance's
W3 diff was done by hand. `.agents/scripts/tokengate-e126687-diff.py` makes that
step executable. It checks three things that are not the same check:

1. `greedy_ids.npy`, the bar, by sha256 and by mismatched position.
2. the six `p{i}_prompt.i32`, the **input**. A matching output on a moved input
   would be a coincidence, not agreement.
3. `greedy_dist.npy`, the **selector**, recomputed from the candidate's own K
   runs rather than read from the committed file.

Exit 0 is all three holding; exit 1 is a real difference, which is a finding;
exit 2 is a missing or malformed input, which is the instrument failing rather
than the comparison failing. That separation is deliberate: an instrument whose
failure looks like a result is this row's recurring trap, and three of RUNHALF's
four reds were its own instruments.

`.agents/scripts/tokengate-e126687-job.sh` is the lease job. It **refuses to run
on anything but compute capability 12.1** rather than adapting to the device it
finds, because a capture on the wrong silicon is the failure this wave exists to
avoid; the refusal is the gate's device assertion, expressed where it cannot be
skipped. It is resumable in two stages (`STAGE=build`, `STAGE=capture`) and
persists the wheel to `/workspace` the moment it exists, because `dgx:gpu0` has
crashed under sequences shorter than the ~1.3 h source build this needs.

## 3. Risks

- **The queue.** `dgx:gpu0` had eleven jobs ahead of this one at submission. The
  wave's deliverable degrades gracefully: the design, the two findings and the
  proven instrument stand whether or not the job runs.
- **The build.** No aarch64 wheel exists for this revision, so GB10 needs a
  from-source build (~1.3 h at the pin, 94 min at the candidate on Thor). The
  worker has no `github.com` egress, so all nine `FetchContent` repositories and
  the vLLM tree itself are staged from `/workspace`; RUNHALF's staging is reused
  unchanged.
- **FlashAttention on GB10.** `flash-attention/CMakeLists.txt:140` computes
  `FA2_ARCHS` as a loose intersection of `"8.0+PTX"` with the target archs, so a
  `12.1` build reaches `sm_121` by a driver JIT of `compute_80` PTX — the mode
  recorded as failing on GB10 with `cudaErrorUnsupportedPtxVersion`. If the
  engine cannot select its default backend the job records which backend
  produced the tokens; a capture on a non-default backend is reported as such
  and is not silently labelled the gate.
- **A false pass by coincidence.** Ruled out by checking the input tokenization
  alongside the output, §2.5 item 2.

## 4. Gates

| Gate | Command | Result |
|---|---|---|
| The differ detects a drifted token id | mutate `greedy_ids[2,7]`, rerun | **rc 1**, `IDS_DIFF prompt[2] pos 7: committed 4 candidate 5` |
| The differ detects a moved input | mutate `p3_prompt.i32[1]`, rerun | **rc 1**, `PROMPT[3] … EQUAL False` |
| The differ detects a lost selector | mutate `greedy_dist[0,3,4]`, rerun | **rc 1**, `SELECTOR K=5 multi_valued_cells 1` |
| A missing input is NOT a finding | delete `greedy_dist.npy`, rerun | **rc 2**, `FATAL missing input` |
| Identical inputs pass | committed vs a copy of itself | **rc 0**, `TOKENGATE_VERDICT PASS` |
| The token harness reads no pin constant | full read + the §2.1 probe with controls | zero hits, controls 1/5/4 |
| The capture at the target on GB10 | `efc30c74-005e-4e80-bc28-bd34f5b76b77` | **PENDING**, queued |

Every rc above was read directly, never after a pipe.

## 5. Stop conditions

- Stop on a `DIFF_RC=1`: a drifted golden is a finding to record and re-gate
  against, not a defect to repair in this wave.
- Stop rather than capture on any device whose compute capability is not 12.1.
- Stop rather than advance the pin. That is not this wave's authority.

## Owed

- **The capture itself**, until job `efc30c74-005e-4e80-bc28-bd34f5b76b77`
  returns (#2794).
- **Our arm re-run against whatever golden that job produces.** If the bytes are
  identical the existing green `test_opt_paged_engine` measurement carries over
  to those bytes unchanged; if they drift, the gate must be re-passed on GB10
  with our code byte-unchanged, which is a second lease (#2794).
- **A re-capture of the OPT golden at the ACTIVE pin `5559679229`**, which §2.4
  shows was skipped. A candidate capture that reproduces the committed bytes
  answers this too; one that does not leaves it open (#2794).
- **The other strict goldens at the target** — 27B W4A4, 32B-NVFP4A16, 35B,
  Coder. The pin advance re-captured all four at `5559679229`; none has been
  re-captured at `e126687a9a`, and this wave does not attempt it (#2794).
