# Localising the ROCm/CPU tier divergence to a layer and an op

Row `BACKEND-ROCM`. Issue
[#2590](https://github.com/mudler/vllm.cpp/issues/2590).

Sibling records:
[#2546](https://github.com/mudler/vllm.cpp/issues/2546) (the gate run that
measured the term),
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) and
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (both fixed and landed;
neither explains this),
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the decode number this
gate blocks).

Predecessor specs:
[`rocm-gfx1151-q4k-token-gate-v2.md`](rocm-gfx1151-q4k-token-gate-v2.md), which
measured the term and lists it under `## Owed`, and
[`qwen38-27b-q4km-logit-dump.md`](qwen38-27b-q4km-logit-dump.md), whose
`## Outcome` names the per-layer hidden-state bisect as the next dispatch and
records why one statistic was not enough.

## Now

`ACTIVE` — the instrument is being built. No measurement has been taken.

## 1. Why this row is different from every predecessor on this arm

Every previous divergence measurement on the Q4_K_M arm compared **us against an
oracle**, and the oracle at this pin is not deterministic across its own kernel
paths: three of the six contested steps in the v2 gate are llama.cpp disagreeing
with itself, and at those three our two tiers emit the same token. So an
oracle-scored result mixes two effects and the denominator itself is contested.

Three steps are not like that. At `p1/45`, `p3/45` and `p4/14` **our own ROCm
tier and our own CPU tier compute a different argmax over an identical prefix on
an identical artifact, with no oracle in the comparison** (v2 evidence, the
`A vs D` column). Two implementations of one computation disagree. One is wrong,
or both are, and the comparison needs nothing external to adjudicate it.

That makes this row oracle-free. It needs no llama.cpp build, no token gate, no
scoring policy, and no ratification of any band. It needs one instrument that
neither tier has ever had: the per-layer hidden state, from both tiers, on the
same prefix.

The term is not one-signed, and any explanation has to survive that. ROCm loses
`p1/45` and `p3/45` and **wins** `p4/14`, which is one of the three steps the
CPU tier is convicted at. "ROCm is less accurate" is refuted before this row
starts.

## 2. Scope

In scope:

1. Repairing the two env-gated activation dumps this file already carries in the
   dense paged path so that a cross-tier comparison is possible and auditable
   (§3). They are dumps that already exist; none of them can currently support
   this comparison, and §3.1 says exactly why for each.
2. One host-side comparator that joins two dump directories and prints the
   per-layer delta profile (§4).
3. One `rc` job on `strix:gpu0` that builds two binaries from **one** source
   tree — the HIP arm and a CPU-only arm — runs both over the same prompt with
   the dumps on, and produces the profile (§5).
4. Reporting where the two tiers separate, and whether that is a defect or
   arithmetic (§6), against outcomes pre-registered in §7.
5. Fixing it, if §6 names a defect that is ours.

Out of scope, deliberately:

- **The token gate's scoring policy.** Adjudicated and landed under #2534: the
  ratified band fails this arm on `n_divergent`. No scoring change is in scope
  and none is argued for here.
- **Every speed, latency and memory number.** `AGENTS.md` §Gates admits none from
  this arm until its declared token gate passes. The harness prints none and
  none is transcribed.
- **#2511 and #2534.** Both fixed, both landed, neither re-litigated.
- **The oracle.** No llama.cpp build, no oracle leg, no margin recomputation.
  Every number this row produces is ours against ours.

## 3. The instrument

### 3.1 What exists, and why none of it can answer this question

Three dumps already exist in `src/vllm/model_executor/models/qwen3_5.cpp`, and
each of them fails this comparison in a different way. They are listed with their
defects because "the instrument already exists" is exactly the belief that would
make somebody run it and read a wrong answer.

| dump | where | why it cannot answer this |
|---|---|---|
| `VT_DUMP_ACT`, per layer | the dense paged loop | writes `hidden` ONLY. In this model the state after layer *l* is `hidden + res`: the two halves are carried separately and summed at the final norm. Half a stream cannot be compared. |
| `VT_DUMP_ACT`, `layer_-1_{hidden,res}` | the MoE paged loop | a pre-layer snapshot only, and on the wrong model class — a 27B Q4_K_M GGUF is the DENSE arm. |
| `VT_DUMP_ACT_SUB`, four stages | `RunDenseLayerPaged` | keyed by a `static thread_local` counter that is never reset, so a file name encodes an invocation ordinal rather than (step, layer), and any extra forward call on either side silently shifts the whole join. |

All three share two defects that matter more than any of the above:

- **A failed open writes nothing and says nothing.** Each writer is
  `if (f != nullptr) { fwrite; fclose; }`. A dump into a directory that does not
  exist, or a short write onto a full disk, is indistinguishable from a dump that
  was never switched on. `AGENTS.md` §Gates and
  [`verification.md`](../verification.md) name this class, the logit-dump row
  burned a lease on exactly it (`ALIGNMENT=BROKEN checked=0 bad=6`), and it is
  the first thing repaired here.
- **Nothing records what was dumped.** The blobs are raw bytes with no dtype, no
  shape and no provenance. A reader must infer the element width from the file
  size, and two dumps of different dtypes compare as garbage without either side
  saying so.

### 3.2 What this row builds

One writer, used by every activation dump in the dense path.

1. **It refuses by name.** A failed `fopen`, a short `fwrite` and a failed
   `fclose` are each a `VT_CHECK` naming the env variable, the path and the
   errno. A dump that writes nothing fails the run.
2. **It writes a manifest.** `<dir>/manifest.tsv`, one row per blob:
   `step, layer, stage, dtype, rows, cols, bytes, path`. The comparator joins on
   the first three and refuses on any disagreement in the rest.
3. **It narrates once.** One stderr line per process, before the first blob:
   the directory, the device type the model is running on, the model class, the
   layer count, and which of the two knobs is set. A reader of the log can see
   what the instrument was pointed at without reading the source.
4. **It counts.** One stderr line at the end of each forward: how many blobs this
   step wrote. Zero is a refusal, not a silence.

Then the two knobs are keyed correctly and made complete:

- `VT_DUMP_ACT=<dir>` writes **both** halves of the residual stream, per
  (step, layer): `s<step>_l<layer>_hidden.bin` and `s<step>_l<layer>_res.bin`.
  The comparator reconstructs `hidden + res`, which is the quantity the final
  norm consumes and the only one that is the model's hidden state.
- `VT_DUMP_ACT_SUB=<dir>` writes the four existing stages per (step, layer):
  `post_input_norm`, `block_out`, `post_attn_norm`, `mlp_out`. `block_out` is
  the GDN block's output on a linear-attention layer and the full-attention
  block's output on the others, which is what separates the two families.

`step` is the forward-call ordinal of the dense paged forward, taken explicitly
rather than inferred from a running counter. `layer` is the loop index, passed
into `RunDenseLayerPaged` as an argument the way the MoE `RunLayerPaged` already
takes one, so the sub-stage files stop depending on a thread-local ordinal.

Neither knob is new: both are already on
[`../../scripts/env-doc-allowlist.txt`](../../scripts/env-doc-allowlist.txt).
This row adds no environment variable.

### 3.3 The dump must not change what it measures

The dumps force a device synchronize per layer, so they are inert by default and
must never be set on a graph-capturing run. Two controls, both of which fail the
job rather than being reported:

- **Identity.** The generated ids with the dumps ON are byte-identical to the
  same run with them OFF, per tier. Without this the instrument could be
  reporting its own perturbation.
- **Eager.** The ROCm arm captures hipGraphs by default
  (`RocmBackend::SupportsGraphCapture()` is true; `VLLM_CPP_CUDAGRAPH` gates it),
  and the CPU arm cannot capture at all. The dumped ROCm leg therefore runs
  `VLLM_CPP_CUDAGRAPH=0`, and its ids are compared against the ROCm leg with
  graphs ON. **If those two differ, that is the finding and the row reports it
  instead**: it would mean the divergence is a capture/replay effect and not a
  kernel one, and no per-layer profile could be read until it was resolved.

## 4. The comparator

`scripts/tier-hidden-delta.py`. Host-side, no device, no build.

It reads two dump directories and their manifests, joins on
`(step, layer, stage)`, and for every joined pair reports `max_abs`, `rms`,
`rel_l2` and the index of the worst element. It prints, in its own output and in
words, **which directory it took as A and which as B**, how many rows each
manifest declared, how many joined, and how many were dropped and why. A
comparator that silently intersects two half-populated directories reports a
clean profile over almost nothing; the drop count is what makes that visible.

It refuses, by name, on: a missing manifest, a manifest with zero rows, a shape
or dtype disagreement on a joined key, a join that covers fewer keys than either
side declared, and a blob whose byte length disagrees with its manifest row.

## 5. The measurement

One `rc` lease on `strix:gpu0`, detached, never `ssh`.

**One source tree, two builds.** The tier is selected at build time on this
codebase, not at run time: the CPU tier is a `-DVLLM_CPP_HIP=OFF` build. Both
binaries come from one clone of this branch at one revision, asserted by
`git rev-parse` inside the job, so "the two tiers" cannot become "two trees".
This is what makes the comparison oracle-free AND source-identical.

**One host.** Both arms run on the same box, in the same lease, on the same
boot, over the same artifact staged to worker-local `/tmp` and re-hashed there.
The recorded CPU-tier result was measured on `thor` (aarch64) and the ROCm one
on `strix` (x86-64), so the recorded pair also crosses an architecture. Running
both here on x86-64 removes that variable, and §7's P3 is the outcome where it
turns out to have been carrying part of the effect.

**One prompt at a time.** `p1` ("The three primary colors are"), 48 tokens,
greedy, concurrency 1, MTP off, the same `--num-blocks 256` and seed the
predecessor harnesses used. `p3` and `p4` follow on the same binaries. All three
are run: §Stop conditions forbids reporting a result from one step.

Assertions inside the job, each failing it rather than being reported:

1. `git rev-parse HEAD` in the worker clone equals the branch head this spec
   names.
2. Both builds are clean (`rm -rf` first) and both `libvllm.so` are hashed. The
   HIP and CPU libraries must differ; two equal hashes mean one build did not
   happen. `vllm-cli` is a ~26 KB thin client and is never the identity.
3. The ROCm leg reports `reference_tier_hits=0` with a non-zero `device=5` op
   count, with the dump on. A leg that fell back to the portable tier did not
   measure the ROCm tier.
4. Both identity controls of §3.3 pass.
5. Each manifest declares `layers x steps` rows and the comparator's drop count
   is zero.

## 6. Reading the profile: separating a defect from fp non-associativity

Two tiers reduce in different orders and will differ in the last bits at every
layer. **That difference is not the finding**, and a report of "the first
non-zero layer" would be a report of layer 0 every time. Three readings decide
it, and all three are printed:

- **The run-to-run floor.** Each tier is run twice with the dumps on and
  compared against itself. Both are expected to be exactly zero (each tier was
  self-reproducible across six legs in the v2 gate). A zero self-delta means the
  cross-tier profile has no run-to-run component and every part of it is the
  tier. A non-zero one changes the question and is reported as the headline.
- **Shape, not magnitude.** Under pure rounding, `rel_l2` accumulates smoothly
  across the 64 layers. A **discontinuity** — one layer or one stage where the
  delta jumps by a factor the neighbouring layers do not — is a defect signature.
  A smooth ramp is not, however large it gets.
- **Amplitude against the thing that flipped.** A per-layer delta only matters if
  it can move the final logit gap by the size of the gap that actually flipped:
  0.131054 at `p1/45`, 0.006284 at `p3/45` and 0.092752 at `p4/14`, as this
  run's HIP oracle measured them. The profile is therefore read together with the
  final logit delta at the same step, which `VT_DUMP_LOGITS` (#2534, already
  landed and reachable from both sampling paths) supplies on both arms.

**The GDN split is reported separately.** This model is a GDN hybrid: linear-
attention layers and full-attention layers are different code, and the
predecessor note has flagged that split since 2026-08-23. Every table separates
them, and a term that lives in only one family is a much stronger localisation
than one that lives in both.

## 7. Pre-registered outcomes

Written before any of our hidden states have been seen. The logit-dump row's
`## Outcome` records why this matters and also why it is not sufficient: every
branch there was framed on one statistic, and the cause lived in a property that
statistic could not express. So §7.5 admits a finding outside the statistic, and
it is not a courtesy clause.

**P1 — a discontinuity.** One layer, or one stage inside one layer, where the
cross-tier `rel_l2` jumps against its neighbours. Then the term is localised to
that op, the GDN/full-attention split says which family carries it, and the row
proceeds to a red-before/green-after fix. This is the shape the gfx1100 0.8B
divergence had, where `fa0_q` read `rms-rel 1.196` with its siblings clean and
the cause was a kernel dispatched on the source dtype instead of the output
dtype.

**P2 — a smooth ramp.** No discontinuity; `rel_l2` grows monotonically from
layer 0 at a rate consistent with per-layer rounding, and reaches a final logit
delta of the size of the contested gaps. Then the divergence is irreducible
ordering between two kernel families, the row says so plainly with the profile
as evidence, and it states what that means for token-exactness BETWEEN our
tiers — which is a different and weaker claim than token-exactness against an
oracle, and one this project has never written down.

**P3 — it does not reproduce.** The x86-64 CPU arm agrees with the ROCm arm at
all three steps. Then part of the recorded `A vs D` divergence was the aarch64
CPU kernels, not the ROCm ones, the recorded comparison is restated as a
three-way one, and the row reports that rather than forcing P1 or P2 onto it.
This outcome is admissible and is not a failure of the experiment.

**P4 — the trunk is clean and the head is not.** The per-layer profile stays at
the rounding floor through layer 63 and the separation appears at the final norm
or the `lm_head`. Then this is #2534's resolution term on the ROCm side rather
than a trunk defect, and the fix belongs at the head.

**P5 — admitted outside the statistic.** The profile is not stationary across
the 48 steps: clean at most steps and spiking only at the contested ones. That
would say the term is input-dependent — a shape none of P1 to P4 assumes — and
would redirect the bisect from "which layer" to "which input". It is reported if
it occurs.

## 8. Risks

- **The instrument perturbs the arm.** Controlled in §3.3. Without the identity
  control, the whole profile could be an artefact of the synchronize.
- **The divergence needs the incremental KV cache.** These are decode steps
  reached after 45 tokens of generation. Re-prefilling the prefix in one shot
  would run different kernels and might not reproduce the flip, so both arms
  generate naturally from the prompt and the dump keys on the step.
- **The container has no `github.com` egress.** Sources are staged as a bundle
  from the share and the staged tree is verified against its own revision, the
  way the predecessor jobs do it.
- **The box.** `gfx1151` completed 6 of 6 legs of a heavier workload after
  #2511, with no knobs. If it faults here that contradicts a landed result and
  is the headline, not a nuisance.
- **A CPU-only 27B Q4_K_M decode is slow.** Only the three contested prompts are
  run, and the job is resumable per arm.

## 9. Gates

This row produces a **localisation**, and a fix only if §6 names one. It changes
no gate. `TOKEN_GATE` stays `FAIL` as the v2 evidence left it, and no speed,
latency or memory axis becomes admissible.

The instrument's own gate is a hermetic test of the writer and the comparator:
the refusal fires on an unwritable directory, the manifest round-trips, and the
comparator refuses a shape mismatch and a short join. Each assertion is
mutation-proven: deleting the guard reds exactly its own case.

## 10. Evidence required

- The rc job id, the device, the boot id, the raw log paths, both `libvllm.so`
  hashes and the asserted source revision.
- The two identity controls and the eager/graph control, verbatim.
- The self-delta (run-to-run floor) for each tier.
- The per-layer delta profile for at least one contested step, both tiers, with
  the GDN and full-attention layers separated.
- The first layer and stage where the tiers separate beyond the floor, with the
  calibration of §6 that distinguishes a defect from a ramp.
- The outcome named against §7.
- If fixed: red-before and green-after, and what happens to all three steps.

## 11. Stop conditions

- **Do not report a fix from one step.** All three are checked and each one's
  result is stated, including a step the fix does not move.
- **Do not weaken the token gate**, and do not touch its scoring. Not in scope.
- **Do not quote a speed number**, even as a by-product.
- **"Localised but not fixed" is a complete answer.** Report it rather than
  attempting a speculative fix.
- **If the cause is irreducible fp ordering, say so plainly.** That is a real
  finding and it changes what token-exactness between our tiers can mean.
- If the eager/graph control of §3.3 fails, stop and report that; no per-layer
  profile is readable until it is resolved.
- `NEEDS_CONTEXT` rather than guessing.
