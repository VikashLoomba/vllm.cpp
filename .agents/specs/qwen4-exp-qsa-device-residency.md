# The QSA block's three host reads, on a device-resident operand

**Campaign row:** `MODEL-MM-QWEN4-EXP`
**Issue:** [#2421](https://github.com/mudler/vllm.cpp/issues/2421)
**Parent spec:** [`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md), which carries
this work under `## Owed` as "the QSA CUDA arm must give the translation a
device-side home or argue it away".
**Base:** `origin/main` at `4ab04afd66f0eb385c48cc2833817db2192219f4`

## Scope

`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp` resolves three values on
the host. Each site refuses by name when its tensor is not CPU-resident:

| site | anchor at the base SHA | value read |
|---|---|---|
| `IndexerRows` | `qwen4_exp_qsa_block.cpp:83` | group 2's block table, `[1, pages]` i32 |
| `CheckRopeLayoutsAgree` | `qwen4_exp_qsa_block.cpp:147` | `kRopeProbeRows` rows of the packed cache and of the separate `cos`/`sin` |
| `Qwen4ExpQsaIndex` | `qwen4_exp_qsa_block.cpp:273` | `kv_lens`, `[T]` i32 |

In scope: make the three reads work on a device-resident operand, by copying the
words they read to the host explicitly. Out of scope, and named under `## Owed`
rather than dropped: a device-side page translation, a device-side rope
cross-check, and any throughput claim.

**Not in scope, and deliberately so: the four `vt::` op arms** (#2380 / PR #2391)
and the load-time quantized gather gate (PR #2396). This change is one of three
independent things in the way of a CUDA step and it claims only its own.

## Why refusing is not the answer, and why deleting is not either

The three refusals are correct as written. A host loop that dereferenced a device
pointer would fault or, on a unified-memory device, read plausible bytes and
produce a silently wrong selection — which is the failure this row already paid a
day for once (dropped repack markers, NaN at layer 0, every token id 0, nothing
thrown). So the refusals are not deleted.

They are also not the smallest thing that works. Each of the three values is
O(pages) or O(T) machine words that pick out an INDEX or feed a HOST comparison,
and none of them has a device-side consumer waiting for it:

- the block table's output is a `[n]` i32 row vector that is immediately uploaded
  as the `idx` operand of `vt::IndexSelect` / `vt::IndexCopy`;
- the rope probe feeds `std::fabs` on the host and produces no tensor at all;
- `kv_lens`'s output is `win_start` / `win_end`, two `[T]` i32 vectors that are
  immediately uploaded.

So the honest move is to say the read is a host read, copy the words it reads,
and state the cost. `Backend::Copy` infers its direction from the pointers
(`src/vt/cuda/cuda_backend.cu:116`, `cudaMemcpyDefault`), so one entry point
covers every arm, and `Backend::Synchronize` is what makes the bytes readable.

## Design

One file-local helper, `HostWords<T>(Dev, const Tensor&, offset, count)`. On a
`kCPU` tensor it is a `memcpy` from the tensor's own bytes and enqueues nothing.
On any other device it enqueues one `Backend::Copy` and one
`Backend::Synchronize`. The three call sites pass the exact element range they
read, so the copy's size is visible where the read is.

Three consequential decisions:

1. **`kv_lens` becomes ONE tensor, not two.** At the base SHA `QsaBlockCore`
   builds `kv_lens_host`, wraps it as a CPU `Tensor` for `Qwen4ExpQsaIndex`, AND
   uploads the same bytes as a `DBuf` for `vt::Qwen4ExpQsaGatherAttention`. The
   CPU wrapper is dropped and the `DBuf` is passed to both. On CPU that is
   byte-identical and costs nothing (`DBuf` on a CPU queue is host memory whose
   `device.type` is `kCPU`, so `HostWords` takes the memcpy arm). On a device arm
   it costs one round trip and, crucially, it is what makes the new branch
   REACHED from the production forward instead of exercised only by a test.

2. **The block table stays a `DBuf` in `qwen4_exp_registry.cpp`.** The parent spec
   records a cheaper fix — hand the block a host tensor over the runner's own
   vector — and this change does not take it, for the same reason as (1): a host
   tensor there would make the device branch dead on the production path. The
   registry is not touched.

3. **The block table is downloaded ONCE per block call, not twice.**
   `IndexerRows` is called twice on the paged arm (the scatter at `past_len`, the
   gather at `0`). It becomes a function over an already-resolved host vector, and
   `QsaBlockCore` resolves that vector once.

## Upstream anchors

This change introduces no arithmetic. The three values and their meanings are
unchanged, so the algorithm oracle is unchanged: `transformers` 5.16.0,
`models/qwen4_exp/modeling_qwen4_exp.py`, as the file's own header states. The
anchors the touched code cites — `:684`, `:670-676`, `:693`, `:653`, `:679` —
are untouched and re-verified by the block's existing golden gate, which compares
values against that oracle's captured pre-top-k `scores`.

The residency policy itself is a vLLM mirror in the following sense and no other:
vLLM's own paged-attention wrappers read a block table on the host wherever a
Python-level loop needs one and hand the device the resulting index tensor. This
change is not a port of a named upstream function, and this spec says so rather
than manufacturing an anchor for it.

## Tests

The CPU arm is the behavioural oracle. Three obligations:

1. **CPU stays byte-identical.** `test_qwen4_exp_qsa_block`,
   `test_qwen4_exp_qsa_device`, `test_qwen4_exp_layer_loop` and
   `test_qwen4_exp_runner` must be green and unchanged. On a CPU queue every
   tensor is `kCPU`, so `HostWords` takes the memcpy arm and no value moves.

2. **The device arm agrees with the CPU arm.** A CUDA case that runs the same
   block over the same inputs on both queues, comparing:
   - the SELECTION by SET EQUALITY per query token, plus the printed MARGIN
     between the last selected and the first rejected logit. A top-k error is
     bimodal — it flips or it floors — so a float tolerance on `block_ids` gates
     nothing;
   - the LOGITS and the BLOCK OUTPUT by `max|diff|`, against the parent spec's
     existing relative bounds.

3. **The refusals that remain still fire.** The block still refuses what it
   cannot do; those cases are unchanged and must stay red-on-mutation.

## Reachability

`ModelRegistry::Forward` on a `--device cuda` queue is the entry point. The
mutation is the deletion of the production call site: remove the
`RunQwen4ExpQsaBlockPaged` call in `qwen4_exp_forward.cpp`'s layer loop in a
scratch copy and the focused gate must go RED. Proving applied-ness is by a
COUNTED property, never by a patch exit code.

**This change alone does not make a CUDA step run.** Two other things are in the
way and are owned elsewhere: the four missing `vt::` CUDA op arms (#2380, PR
#2391) and the load-time quantized gather gate (PR #2396). This spec claims only
that the QSA block stops refusing a device-resident operand, and the `## Now`
section below states what a CUDA step actually reaches.

## Stop conditions

Stop and report rather than widening scope if:
- the CUDA forward stops somewhere other than predicted — that finding outranks
  the fix;
- making a read work on device requires a new `vt::` op or a change under
  `src/vt/cuda/` — that is another wave's territory;
- a device-vs-CPU selection difference appears. A selection that moves is a
  correctness defect, never a tolerance to widen.

## Now

Implementation in flight on `row/MODEL-MM-QWEN4-EXP-QSADEV`.

## Owed

- **THE COPY COSTS A QUEUE SYNCHRONIZE PER QSA LAYER PER STEP on a device arm.**
  Two, in fact: one in `QsaBlockCore` for the rope probe and the page table, and
  one in `Qwen4ExpQsaIndex` for `kv_lens`. On a 48-layer model with 12
  `qwen_sparse_attention` layers that is 24 synchronizes per step. NOTHING HAS
  MEASURED IT, because no CUDA `qwen4_exp` step exists to measure. The fix is the
  entry the parent spec already carries: fold the page resolution into
  `vt::Qwen4ExpQsaCompress`'s own address mode, which removes the table download,
  and give the rope cross-check a device-side home or argue it away. Owed under
  [#2421](https://github.com/mudler/vllm.cpp/issues/2421); no speed claim on this
  row is admissible before it.
- **THE ROPE CROSS-CHECK IS NOT MEMOIZED, and that is deliberate.** The forward
  hands every QSA layer the same three tensors, so a memo keyed on their data
  pointers would turn 12 checks per step into one. It is not taken: the device
  pool recycles pointers, so a different table at a recycled address would pass a
  pointer-keyed memo. That is the mute-switch shape this file already refuses
  once, and paying the check is cheaper than being wrong about it.
- **NO DEVICE ARM OTHER THAN CUDA WAS RUN.** `HostWords` is device-agnostic by
  construction — it reads `device.type` and calls `Backend::Copy` — but ROCm,
  Metal, Vulkan and Tenstorrent are unmeasured here and this spec claims nothing
  about them.
