# Spec — the gfx1151 plain Q4_K decode fault

Row `BACKEND-ROCM`. Issue
[#2511](https://github.com/mudler/vllm.cpp/issues/2511).
Sibling records: [#2377](https://github.com/mudler/vllm.cpp/issues/2377) (the
DFlash2 arm, same signature, different candidate) and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the throughput number
this defect makes provisional).

## Now

`ACTIVE` — the site is measured, the kernel at that site is now EXONERATED at
the production launch geometry, and the cause is still not named.

**Measured:** the failure rate; that one prefill forward reproduces it; that both
symptoms are one defect; that every failing leg dies inside
`KQuantGemmK<OutT, 2>` with its three pointer arguments exactly sized; that
`DotQ6K`'s private array is NOT the mechanism; and — new, W3 — that the same
kernel launched at the same geometry, with no model, no loader and no scheduler,
does not fault in 9,000 consecutive launches on a board that failed 5 of 6 model
legs in the SAME lease.

**Not measured:** WHY the step faults. The search has moved off the kernel and
onto what reaches the kernel. `## The standalone shaped launch` states what W3
ruled out and what it leaves.

**No fix is claimed.** The `Fmt == 3` arm this row added defaults OFF, and W2
measured that it fixes nothing; it stays OFF.

## Scope

In scope: naming the cause of the intermittent `GPU Hang` / `Memory access
fault` that `Qwen3.8-27B-Q4_K_M.gguf` decode hits on `strix:gpu0` (`gfx1151`,
ROCm 7.2.4), and fixing it if the cause is in this tree.

Out of scope: the ROCm decode throughput deficit (#2497), the DFlash2 arm
(#2377), the `IQ3_S` loader refusal (#2510), and any AMD clock-state harness
(#2381).

## Reproduction

Binary `vllm-cli` sha256
`a703b83dd8954ba6dd3cbe82efcd38083c1d55492bbbaecf5c406f7c6efd646f`, built from
`11fed3ba56b8f823c07032416982a44a8c0967b5`, Release, `VLLM_CPP_HIP=ON`,
`VLLM_CPP_HIP_ARCHITECTURES=gfx1151`. Artifact
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, staged to
worker-local `/tmp` and verified there.

```sh
vllm-cli --model <gguf> --prompt 'The capital of France is' \
         --max-tokens <N> --temperature 0 --repeat <R> --max-num-seqs 1
```

Every leg runs inside an `rc` lease on `strix:gpu0`, one job at a time.

## Findings

**Rate.** 21 legs in one lease, rc job `4237e4e1-7035-4380-87e8-7c5971ad5ff6`,
boot id `a5bc8128-f6ad-4767-8614-6923f88032e1`. The `#2511` shape
(`--max-tokens 64 --repeat 4`) failed **6 of 8**. The minimal shape
(`--max-tokens 1 --repeat 1`) failed **6 of 6**, so one prefill forward pass is
enough and the repro costs a model load rather than a 4x64-token generation.
`--repeat 4` legs that failed did so at varying points, twice before the first
logits cast and four times inside decode.

**Two symptoms, one site.** `GPU Hang` (14 legs) and
`Memory access fault ... Reason: Page not present or supervisor privilege`
(5 legs) are the same defect. `dmesg` reports `[gfxhub] page fault (src_id:0
ring:88 vmid:8 pasid:32770) in process vllm-cli`, followed by
`GPU reset(9) succeeded`.

**The faulting kernel is named.** Three legs run under `AMD_LOG_LEVEL=4` plus
`AMD_SERIALIZE_KERNEL=3` (so exactly one kernel is in flight and the host is
sitting on its completion signal) each ended on the SAME shader:

```text
ShaderName : void vt::rocm::(anonymous namespace)::KQuantGemmK<unsigned short, 2>(
    unsigned short*, unsigned char const*, vt::cpu::BlockQ8_K const*,
    long, long, long, unsigned long, unsigned long)
```

`Fmt == 2` is the **Q6_K** arm of `src/vt/rocm/rocm_grouped_gemm.hip`.

**Its arguments are exactly sized, so this is not an argument overrun.** From
the same log, two different legs, two different tensors:

| leg | grid / block | m | n | nsb | w_row_bytes | weight `obj` span | required `n * w_row_bytes` |
|---|---|---|---|---|---|---|---|
| `p4-log-1` | {6400,1,1} / {32,4,1} | 5 | 5120 | 68 | 14280 | 73,113,600 B | 73,113,600 B |
| `p4-log-2` | {1280,1,1} / {32,4,1} | 5 | 1024 | 20 | 4200 | 4,300,800 B | 4,300,800 B |

The activation `obj` likewise equals `m * nsb * sizeof(BlockQ8_K)` to the byte on
the first, and the output `obj` covers `m * n * sizeof(OutT)` on both. The grid
is exactly `(m*n + 3) / 4` warps for `m*n` outputs. Two independent static
audits computed every index in `DotQ4K`, `DotQ5K`, `DotQ6K` and
`DotQ6KIsumRange` against the block layouts and found every one inside its
buffer. **The reported fault addresses are nowhere near any of the three
arguments** -- `0x7383103b5000` sits 21 GiB from the output and 28 GiB past the
weight, and `0xbc623c10b000` is not a user address at all.

**What makes the Q6_K arm different from every other kernel on this path** is
`DotQ6K`'s `int8_t aux8[kQK_K]` -- 256 bytes of private memory PER THREAD, which
the compiler places in scratch, at 128 threads per block over 6400 blocks. No
other kernel this model dispatches declares a private array anywhere near that
size, and `DotQ6KIsumRange` right beside it does the same arithmetic 32 weights
at a time in 32 bytes. That is the A/B this row runs.

**Why the arm split in #2511 falls out of this.** Q6_K appears only in the
`Q4_K_M` mix (it carries `ffn_down` and the attention projections at Q6_K); a
bf16 checkpoint never reaches `KQuantGemmK` at all, which is why #2377's
`plain_control` completed on the same board.

## Refuted

Each of these was tested and did not hold. They are recorded because the next
reader will otherwise re-derive them.

- **#2377's candidate (CPU reference tier + un-overridden `RocmBackend::
  FlushPending`).** `VT_OP_PROVIDER_STATS=1` reports zero reference-tier notices
  on every leg, and `GetOp`'s drain call site is guarded by `slot.ref_selected`
  (`src/vt/op_provider.cpp`), so on a run that reaches no reference-tier op the
  override cannot execute. The two `FlushPending` commits (`f424b53fe`,
  `1c0d5c412`) landed AFTER the binary #2511 measured and cannot change it.
- **A host-runs-ahead race.** `AMD_SERIALIZE_KERNEL=3` blocks the host on every
  dispatch. It did NOT fix the failure: 2 of 4 legs still died, against 0 of 6
  for the unserialized minimal shape. A difference that size is not established
  at n=4 and is recorded as such, but the arm plainly does not eliminate the
  defect.
- **The prefill flash / SharedK / WMMA attention arms.** Every one is gated on
  `total_q >= 64` and `d == 256 || d == 512`
  (`src/vt/rocm/rocm_paged_attn.hip`). This prompt is 5 tokens at `d == 128`, so
  none of them can launch.
- **A decode graph, and `PersistentStepInput`'s staging block.**
  `RocmPlatform::support_static_graph_mode()` returns false
  (`src/vllm/platforms/rocm.cpp`), so no graph is captured and the staged step
  input path is not entered.
- **The CUDA async device mirror and its second stream.** Both the mirror and
  `MoeAuxStream` are inside `#ifdef VLLM_CPP_CUDA` / `VT_MARLIN_NVFP4` and are
  not compiled for ROCm.
- **A dangling `gdn_bt` / `gdn_cam` behind `gdn_meta`.**
  `CommonAttentionMetadata` and `GDNAttentionMetadata` hold every array BY VALUE
  (`std::vector` / `std::optional<std::vector>`), so the block-scoped locals in
  `execute_model` are copied, not borrowed.
- **A wavefront-size assumption.** Every reduction in the ROCm kernels is a
  16-to-1 butterfly over 32 lanes with `dim3(32, N)` blocks, which is correct at
  wave32. The exposure runs the other way: these kernels would break on wave64.
- **A data-terminated loop in the quant path.** `DF16ToF32`'s
  `while ((mant & 0x400) == 0)` is guarded by an earlier `mant == 0` return and
  terminates within 10 shifts.
- **A barrier some waves do not reach.** Every early `return` ahead of a
  `__syncthreads()` in the reachable kernels is block-uniform.

## Gates

1. The minimal shape (`--max-tokens 1 --repeat 1`) completes on N consecutive
   legs, where N is chosen so that the pre-fix failure rate would have shown at
   least one failure with probability >= 0.99.
2. The `#2511` shape (`--max-tokens 64 --repeat 4`) completes on repeated
   consecutive legs.
3. A red-before reproduction exists for the named cause, and the repair turns it
   green.

## Owed

Found by a fresh audit during this row, read in the source rather than relayed,
and NOT the cause of #2511. Each needs an issue and an owner; none is fixed
here, because a fix built on this row's flow would ride an unrelated diff.

- **`ResidentWeightF32` frees an async copy's source on the next line.**
  `src/vllm/model_executor/models/qwen3_5.cpp:1226-1246` and the byte-identical
  `include/vllm/model_executor/models/dense_attn_block.h:342-360`:
  `std::vector<float> f = WeightF32(w); … d.b.Copy(d.q, p, f.data(), nb);` and
  then `f` is destroyed at the closing brace with no `Synchronize`. W7 measured
  that `Backend::Copy` completes inside the call on THIS board, so it is latent
  here; on a discrete ROCm card, where the copy is a real DMA, it is live.
- **`EnsureQuantScratch`'s lock protects the map and not the entry.**
  `src/vt/rocm/rocm_grouped_gemm.hip:594-608`: `ScratchFor` returns a reference
  after its `lock_guard` is destroyed, and the read-modify-write of `sc.buf` /
  `sc.bytes` happens unlocked. Safe at one engine thread, which is what this
  workload has, and stated nowhere. The same function keys a `static
  std::unordered_map` on `hipStream_t`, a handle the runtime recycles after
  `hipStreamDestroy` (`rocm_backend.hip:219-223`), so a recycled value maps to a
  `hipMallocAsync` block ordered on a stream that no longer exists.
- **`LtWorkspace` grows by freeing with work possibly in flight.**
  `src/vt/rocm/rocm_matmul_hipblaslt.hip:298-309`: one `thread_local` workspace
  serves every hipBLASLt shape, and a grow does `hipFree(ws)` then
  `hipMalloc(&ws, need)` with no ordering against a previously enqueued
  `hipblasLtMatmul` that may still be writing it. Not reached on this workload —
  the dense Q4_K projections go to `rocm_grouped_gemm.hip` — and its absence is
  checkable for free, because `MatmulBTLt` prints `hipBLASLt COL ok M=…` on
  every first-seen shape.
- **Unsynchronised pointer-table uploads on the ROCm fp8 MoE path.**
  `src/vt/rocm/rocm_fp8_channel_gemv.hip:509-524`: five `hipMemcpyAsync` calls
  from caller arrays into `thread_local` device tables that a kernel then
  dereferences AS POINTERS, with no sync and no pinned staging. The identical
  shape sits in `src/vt/rocm/rocm_matmul_hipblaslt.hip:653-676`
  (`MatmulBTPointerBatchKernelRocm`), which has no caller anywhere in the tree
  and is therefore dead code one call site away from being real. Neither is
  reachable on a dense model with no routed-expert layers.

Filed earlier in this row:

- [#2522](https://github.com/mudler/vllm.cpp/issues/2522) — `GdnStateGatherK` /
  `GdnStateScatterK` index the GDN state cache at an unbounded `state_idx[row]`
  and are never passed the slot count, while the sibling `GdnScanK` in the same
  family bounds the identical index. Found while diagnosing this row and NOT its
  fault site, so it is filed rather than folded in. It mirrors `cuda_gdn.cu`
  exactly, so the repair belongs on both backends together and not inside a
  gfx1151 row.

## Gate state on this change

`scripts/agent-preflight.sh --staged` on `d6fc76152`: **no gate FAILED** and no
`gate(s) failed` line. It still reports `NOT a green preflight`, and the reason is
five SKIPS, each printed with its cause:

| skipped gate | why |
|---|---|
| `check-arm-isa-build.py` | needs `--compile-commands` |
| `check-cpu-isa-build.py` | needs `--compile-commands` |
| `check-cuda-fat-gencode.py` | needs `--compile-commands` |
| `check-triton-aot-multiarch.py` | needs `--vendored-root` |
| `check-pr-size.py` | needs `--base` / `--head`; there is no pull request |

**Say the consequence plainly: nothing on the dev box compiles the `.hip` edit.**
The three build gates want a compile-commands database, and this row was worked
under an instruction not to compile locally (another agent holds the box's
compile slot, and parallel builds have OOM-killed it). So the ONLY compile
evidence for `Fmt == 3` is the rc job's own rebuild on `strix:gpu0`, which is why
`q6k.sh` aborts on a non-zero build and again if the rebuilt binary is
byte-identical to the original -- a leg run after a build that silently did
nothing would read as a result and be none.

One earlier preflight run reported `test_cpu_x86_llamacpp_floor` FAILED at
`test_a_contended_leg_is_discarded_and_never_summarised`. It passes standalone
(10 tests, `OK`) and passes in the next full preflight. The box load was 21-31
from other agents' builds at the failing run and 12.7 at the passing one, which
is the already-recorded load flakiness of that harness test. It is not reachable
from this diff, which touches one ROCm `.hip` file, one allowlist line and this
spec.

## The A/B arm computes the same number, checked rather than asserted

An A/B is only a measurement of private memory if the two arms compute the SAME
thing; otherwise a green `newon` leg could be a different kernel giving different
tokens. So the substitution was checked off-device, on the host, before any leg
ran.

`DotQ6K` and `DotQ6KIsumRange` were extracted VERBATIM from
`src/vt/rocm/rocm_grouped_gemm.hip` (by `sed` range, not retyped) into a
standalone program with the real `BlockQ6_K` / `BlockQ8_K` layouts and the real
`DF16ToF32`, and driven over 200,000 random superblocks. It compares
`DotQ6K(xb, yb)` against `(DF16ToF32(xb->d) * yb->d) * DotQ6KIsumRange(xb, yb, 0,
8)` -- exactly the substitution `Fmt == 3` makes -- by `memcmp` on the float, not
by a tolerance.

```text
compared DotQ6K(float) against (DF16ToF32(d)*y.d)*DotQ6KIsumRange(0,8):
    200000 blocks, 0 BIT-DIFFERENT
```

**Negative mutation, because a comparison that cannot fail proves nothing.**
Changing `qh_shift` from `2 * r` to `2 * r + 1` inside `DotQ6KIsumRange` ALONE
gives `200000 blocks, 200000 BIT-DIFFERENT` and a non-zero exit. The unmutated
source is byte-unchanged afterwards (sha256
`7e2eaf51fd56816a04f83ae826b572a2961ddbdfe6593dbb5dfc10d1936af5bd`) and green
again.

**What this does NOT establish**, stated so nobody reads it as more: it is the
HOST compiler's answer for these two bodies, not the HIP compiler's answer for
`gfx1151`. The on-device check is the leg's own output text, which `q6k.sh`
prints per leg, and which must read ` Paris` on both arms.

## Why a green ROCm suite coexists with this

`tests/vt/test_backend_cross_device.cpp` DOES cover `kMatmulBTQuant` on ROCm for
Q6_K, and it passes. It runs `M = 3, N = 8, K = 512`, which is `nsb = 2` and a
grid of `ceil(3*8/4) = 6` blocks. The cooperative case beside it runs `N = 64`
at `K` in {4096, 12288}, and only at `m == 1`.

The launch that dies is `m = 5, n = 5120, nsb = 68` -- **6400 blocks**, and
`nsb = 68` against the covered `nsb = 2`. The gate is three orders of magnitude
below the production launch on the grid axis and 34x below it on the superblock
axis, so it exercises the kernel's ARITHMETIC and has never exercised its launch.
That is the gap, and it is the shape any regression test for #2511 has to take:
the `## Acceptance` line in the issue asks for a test that "runs the arm enough
times to see a 1-in-3 failure", and a test at `N = 8` would not see it however
many times it ran.

## The private-memory mechanism is REFUTED, and the owed number is now read

rc job `3415e7c5-2ad8-4a09-b97f-173d15e5edd4`, boot
`a5bc8128-f6ad-4767-8614-6923f88032e1`, minimal shape, three arms interleaved:

| arm | `DotQ6K` private scratch | failures |
|---|---|---|
| `newon` (`Fmt == 3`, register-resident) | 32 B intended | **6 of 6** |
| `newoff` (`Fmt == 2`, unpatched body) | 256 B | **3 of 5** |
| `orig` | 256 B | no data — the control lib dir lacked the client's other `.so`s, `rc=127` on all five |

Shrinking the per-thread private array did not fix the fault; the arm that
shrank it failed every leg it ran. `VT_ROCM_Q6K_SMALL_PRIVATE` stays default OFF
on evidence rather than on caution.

**Why n mattered.** At round 2 the tally read `newoff` 0 of 2 against `newon`
2 of 2 — a clean-looking confirmation of the OPPOSITE conclusion. `newoff`'s
first failure did not arrive until round 3. Two legs per arm is an anecdote at
this failure rate.

**The `private_segment_fixed_size` this spec listed as owed is now read**, and it
cost no GPU compute and no `roc-obj`: `hipFuncGetAttributes` reads
`localSizeBytes` straight out of the loaded code object. On `gfx1151`, ROCm
7.2.4, `/opt/rocm/lib/llvm/bin/clang++ -x hip -O2 --offload-arch=gfx1151`:

| instantiation | private segment | registers |
|---|---|---|
| `KQuantGemmK<unsigned short, 2>` | **272 B** | 95 |
| `KQuantGemmK<unsigned short, 3>` | **0 B** | 82 |

So the A/B did change the thing it claimed to change — 272 bytes of scratch per
thread against none at all — and changing it changed nothing about the fault.
That is the strongest form the refutation can take. Stated limit: these are the
probe translation unit's numbers at `-O2`, not the tree's full build's, so they
characterise the substitution rather than pin the shipped binary.

## The standalone shaped launch: the kernel is where it LANDS, not where it is MADE

rc job `3c8e7ea6-f7be-4f8d-8a7a-f4724a130174`, boot
`a5bc8128-f6ad-4767-8614-6923f88032e1`, log
`/mnt/nas_share/rc/rocm-strix-shape/evidence/shape.log`.

The in-tree gate runs `KQuantGemmK` at `M=3, N=8, K=512` — `nsb = 2`, six
blocks. The launch that dies runs `m=5, n=5120, nsb=68` — grid `{6400,1,1}`,
block `{32,4,1}`, weight `obj` 73,113,600 B, and a **bf16** output, which is a
different template instantiation from the f32 one the gate exercises. So the
gate had tested the kernel's arithmetic and had never tested its launch.

`/mnt/nas_share/rc/rocm-strix-shape/probe.hip` closes that gap outside the tree:
`gen.sh` sed-extracts the device bodies VERBATIM from
`src/vt/rocm/rocm_grouped_gemm.hip` (anchored by `grep`, and the anchors are
asserted unique) into a standalone program with no model, no loader and no
scheduler. Five arms, interleaved over six rounds, 300 launches a leg:

| arm | what it varies | legs | failures |
|---|---|---|---|
| `q6k_managed` | production allocator (`hipMallocManaged`) | 6 | **0** |
| `q6k_devmem` | plain `hipMalloc` weights | 6 | **0** |
| `q4k_managed` | same `m/n/nsb`, Q4_K body | 6 | **0** |
| `q6k_ballast` | `q6k_managed` + a 16 GiB managed ballast | 6 | **0** |
| `small` | the in-tree gate's `M=3,N=8,K=512` | 6 | **0** |

30 legs, 9,000 launches, `drift=0` on every one (each leg also byte-compares
every iteration's output against its first, so a launch that read state it did
not own would show even without a fault).

**The board was in its failing state during the same job.** PHASE 3 of that same
lease ran the model repro and it failed **5 of 6** legs
(`MLEG sdmaoff-r1 GPUHANG`, `sdmaon-r1 MEMFAULT`, `sdmaoff-r2 OK`,
`sdmaon-r2 GPUHANG`, `sdmaoff-r3 MEMFAULT`, `sdmaon-r3 MEMFAULT`). So the clean
probe is not a healthy-board artefact; it is a contrast measured inside one boot.

What this rules out, at the production geometry: the grid size, the block shape,
the superblock count, the bf16 instantiation, the weight extent, the Q6_K format
itself, the allocator branch, and the total managed footprint — each of them in
isolation. What it leaves is everything the step does that the probe does not:
the loader, the other ~600 dispatches of a forward pass, the host-side traffic
between them, and whatever they leave behind in queue, descriptor or allocator
state.

## Also refuted, W3

- **`HSA_ENABLE_SDMA=0`.** 2 of 3 legs still failed, against 3 of 3 with SDMA on.
  Not a fix, and not distinguishable from the baseline at n=3.
- **The Q6_K tensor SHAPES rather than the format.** `q4k_managed` runs the same
  `m`, `n` and `nsb` through the Q4_K body and is equally clean, and `q6k_*` at
  the Q6_K shape is clean too, so neither the shape nor the format reproduces it
  standalone.
- **The managed-memory FOOTPRINT alone.** A 16 GiB `hipMallocManaged` ballast,
  device-memset so its pages are real, changes nothing.

## What the fault ADDRESS is, and why nobody had read it

The `AMD_LOG_LEVEL=4` legs kept only their last 400 KB
(`evidence/p4-log-*.err.tail`), which is about 0.5 s of trace. Run through
`/mnt/nas_share/rc/rocm-strix-shape/mapfault.py`, which rebuilds the allocation
table out of the log's own `hipMallocManaged` / `ihipMallocManaged ptr=` /
`Returned hipSuccess : 0x…` triples:

| leg | fault address | in that window |
|---|---|---|
| `p4-log-1` | `0x7383103b5000` | in no recorded allocation, but only **28 MB** past the highest allocation the window records (`0x73830e756000`) |
| `p4-log-3` | `0xbc623c10b000` | **74 TB** from every allocation in the window |

The framing "21 GiB from the output" is true and misleading: the process's
managed allocations span 27 to 46 GiB of virtual address space in these windows
alone, so 21 GiB from one argument is ordinary. `0x7383103b5000` sits at the TOP
of that span and is a plausible address for an allocation made earlier in the
load, which the 400 KB window cannot see. `0xbc623c10b000` is not.

Two different fault addresses with two different characters is itself a finding:
they are unlikely to be one wild index.

The full trace is what settles it, and it is cheap — filtering
`AMD_LOG_LEVEL=4` live down to allocation, free, copy, dispatch and fault lines
keeps the file to a few MB. rc job `66658d40-1481-4ff7-841c-252e5744c3a5`
(`fault.sh`) does that on four legs and maps each fault address against the
whole table, and then runs an env-knob matrix (`base`, `HSA_ENABLE_SDMA=0`,
`GPU_MAX_HW_QUEUES=1`, `AMD_DIRECT_DISPATCH=0`,
`HSA_ENABLE_SCRATCH_RECLAIM=0`) five rounds interleaved. Results land in
`/mnt/nas_share/rc/rocm-strix-shape/evidence2/fault.log`.

Recorded so the run is readable: that job's `vllm-cli` is sha256 `a703b83d…`,
the binary this issue reports, but its `libvllm.so` is `41bb4052…`, the library
W2's A/B rebuilt. With `VT_ROCM_Q6K_SMALL_PRIVATE` unset that library takes the
`Fmt == 2` path, so it is W2's `newoff` arm, which failed 3 of 5 — it reproduces,
and it is not the byte-identical `#2511` library.

## The regression test this row lands

`tests/vt/test_backend_cross_device.cpp`, "keep-quant Q6_K GEMM runs at the
production launch geometry": `M=5, N=5120, K=17408` through `vt::MatmulBTQuant`
with a **bf16** output, repeated (default 8 launches, `VT_TEST_Q6K_LAUNCHES`
raises it for a soak). It checks the first launch against the CPU oracle on a
64-column slice of the same weight buffer — exact, not sampled, because output
column `j` reads only weight row `j` — and byte-compares every later launch
against the first. It skips the CPU device arm on purpose: a full-width CPU pass
is 445M scalar MACs per CI run to compare the reference with itself.

It is a launch gate, not a second arithmetic gate. On the evidence above it is
expected to pass on `gfx1151`; what it stops is the class of regression the old
`N=8` case could not see.

## Stop conditions

- A cause that is in the AMD driver or firmware rather than in this tree ends the
  row with that finding recorded and the issue updated; it is not fixed here.
- No fix lands on a single passing leg. One green leg proves nothing at this
  failure rate.

## Queued measurements

| rc job | script | what it decides | lands in |
|---|---|---|---|
| `66658d40-1481-4ff7-841c-252e5744c3a5` | `fault.sh` | what the fault ADDRESS is, mapped against the whole allocation table, plus the env-knob matrix | `/mnt/nas_share/rc/rocm-strix-shape/evidence2/fault.log` |

Both of the jobs this section used to list have run and are reported above:
`q6k.sh` under `## The private-memory mechanism is REFUTED`, and `codeobj.sh`'s
question is answered there too, by `hipFuncGetAttributes` rather than by
`roc-obj`, so the code-object route is no longer owed.

## W4: three fault addresses, and none of them is in any allocation

rc job `66658d40-1481-4ff7-841c-252e5744c3a5`, `fault.sh`, log
`/mnt/nas_share/rc/rocm-strix-shape/evidence2/fault.log`. Four legs under
`AMD_LOG_LEVEL=4` + `AMD_SERIALIZE_KERNEL=3`, with the trace filtered LIVE down
to allocation, free, copy, dispatch and fault lines so the whole run fits in a
few MB instead of gigabytes. `mapfault.py` then rebuilds the allocation table
from the log's own `hipMallocManaged (…, size)` / `ihipMallocManaged ptr=` /
`Returned hipSuccess : 0x…` triples and locates each fault address in it.

| leg | class | live allocations at the end | live bytes | fault address | in an allocation? |
|---|---|---|---|---|---|
| `log-r1` | MEMFAULT | 423 | 9.44 GiB | `0xed0000fe0000` | **no** — 113 TB from the nearest |
| `log-r2` | GPUHANG | 590 | 13.50 GiB | (no address; hang) | — |
| `log-r3` | OK | 55 (966 freed) | 0.04 GiB | — | — |
| `log-r4` | OK | 55 (966 freed) | 0.04 GiB | — | — |

Two facts fall out, and both are new.

**The fault addresses are WILD, not a buffer that lost its pages.** Across three
recorded faults they are `0x7383103b5000` (W1), `0xbc623c10b000` (W1) and
`0xed0000fe0000` (W4). The last two are 74 TB and 113 TB from every HIP
allocation the process ever made. The idea that the GPU is touching a real
managed weight whose pages went away does not survive this: it would fault
INSIDE an allocation. Three different addresses with three different characters
also argues against one fixed bad index.

**The fault happens while the model is still becoming resident.** A completed
leg makes 1,021 allocations; the faulting legs died at 423 and 590 of them, with
9.4 and 13.5 GiB live. `RocmPlatform::needs_weight_staging()` is false, so
weights become device-resident LAZILY at first use — inside the prefill — which
puts `Alloc` + an async `Copy` + a page release in between the very kernel
launches that fault. That is where W5 looks.

`dmesg` on the faulting leg adds `amdgpu: sq_intr: error, detail 0x00000000,
type 2, priv 1, wave_id …` before the reset. That is a shader-unit memory
violation raised by a WAVE, not only a VM fault seen by the hub, which is worth
recording because it constrains the mechanism.

## The env-knob matrix, and the one knob that moved the rate

Same job `66658d40`, PHASE B: five knobs, five legs each, interleaved with the
within-round order rotating, minimal shape, no rebuild.

| arm | env | legs | failures |
|---|---|---|---|
| `base` | — | 5 | 4 |
| `hwq1` | `GPU_MAX_HW_QUEUES=1` | 5 | 3 |
| `nodirect` | `AMD_DIRECT_DISPATCH=0` | 5 | 3 |
| `noreclaim` | `HSA_ENABLE_SCRATCH_RECLAIM=0` | 5 | 2 |
| **`sdma0`** | **`HSA_ENABLE_SDMA=0`** | **5** | **0** |

`HSA_ENABLE_SDMA=0` is the only arm that took no failure, and it is the arm that
turns OFF the engine that reads the HOST SOURCE of an `hipMemcpyAsync` directly.
With SDMA disabled the runtime stages a pageable H2D copy instead, so releasing
the source afterwards stops being an exposure. That is the mechanism W5 names,
approached from the other side.

**`HSA_ENABLE_SDMA=0` IS A WORKAROUND, NOT A FIX, AND IT IS NOT FREE.** It turns
off the DMA engine, so every host-to-device copy falls back to a staged blit on
the compute queue. This row has NOT measured what that costs, and #2497 is the
open throughput axis it would move. The flag must never be quoted as
performance-neutral, must not be set by default, and must not close this issue:
it removes an exposure without saying what created it.

**Count the legs before quoting the rate.** Each leg appears TWICE in
`fault.log`, because the script's `===== SUMMARY =====` block reprints every
`MLEG` line it already logged live. `grep -c` over the whole log therefore
doubles every arm, and so does `rc logs`, which serves the same file. The table
above is the in-script tally, which runs before the reprint, and it is 5 legs an
arm. A pooled reading that says 10 legs an arm is that doubling, and the ratio
it reports is the same ratio, so it is not more evidence — it is the same
evidence counted twice. Verified: 50 `MLEG` lines, 25 unique, 5 unique per arm.

**Do not read this as proven, and here is the exact reason.** The W3 job's
PHASE 3 ran the same knob against the same binary and got `sdmaoff` **2 of 3**
against `sdmaon` 3 of 3. Those three legs are tagged `sdmaoff` and PHASE B's are
tagged `sdma0`, so any tally keyed on the arm name drops the contradicting ones
silently. They are the same knob and they belong in the same denominator. Pooled over both jobs the knob reads 2 failures in 8
against 7 in 8 for the unset arm — suggestive (Fisher p ~ 0.04) and not a
result. Two jobs disagreeing at n=3 and n=5 is precisely the shape W2 already
paid for once. The knob is a strong steer for the next A/B, not a fix, and
`scratch reclaim`, `hw queues` and `direct dispatch` all moved the rate a
little, which is what schedule noise looks like.

**The fault addresses now number five, and four of them are impossible.**
`0x7383103b5000`, `0xbc623c10b000`, `0xed0000fe0000`, `0xf9f0dfef3000`,
`0xf6037c75f000`. Every one but the first is above `0x0000_7fff_ffff_ffff`, the
top of a 4-level-paging Linux user address space, and each is page-aligned.
Five faults at five different addresses, four of them outside any addressable
user range, is a wild pointer or a torn-down mapping, not one bad index.

## A confound every future A/B on this box has to carry: failures cluster in a job's EARLY model legs

Tabulated over the model legs of the two W3/W4 jobs, in the order they ran:

| job | model legs, in order | failures at ordinal |
|---|---|---|
| `3c8e7ea6` PHASE 3 | F F O F F F | 1, 2, 4, 5, 6 |
| `66658d40` PHASE A + B | F F O O · F O O F O O O … | 1, 2, 5, 8 |

Both jobs fail their FIRST TWO model legs and then get progressively cleaner.
This is not something any single-arm design can see, and an A/B that runs arm A's
legs before arm B's would charge the whole effect to arm A. It is why `q6k.sh`,
`shape.sh` and `adopt.sh` all interleave, and why `adopt.sh` additionally SWAPS
the within-round order on alternate rounds.

Do not read this as a cause. It is a schedule effect with at least three
candidate sources — the GGUF's page-cache warmth, the state the board is left in
by a `GPU reset(9)`, and clock or power state — and separating them is its own
measurement. It is recorded here so that nobody reads a late-running arm's clean
legs as a fix.

## W5: the candidate the audit found — an async copy whose source is released underneath it

**Not yet measured. The A/B is queued (rc `ea99c89a-d170-4322-93cb-e9bde698be7c`,
`adopt.sh`), and this section says candidate, not cause.**

`RocmBackend::Copy` (`src/vt/rocm/rocm_backend.hip:211-213`) is a bare
`hipMemcpyAsync` with no synchronisation and no event. Two weight-staging call
sites hand it a source and then release that source without waiting:

- `src/vllm/model_executor/models/qwen3_5.cpp:1188-1202` and the SHARED
  `include/vllm/model_executor/models/dense_attn_block.h:253-268`:
  `Alloc` → `Copy(d.q, p, w.bytes.data(), nb)` → `AdoptDeviceBytesAsHost(d.b, w)`.
  For an OWNED buffer that function
  (`src/vllm/model_executor/models/qwen3_5_weights.cpp:424-461`) runs
  `::madvise(…, MADV_DONTNEED)` over the interior pages of the copy SOURCE and
  then move-assigns `self.bytes`, destroying the vector — `free()`, and `munmap`
  for anything over glibc's mmap threshold.
- `src/vllm/model_executor/models/qwen3_5.cpp:1237-1243` and
  `include/vllm/model_executor/models/dense_attn_block.h:342-360`
  (`ResidentWeightF32`): the source is a function-local `std::vector<float>`
  that is destroyed at the closing brace, with no `Synchronize` between.

Why this is a ROCm-shaped defect rather than a general one: the adoption branch
is gated on `Backend::DeviceMemoryIsHostAddressable()`, which is TRUE on
`gfx1151` and FALSE on CUDA (`src/vt/cuda/cuda_backend.cu` pins CUDA to the
`false` default), so the CUDA lane never takes it.

Why it would look like this defect: `MADV_DONTNEED` on a private page tears the
PTE down immediately, so an in-flight DMA reading it hits a page that is not
present; the fault has nothing to do with the arguments of whatever kernel
happens to be resident, and the biggest, longest kernel on the path
(`KQuantGemmK<u16,2>`, 6400 blocks, 2.9 ms in the trace) is the most likely one
to be resident; it is intermittent because it is a race with the copy engine;
and `AMD_SERIALIZE_KERNEL=3` does not touch it, because that serialises KERNELS
and not copies.

**The owned branch is reachable, and it is not small.** The W4 trace of a
COMPLETED leg records 1,018 `hipMallocManaged` calls totalling 24.9 GiB, and two
of them are 2,542,796,800 B and 1,042,944,000 B. Allocations that size are the
bf16 expansions, not the borrowed keep-quant tensors, so they are exactly the
buffers whose host source is an owned block far above glibc's mmap threshold —
the case where the release is a `munmap` rather than a free-list return. A 2.4
GiB `hipMemcpyAsync` does not complete in the microseconds before the
`madvise` on the next line.

**Every large weight upload on this path sources from the HEAP, and a third of
them from an `mmap`ed heap chunk.** Measured over the 402 copies larger than
1 MB in the W4 trace of a completed leg (`evidence2/log-r3.trace.gz`):

| quantity | value |
|---|---|
| bytes copied host to device | 24.19 GiB |
| span the sources occupy | 28.35 GiB, wider than the 15.9 GiB file |
| sources at a page boundary exactly (a raw file mapping) | **0 of 402** |
| sources at page boundary + `0x10` (a glibc `mmap`ed malloc chunk's payload) | **145 of 402** |

145 sources at exactly `+0x10` is not chance (1 in 4096 would give about 0.1),
and a chunk glibc served by `mmap` is a chunk `free()` returns by `munmap` — a
real hole, not a free-list entry.

Split by tensor size, the two populations separate cleanly, and they separate
exactly the way the source says they should:

| copy size | copies | sources at page + `0x10` | reads as |
|---|---|---|---|
| 50,135,040 (Q4_K row block) | 160 | 0 | BORROWED from the GGUF mapping |
| 73,113,600 (the Q6_K tensor that faults) | 32 | 0 | BORROWED |
| 17,694,720 / 35,389,440 / 4,300,800 / 2,949,120 | 64 | 0 | BORROWED |
| **62,914,560** | **96** | **96** | OWNED, `mmap`ed chunk |
| **104,857,600** | **48** | **48** | OWNED, `mmap`ed chunk |
| **2,542,796,800** | **1** | **1** | OWNED, `mmap`ed chunk |

That is the code's own split, measured: the keep-quant tensors borrow the
mapping and take `AdoptDeviceBytesAsHost`'s `w.bytes.borrowed()` early return,
and the bf16 expansions are owned and do not. **145 uploads totalling 12.7 GiB
per run take the branch that madvises and unmaps its own copy source**, and they
take it lazily, inside the forward.

The largest is unambiguous, and its two log lines sit next to each other:

```text
hipMallocManaged ( …, 2542796800, 1 )   ->  ptr=0x7640a4700000
hipMemcpyAsync ( 0x7640a4700000, 0x76440051f010, 2542796800, hipMemcpyDefault, … )
```

2.37 GiB enqueued from a heap block at page + `0x10`, with the host returning
immediately. It is allocation 114 of 1018 on a completed leg, and the other
billion-byte upload is allocation 1015 of 1018 — first and last use of the
embedding and the LM head, which is exactly the signature of residency happening
lazily inside the forward rather than at load. The faulting legs died at
allocation 423 and 590, in between.

**What does not yet fit, stated rather than glossed:** W4's fault addresses are
in no allocation at all, where a released copy source would fault at a real host
VA. Either the reported address is not the source VA, or W5 is not the whole
story. The A/B decides it without a rebuild: `VT_ADOPT_DEVICE_BYTES=0`
(`qwen3_5_weights.cpp:405-408, :431-433`) turns the adoption off on the binary
this issue reports, nine legs an arm, alternating. Nine clean `adopt0` legs
against a baseline that fails ~5 of 6 would be decisive; anything less is not,
and W2 is the standing reminder of why two legs an arm is an anecdote here.

Note that `VT_ADOPT_DEVICE_BYTES=0` deliberately does NOT disable
`ReleaseDirectUploadSource` (`qwen3_5_weights.cpp:399-401`), so the A/B covers
the OWNED-buffer branch only. For a GGUF the keep-quant weights are BORROWED
from the mapping and take the early return at `:429`, so what the knob actually
moves is the owned bf16 expansions — the norms, and `token_embd` / `output` if
either resolves to `kExpandBf16`.

## W6: the experiment that separates OUR defect from an SDMA defect

rc job `1fabe3c2-d44f-4667-8291-b97d21782f8e`, `uaf.sh`, results in
`/mnt/nas_share/rc/rocm-strix-shape/evidence4/uaf.log`.

`HSA_ENABLE_SDMA=0` removing the fault is consistent with two different worlds,
and only one of them is this tree's to repair:

- **(a) ours.** We enqueue `hipMemcpyAsync` and then release its source, and
  SDMA executes our copy faithfully into the hole we made. The fix is an
  ordering change; the flag is a workaround for our own bug.
- **(b) not ours.** An SDMA or firmware defect on `gfx1151`. The outcome is the
  flag plus an upstream report, and no code change here.

PHASE A decides it with no model. `uafprobe.hip` — generated by the same
`gen.sh` that produced the W3 probe, so the kernel bodies are still
sed-extracted verbatim — replays `ResidentWeight`'s upload shape at the sizes W4
measured (96 x 62,914,560 B, 48 x 104,857,600 B, 1 x 2,542,796,800 B, all from
`std::vector` blocks that glibc serves by `mmap`), launches the production Q6_K
GEMM after each upload so a fault has the same thing in flight to land on, and
varies ONE thing:

| arm | what it does after the async copy | what it isolates |
|---|---|---|
| `uaf` | `madvise(MADV_DONTNEED)` + `free()` immediately | our pattern |
| `sync` | `hipStreamSynchronize`, THEN release | the fix shape |
| `nofree` | never release | the release itself |
| `uafnosdma` | `uaf` under `HSA_ENABLE_SDMA=0` | the engine |

`uaf` faulting while `sync` and `nofree` stay clean is a red-before for a defect
in this tree and makes world (a) the answer. All four clean says the loader
shape alone does not reproduce it and moves weight to (b). `uafnosdma` clean
while `uaf` faults says the DMA engine is the executor, not the author.

PHASE B is the model-level counterpart, `VT_ADOPT_DEVICE_BYTES=0` against an
unset baseline, alternating with the within-round order rotating.

## W6 RESULT: W5 is REFUTED, from both sides

rc job `1fabe3c2-d44f-4667-8291-b97d21782f8e`, boot
`a5bc8128-f6ad-4767-8614-6923f88032e1`, log
`/mnt/nas_share/rc/rocm-strix-shape/evidence4/uaf.log`.

**PHASE A, the loader shape standalone, four arms interleaved over six rounds:**

| arm | what it varies | legs | failures |
|---|---|---|---|
| `uaf` | `madvise(MADV_DONTNEED)` + `free()` right after the async copy | 6 | **0** |
| `sync` | `hipStreamSynchronize`, THEN release | 6 | **0** |
| `nofree` | never release | 6 | **0** |
| `uafnosdma` | `uaf` under `HSA_ENABLE_SDMA=0` | 6 | **0** |

24 legs, 145 uploads each at the measured sizes (96 x 62,914,560 B, 48 x
104,857,600 B, 1 x 2,542,796,800 B) from `std::vector` blocks glibc serves by
`mmap`, with the production Q6_K GEMM launched after every upload. Nothing
faulted. The release that W5 named does not fault on its own.

**PHASE B, the model-level A/B, alternating with the order rotating:**

| arm | legs | failures |
|---|---|---|
| `base` | 6 | 1 |
| `adopt0` (`VT_ADOPT_DEVICE_BYTES=0`) | 6 | **4** |

Turning the adoption off did not fix the fault; the arm without it failed MORE.
That is the same shape as W2, where the arm that removed the suspected mechanism
also failed more, and the reading is the same: the mechanism is not the cause.

**So W5 is refuted.** `ResidentWeight`'s unwaited `hipMemcpyAsync` followed by
`AdoptDeviceBytesAsHost` is not the cause of #2511, measured standalone and
measured in the model. The section above stays in this spec because the next
reader will otherwise re-derive it: it is a genuine ordering smell, it is
reachable, and it is not this bug.

**Read PHASE A with its limit.** A clean `uaf` arm proves that the release is
harmless OR that the copy never raced, and this probe does not separate those:
it never measured whether `hipMemcpyAsync` from PAGEABLE host memory is
asynchronous on this runtime. HIP is permitted to make that copy synchronous,
and if it is, then no call site in this tree that hands `Copy` a pageable source
can race at all, and the whole family is refuted rather than just this member.
Timing does not settle it here, because every arm ends on the same
`hipDeviceSynchronize` and all four measured 8-10 s. That measurement is one
`std::chrono` bracket around a single 2.4 GiB `hipMemcpyAsync` call and it is
worth taking on the next lease.

**Note also that `base` failed only 1 of 6 in this job**, against 4 of 5 in W4
and 5 of 6 in W3, on the same binary and the same boot. The rate is not stable
between jobs. Every arm ratio in this spec has to be read against that, and it
is a second reason no arm may be called a fix from one job.

## W7: `Backend::Copy` is SYNCHRONOUS on this board, and that closes a whole family

rc job `c56b596c-0f57-4d05-ab54-744fb5c70c17`, `sdma.sh` PHASE A, log
`/mnt/nas_share/rc/rocm-strix-shape/evidence5/sdma.log`. A 2,542,796,800 B
`hipMemcpyAsync` into a `hipMallocManaged` destination, timed with one
`std::chrono` bracket around the enqueue and a second around the following
`hipStreamSynchronize`, three repetitions, three process runs:

| source | enqueue (ms) | total (ms) | fraction spent INSIDE the enqueue |
|---|---|---|---|
| pageable (`std::vector`, page + `0x10`) | 90.9 - 93.5 | 91.0 - 93.5 | **0.9995 - 0.9999** |
| pinned (`hipHostMalloc`) | 87.1 - 89.5 | 87.1 - 89.5 | **1.0000** |

The call does not return early. It moves 2.37 GiB in about 90 ms, which is
28 GB/s — host memory bandwidth, i.e. a CPU `memcpy`, not a DMA transfer. On
this APU both operands are host-addressable, so the runtime copies inline and
`hipStreamSynchronize` has nothing left to wait for. The PINNED row is the
control: a copy that is asynchronous by contract behaves the same way, so this
is the runtime's choice for this destination and not a property of pageable
memory.

**Two things follow, and one of them is a whole family closed.**

The "we released the source underneath an in-flight copy" family is refuted by
construction, not member by member: there is no in-flight copy to race. W5's
`AdoptDeviceBytesAsHost`, `ResidentWeightF32`'s local `std::vector`, and every
other `Backend::Copy` call site that hands over a stack address or a temporary
are all ordering smells and none of them can produce this fault while the copy
completes inside the call. W6 refuted W5 by measurement; this refutes the
reason.

And it removes the obvious reading of the `HSA_ENABLE_SDMA=0` result. SDMA is
not moving our weights, so the flag cannot be suppressing a copy-source race.
What SDMA does move on an HMM board is MANAGED PAGE MIGRATION, which is the
other half of the pattern W8 tests.

**Limit:** the syncprobe's stream is idle. The model's stream has work in
flight, and a busy stream could take a different path. That is worth a second
bracket and is not what the arms above measured.

## W8: what the faulting legs were ACTUALLY doing, and the correction it forces

The `AMD_LOG_LEVEL=4` traces of the two faulting legs
(`evidence2/log-r1.trace.gz`, `log-r2.trace.gz`) were read all the way through
rather than in their last 400 KB, and they show a pattern nobody had counted.
`RocmPlatform::needs_weight_staging()` is false, so weights become device
resident LAZILY at first use — inside the forward. The trace is this, over and
over, on one thread and one stream:

```text
hipMallocManaged ( …, 2949120, 1 )      ->  ihipMallocManaged ptr=0x7e5918e00000
hipMemcpyAsync ( 0x7e5918e00000, 0x7e5ba4071060, 2949120, hipMemcpyDefault, … )
ShaderName : QuantizeQ8KK(…)
ShaderName : KQuantGemmK<unsigned short, 0>(…)   Arg1: ptr:0x7e5918e00000
```

**249 of the 570 dispatches on the faulting leg take a pointer argument that
lies inside one of the four most recently returned `hipMallocManaged`
allocations.** Forty-four per cent of this model's dispatches read a virtual
address that did not exist a millisecond earlier, with work already in flight on
the same queue.

**This forces a correction to #2511's headline.** The last dispatch before the
fault is `KQuantGemmK<unsigned short, 0>` on `log-r1` — `Fmt == 0` is **Q4_K**,
not Q6_K — and `KQuantGemmK<unsigned short, 2>` on `log-r2`. Both under
`AMD_SERIALIZE_KERNEL=3`, so both are genuinely the kernel in flight. The
arguments are exact on the Q4_K leg too: `Arg1 obj` spans 2,949,120 B for
`n = 0x400`, `nsb = 0x14`, `w_block_bytes = 0x90` (144, `sizeof(BlockQ4_K)`),
and `w_row_bytes = 0xb40`. So the site is not the Q6_K arm; it is whichever
`KQuantGemmK` happens to be consuming a fresh allocation. W1 saw Q6_K on three
legs and generalised from three.

That also retires the last reason the private-memory A/B looked meaningful: the
format was never the variable, which is why both bodies failed identically.

The probe is `freshprobe.hip`, generated by the same `gen.sh`, so the kernel
bodies are still extracted verbatim. It alternates the two shapes the faulting
legs were running (Q4_K at `n=1024, nsb=20`; Q6_K at `n=5120, nsb=68`) for 250
iterations and varies ONE thing — when the weight's VA comes into existence:

| arm | weight allocation | what it isolates |
|---|---|---|
| `fresh` | `hipMallocManaged` + copy + consume in the same breath | the model's pattern |
| `pre` | every weight allocated and copied before the first dispatch | the staged pattern |
| `freshsync` | `fresh` + `hipStreamSynchronize` after the alloc | the ordering fix |
| `freshnosdma` | `fresh` under `HSA_ENABLE_SDMA=0` | the engine |

rc job `d535e65c-33a6-40a0-918e-0e21b82a6df2`, results in
`/mnt/nas_share/rc/rocm-strix-shape/evidence6/w8.log`. `fresh` faulting while
`pre` stays clean names lazy residency as the trigger and makes the repair a
staging change in this tree — `needs_weight_staging()` is the policy that
produces the interleave. All four clean rules the pattern out.

## The device attributes, finally read on the board

rc job `d535e65c-33a6-40a0-918e-0e21b82a6df2` PHASE -1, output
`/mnt/nas_share/rc/rocm-strix-shape/evidence6/attrs.out`. `Radeon 8060S
Graphics`, `gfx1151`:

| attribute | value |
|---|---|
| `Integrated` | 1 |
| `ManagedMemory` | 1 |
| `ConcurrentManagedAccess` | 1 |
| **`PageableMemoryAccess`** | **0** |
| `PageableMemoryAccessUsesHostPageTables` | 0 |
| `DirectManagedMemAccessFromHost` | 0 |

Two consequences, and the first retires a candidate that was otherwise the best
one this issue has had.

**`ResidentWeight`'s alias branch is DEAD on this board.**
`src/vllm/model_executor/models/qwen3_5.cpp:1139-1183` will hand a kernel
`w.bytes.data()` — a raw `PROT_READ MAP_PRIVATE` GGUF page-cache address that
nothing pins and nothing registers with HIP — whenever
`host_memory_is_device_addressable()` is true and
`MakeHostBytesDeviceAliasable` finds the borrow 256-aligned
(`qwen3_5_weights.cpp:256`). On an XNACK-less part a reclaimed page there is a
hard memory violation, and a fault on an unregistered VA is exactly the
"nowhere near any argument, not a valid user address" shape. But
`vt::rocm::HostMemoryIsDeviceAddressable` is
`caps.integrated && caps.pageable_memory_access`
(`src/vt/rocm/rocm_backend.hip:474-480`), and `pageable_memory_access` reads 0.
The branch cannot be entered here, `VT_QWEN35_ALIAS_HOST_WEIGHTS` would change
nothing, and the A/B that was staged for it correctly skipped itself. Ruled out
by an attribute rather than by legs, which is the cheap way round.

**And the hardware fact underneath the whole row is now measured rather than
inferred.** `PageableMemoryAccess = 0` and
`PageableMemoryAccessUsesHostPageTables = 0` mean this part is XNACK-less: the
GPU CANNOT take a recoverable page fault. Meanwhile `UseManagedAlloc` is on, so
every `Backend::Alloc` block is `hipMallocManaged` — migratable memory — on a
device that cannot fault and recover. Any access to a page not currently mapped
in the GPU's tables is a hard violation, reported at whatever address the walk
produced, against whatever kernel was resident. That is the class W8 measures,
and it is the reason `HSA_ENABLE_SDMA=0` is interesting: SDMA is what moves
managed pages on an HMM board.

## W8 RESULT: lazy residency is NOT the trigger either

rc job `d535e65c-33a6-40a0-918e-0e21b82a6df2`, log
`/mnt/nas_share/rc/rocm-strix-shape/evidence6/w8.log`. Four arms, eight rounds,
interleaved with the order rotating, 250 alloc-copy-consume iterations a leg:

| arm | weight allocation | legs | failures |
|---|---|---|---|
| `fresh` | allocate, copy and consume in the same breath | 8 | **0** |
| `pre` | everything allocated and copied before the first dispatch | 8 | **0** |
| `freshsync` | `fresh` + `hipStreamSynchronize` after the alloc | 8 | **0** |
| `freshnosdma` | `fresh` under `HSA_ENABLE_SDMA=0` | 8 | **1** (GPUHANG) |

Consuming a managed buffer that came into existence a moment earlier does not
reproduce the fault, and staging everything first does not prevent anything,
because there was nothing to prevent. Lazy residency is the pattern the model
runs, and it is not sufficient.

**One thing here IS new: that GPUHANG is the FIRST standalone fault this row has
seen.** Across W3 (30 legs), W6 (24) and W8 (32) — 86 standalone legs — exactly
one faulted, against a model rate that has run between 17% and 83%. So the
probes are not incapable of faulting; they reproduce the conditions weakly. Read
the arm it landed in with care: it landed in `freshnosdma`, the arm that should
be safest, at n=8. That reads as the box's floor, not as an arm effect, and it
is recorded rather than interpreted.

## What is left, after everything this row has eliminated

The eliminations are worth reading together, because they converge:

- the kernel, at its own launch geometry (9,000 launches, no fault);
- the Q6_K FORMAT, and with it the private-memory arm (the site is `Fmt == 0` on
  one faulting leg);
- every "we released the copy source" candidate, by construction, because
  `Backend::Copy` completes inside the call on this board;
- `AdoptDeviceBytesAsHost`, measured twice and a no-op for a GGUF borrow anyway;
- `ResidentWeight`'s mmap-alias branch, by an attribute the board reports;
- lazy residency itself, measured four ways.

What survives is the one thing every arm has in common and no probe has managed
to stress hard enough. **`gfx1151` cannot take a recoverable page fault
(`PageableMemoryAccess = 0`), and this tree puts about 25 GiB of migratable
`hipMallocManaged` memory on it, touched from both sides, while a queue is
live.** `HSA_ENABLE_SDMA=0` is the only arm that has moved the rate, and on an
HMM board SDMA is what moves managed pages. Every fault address is
page-aligned, four of five are outside any user address space, and none lies in
any allocation the process ever made — which is what a walk over a mapping that
is being changed underneath it produces, and is not what an out-of-bounds index
produces.

**The next A/B is one predicate wide and it is in this tree.**
`UseManagedAlloc` (`src/vt/rocm/rocm_backend.hip:142`) is what routes every
`Backend::Alloc` to `hipMallocManaged` instead of `hipMalloc`, and it is a pure
function of three attributes with no env knob. Put it behind one, default ON so
no shipped behaviour changes, and run it interleaved against the baseline the
way this row has run every other arm. If plain `hipMalloc` removes the fault,
the cause is managed memory on a part that cannot fault and recover, the repair
is in this tree — do not take the managed branch on a device reporting
`PageableMemoryAccess = 0` — and #2511 closes with a fix rather than with the
SDMA workaround. If it does not, the remaining surface is the driver, and the
row ends with the report its `## Stop conditions` already provides for.

That needs a rebuild of one translation unit, which `q6k.sh` already showed how
to do inside a lease, so it is a dispatch and not a research question.

## Next hypotheses, in the order they are worth testing

The kernel is exonerated at its own geometry, so the remaining surface is what
the step does around it. Ranked by what would discriminate most per lease-minute:

1. **Map the fault address against the FULL allocation table** (running). If
   `0x7383103b5000` lands inside a real managed allocation, this is a buffer
   losing its pages, not a wild index, and the next question is which buffer.
   If it lands in none, it is a wild address and the question is whose.
2. **An async copy outliving its host source.** `RocmBackend::Copy`
   (`src/vt/rocm/rocm_backend.hip`) is a bare `hipMemcpyAsync` with no
   synchronization, and the log shows the loader driving 73 MB copies from
   pageable host addresses. Every call site that passes a stack address, a local
   `std::vector::data()`, or a buffer it reuses on the next iteration is a
   candidate, because the copy engine is NOT what `AMD_SERIALIZE_KERNEL=3`
   serializes. This needs a call-site audit and then a red-before, not another
   leg.
3. **Bisect the step.** The probe runs one dispatch; the step runs hundreds. A
   probe that replays the step's dispatch SEQUENCE, or a run with parts of the
   model's forward disabled, separates "some earlier kernel" from "the host
   traffic between them".

## Evidence

Raw logs under `/mnt/nas_share/rc/rocm-strix-hang/evidence` (W1),
`.../evidence3` (W2's A/B), the earlier `#2497` legs under
`/mnt/nas_share/rc/rocm-strix-q4k/evidence`, and W3 under
`/mnt/nas_share/rc/rocm-strix-shape/`, which also holds the probe itself
(`probe.hip`), its generator (`gen.sh` + `prologue.inc` + `epilogue.inc`,
which regenerate it from a checkout so the extraction is reproducible), the
runner (`shape.sh`), the fault-address mapper (`mapfault.py`) and `fault.sh`.
