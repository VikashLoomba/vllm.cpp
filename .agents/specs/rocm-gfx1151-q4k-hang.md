# Spec — the gfx1151 plain Q4_K decode fault

Row `BACKEND-ROCM`. Issue
[#2511](https://github.com/mudler/vllm.cpp/issues/2511).
Sibling records: [#2377](https://github.com/mudler/vllm.cpp/issues/2377) (the
DFlash2 arm, same signature, different candidate) and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the throughput number
this defect makes provisional).

## Now

`ACTIVE` — diagnosis. No fix is proposed in this document yet; §"Findings"
records what is measured and §"Refuted" records what is ruled out.

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

## Stop conditions

- A cause that is in the AMD driver or firmware rather than in this tree ends the
  row with that finding recorded and the issue updated; it is not fixed here.
- No fix lands on a single passing leg. One green leg proves nothing at this
  failure rate.

## Evidence

Raw logs under `/mnt/nas_share/rc/rocm-strix-hang/evidence`, and the earlier
`#2497` legs under `/mnt/nas_share/rc/rocm-strix-q4k/evidence`.
