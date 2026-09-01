# GDN `in_proj_qkvz` operand lifetime (issue #2476)

Row: `MODEL-MM-QWEN4-EXP`
Issue: https://github.com/mudler/vllm.cpp/issues/2476

## Now

`ACTIVE` — the fix for the illegal memory access that blocks the production
CUDA arm of `qwen4_exp`.

## Scope

One defect: the buffer the GDN `in_proj_qkv` / `in_proj_z` GEMM is pointed at
is owned by a per-step temporary and is freed while the GEMM is still queued.

Out of scope: #2496 (the garbage-token sequence), the QSA block, the packed
GDN decode ceiling, and the per-step MoE adapter rebuild that
`qwen4_exp_forward.cpp` already records as owed.

## The measurement this starts from

`compute-sanitizer memcheck` on `thor:gpu0` (sm_110, CUDA 13.0.88), released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, binary
`70e522df28d7748d8925ce9fc54dffd4909b3e8fd53d72fafd87bb13bed62056` from
`e934fb0020ba099125994e9a44e93e8c89d87977`. Evidence on the share at
`/mnt/nas_share/rc/q4exp-sanitize/out-san/`.

Facts that constrain the diagnosis, read off `server.log`:

- The prefill (`prompt_tokens=5`) and **five** decode steps complete. The first
  sanitizer report is at `server.log:58`, inside the sixth decode step.
- The faulting kernel is `nvjet_sm110_tst_512x8_64x3_2x1_v_bz_TNT`, reached
  through `cublasLtMatmul` <- `MatmulBTKernelCuda` <- `MatmulBf16D` <-
  `ProjectGdnQkvz` <- `GdnBlockPaged` <- `Qwen4ExpTextModelForward` <-
  `ModelRegistry::Forward`.
- The reports are **block 17 threads 128..159, then block 18 threads 128..135**
  (`--print-limit 40` truncates there). Blocks 0..16 do not fault. At the
  released geometry `conv_dim = 2*16*128 + 48*128 = 10240`, so the col-major
  `M` axis of this GEMM is 10240 and a 512-wide tile ladder has 20 blocks: the
  first ~17 tiles of the operand are addressable and the tail is not.
- `[vt load] w0f-alias per-call totals` grows `rehomed` by about 1 GiB on
  **every** step (3.381 -> 3.673 -> ... -> 6.079 -> 6.406 GiB). A weight that
  has been re-homed once is 256-byte aligned and takes the
  `kAliasedInPlace` branch forever after, so a `rehomed` counter that keeps
  climbing proves that **fresh owned host buffers appear each step**.

## The defect

`qwen4_exp_forward.h` states the invariant this code breaks:

> nothing is transposed, reordered or reallocated here — the returned
> `OwnedTensor`s are COPIES OF THE HANDLES and share the loader's bytes.

`Qwen4ExpGdnBlockWeights` implements that with plain assignment
(`w.in_proj_qkv = g.in_proj_qkv;`). `OwnedTensor`'s implicit copy constructor
deep-copies `OwnedBytes`, which for the GGUF arm is an **owned**
`std::vector<uint8_t>` (`reorder` is on at the released 16-vs-48 head ratio, so
`LoadGdn` expands every GDN projection to bf16 through `Bf16From`). So the
adapter reallocates ~115 MiB of GDN weights per linear layer per step, and:

1. the copy's `d_dev` / alias memo is written to a temporary and thrown away,
   so residency is re-established from scratch every step; and
2. `ResidentWeight`'s host-alias arm — the arm this box takes, because
   `StagingFitsModel` is false for a 65 GiB checkpoint on a 122 GiB box —
   returns `MakeTensor(w.bytes.data(), ...)`, i.e. a **host** pointer into the
   temporary's buffer, and hands it to `cublasLtMatmul`.

`const GdnLayerWeights gw` is scoped to the `if (linear)` arm of the layer
loop. It is destroyed at the end of that arm, which runs `::operator delete` on
the aligned block (or `~vector` on the copy). Every GDN kernel of that layer is
still only *queued*. `free()` of a 52 MiB block `munmap`s it, so the GEMM then
reads unmapped host virtual addresses.

This is an **extent-correct, lifetime-wrong** operand: `M`, `N`, `K` and every
declared byte length agree with each other and with the allocation at launch
time. What disagrees is *when* the allocation stops existing. It is a fixed
index, not a race: `CUDA_LAUNCH_BLOCKING=1` completes the kernel inside the
launch, before the scope exits, which is exactly why it suppresses the fault.
The partial validity (blocks 0..16 fine, 17+ faulting) is the freed range being
partly re-mapped by the next layer's copies before the kernel runs.

## Design

Two edits, one invariant: *the operand a queued kernel reads is owned by the
model, never by a per-step temporary.*

1. **Share the bytes.** `Qwen4ExpGdnBlockWeights` builds each field as a
   zero-copy view over the loader's buffer (`OwnedBytes::KeepAlive()` +
   `OwnedBytes::Borrow`), which is what the header already claims and what
   `Qwen4ExpMoeBlockWeights` already does for the shared expert's projections.
   The helper is lifted to `qwen3_5_weights.{h,cpp}` as
   `BorrowWholeOwnedTensor` so there is one implementation, and
   `qwen4_exp_moe.cpp`'s `BorrowWhole` delegates to it.
2. **Build it once.** The adapter is cached on `Qwen4ExpLayerWeights`, so the
   residency memo lives as long as the model and no temporary ever owns a
   device allocation either.

Edit 1 alone fixes the reported fault on the aligned arm. Edit 2 is what makes
the fix independent of `malloc` alignment and of `cudaFree`'s implicit
synchronisation, and it removes the per-step re-upload the same lines cause.

## Risks

- Sharing turns the loader's owned buffer into a shared read-only one
  (`OwnedBytes::Share()`), so nothing may write through it afterwards. Nothing
  does: the GDN weights are read-only from the end of the load.
- Caching pins the adapter for the model's life. It holds no bytes of its own,
  so the residency cost is zero.

## Tests

- `tests/vllm/models/test_qwen4_exp_layer_loop.cpp`: the adapter's every field
  points at the loader's own buffer (pointer identity), and after a forward
  through `Qwen4ExpTextModelForward` the layer's GDN buffers are shared rather
  than duplicated. Red before, green after.

## Gates

```sh
ctest --test-dir build -R qwen4_exp --output-on-failure
```

## Stop conditions

A GPU rerun on `thor:gpu0` is required to claim the production arm completes a
forward. Without it this row claims the defect and the fix, not the arm.

## Owed

- The GPU rerun of `compute-sanitizer` on the fixed binary, and the token
  sequence with no `CUDA_LAUNCH_BLOCKING`.
- Whether #2496's garbage tokens share this root cause. The mechanism predicts
  they can (a freed range re-mapped and refilled before the GEMM runs returns
  another layer's bytes rather than faulting), but that is a hypothesis until
  the fixed binary is measured.
