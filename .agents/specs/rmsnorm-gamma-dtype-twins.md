# The two RmsNorm twins that still weld the gamma dtype, and the kF16 divergence

**Row:** `MODEL-MM-QWEN4-EXP` (wave RMSTWINS)
**Issues:** [#2492](https://github.com/mudler/vllm.cpp/issues/2492),
[#2503](https://github.com/mudler/vllm.cpp/issues/2503)
**Parent spec:** [`qwen4-exp-cuda-rmsnorm-weight-dtype.md`](qwen4-exp-cuda-rmsnorm-weight-dtype.md),
which lists both under `## Owed`
**Base:** `origin/main` at `f6105b2e5`

## Scope

Close the last two items of correctness debt behind the `qwen4_exp` CUDA row.

1. **#2492.** `RmsNormRowKernel` in `src/vt/rocm/rocm_rmsnorm.hip` and
   `RmsNormQuantFp8RowKernel` in `src/vt/cuda/cuda_ops.cu` both still take the
   gamma as `const Tin*` -- the ACTIVATION's element type -- so both dispatchers
   have to refuse `w.dtype != x.dtype`. Give each its own `Tw`, and NARROW the
   refusal instead of deleting it, exactly as
   [#2493](https://github.com/mudler/vllm.cpp/pull/2493) did for the CUDA
   `RmsNorm`.
2. **#2503 half 1.** `vt::RmsNorm` admits `IsFloat(weight.dtype)`
   (`src/vt/ops.cpp:1030`), and `IsFloat` is `{kF32, kF16, kBF16}` (`:24`). The
   CPU kernel widens a kF16 gamma like any other (`cpu_ops.cpp:554-557`); the
   CUDA dispatcher refused it. **Support it** rather than record it, and say why.
3. **#2503 half 2.** Settle the claimed blast radius of the bf16-activation /
   f32-gamma pairing #2477 newly admitted, by measurement rather than assumption.

Out of scope, each recorded under `## Owed`: the kF16 ACTIVATION divergence, the
ROCm `RmsNormPlusAdd`/`DualRmsNormPlusRes` welds, and #2488/#2476/#2496.

## The oracle

vLLM is this row's primary oracle and it never couples the two dtypes.
`Qwen4ExpRMSNorm.forward` upcasts the gamma on its own:

```python
return (normalized * (1.0 + self.weight.float())).to(input_dtype)
```

`vllm/models/qwen4_exp/nvidia/ple_layer.py:80`. `.float()` accepts ANY float
gamma dtype, `float16` included, which is the whole basis for supporting kF16
below rather than recording its refusal.

**This citation is OFF-PIN and is a forward reference.** It is read at vLLM
`origin/main` `cdefd9d499`. This project's pin is `5559679229`, which has no
`vllm/models/qwen4_exp/` at all -- vLLM landed the architecture after the pin.
The path is `vllm/models/`, NOT `vllm/model_executor/models/`.
[#2502](https://github.com/mudler/vllm.cpp/issues/2502) is reconciling the pin.
Nothing here is gateable against `5559679229`.

## Design

### #2492, both twins

Same shape both times, and it is the shape `cuda_ops.cu` already carries:

* the kernel takes `template <typename Tin, typename Tw, ...>` and `const Tw* w`;
* the launcher forwards `Tw`;
* a new `DispatchRmsNormWeight` / `DispatchRmsNormQuantFp8Weight` switches on
  `w.dtype` and instantiates the right `Tw`;
* the old `VT_CHECK(w.dtype == x.dtype, ...)` becomes that switch's `default:`
  arm, which names the dtype it got.

The refusal is NARROWED, not deleted. Widening the check on its own would have
read a bf16 gamma through a `const float*` and run off the end of it. `Load()`
is overloaded for every gamma element type, so each kernel body and its f32
arithmetic are unchanged, and every pairing that already worked stays
bit-identical.

### #2503 half 1: support kF16, do not record it

Four reasons, in order of weight:

1. **The oracle upcasts any float gamma.** `self.weight.float()` has no dtype
   arm. Mirroring vLLM means accepting what vLLM accepts.
2. **The CPU sibling already accepts it.** `cuda_qwen4_exp.cu:60-62` states the
   rule: "a device arm that refused a dtype its CPU sibling accepts would be a
   divergence to record". Removing the divergence is strictly better than
   recording it, and it removes it for ROCm at the same time.
3. **This tree already promises kF16 on CUDA.** `CudaPlatform::supported_dtypes`
   returns `{kBF16, kF16, kF32}` (`src/vllm/platforms/cuda.cpp:112`), mirroring
   upstream `interface.py:181-187`. `CpuPlatform` and `RocmPlatform` say the
   same.
4. **The change cannot regress a working pairing.** It adds one `case` to a
   `switch` and one `__device__ Load` overload. No existing arm's code changes.

The alternative the issue offers -- narrowing `vt::RmsNorm`'s admission so the
seam stops promising kF16 -- was rejected because it would make the seam refuse
what both the CPU kernel and the oracle accept, to make a device arm's gap look
intentional.

**What this does NOT do.** kF16 is admitted as a GAMMA only. `x` and `out` stay
f32/bf16 on the CUDA and ROCm arms, while the CPU kernel widens a kF16
activation too and `IsFloat(x.dtype)` admits it at the seam. That is the SAME
divergence class, one operand over, and it is recorded under `## Owed` rather
than fixed: closing it means a f16 `Tin` instantiation through both vectorised
decode-fast paths, the residual round-trip and `LoadVec`, which is a different
change with a different risk profile.

### #2503 half 2: the blast radius

See `## Evidence`. The short version is that the claim in the issue does not
survive contact with the loaders.

## Risks

* **The ROCm edit cannot be compiled anywhere.** No ROCm device is in the `rc`
  fleet, no ROCm toolchain is on this host, and `.github/workflows/` has no lane
  that invokes `hipcc`. The file's own header already says **UNBUILT**. This
  change lands ungated and uncompiled, and that is stated in the commit body.
  It is a mechanical mirror of an edit that CI does compile on the CUDA side,
  which is the most that can be said for it.
* **The CUDA edits are compiled by `cuda-fat-build` and executed by nothing.**
  There is no CI lane that runs GPU tests, so every CUDA case in this suite is
  `[SKIP]` in practice.
* **`__half` is a new element type in both files.** `cuda_fp16.h` was not
  previously included by `cuda_ops.cu` and `hip_fp16.h` was not previously
  included by `rocm_rmsnorm.hip`. A missing include is a compile error, not a
  silent wrong answer.
* Instantiation count for `RmsNormRowKernel` goes from 16 to 24 and for
  `RmsNormQuantFp8RowKernel` from 4 to 12. Compile time and archive size grow;
  no runtime path changes.

## Tests

`tests/vt/test_ops_rmsnorm_weight_dtype.cpp` (extended) and
`tests/vt/test_ops_rmsnorm_quant_fp8_weight_dtype.cpp` (new).

Every case names the production site it models, and every CPU case compares
against an independently recomputed reference rather than against the op itself.

| Case | Pairing | Models |
|---|---|---|
| 1 | f32 act / bf16 gamma | `qwen4_exp_qsa_block.cpp:705` -- the site that refused in #2477 |
| 2 | bf16 act / f32 gamma | no production site; see `## Evidence`. Kept as the contract the seam promises |
| 3 | bf16 act / f16 gamma | no production loader; the `IsFloat` seam admission at `ops.cpp:1030` and the CPU arm at `cpu_ops.cpp:554-557` |
| 4 | fp8: bf16 act / f32 gamma | `vt::RmsNormQuantFp8`'s seam admission at `ops.cpp:748` |
| 5 | fp8: bf16 act / f16 gamma | the same |

Case 3 CHANGES POLARITY in this wave: #2493 pinned it as a CUDA refusal with
`CHECK_THROWS_WITH(..., doctest::Contains("unsupported weight dtype"))`; it now
pins CPU/CUDA agreement. That is the point of the change and the test says so.

## Gates

```sh
ctest --test-dir build -R 'test_ops_rmsnorm' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

Filled in by the implementation commits.

## Stop conditions

A ROCm compile failure found later by someone with an AMD box is a reportable
result of this wave, not a defect discovered outside it. The edit is recorded
here as uncompiled precisely so that it is attributable.

## Owed

- [#2542](https://github.com/mudler/vllm.cpp/issues/2542) --- the kF16
  ACTIVATION divergence. `IsFloat(x.dtype)` admits kF16 at `ops.cpp:1030`, the
  CPU kernel widens it (`cpu_ops.cpp:561-562`, `:576`), and both
  `RmsNormKernelCuda` and `RmsNormKernelRocm` refuse it by name. Same class as
  #2503 half 1, one operand over, and a much larger change: `Tin` also drives
  `ResRound`, `LoadVec`, both decode-fast guards and the `out.dtype` switch.
- [#2543](https://github.com/mudler/vllm.cpp/issues/2543) ---
  `RmsNormPlusAddRocm` (`rocm_rmsnorm.hip`) and `DualRmsNormPlusResRocm` weld
  every operand's dtype to one `T`, while `vt::RmsNormPlusAdd`'s non-ROCm
  composed reference (`src/vt/fused_ops.cpp:19-31`) is `RmsNorm` + `Add` and
  therefore accepts a mixed gamma. A third instance of the weld, in the same
  uncompilable file. Not fixed here: the two twins #2492 names are the scope, and
  every additional blind ROCm line is unverifiable surface.
- #2488, #2476, #2496 stay with the parent spec.
- A CI lane that executes GPU tests, and a ROCm lane that merely COMPILES
  `src/vt/rocm/`. Until the second exists, this file is edited by reading.

## Now

`ACTIVE`.
