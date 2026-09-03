# Gemma-4 FP8-native expert arm: guard the upload on the ops it actually calls

| Field | Value |
|---|---|
| Row | `ENG-EXPERT-STREAM` |
| Issue | [#2623](https://github.com/mudler/vllm.cpp/issues/2623) |
| Supersedes | the `## Owed` row at [`expert-streaming.md`](expert-streaming.md) that tracked this as the dead `#1218` |
| State | ACTIVE |

## Now

The guard, its three predicates and its red-first reachability gate land together.
The `## Owed` row in `expert-streaming.md` that recorded the gap is closed by this
change and its dead premise is corrected in the same commit.

## Scope

In: an availability guard on `EnsureGemma4Fp8NativeOnDevice`
(`src/vllm/model_executor/models/gemma4_moe.cpp:612`); the two `vt::Has*`
predicates that guard needs and that the tree does not yet have; making the two
ops they describe dispatch on their own predicate; the false comment on
`DevExpertLru::FreeBytes` (`:443-449`); the `## Owed` row in
`expert-streaming.md` that this closes.

Out: writing a CUDA or CPU arm for any of the three ops (that is #1205 and is not
this row). Out: the BF16 sibling's guard, which already exists at `:572`. Out:
`vt::ExpertGeGLUFp8TopKM1`, which returns `false` off ROCm instead of throwing and
is already handled by its callers.

## The defect

`EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:549`) refuses before its upload
when the arm it promises does not exist:

```cpp
if (!vt::HasMatmulBTAlphaBeta(d.q)) return false;   // :572
```

Its FP8-native twin `EnsureGemma4Fp8NativeOnDevice` (`:612`) has no such term:

```cpp
bool EnsureGemma4Fp8NativeOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H) {
  if (!ExpertLru().Enabled()) return false;
```

and the native arm is the DEFAULT (`:1071-1077`, `VT_GEMMA4_FP8_NATIVE` unset
returns `true`). Returning `true` is a promise the caller may run the
device-resident arm. Every consumer of the slots it fills ends in an op with one
arm in the tree:

| consumer | reached from | ends in | refusal |
|---|---|---|---|
| `ExpertGeGLUFp8TopKFusedGelu` (`:134`) | `:1461` | `vt::MatmulBTFp8Channel` (`:165`, `:176`) | `fused_ops.cpp:177` |
| `ExpertGeGLUFp8Native` (`:96`) | `:1586` | `vt::DequantFp8ChannelBf16` (`:118`, `:120`) | `fused_ops.cpp:194` |
| `ExpertGeGLUFp8Native` (`:96`) | `:1586` | `vt::MatmulBTAlphaBeta` (`:129`) | `fused_ops.cpp:152` |

None of those calls is inside the upload's `try`/`catch (...)`, so the throw
leaves the decode step rather than degrading to the host fallback that already
sits in the `else` arms.

## It is LIVE, not latent, and that is the change since the record was written

`expert-streaming.md`'s `## Owed` row says the twin is "latent for the same reason
and for exactly as long: its `MakeRoom` also needs `Backend::DeviceMemoryInfo`".
That premise is dead. `CudaBackend::DeviceMemoryInfo` exists:

```cpp
// src/vt/cuda/cuda_backend.cu:93
bool DeviceMemoryInfo(size_t* free_bytes, size_t* total_bytes) const override {
```

so on a stock CUDA build every admission term now passes by default:
`VT_GEMMA4_FP8_NATIVE` unset is `true` (`:1071-1077`); `ExpertLru().Enabled()` is
`BudgetBytes() > 0` and `VT_GEMMA4_EXPERT_VRAM_MB` unset is 2048 MiB (`:417-439`);
`MakeRoom`'s `FreeBytes` (`:450`) now answers on CUDA. The upload admits, and the
first consumer throws.

`DevExpertLru::FreeBytes` still carries the false claim in a comment
(`:443-449`, "ROCm ONLY: `CudaBackend` does not override that seam"). It is
corrected here, in the change whose argument depends on it being false.

## Design: a predicate per op, not the sibling's predicate

`expert-streaming.md`'s `## Owed` row rejects the one-line repair by name, and it
is right:

> reusing `HasMatmulBTAlphaBeta` there would be a guard naming the wrong arm,
> which is the defect this row's own review just corrected in a refusal message.

The twin depends on three ops, so an honest guard needs a term per op. Two of the
three have no predicate, so this change adds them, in the shape
`HasMatmulBTAlphaBeta` already established (`fused_ops.cpp:102-109`):

- `vt::HasMatmulBTFp8Channel(const Queue&)`
- `vt::HasDequantFp8ChannelBf16(const Queue&)`

and makes `MatmulBTFp8Channel` and `DequantFp8ChannelBf16` **dispatch on their own
predicate** rather than on a second copy of its condition, which is what stops a
predicate drifting from the function it describes. `MatmulBTAlphaBeta` already
does this (`fused_ops.cpp:116`).

The guard is then:

```cpp
if (!vt::HasMatmulBTFp8Channel(d.q) || !vt::HasDequantFp8ChannelBf16(d.q) ||
    !vt::HasMatmulBTAlphaBeta(d.q))
  return false;
```

Three terms, one per op the promise commits the caller to, each keyed on whether
the arm EXISTS rather than on a device name or a build macro. Writing any one of
the three kernels for a new device changes nothing here; writing all three wakes
the arm with no edit at this call site.

## Reachability

The guard is entered through `vllm::RunGemma4Moe`, the layer entry point
`src/vllm/model_executor/models/gemma4.cpp` calls. The test does not construct
the `Dev`, the LRU or any `vt::` op by hand.

## Tests

`tests/vllm/models/test_gemma4_moe_fp8_native_arm_guard.cpp`, a twin of the
existing `test_gemma4_moe_device_arm_guard.cpp` with one difference:
`VT_GEMMA4_FP8_NATIVE` is forced to `1` rather than `0`, so the run takes the
default arm this row guards instead of the BF16 arm the existing file guards.

It must be its OWN binary. `RunGemma4Moe` freezes `fp8_native` in a
function-local `static const` on the first call in the process, so two cases with
different values of that knob cannot share an executable.

Preconditions asserted before the assertion, so an instrument that failed to arm
cannot report the guard proven by a run that never reached it:

1. all three predicates are false for this queue (true on every build, HIP
   included: the only arms are kROCM's);
2. the STOCK CPU backend does not answer `DeviceMemoryInfo` — this is why the
   hazard is latent on a CPU box and why the decoration is needed;
3. inside the decoration, the probe DOES answer and reports the budget, which is
   the post-`CudaBackend::DeviceMemoryInfo` state the guard exists for;
4. the baseline output is not all zeros, or the equality below is vacuous.

Assertion: `REQUIRE_NOTHROW` through the real layer, and the degraded answer is
byte-identical to the baseline. Refusing a device arm must not change a value.

## Gates

```sh
cmake --build build --target test_gemma4_moe_fp8_native_arm_guard test_gemma4_moe_device_arm_guard test_gemma4_rocm_fp8_seams
./build/tests/test_gemma4_moe_fp8_native_arm_guard
./build/tests/test_gemma4_moe_device_arm_guard
./build/tests/test_gemma4_rocm_fp8_seams
scripts/agent-preflight.sh
```

A doctest run reporting `assertions: 0` is a SKIP wearing a pass; read both
counts.

## Mutations the reviewer must run, rebuilding each time

1. Delete the three-term guard. Expect RED with a `ROCm-only in this build`
   throw out of the decode step.
2. Delete only the `HasMatmulBTFp8Channel` term. Expect RED — that is the op the
   `:1461` consumer reaches first.
3. Force each new predicate to `return true`. Expect RED.
4. Delete the production call site in `gemma4_moe.cpp`. A gate that stays green
   without it measures a class, not a capability.

A compiler catching a mutation is not a test detecting it. Void the parameter and
re-run.

## Risks

- **The three predicates are the same expression today.** They are separate
  symbols because they describe different ops, not because they differ now. That
  is the point: a CUDA `MatmulBTFp8Channel` alone must not wake the arm.
- **No Gemma-4 per-expert FP8 checkpoint is pinned in `docs/USAGE.md`.** So this
  fixes a path no pinned checkpoint exercises today. The `## Owed` row already
  records that; it is why the gate is a decorated-backend reachability test and
  not a checkpoint run.
- **Degradation is not free.** The host fallback rounds twice more per expert
  than the device arm. That is the same trade the BF16 sibling already makes, and
  it answers instead of throwing.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if the reviewer concludes the
guard belongs in `ExpertGeGLUFp8Native` instead of in the upload. The upload is
where the sibling puts it and where the `true` promise is made, but the argument
is a real one.

## Owed

- A CUDA arm for `MatmulBTFp8Channel`, `DequantFp8ChannelBf16` and
  `MatmulBTAlphaBeta`. Owner `ENG-EXPERT-STREAM`, tracked by #1205, unchanged by
  this row.
- A pinned per-expert FP8 Gemma-4 checkpoint, without which neither arm can be
  measured. Owner `ENG-EXPERT-STREAM`.
