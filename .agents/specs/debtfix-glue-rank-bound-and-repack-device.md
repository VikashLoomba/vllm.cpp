# DEBTFIX — the glue tensor writes past `kMaxRank`, and `quant_repack` cannot see the device

**Row:** `-` (two standing defects, each already filed; no roadmap row owns either)
**Issues:** [#2435](https://github.com/mudler/vllm.cpp/issues/2435),
[#2406](https://github.com/mudler/vllm.cpp/issues/2406)
**State:** `ACTIVE`
**Base:** `origin/main` at `63889449c`

## Scope

Two unrelated correctness defects that share one property: both are invisible to
every gate that currently runs, and both were found by reading rather than by a
red.

1. **#2435** — `vllm::dense_attn::MakeTensor`
   (`include/vllm/model_executor/models/dense_device_glue.h:47-60`) writes
   `t.shape[i]` and `t.stride[i]` for `i` up to `shape.size() - 1` with no bound,
   while `vt::Tensor` fixes `kMaxRank = 4`. Any rank-5 shape writes past both
   fixed arrays. Give the write site the bound its sibling
   `vt::Tensor::Contiguous` (`src/vt/tensor.cpp:19-20`) has carried all along,
   and repair every caller the bound then refuses.
2. **#2406** — `GgufLoadPolicy::FromEnv`
   (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:366`) resolves
   `quant_repack` from a pure host-ISA probe with no device term, so an aarch64
   i8mm host loading `--device cuda` repacks Q8_0 weights into the ARM
   `block_q8_0x4` interleave and stages them to a card whose kernels read plain
   `block_q8_0`. Give the line the resolved device its sibling `elem_kn_repack`
   already reads two lines below.

### Out of scope, recorded under `## Owed`

* Raising `vt::kMaxRank`. Four is a vt design constant; every op, every kernel
  and `sizeof(vt::Tensor)` depend on it.
* Dequantizing the `qwen4_exp` hyper-connection weights at load. See
  `## Why the device gate and not the dequant`.
* The aarch64 CUDA measurement #2406 asks for. This wave has no lease on a
  fleet aarch64 CUDA box and does not claim one.

## Defect 1 — the unbounded write

### What it does

`MakeTensor` sets `t.rank = shape.size()` and then walks `i` from `rank - 1`
down to `0`, writing `t.shape[i]`, writing `t.stride[i]`, and reading
`t.shape[i]` back into the running product. At rank 5 the `i == 4` iteration
writes eight bytes past the end of `shape` (into `stride[0]`) and eight bytes
past the end of `stride` (into the three storage-marker bools). The `i == 0`
iteration then overwrites `stride[0]` again, so the shape damage is
self-healing and only the marker write survives — `stride[4] = 1` sets
`Tensor::repacked = true` on a tensor that was never repacked.

That is why no value gate sees it. The tensor comes out arithmetically usable,
and the corrupted member is a storage marker that only `kMatmulBTQuant` reads.

### The population

Thirteen call sites, all in tests, all the same paged-KV allocation
`{2, num_blocks, block_size, num_kv_heads, head_size}`:

| file | lines |
|---|---|
| `tests/vllm/models/test_qwen4_exp_layer_loop.cpp` | 495, 685, 828, 1116, 1484, 1721, 2038, 2131, 2295, 2547 |
| `tests/vllm/models/test_qwen4_exp_inject_residency.cpp` | 216, 329 |
| `tests/vllm/models/test_qwen4_exp_matmul_bt_dtype.cpp` | 327 |

Every one of them exists to size and fill one buffer. The tensor view it
produces is never consumed as a tensor: the fixture reads `kv_b.t().data` and
carries the five dimensions to `PagedKvCache` as separate scalars
(`include/vllm/model_executor/models/qwen3_5.h:78-98`). So `{2 * num_blocks,
block_size, num_kv_heads, head_size}` is the same bytes, the same element count
and the same truth, at a rank the type can hold.

### Design

`MakeTensor` gains

```cpp
VT_CHECK(shape.size() <= static_cast<size_t>(vt::kMaxRank), "...");
```

before it writes anything, mirroring `Tensor::Contiguous`. The thirteen call
sites fold the leading `2` into the block dimension.

**The bound belongs here and not in `vt/tensor.h`.** `vt::Tensor` is a plain
aggregate with public arrays; a bound is enforceable only where something
writes. `Tensor::Contiguous` is the one other writer and it already has one.
This closes the parallel path rather than adding a third.

## Defect 2 — the device-blind repack

### Why the device gate and not the dequant

[#2406's own comment](https://github.com/mudler/vllm.cpp/issues/2406) records a
second option: dequantize `hc_*_down` / `hc_*_up` to bf16 at load, which mirrors
vLLM and removes all 194 repack-hazard tensors of the released artifact at once.
This spec rejects it, for reasons and not for size.

1. **The defect is in the policy line, not in one architecture's tensors.** The
   dequant removes `qwen4_exp`'s population and leaves the same device-blind
   line deciding for every Q8_0 weight of `qwen3_5`, `glm5_next`,
   `glm_moe_dsa`, `deepseek_v4` and `laguna` on the same host. The device gate
   fixes the class.
2. **The device gate provably cannot move the CPU control; the dequant can.**
   On `dev == kCPU` the gated expression is character-for-character the old one,
   so the released artifact's `--device cpu` decode is unchanged by
   construction. Dequantizing 194 weights replaces `kMatmulBTQuant` with a bf16
   GEMM on every hyper-connection projection — different arithmetic, and the
   eight-token control `11751 13 15767 411 2029 11 1092 369` would have to be
   re-earned on a box this wave has no lease for.
3. **The upstream evidence is about intent, and it is off-pin.** vLLM's
   hyper-connection linears pass `quant_config=None` explicitly
   (`vllm/model_executor/models/nvidia/hyperconnection.py:102`, `:113`, `:122`)
   with `params_dtype=torch.bfloat16` hardcoded, and the decode GEMM demands
   packed row-major bf16 on both operands (`nvidia/low_latency_gemm.py:95-96`).
   That is real, and it is a **forward reference**: the revision that carries it
   is 1,566 commits past this tree's parity pin and unreachable from it. vLLM
   loads safetensors and never meets a Q8_0 hyper-connection weight, so it
   states no position on the GGUF arm. The oracle that does is
   `llama-cpp-qwen4exp`, which keeps them typed and declares all six
   `GGML_OP_MUL_MAT`.
4. **Correctness first.** Turning repack off on a non-CPU device costs the i8mm
   interleave on any Q8_0 weight that reaches the documented CUDA-to-CPU drain
   (`src/vt/cuda/cuda_quant_dot.cu`). That is a speed question on an already-
   degraded path, and AGENTS.md `## Gates` settles the order.

The dequant stays a live option for the `qwen4_exp` row as a residency and
parity question. It is recorded under `## Owed`, not refused.

### Design

```cpp
p.quant_repack = QuantRepackForDevice(p.keep_quant, p.cpu_ref,
                                      vt::cpu::QuantRepackActive(), dev);
```

with

```cpp
bool QuantRepackForDevice(bool keep_quant, bool cpu_ref,
                          bool host_repack_active, vt::DeviceType dev);
```

declared beside the policy and defined in its translation unit.

**Why a named predicate and not `&& dev == kCPU` inline.** `QuantRepackActive()`
is a hard compile-time `false` on every non-aarch64 target
(`src/vt/cpu/cpu_quant_repack_arm.cpp:275`), so an inline gate asserted through
`FromEnv` is **vacuous on x86 CI**: `quant_repack` is already false there for the
wrong reason, and the assertion would be a mute switch that passes whether the
device term exists or not. Taking `host_repack_active` as a parameter is the same
move `RouteGgufTensor` already made for the device — the ROCm routing case
(`tests/vllm/test_gguf_keep_quant.cpp:238`) says so in its own comment: "the
whole point of removing the probe is that a device's routing is checkable from a
host that is not that device."

## Tests

**#2435.** The red is a sanitizer run, not an assertion. `test_qwen4_exp_layer_loop`
reports `SUCCESS!` and exits 1. The gate is therefore the process exit code of
the `address,undefined` build, plus a count: the number of
`dense_device_glue.h:5[678]: runtime error` lines, which must go 3 → 0. A doctest
case is added beside it (`tests/vllm/models/test_dense_device_glue_rank.cpp`)
asserting that a rank-5 shape now `CHECK_THROWS` and that a rank-4 one is
unchanged, so the bound has an assertion of its own on every lane including the
ones with no sanitizer.

**#2406.** A doctest case over `QuantRepackForDevice`'s full truth table, run on
every host. The discriminating row is
`QuantRepackForDevice(true, false, /*host_repack_active=*/true, kCUDA) == false`
against `... kCPU) == true`. `tests/vllm/test_gguf_keep_quant.cpp:1361`'s
`expect.quant_repack` mirror is updated to the same predicate so the env-load
comparison stays apples-to-apples on an i8mm host.

## Gates

```sh
cmake -S . -B build-san -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF \
      -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-san -j 2
UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 \
  VT_POOL_BYPASS=1 ctest --test-dir build-san --output-on-failure
```

plus the ordinary CPU suite, and `scripts/agent-preflight.sh --staged`.

## Risks

* The rank bound turns a silent corruption into a throw. If any caller outside
  the thirteen builds a rank-5 shape from a runtime vector, the static scan
  cannot see it and the suite run is what finds it. That is the point of running
  the whole suite rather than the one test.
* Gating `quant_repack` on the device removes the i8mm interleave from any
  aarch64 `--device cuda` load. No arm of this fleet gates that path today, so
  the loss is unmeasured; it is named here rather than assumed to be zero.

## Stop conditions

Stop and report if the rank bound reds a test whose rank-5 shape is genuinely
consumed as a rank-5 tensor. That would make this a `kMaxRank` question and not
a bounds question, and it is a different row.

## Owed

* Raising `vt::kMaxRank`, if a production path ever needs rank 5.
* Dequantizing the `qwen4_exp` hyper-connection weights at load
  (`MODEL-MM-QWEN4-EXP`), with the residency and token evidence that needs.
* The aarch64 CUDA measurement of the repack trade that #2406 asks for.

## Now

`ACTIVE`.
