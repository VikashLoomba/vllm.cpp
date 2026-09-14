# gfx1100 quantized WMMA admission receipts

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3`.
Issue: `ISSUE-LOCAL-01M2F0PQWGSCXG0N4951NF9DPZ`.
Spec: [Reuse quantized WMMA prefill](../../../.agents/specs/rocm-rdna3-quant-wmma.md).
Implementation: `c3fe98ba6c55ce71e75746e1b944a27640464e0f`.
Spec ancestor: `a5b5c92f2604386fcf162d5e12a42988f412db3b`.
Measured on 13 September 2026 on the local RX 7900 XTX, physical `gfx1100`.
Every GPU invocation held `/home/vikash/gpu.lock` and selected device 0 with
`HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0`.

## Results

| Gate | Result | Evidence |
|---|---|---|
| G1 architecture policy | Satisfied. The CPU suite passes 16 cases and 109 assertions. The initial gfx12-only stub fails four assertions. | [Red](arch-red.log), [green](arch-green.log) |
| G1 physical dispatch | Satisfied. Before admission, all four cases fail ten dispatch assertions. After admission, all 46 assertions pass. | [Red](hardware-red.log), [green](hardware-green.log) |
| G2 unchanged tile bodies | Satisfied. The executing Q4_K and Q6_K bodies are byte-identical to the committed spec ancestor. | [Body hashes](tile-bodies.json) |
| G2 signed-int8 instructions | Satisfied. The emitted gfx1100 object contains eight `v_wmma_i32_16x16x16_iu8` instructions across Q4_K and Q6_K, each with F32 and BF16 outputs. | [ISA excerpt](gfx1100-wmma-isa.txt), [artifact hashes](artifacts.json) |
| G2 original upstream fixture gate | Satisfied. All 240 native tensor cases pass against the original dense reference and the primary plugin output. | [Summary](original-mmq-summary.json), [comparison](original-mmq-comparison.json), [log](native-comparison.log) |
| G2 scalar control | Satisfied. A separate process with `VT_ROCM_QUANT_WMMA=0` passes all four physical cases and 46 assertions. | [Scalar log](hardware-scalar.log) |
| G3 public production entry | Satisfied. Prompts of 16 and 37 tokens each enter two Q4_K calls and one Q6_K call. All 1024 logits and eight completion tokens match the scalar child process. | [Red](public-red.log), [green](public-green.log) |
| G5 broad HIP gate | Satisfied with the five baseline resource skips listed below. All 29 registered tests have zero failures after a clean rebuild. | [HIP gate](hip-gate-green.log) |
| G5 repository preflight | Executed with exit 0 and 12 unchanged skips. The skips remain unavailable checks, rather than passes. | [First](preflight-first.log), [staged](preflight-staged.log) |
| G5 implementer mutations | Satisfied. Four CPU policy mutations and four public production mutations fail. The immutable original files retain their hashes. | [CPU mutations](cpu-mutations.json), [production mutations](production-mutations.json) |
| G4 full-model correctness and performance | Pending the coordinating operator's model and oracle runs. No throughput, latency, or memory claim is made here. | Owning spec |
| G5 fresh review and operator verification | Pending the coordinating operator's separate receipts. Implementer results do not discharge these obligations. | Owning spec |

The existing physical fixtures preserve the partial four-wave block at
`M=32, N=48, K=512`, both output dtypes, joint tails, and separate bottom and
right tails. Their normalized mean squared error limit remains `5e-4`.
The hardware test selects the physical device independently of production
admission. Restoring gfx12-only admission cannot skip the gfx1100 test.

The first broad invocation omitted `test_qwen35_moe_kq_device`, the executable
behind the `test_rocm_qwen35_moe_kq_device` CTest alias. That invocation failed
with one test not run. The [failed receipt](hip-gate-missing-alias.log) remains.
Building the actual executable and rerunning the complete gate produced the
passing receipt. No product change was needed.

The five baseline skips are `test_rocm_moe_bf16`, `test_rocm_moe_upstream`,
and `test_rocm_f16_model`, which require model fixtures, plus
`test_rocm_attn_gate_split_device_binding` and `test_rocm_f16_device`, which
require two visible devices. This gate deliberately exposed only device 0.

## Build and execution recipe

Worktree: `/home/vikash/vllm.cpp-rdna3-wmma-impl`.
Build directory: `build-rdna3-wmma`. The default build type preserves test
assertions. [Compiler details](compiler.txt) identify Clang 23 and its exact
revision. The artifact manifest seals the rocWMMA 2.2.1 headers and binaries.

```sh
cmake -S . -B build-rdna3-wmma -G Ninja \
  -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DROCM_PATH=/opt/rocm \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF \
  -DVLLM_CPP_MLX=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build build-rdna3-wmma --target test_rocm_arch \
  test_backend_cross_device test_capi_rocm_quant_wmma rocm_quant_wmma_capture -j4
build-rdna3-wmma/tests/test_rocm_arch
flock /home/vikash/gpu.lock env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  timeout 120s build-rdna3-wmma/tests/test_backend_cross_device \
  '--test-case=*WMMA tile arm*,*M and N are not multiples*,*only one of M/N*'
flock /home/vikash/gpu.lock env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  VT_ROCM_QUANT_WMMA=0 timeout 120s build-rdna3-wmma/tests/test_backend_cross_device \
  '--test-case=*WMMA tile arm*,*M and N are not multiples*,*only one of M/N*'
flock /home/vikash/gpu.lock env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  timeout 120s build-rdna3-wmma/tests/test_capi_rocm_quant_wmma
```

After the header change, the implementer clean-rebuilt every ROCm test target
with `--clean-first` and `-j4`. The [target list](build-targets.json) includes
the CTest alias's actual executable.

```sh
python3 - <<'PYBUILD'
import json
import subprocess
from pathlib import Path
targets = json.loads(Path("docs/bench-evidence/rocm-rdna3-quant-wmma/build-targets.json").read_text())
subprocess.run(["cmake", "--build", "build-rdna3-wmma", "--clean-first",
                "--target", *targets, "-j4"], check=True)
PYBUILD
flock /home/vikash/gpu.lock env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  timeout 180s ctest --test-dir build-rdna3-wmma \
  -R 'rocm|cross_device' --output-on-failure
```

## Primary oracle and memory format

The developer selected task snapshots vLLM
`39545e475d3627287ff69c25465dc0bd405f67e1` and GGUF plugin
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
The operator built and ran their isolated runtime. The manifest reports
PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`, and physical `gfx1100`.
These are the primary runtime's dependency versions. The native build uses
the separate local HIP 7.15 and rocWMMA 2.2.1 toolchain.

The port preserves plugin `tests/test_kernels.py::test_mmq`, lines 145 to 196,
and `tests/utils.py::seed_everything`. Every original Q4_K and Q6_K tensor
runs at `M=7,83,128,2048`, `K=256,1024`, with F16, BF16, and F32 input,
and seed zero. The four source GGUF files retain the hashes in the
[fixture summary](original-mmq-summary.json).

| Output dtype | Original absolute tolerance | Original relative tolerance |
|---|---|---|
| F16 | 1 | 0.1 |
| BF16 | 1.5 | 10000 |
| F32 | 1.2 | 20 |

All 240 cases select the compiled plugin extension. The executing source chain
is `vllm_gguf_plugin/ops.py::ggml_mul_mat_a8`, line 200, to
`vllm_gguf_plugin/csrc/gguf/gguf_kernel.hip::ggml_mul_mat_a8`, line 221.
That function creates output with the input dtype and quantizes activations
through `quantize_row_q8_1_cuda`. Its Q4_K and Q6_K branches call
`ggml_mul_mat_q4_K_q8_1_cuda` and `ggml_mul_mat_q6_K_q8_1_cuda` in
`vllm_gguf_plugin/csrc/gguf/mmq_hip.cuh`, lines 490 and 591.
The corresponding device entry points are `mul_mat_q4_K` and `mul_mat_q6_K`.

The native path retains its existing Q8_K activation format and F32 or BF16
output contract. This row changes architecture admission only. The test adapter
narrows native F32 output to F16 for the original upstream F16 comparison.
BF16 emits directly from the native kernel. No model buffer dtype changes.
The synthetic public model uses the normal BF16 model path.

The adapter commands use the runtime and input locations supplied by this task.
The runtime wrapper runs inside the caller's GPU mutex.

```sh
flock /home/vikash/gpu.lock timeout 300s \
  /home/vikash/oracle/rdna3-wmma-latest/python-gpu-under-lock.sh \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-primary.py capture \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-original-fixture-manifest.json \
  /home/vikash/models/test-gguf-sample \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-primary
flock /home/vikash/gpu.lock env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  timeout 300s build-rdna3-wmma/tests/rocm_quant_wmma_capture \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-primary/manifest.json \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-native
flock /home/vikash/gpu.lock timeout 300s \
  /home/vikash/oracle/rdna3-wmma-latest/python-gpu-under-lock.sh \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-primary.py compare \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-primary/manifest.json \
  /home/vikash/oracle/rdna3-wmma-latest/mmq-native
```

The copied primary script equals
`tools/rocm_quant_wmma/primary.py` at the implementation commit.
The original fixture manifest comes from the existing
`tools/rocm_quant_gather/primary.py::export_upstream` corpus exporter.
The raw matrices remain in the recorded oracle directory, with manifest and
output hashes sealed in this row's receipts.

## Mutation method

[CPU recipe](cpu-mutation-recipe.py) builds copied policy headers against the
unchanged architecture test. Each mutation fails that suite.
[Production recipe](production-mutation-recipe.py) compiles a copied grouped
GEMM translation unit, substitutes its object into a copied static archive,
and links the unchanged public test against that archive.
The originals remain byte-identical throughout every mutation.

| Mutation | Observed failure |
|---|---|
| Reject gfx1100 in the CPU predicate | gfx1100 admission assertions fail |
| Admit unmeasured gfx1101 | exclusion assertion fails |
| Accept malformed feature suffixes | malformed-name assertions fail |
| Widen the attention predicate | attention exclusion assertions fail |
| Restore gfx12-only runtime admission | public Q4_K dispatch assertion fails |
| Delete only the Q4_K launch, retaining its counter | scalar completion token comparison fails |
| Delete only the Q6_K launch, retaining its counter | completion fails because the unwritten output produces an invalid token |
| Remove the scalar-control environment override | the scalar child's dispatch assertion fails |

Removing a launch leaves its grid variable unused. The scratch copy retains
the grid computation with a void cast. Removing Q4_K instantiation also leaves
two internal helpers unused. Scratch compilation suppresses only
`-Wunneeded-internal-declaration`. The production build retains `-Werror` and
its ordinary warning flags. The initial scratch compile errors are harness
failures, not mutation evidence. Each recorded mutation subsequently compiles,
executes, and exits 1 for the stated runtime assertion.

## Repository preflight

The first full preflight, invoked before edits, completes with exit 0 and
12 explicit skips. [Full receipt](preflight-first.log).
Both full invocations use `GIT_CONFIG_GLOBAL=/dev/null` because the repository's
onboarding test assumes Git's `master` default while the user's global setting
selects `main`. No global configuration or checker changes are made.

Seven skipped suites require NumPy, which this host Python cannot import.
Five checks require arguments that `agent-preflight.sh` does not supply:
`check-arm-isa-build.py`, `check-cpu-isa-build.py`, `check-cuda-fat-gencode.py`,
`check-pr-size.py`, and `check-triton-aot-multiarch.py`.
The relevant gfx1100 instruction path is separately inspected in this row.
These skips do not constitute executed gate results.
The full staged preflight completes with exit 0, no failed checks, and the
same 12 explicit skips as the first run. It compiles all eight translation
units in scope under the default host configuration.
[Full staged receipt](preflight-staged.log). The record, symbol-anchor, and
staged NOW checks also pass after the evidence and spec updates.
