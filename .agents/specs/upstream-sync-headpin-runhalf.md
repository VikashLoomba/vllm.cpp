# Sync cycle `e126687a9a`, wave RUNHALF

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2611](https://github.com/mudler/vllm.cpp/issues/2611).
Predecessor: [#2593](https://github.com/mudler/vllm.cpp/issues/2593), wave
HEADPIN, which landed [#2594](https://github.com/mudler/vllm.cpp/pull/2594) and
[`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md), whose `## Owed`
names this work as its first item.

## Now

The pin does **not** advance in this wave and nothing here is a reason to move
it. The active parity pin remains
`5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question.** Does `e126687a9a828d513c01a07cd69f025f27d63280` demonstrably
**run a model** on this fleet? AGENTS.md §"When vLLM has no implementation"
requires an oracle to "demonstrably build and run the model" before it is
gateable, and adds that "constructing a config proves nothing". #2594 measured
install, build and import at this target and said in as many words that this is
not the run half: no weights were loaded, no forward pass ran, no token was
compared. That is the sole reason the candidate's `gateable` stays `no`.

**What makes it reachable now, and what still blocks it.** `thor:gpu0`'s leased
container exposes the device (`CUDA_AVAIL=True`, `NVIDIA Thor, 595.78`) where
`orin`'s does not, and `nvidia-cuda-nvcc 13.3.73` installs as an aarch64 wheel
inside the target's own `requirements/cuda.txt`
([`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md) §5.5, §5.7).
The blocker is that every #2594 job ran `VLLM_USE_PRECOMPILED=1`, which on
aarch64 yields an editable wheel and no `vllm._C` (`EXT_RC=1`). **A package
without its extension cannot run a model**, so this wave needs a source build,
and no job has ever run one at this target. Locating a compiler is not compiling
with it.

In scope:

1. One `rc` lease on `thor:gpu0`. Source-build vLLM at the target from a
   verified tree, install the wheel, and report `SRCBUILD_RC`, `EXT_PRESENT`
   and `RUN_RC` separately and literally.
2. Generate tokens greedily from a real checkpoint through vLLM's normal path,
   and record the model, prompts, sampling parameters, engine configuration and
   the output token ids.
3. Record the result, and record what gateability it establishes and what it
   does not.

Out of scope:

- **Advancing the pin.** The 290-entry PORT-NOW queue for
  `5559679229..e126687a9a` is unworked and is not this wave's.
- Working that queue, re-deriving its dispositions, or editing
  [`../porting-inventory.md`](../porting-inventory.md).
- Any product code. This wave touches records and documents only.
- Any throughput, latency or memory number. A run is a correctness precondition
  here, not a benchmark, and nothing measured in this wave may be quoted as a
  speed axis.

## 2. Design

### 2.1 Three levels of answer, and which one this wave targets

The question "does it run the model" has three levels on this fleet, and the
report says which one it reached rather than blurring them:

1. **`qwen4_exp` itself**, from a real checkpoint. **Not reachable.** Every
   published safetensors arm of Qwen3.8-Flash-Next exceeds the largest fleet
   box, and upstream's own `tests/models/registry.py` marks all three Qwen4Exp
   architectures `is_available_online=False` at this very revision. Nothing this
   wave can do changes that.
2. **Any model** generating tokens through vLLM's normal path at this revision.
   This is what AGENTS.md's sentence means, and it is this wave's target.
3. A forward pass producing logits without generation. Weaker, and recorded as
   such if it is all that is reached.

A fourth reading sits between 1 and 2 and this wave attempts it as a stretch:
the `qwen4_exp` graph executing on **random** weights, from the model's own
published `config.json` shrunk in depth and width, under `load_format="dummy"`.
That is not a parity statement and not a token gate — the tokens carry no
information — but it is the strongest statement about this architecture the
fleet can currently support, and it is reported with that qualifier attached.

### 2.2 The build

`VLLM_USE_PRECOMPILED=0`, `VLLM_TARGET_DEVICE=cuda`, `MAX_JOBS=4`,
`NVCC_THREADS=1`, and `TORCH_CUDA_ARCH_LIST` read from the device rather than
written down. The shape follows the only recorded source build of upstream vLLM
inside a lease, `.agents/benchmark-record.md`'s DFlash2 oracle wheel on
`dgx:gpu0`: `pip wheel --no-deps --no-build-isolation -w dist .`, then install
the wheel into a venv. `MAX_JOBS=4` is AGENTS.md's limit; unconstrained
parallelism has OOM-rebooted a fleet box.

**`VLLM_FA_CMAKE_GPU_ARCHES` is deliberately NOT set.** Upstream's
`vllm-project/flash-attention` hard-codes `FA2_ARCHS "8.0+PTX"`, so the built
FlashAttention reaches this device only through a driver JIT of `compute_80`
PTX, which is the mode that failed on GB10 with
`cudaErrorUnsupportedPtxVersion` (`/workspace/oracle-vllm/README-WHEELS.md`).
Overriding the arch list might produce native SASS and might instead fail the
whole compile for a device nobody has built FA2 for. The question this wave owes
is whether a model runs, not which attention backend is fastest, so the build
takes the low-risk path and the run walks a backend ladder: the default first,
then `TRITON_ATTN`, then `FLASHINFER`, reporting which one answered.

### 2.3 Nothing the build needs is fetched from the network

The worker's `github.com` egress is not guaranteed and its absence reads as an
authentication failure rather than a network one
([`../environment.md`](../environment.md)); a `thor:gpu0` job failed `git fetch`
that way on 2026-09-02, the same day #2594's jobs cloned successfully. The
target tree, the four `FetchContent` dependencies CMake would clone
(`cutlass v4.4.2`, `vllm-project/flash-attention 06bdd47c`,
`vllm-project/FlashMLA 0397728d`, `vllm-project/FlashKDA ee0be888`) and the
checkpoint are therefore staged on `/workspace` from the developer box, and the
build is pointed at them with `VLLM_CUTLASS_SRC_DIR`,
`VLLM_FLASH_ATTN_SRC_DIR`, `FLASH_MLA_SRC_DIR` and `FLASH_KDA_SRC_DIR`.

The tree is staged as a **git bundle**, not a tarball, for two reasons. A
release tarball cannot build vLLM, because `setuptools_scm` needs git. And the
bundle carries the tags, so `git describe` and the version string are derived in
the built environment rather than transcribed from the developer box, which is
the failure #2589 §7.3 measured when a locally-only tag rewrote a version
prefix.

### 2.4 The identity is asserted, not named

The job reads `HEAD` from the restored tree and **aborts** when it is not the
target, before it installs or compiles anything. `vllm.__version__` is then read
from `cd /`, so it is the installed package and not a source tree, and it must
carry `e126687a9`. A build that cannot say which commit it compiled is not an
oracle.

### 2.5 Instruments

This row's recurring failure is an instrument whose failure looks like a result,
and #2594's own `BUILDDEPS_RC=1` was one. Three rules follow, and the job
implements each:

- **Every rc is printed separately and literally**, immediately after its
  command and never after a pipe. A leg that could not run prints
  `SKIPPED_<reason>`, never a `0` and never a red that would read as a target
  defect.
- **The run is watchdogged on `MemAvailable`, and the floor is recorded.**
  `thor` is a unified-memory box on which `gpu_memory_utilization` reserves HOST
  RAM, and the recorded failure mode of this fleet is a host consumed in the
  step after `torch.compile` — a reboot on `dgx` at both `0.75` and `0.30`, so
  the fraction is not the lever. The engine therefore runs at `0.10` of 132 GB,
  which is far more than a 125M model needs and far below any fraction that has
  taken a box down, and the watchdog floor sits at 20,000 MB, which is outside
  that configuration's operating point. A guard whose threshold sits inside the
  guarded thing's operating point manufactures the finding it was meant to
  detect.
- **The first leg is `enforce_eager=True`**, which removes `torch.compile` and
  graph capture from the variables. Only after an eager leg has proved the
  engine runs does a compiled leg follow. Neither is a benchmark denominator and
  neither may be quoted as one.

### 2.6 The workload is the one already committed here

Prompts, dtype, sampling parameters and the per-prompt batch=1 regime are copied
from [`../../scripts/opt-oracle-capture.py`](../../scripts/opt-oracle-capture.py),
which captured `tests/parity/goldens/opt_greedy` at the **pin** on `dgx`. Laying
this wave's token ids beside that golden is **informative and not a gate**: the
revision differs by 1465 commits and the silicon differs, so a divergence is not
a defect and an agreement is not a parity result.

## 3. Risks

- **The source build may fail.** Nobody has built upstream vLLM on `thor` — the
  repository has no `TORCH_CUDA_ARCH_LIST` value for `sm_110` anywhere, and no
  recorded build duration or toolkit choice for one. A failure is a legitimate
  result and is reported as the **first** error rather than the last line.
- **The box may be lost.** `thor` reboots rather than OOM-kills, and it has gone
  `worker_lost` mid-build before. The wheel is persisted to `/workspace` the
  moment it exists, so a build that succeeded is not repaid by a run that dies.
- **`e126687a9a` may run a model and still not be pinnable.** It is, and §5
  says so: this wave measures a precondition, not a licence.

## 4. Gates

- `SRCBUILD_RC`, `EXT_PRESENT`, `IMPORT_RC` and `RUN_RC` reported separately,
  each read literally, from one `rc` lease on `thor:gpu0`.
- The run recipe recorded in full: model identity, prompts, sampling parameters,
  engine configuration, backend, and the output token ids.
- `git diff` is not a gate here. This wave's gate is the job's own output, and
  the report carries it verbatim.

## 5. Stop conditions

- **Do not advance the pin.** Not on any result in this wave.
- **One lease, on `thor:gpu0`.** Never a second device, never two at once.
- **Do not fake a run.** If the source build fails, report the failure precisely
  and stop. A build that did not produce a wheel makes every leg below it an
  absence, and the job prints them as absences.
- Touch [`../oracles/vllm.md`](../oracles/vllm.md) only if the run succeeds, and
  then only to record what this candidate established. The `oracle-pin` block
  keeps naming `5559679229`.

## Owed

Carried from [`upstream-sync-headpin.md`](upstream-sync-headpin.md) and not
discharged here:

- The PORT-NOW queue for `5559679229..e126687a9a`, 290 entries
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Whether upstream `main` installs under the same repair (#2611).
- `scripts/check-pr-size.py` cannot classify `.agents/scripts/**`
  ([#2607](https://github.com/mudler/vllm.cpp/issues/2607)), so this wave's job
  script is reproduced verbatim in its report rather than committed beside the
  two already tracked there.
