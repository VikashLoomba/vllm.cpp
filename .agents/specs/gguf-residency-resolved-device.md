# ENG-GGUF-RESIDENCY-RESOLVED-DEVICE — the GGUF residency policy reads the engine's resolved device, not the platform probe

Issue: OWED. The GitHub API refuses writes for this account at the time of this
row (`gh issue create` returns 403 while `git push` over SSH works), so the
issue cannot be filed from here. The obligation is recorded under `## Owed`
below and the row's ID is the anchor until it can be filed.
Base: `9fb40279d` (`origin/main` at the claim).

## Scope

Two seams answer the question "which device will this load run on", and they
answer it differently by construction.

The ENGINE resolves it. `ResolveModelDeviceType`
(`src/vllm/entrypoints/model_loader.cpp`) routes an explicit selection through
`LoadedEngine::ResolveExplicitDeviceType`, whose `kCPU` arm carries its own
comment: "Explicit CPU never consults the accelerator probe: even on a
CUDA-capable build/process this selects the CPU queue."

The GGUF RESIDENCY POLICY probes it. `GgufLoadPolicy::FromEnv()` and
`RouteGgufTensor` both call `vllm::platforms::CurrentPlatform().device_type()`,
and `CurrentPlatform()` answers `kCUDA` whenever the CUDA platform registered —
which `src/vllm/platforms/cuda.cpp`'s `Registrar` does on any process where
`cudaGetDeviceCount` reports a usable GPU, regardless of what the caller asked
for. `qwen4_exp_registry.cpp` probes it the same way when it hands the
architecture's load-time refusal its device argument.

So on a CUDA-capable process, `--device cpu` selects the CPU queue and gets the
CUDA residency policy. The two disagree on every GGUF model, not on one:

* `RouteGgufTensor`'s device gate (`DeviceKeepQuantSupported`,
  `DeviceQuantGatherSupported`) decides per tensor whether a block encoding
  stays resident. On `kCUDA` the gather arm is false, so a keep-quant embedding
  table expands to bf16 that a CPU load would have kept.
* `GgufLoadPolicy::FromEnv` resolves `keep_quant` from
  `GgufQuantComputeAvailable()`, `keep_f16` from `DeviceKeepF16Supported`,
  `nvfp4_fp4` from `GgufNvfp4ComputeAvailable()` and `elem_kn_repack` from a
  `kCPU` equality test — four flags, all on the probe.
* `qwen4_exp` turns the disagreement into a refusal the user cannot satisfy.
  `qwen4_exp_weights.cpp`'s PLE guard throws on any device without a block
  gather, and its remedy text reads "Load this model with `--device cpu`" — the
  thing the user just did.

This is the same defect class the device-fit refusal already fixed one level up.
`#1136` found `CheckDeviceWeightFit` "refusing a checkpoint by naming a device
nothing was going to run on", and the repair was to resolve the device once and
pass it. `model_loader.cpp` now binds `const platforms::Platform& target =
platforms::GetPlatform(ResolveModelDeviceType(...))` and hands `target` to the
fit check — and then, fifty lines further down, builds the residency policy for
the same load with `GgufLoadPolicy::FromEnv()`, which probes. The fix landed for
the bound and not for the policy the bound describes.

It is latent rather than live only because every run that exercised these paths
was a CPU-only build, where the probe also answers `kCPU`. Two CUDA waves are
enabling CUDA in the same build right now.

### Out of scope

* `DeviceQuantGatherSupported`'s own answer (wave KGATHER owns it).
* The `qwen4_exp` op kernels and their registration (wave CUOPS owns them).
* The safetensors arm. `ModelSource::device` is set by `FromGguf` and read by
  the GGUF loaders only; no safetensors path consults a residency policy.
* Non-GGUF `CurrentPlatform()` readers. This row does not sweep the tree for
  the probe; it fixes the loads whose residency it decides.

## Upstream chain

No vLLM mirror. vLLM resolves `DeviceConfig` once at config-creation time
(`vllm/config/device.py:61-66` @ `555967922`, `vllm/engine/arg_utils.py:1878`)
and every later consumer reads that resolved value; nothing in the weight load
re-probes the accelerator. This row makes our GGUF residency policy read the
resolved value the same way, and it is the polarity `ResolveExplicitDeviceType`
already documents on our side.

## Design

The resolved device travels as a value along the path the load already takes.
No new global.

1. `GgufLoadPolicy` gains a `device` field, and `FromEnv` gains a required
   `vt::DeviceType` parameter. The no-argument `FromEnv()` is REMOVED rather
   than kept as a defaulted overload: a default is what let every caller get
   the probe by saying nothing, so removing it makes the compiler the checker.
   Every one of the policy's four device-dependent flags resolves from that
   parameter.
2. `RouteGgufTensor` gains a required `vt::DeviceType` last parameter, replacing
   its internal `CurrentPlatform()` call. Same reasoning: no default.
   `GgufLoadPolicy::Route` passes its own `device`, so a policy and the tensors
   it routes can never disagree.
3. `ModelSource` gains a `device` field, required by `ModelSource::FromGguf`.
   `ModelSource` is already the per-load CONTEXT and not only the checkpoint —
   `load_queue` beside it is an engine-selected execution resource and
   `multimodal` is a borrowed engine config — so the engine's resolved device
   belongs there. This is what reaches the eight GGUF registries, which are the
   only production callers with no policy argument of their own.
4. Every GGUF registry builds `GgufLoadPolicy::FromEnv(source.device)` and
   passes it to its loader. `qwen4_exp_registry` additionally passes
   `source.device` in place of the probe for the PLE refusal.
5. `entrypoints/model_loader.cpp` passes the device it ALREADY resolved for the
   fit check to `ModelSource::FromGguf`, to the policy it builds for the fit
   check, and to the MTP head's policy.

The alternative — a process-global "resolved device" that `CurrentPlatform()`-
style readers consult — was rejected. It has no upstream analogue, it makes the
value ambient exactly where the bug was ambience, and it cannot express two
engines in one process on different devices.

## Risks and decisions

* **Removing `FromEnv()` touches 35 test call sites.** That is the cost of
  making the compiler enforce the parameter. Each becomes an explicit
  `FromEnv(vt::DeviceType::kCPU)`, which states what the test always meant on a
  CPU-only build and now says so.
* **`ModelSource::device` defaults to `kCPU` on the struct.** The safetensors
  factories do not set it and no safetensors path reads it. `FromGguf` requires
  it, so no GGUF load can reach a loader without one.
* **Behaviour change on a CUDA process with `--device cpu`.** That is the fix.
  On every other combination the resolved device equals the probe, so the load
  is byte-identical: `--device cuda` resolves `kCUDA`, `--device auto` resolves
  through `ResolveAutoDevice` which already agrees with the probe except where
  `CreateQueue()` fails (in which case the resolved answer is `kCPU` and is the
  more correct one, for #1136's reason).
* **No GPU here.** The device-aware case is expressible on a CPU-only build
  because the device is now a parameter: a test can hand the production
  registry hook a CUDA-resolved `ModelSource` and read the refusal. What it
  cannot do is prove the CUDA platform registers and `CurrentPlatform()` then
  answers `kCUDA`; that half is recorded under `## Owed`.

## Tests

`tests/vllm/models/test_qwen4_exp_gguf_weights.cpp` — the production hook:

1. RED: `ModelRegistry::Load` with a `ModelSource` whose resolved device is
   `kCUDA` must refuse with the PLE gather message. Before the registry fix
   this passes the probe's `kCPU` and the load succeeds.
2. `ModelRegistry::Load` with a `kCPU`-resolved `ModelSource` must load — the
   `--device cpu`-on-a-CUDA-box case, which is the one that is wrong today.
3. Every non-CPU resolved device refuses through the same hook.

`tests/vllm/test_gguf_keep_quant.cpp` — the route and the policy:

4. `RouteGgufTensor(..., kCUDA)` and `RouteGgufTensor(..., kCPU)` on the same
   quantized embedding table give different residencies, so the parameter is
   load-bearing.
5. `GgufLoadPolicy::FromEnv(kCUDA)` and `FromEnv(kCPU)` differ in `keep_quant`
   on a build where the quantized GEMM is CPU-only, and `Route` carries the
   policy's device.

## Gates

```sh
ctest --test-dir build -R 'test_qwen4_exp_gguf_weights|test_gguf_keep_quant' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

Recorded in the pull request body: the red transcript, the green transcript,
and the build rcs, each read from the command's own status.

## Owed

* **File the issue.** The GitHub API refuses writes for this account
  (`gh issue create` -> 403 while `git push` over SSH works). The issue must be
  filed against this row ID and linked here, in the row spec, and in the pull
  request body, once writes are restored.
* **The hardware half of the test.** Nothing here proves that on a real
  CUDA-capable process `CurrentPlatform()` answers `kCUDA` while
  `ResolveModelDeviceType(..., Device::kCPU)` answers `kCPU`. The test pins that
  the RESOLVED device decides; it takes the resolution itself from
  `ResolveExplicitDeviceType`, whose own suite
  (`tests/vllm/entrypoints/test_loaded_engine_dense.cpp`) pins the `kCPU` arm.
  An end-to-end `--device cpu` GGUF load on a fleet GPU box closes it.
* **The rest of the tree's `CurrentPlatform()` readers.** This row fixes the
  GGUF residency path. A sweep for other load-time or config-time readers that
  should take the engine's resolved device is not done here and has no owner.

## Stop conditions

* Stop and report if the mismatch turns out not to be reachable — if some
  caller upstream of the policy already forces the probe and the resolved
  device to agree. It does not; `ResolveExplicitDeviceType`'s `kCPU` arm is
  unconditional.
* Stop if a CUDA build is required to demonstrate the red. It is not: the
  device becomes a parameter, so both arms run on a CPU-only build.

## Now

ACTIVE. Implementation and tests in one pull request with this spec.
