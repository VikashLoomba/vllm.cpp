# Spec — can the PRIMARY oracle run on gfx1151?

Row `BACKEND-GATE-ROCM-LLAMACPP`. Issue
[#2740](https://github.com/mudler/vllm.cpp/issues/2740).

Sibling records: [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the
quant-matched gfx1151 decode number, blocked behind a failing token gate),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (the CPU tier's
divergence against the same secondary oracle),
[#2546](https://github.com/mudler/vllm.cpp/issues/2546) (the gfx1151 token gate
that returned `FAIL` at 3 of 6), and
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) (the same
vLLM-plus-GGUF-plugin question on `dgx:gpu0`, CUDA, a different device).

## Now

`ACTIVE` — the prerequisite is MEASURED and it is positive. The remaining
question is the build and the run.

## The question

The Qwen3.8-27B Q4_K_M arm on `strix:gpu0` is gated against llama.cpp `b10451`.
`AGENTS.md` §"When vLLM has no implementation" admits a secondary oracle only
where vLLM implements nothing. **Nobody has ever measured whether vLLM
implements this.** The assumption that it cannot run on gfx1151 was inherited,
not tested, and reading the pin contradicts it:

| what the pin says | where |
|---|---|
| `_ON_GFX1151`, `on_gfx1151()` | `vllm/platforms/rocm.py:214`, `:307` |
| `"0x1586": "AMD_Radeon_8060S",  # gfx1151, Strix Halo` | `vllm/platforms/rocm.py:77` |
| `gfx1151` in `HIP_SUPPORTED_ARCHS` | `CMakeLists.txt:52` |
| `TORCH_SUPPORTED_VERSION_ROCM "2.13.0"` | `CMakeLists.txt:72` |
| `PYTORCH_ROCM_ARCH=…;gfx1150;gfx1151` | `docker/Dockerfile.rocm_base:30` |
| `# Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts)` | `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py:212` |
| `_CAST_DOT_TO_K_DTYPE = on_gfx1x()` | `vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py:26` |
| `"Qwen3_5ForConditionalGeneration": ("qwen3_5", …)` | `vllm/model_executor/models/registry.py:572` |

Each was re-read out of the **committed** object
`5559679229bc961848b121ccdeaa8fa5d79bec98`, not out of a dirty working tree, and
each is re-read again on the worker inside the job.

The last two matter most. `chunk_scaled_dot_kkt.py` is the linear-attention path
this GDN-hybrid model uses, and its RDNA branch exists because someone ran it on
RDNA. A constant naming a device is weak evidence; a kernel tuned for its
wavefront width is not.

## Scope

1. **The prerequisite, measured first and alone.** Does the PyTorch build vLLM
   needs carry device code for gfx1151? The previous attempt at this class of
   question died here on `thor:gpu0` (`sm_110`) and was misdiagnosed three times
   before anyone measured it.
2. Does the pinned vLLM build and install on gfx1151, and do its compiled
   extensions load and compute?
3. Does it run Qwen3.8-27B and emit tokens?
4. If it generates: the six-prompt greedy gate, identical prompts and token
   counts to `qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`.
5. A device-scoped gateability record in `.agents/oracles/vllm.md`.

## Exclusions, which are rules and not preferences

- **No throughput, latency or memory number for vllm.cpp, and no cross-engine
  ratio.** `AGENTS.md` §Gates admits a performance result from an arm only after
  that arm's declared token gate passes. It does not pass. #2497 has already had
  one measurement retracted for exactly this.
- This spec does not rewrite #2534's or #2497's verdicts or evidence.
- It does not advance the vLLM parity pin, and it touches no `src/` or
  `include/`.
- **`HSA_OVERRIDE_GFX_VERSION` is not set anywhere.** It makes the runtime lie
  about the device, which is the one thing an oracle measurement cannot survive.
  Every job asserts its own inherited environment is free of it before starting.

## The stop condition, declared before the data

If torch carries no gfx1151 device code and there is no path to a build that
does, **that is the answer**: record it with the measurement and stop. A
properly measured negative closes #2740. A forced positive does not.

## Risks, and what each would look like if it fired

| risk | how it would present | the guard |
|---|---|---|
| torch ROCm wheels omit gfx1151 | `no kernel image`, or a silent CPU fallback | read the offload bundle out of `libtorch_hip.so` AND `torch.cuda.get_arch_list()`, then compute |
| an instrument reads its own bug as a device verdict | "no wheel exists" off a working index | positive and negative controls on every grep |
| `pip install --no-deps` starves a package | `ModuleNotFoundError` reading as a model failure | never `--no-deps`; name the deps explicitly |
| a same-version reinstall declines silently | the previous run's object is the one called | `--force-reinstall`, then interrogate the INSTALLED object |
| a missing `__main__` guard breaks V1 spawn | reads as a model failure | the generation script is a file with a guard |
| an exit code read through a pipe | the pipe's status, not the program's | `rc=$?` on its own line, never `| tail` first |
| bf16 at 53.8 GB against a 64 GiB carve | OOM mid-load | prefer the 17 GB GGUF arm; watchdog the host |

## Gates

```sh
# the prerequisite
llvm-objdump --offloading <venv>/torch/lib/libtorch_hip.so | grep -c 'amdhsa--gfx1151$'
python -c "import torch; print(torch.cuda.get_arch_list())"    # must contain gfx1151
# the build
VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4 pip install --no-build-isolation -e .
python -c "import vllm._C, vllm._rocm_C"
python -c "from vllm.model_executor.models.registry import ModelRegistry as R; R.resolve_model_cls(['Qwen3_5ForConditionalGeneration'])"
# the run
python gen.py            # must print GEN_IDS for all six prompts
```

## Evidence

`docs/bench-evidence/oracle-vllm-gfx1151-20260903.md`, and the raw job logs
under `/mnt/nas_share/rc/vllm-gfx1151-2740/out/`.

## Instrument failures this run has already made, and what each looked like

Recorded because each one, left uncorrected, would have been written down as a
statement about gfx1151.

| what broke | what it printed | what it actually was |
|---|---|---|
| index grep for `linux_x86_64` | `no ROCm torch index carries a wheel for this interpreter` | the index writes `manylinux_2_28_x86_64`; five torch versions were there the whole time |
| `llvm-objdump --offloading \| head -80` | `objdump_rc=141` | SIGPIPE, not a failure |
| `@triton.jit` body on stdin | `ValueError: @jit functions should be defined in a Python file` | Triton needs a real file; the probe was mine, not the device's |
| `python3-dev` absent | `CMake Error at cmake/utils.cmake:10: Unable to find python matching …`, and separately `fatal error: Python.h` inside Triton's AMD driver | one missing header stopped both the vLLM configure and every Triton kernel on the board. Neither was about gfx1151 |

Each was repaired and re-run rather than reasoned around, and every repaired
probe carries a control.
