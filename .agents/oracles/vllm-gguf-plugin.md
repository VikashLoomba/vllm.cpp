# `vllm-gguf-plugin` — vLLM's own GGUF support, moved out of tree

**This is not a competitor's implementation of GGUF. It is vLLM's.** In-tree
GGUF support was deprecated in vLLM (`vllm-project/vllm#39583`) and migrated to
`vllm-project/vllm-gguf-plugin`, in the same organisation. At our parity pin
`5559679229`, `docs/features/quantization/gguf.md:7` names that repository as
where GGUF support went, `:12` gives the install line and `:19` the serve
recipe. So a GGUF path that this plugin serves is a path the PRIMARY oracle
serves, and `AGENTS.md` §"When vLLM has no implementation" does not admit a
secondary oracle for it.

```oracle-pin
id = vllm-gguf-plugin
role = secondary
upstream = https://github.com/vllm-project/vllm-gguf-plugin
scope = GGUF quantization for the pinned vLLM, which carries no in-tree GGUF at 5559679229; never a behavior source, because vLLM's own model implementations are what it feeds
pin = d4c1f0d082fc7cd4350da56689109a01c1f29d6c
pin_label = post-v0.0.5 HEAD, 2026-08-31
pinned_on = 2026-09-03
gateable = no
evidence = #2624
```

**`role = secondary` is a registry mechanic, not a claim about rank.** The
registry admits exactly one `primary` and it is `vllm`
(`scripts/check-oracle-pins.py:64,183`), so every other file takes the other
value whatever its provenance. `vllm-omni.md` sits in the same position: a
first-party `vllm-project` repository recorded as `secondary`. Read the `scope`
line, not the role word. This plugin supplies no behavior of its own — it feeds
weights to vLLM's own `Qwen3_5ForConditionalGeneration`, so where it runs, the
answer it produces is vLLM's answer.

## Why the pin is a commit and not `v0.0.5`

**The released wheel cannot serve this architecture, and that is measured
rather than assumed.** `v0.0.5` was published 2026-08-10; the Qwen3.5/3.6
adapter landed in `4ec8d61` on 2026-08-19 (`[Models] support Qwen3.5/3.6
multimodal GGUF and MTP (#98)`), and `git ls-tree v0.0.5
vllm_gguf_plugin/weights_adapter/` lists only `base`, `default`, `diffusion`
and `gemma3`. Point the release at a `qwen35` GGUF and no adapter matches, so
`config_parser.py:44-52` falls back to `transformers`'
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` and the generic `default` adapter, whose
name map knows nothing of `ssm_alpha`, `ssm_out` or `attn_qkv`. Whether that
presents as a config error or as a weight-mapping error is NOT measured here;
what is measured is that the file which knows those names is absent from the
release. The pin is therefore the HEAD that
carries the adapter, `d4c1f0d`, whose `weights_adapter/qwen3_5.py:45-64` is
the file that knows `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_out` and
`attn_qkv` are one Gated Delta Net layer.

## What is established, and what is not

**ESTABLISHED — source feasibility on this fleet, no lease needed.**

- **aarch64 is upstream-supported.** `.github/workflows/release.yml:192,247`
  build `cu129` and `cu130` aarch64 wheels on `ubuntu-24.04-arm` inside
  `pytorch/manylinuxaarch64-builder`, and `:354` uploads the aarch64 wheel to
  PyPI. PyPI carries `...-manylinux_2_28_aarch64.whl` for every release from
  `0.0.2` onward. This is the opposite of
  [`exllamav3.md`](exllamav3.md)'s finding, where seven translation units are
  x86-only host code and the project publishes no aarch64 wheel at all.
- **No x86 assumption in the extension.**
  `grep -rn '__x86_64__|__aarch64__|__AVX|__SSE|immintrin|_mm_|_mm256'
  vllm_gguf_plugin/csrc/` returns zero hits over all eleven files: there is no
  host-CPU intrinsic to port. The device code carries 25 `__CUDA_ARCH__`
  guards and **every one is a lower bound that `sm_121` clears** — 23 in
  `csrc/gguf/vecdotq.cuh` at `>= 610` around `__dp4a`, 2 in
  `csrc/gguf/ggml-common.h:1086,1090` at `>= 800` around `__float2bfloat16`.
  The only other architecture branch is CUDA versus ROCm
  (`csrc/cuda_compat.h`, `csrc/gguf/mmq.cuh`). Nothing excludes `sm_121a` by
  construction.
- **It imports against OUR pin, not only against its own contemporary.** All
  44 `from vllm...` symbol imports in `vllm_gguf_plugin/` resolve in the
  pinned checkout at `5559679229`: 42 by an automated walk of each module's
  top-level definitions and star re-exports, and the two the depth-limited
  walk missed, `get_tensor_model_parallel_rank` and
  `...world_size`, by hand at `vllm/distributed/parallel_state.py:2039,2044`.
  `weights_adapter/qwen3_5.py:14-15` needs
  `vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config` and
  `...qwen3_5_moe.Qwen3_5MoeConfig`, and both files exist at the pin.
- **The architecture is registered at the pin.**
  `vllm/model_executor/models/registry.py:572` registers
  `Qwen3_5ForConditionalGeneration`, and `:647` `Qwen3_5MTP`. The artifact's
  own `config.json` declares `model_type = "qwen3_5"`, which is the first
  entry of the adapter's `QWEN35_MODEL_TYPES` (`qwen3_5.py:29-34`) and maps to
  that architecture in `QWEN35_ARCHITECTURES` (`:38-43`).
- **Every tensor of our artifact maps. MEASURED 2026-09-03, offline.**
  Running the adapter's own `build_qwen35_text_mapper`,
  `build_qwen35_vision_mapper` and `build_qwen35_mtp_mapper` over the tensor
  names of `Qwen3.8-27B-Q4_K_M.gguf` (866) and `mmproj-BF16.gguf` (334),
  1200 names in union: **1185 mapped, 0 unmapped**, 15 held back as the
  `blk.64` MTP block, and all 15 of those mapped by the MTP mapper. Samples:
  `blk.0.attn_qkv.weight -> model.language_model.layers.0.linear_attn.in_proj_qkv.weight`,
  `blk.0.ssm_a -> ...linear_attn.A_log`,
  `v.blk.0.attn_qkv.weight -> model.visual.blocks.0.attn.qkv.weight`.
  `_gdn_value_head_layout` (`qwen3_5.py:169-184`) resolves on this config to
  `heads_per_group = 3` from `linear_num_value_heads = 48` over
  `linear_num_key_heads = 16`, so the GDN head-tiling restore is armed rather
  than skipped.
- **Every quantized type in the artifact has a kernel. MEASURED.** The
  backbone's 866 tensors are `Q4_K` 294, `F32` 456, `Q6_K` 67, `Q5_K` 48 and
  `Q8_0` 1; the mmproj's 334 are `F32` 224 and `BF16` 110. All four quantized
  types appear in the plugin's dequantize dispatch table
  (`triton/dequantize/interface.py:74,77,78,79`) and in its GEMM table
  (`triton/gemm/interface.py:64,67,68,69`), so no tensor of ours falls into an
  unimplemented arm. Counting the types is not running them.
- **Upstream tests this shape, on smaller siblings.**
  `tests/test_multimodal_gguf.py:103-115` declares `QWEN35_CONFIG`
  (`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` against `Qwen/Qwen3.5-0.8B`) and
  `QWEN35_MOE_CONFIG` (`...35B-A3B...`), both multimodal, both Q4_K_M, and the
  README's coverage table lists "Qwen 3.5 — Q4_K_M backbone with BF16
  projector" — the exact shape of our checkpoint. They carry
  `pytest.mark.slow` (`:118-123`) and were NOT run here, so this is upstream's
  declaration of coverage and not a green from this session.
- **That 15-tensor block is exactly what the llama.cpp oracle drops.**
  [`llama-cpp.md`](llama-cpp.md) records `b10451` loading 851 of 866 tensors
  and ignoring all 15 of `blk.64`. The plugin maps all 15. The two oracles are
  therefore not offered the same model, and the vLLM side can run the MTP head
  the llama.cpp side cannot.

**MEASURED IN A LEASE, on `thor:gpu0`, 2026-09-03.** Five `rc` jobs ran the
same staged script on `thor:gpu0` (aarch64, NVIDIA Thor, capability 11.0,
driver 595.78); the last is `40a1d8dd-529b-456b-8e46-2789354cce5a`, run dir
`/workspace/ggufplugin/20260903T012806Z`. No `ssh` was used.

- **It BUILDS.** `PLUGIN_BUILD_RC=0` in 29.1 s under nvcc 13.0.88, producing
  `vllm_gguf_plugin-0.0.5-cp310-abi3-linux_aarch64.whl`. The source-feasibility
  question is answered empirically, not by inference.
- **It REGISTERS.** `REGISTRATION_RC=0`; entry point
  `('gguf', 'vllm_gguf_plugin:register')`; `_C_gguf.abi3.so` imports; and
  `torch.ops._C_gguf` exposes `ggml_dequantize` and `ggml_moe_a8_vec`. In the
  engine, vLLM registers the plugin's `GGUFModelLoader` for `load_format=gguf`
  and its `GGUFConfigParser` for `config_format=gguf`.
- **The pinned vLLM wheel runs on a SECOND aarch64 capability.** Built on GB10,
  it imports on Thor as `0.1.dev1+g555967922` and reports
  `cuda True NVIDIA Thor`.
- **THE 17 GB Q4_K_M GGUF LOADS.** `Resolved architecture:
  Qwen3_5ForConditionalGeneration`, then `Model loading took 16.3 GiB memory
  and 745.171284 seconds`, with **no** `No HF name for N Qwen3.5 GGUF
  tensor(s), skipping` warning — the line the adapter emits for anything it
  cannot map. The offline mapping result is confirmed on the real loader. The
  engine chose `Triton/FLA GDN prefill kernel (head_k_dim=128)` and the
  `FLASHINFER` backend, so the Gated Delta Net path is the one that ran.
- **The tokenizations agree with the llama.cpp gate's**, `[6, 5, 6, 7, 11, 7]`
  over the six recorded prompts, from two different vocab sources.

**NOT ESTABLISHED, and it is the half that decides the flag: NO TOKEN.**
`AGENTS.md` requires the oracle to build **and run** the model. A 16.3 GiB
load is not a generation.

- **The forward is blocked on `thor:gpu0`, and NOT by the plugin.** Memory
  profiling dies with `CUDA error: no kernel image is available for execution
  on the device`, raised from `launch_vectorized_kernel` at
  `ATen/native/cuda/CUDALoops.cuh:349`, frame #2
  `at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<c10::BFloat16>>`
  in `torch/lib/libtorch_cuda.so`. That is PyTorch's own bf16 fill in the stock
  PyPI `torch==2.13.0` aarch64 wheel, which carries no `sm_110`. The plugin's
  installed object DOES carry it: `cuobjdump --list-elf` on the installed
  `_C_gguf.abi3.so` reports `_C_gguf.abi3.1.sm_110.cubin`. So this is a
  device-and-toolchain wall on that box, not a statement about GGUF or this
  checkpoint.
- **`dgx:gpu0` is untried.** It is `sm_121a`, which that torch wheel does
  support and where the vLLM wheel was built; `rc` job
  `7a45427e-ad86-4042-aeea-cdf8db535a54` is queued there with the identical
  script and is what owes the token.
- **The `dgx` history is a caution, not a promise.**
  [#1129](https://github.com/mudler/vllm.cpp/issues/1129) recorded that no vLLM
  leg could run a model there by a lease-compliant path, and
  `.agents/specs/mtp-k-gt-1.md` records the host consumed in the step after
  `torch.compile` at two `gpu_memory_utilization` values, the lower one
  rebooting the box. `/workspace/oracle-vllm/README-WHEELS.md` states the
  `FLASHINFER-ONLY` wheel later generated coherent tokens there; that is
  another session's claim, read off the share and not re-derived here, and it
  was a bf16 checkpoint rather than a GGUF one.

The measurement is owed by
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) and the method is in
[`../specs/oracle-vllm-gguf-qwen35.md`](../specs/oracle-vllm-gguf-qwen35.md).

## What this record does NOT license

It does not move the Q4_K_M arm's token gate off llama.cpp. That gate reads
`FAIL` against `b10451` ([#2534](https://github.com/mudler/vllm.cpp/issues/2534),
`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`) and stays exactly
as recorded until this oracle is gateable and a paired run exists. A denominator
that has never emitted a token cannot replace one that has.
