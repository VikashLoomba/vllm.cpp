# Latest upstream oracle preparation

READY for the coordinating task's GPU gates. Preparation ran no GPU workload.
Both native builds finished with exit code 0. No upstream source patch was needed.

## Source identities

- vLLM: `39545e475d3627287ff69c25465dc0bd405f67e1` in `../vllm-source`.
- llama.cpp: `093a2f86c3e37c54fa3e1f9efb17b304f3433abd` in `../llama-source`.
- GGUF plugin: `d4c1f0d082fc7cd4350da56689109a01c1f29d6c` in `../plugin-source`.
- All three source worktrees are clean.
- Live remote `main`, `master`, and `main` respectively matched these commits during preparation.

## Execution

The Python interpreter is
`/home/vikash/oracle/rdna3-wmma-latest/.venv/bin/python` inside image
`sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.
The venv inherits Torch and Triton from that immutable image.

Run `../python-gpu-under-lock.sh <script> <arguments>` under
`/home/vikash/gpu.lock`. The script contains the exact Docker arguments,
environment, and mounts. It mounts the preparation directory read/write,
`/home/vikash/models` read-only, and the shared and coordinating task checkouts
read-only. Add an implementer worktree mount when a script needs another path.
Place writable scripts and output artifacts under the preparation directory.
The launcher was not executed with GPU devices during preparation.

Use `../python-no-gpu.sh` for checks that require no GPU devices.
The successful import command was:

```sh
/home/vikash/oracle/rdna3-wmma-latest/python-no-gpu.sh \
  /home/vikash/oracle/rdna3-wmma-latest/logs/import-provenance.py
```

The native llama.cpp executables are in
`/home/vikash/oracle/rdna3-wmma-latest/llama-build/bin`:
`llama-cli`, `llama-bench`, and `llama-server`.
Their shared libraries resolve without missing dependencies.
Generated `llama-build/common/build-info.cpp` binds commit `093a2f86c`.
The numeric build count comes from the available shallow Git history.

## Build and identity evidence

- `build-vllm.sh`: exact successful source-build command.
- `vllm-build-attempt3.log` and `vllm-build-attempt3.exit`: successful build.
- `llama-build-command.sh`, `llama-configure.log`, `llama-build.log`, and
  `llama-build.exit`: native llama.cpp recipe and results.
- `import-provenance.json`: successful import paths, versions, and hashes.
- `build-provenance.json`: installed source comparison, binary hashes,
  original source status, and installation transformations.
- `plugin-source-artifact-verification.json`: 150 upstream source files and
  138 prior artifact files matched their exact source and recorded hashes.

Installed vLLM version: `0.28.1rc1.dev812+g39545e475`.
All 2473 tracked vLLM Python files match the requested source.
All 111 plugin Python files and 17 gguf Python files match their requested sources.
Every installed native extension matches its wheel and CMake installation output.
CMake removes build-only RPATH data during installation. Host and GPU code
sections remain byte-identical to the native build output.

The vLLM wheel is 21869209 bytes. SHA256:
`010076f9e982d9f4ffaca90014ed4ae5204bc68b09c77843cab7b6bea5b3c276`.
The `llama-bench` SHA256 is
`35fab32eca1ff46846db06da8cb060ee05bcb856ff9a27d1d49792a0c663aac5`.
The plugin extension SHA256 is
`592af79c210496e96eaa7bd409ae0a67e7de8c334aa9f782082c7639ba300138`.

## Dependencies and limits

- llama.cpp uses host HIP 7.15.26333 and Clang 23 at
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.
- vLLM uses the container's HIP 7.2.53211 and Clang 22 toolchain,
  Torch `2.12.0+git6bbd260`, and Triton `3.7.1+gitf0b55c07`.
  vLLM expects Torch >=2.13.0 and emits a warning. Compilation and imports
  succeeded. Model execution remains a required gate.
- The venv now has Hugging Face Hub 1.31.0, MCP 2.2.0, tblib 3.1.0,
  and gguf 0.19.0 built from the requested llama.cpp source.
- Optional Mooncake ROCm remains absent by coordinating-task direction.
- Optional Rust frontend and Rust tool parser were omitted because the
  container has no Rust compiler. The Python LLM engine imports successfully.
- Original oracle source worktrees retain their original commits and clean status.
  Original runtimes, packages, and source files were not overwritten.

## Executing source chain anchors

Paths in this section are relative to the source directories named earlier.

GGUF plugin:

- `vllm_gguf_plugin/plugin.py:130-148`: registers quantization, loader, and config parser.
- `vllm_gguf_plugin/quantization/config.py:74-94`: selects the linear, embedding, or MoE method.
- `vllm_gguf_plugin/quantization/linear.py:35-58`: chooses GEMV or MMQ by batch size and format.
- `vllm_gguf_plugin/ops.py:38-105`: enables compiled kernels by default and includes Q4_K and Q6_K.
- `vllm_gguf_plugin/ops.py:200-208`: enters `_C_gguf.ggml_mul_mat_a8`.
- `vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu:220-289`: quantizes activations to Q8_1 and dispatches.
- `vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu:272-285`: Q4_K and Q6_K dispatch cases.

vLLM:

- `vllm/model_executor/models/registry.py:204`: causal Qwen3.5 registration.
- `vllm/model_executor/models/registry.py:597-600`: conditional Qwen3.5 registrations.
- `vllm/model_executor/models/qwen3_5.py:145-178`: linear attention, full attention, and MLP construction.
- `vllm/model_executor/models/qwen3_5.py:473-526`: conditional model and its text model construction.

These are source anchors. Matching runtime traces and model results remain the
coordinating task's gates.
