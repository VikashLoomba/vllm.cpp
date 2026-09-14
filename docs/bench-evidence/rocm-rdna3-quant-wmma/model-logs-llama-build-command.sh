#!/usr/bin/env bash
set -euo pipefail
prep=/home/vikash/oracle/rdna3-wmma-latest
cmake -S "$prep/llama-source" -B "$prep/llama-build" -G Ninja \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build "$prep/llama-build" --parallel 4 --target llama-cli llama-bench llama-server
