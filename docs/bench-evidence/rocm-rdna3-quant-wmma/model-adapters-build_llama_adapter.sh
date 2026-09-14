#!/usr/bin/env bash
set -euo pipefail
prep=/home/vikash/oracle/rdna3-wmma-latest
/opt/rocm/llvm/bin/clang++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$prep/llama-source/include" -I"$prep/llama-source/ggml/include" \
  -I"$prep/llama-source/vendor" "$prep/adapters/capture_llama.cpp" \
  -L"$prep/llama-build/bin" -Wl,-rpath,"$prep/llama-build/bin" \
  -lllama -lggml -lggml-base -lcrypto -pthread \
  -o "$prep/adapters/capture_llama"
