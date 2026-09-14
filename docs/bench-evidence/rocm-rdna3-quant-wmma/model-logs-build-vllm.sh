#!/usr/bin/env bash
set -euo pipefail
prep=/home/vikash/oracle/rdna3-wmma-latest
docker run --user 1000:1000 --rm --pull=never --network=none --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 -e GIT_NO_LAZY_FETCH=1 \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0="$prep/vllm-source" \
  -e VLLM_TARGET_DEVICE=rocm -e PYTORCH_ROCM_ARCH=gfx1100 \
  -e MAX_JOBS=4 -e CMAKE_BUILD_PARALLEL_LEVEL=4 -e CCACHE_DISABLE=1 -e VLLM_DISABLE_SCCACHE=1 \
  -e TRITON_KERNELS_SRC_DIR=/home/vikash/oracle/gfx1100-active-2773/deps/triton/python/triton_kernels/triton_kernels \
  -e VLLM_USE_PRECOMPILED=0 -e VLLM_USE_PRECOMPILED_RUST=0 \
  -e CARGO_NET_OFFLINE=true -e CARGO_BUILD_JOBS=4 -e CMAKE_BUILD_TYPE=Release \
  -e TMPDIR="$prep/tmp" -e XDG_CACHE_HOME="$prep/cache" \
  -e PIP_NO_INDEX=1 -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e 'CMAKE_ARGS=-DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_HIP_COMPILER_LAUNCHER=' \
  -v "$prep:$prep" \
  -v /home/vikash/.local/bin/uv:/usr/local/bin/uv:ro \
  -v /home/vikash/oracle/vllm-src/.git:/home/vikash/oracle/vllm-src/.git:ro \
  -v /home/vikash/oracle/gfx1100-active-2773/deps:/home/vikash/oracle/gfx1100-active-2773/deps:ro \
  -w "$prep/vllm-source" \
  sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc -c '
set -euo pipefail
test ! -e /dev/kfd
test ! -e /dev/dri
test "$(git rev-parse HEAD)" = 39545e475d3627287ff69c25465dc0bd405f67e1
git diff --exit-code
unset VLLM_VERSION_OVERRIDE SETUPTOOLS_SCM_PRETEND_VERSION VLLM_REQUIRE_RUST_FRONTEND VLLM_USE_RUST_FRONTEND VLLM_USE_RUST_BENCH
export VLLM_VERSION_OVERRIDE=$(/home/vikash/oracle/rdna3-wmma-latest/.venv/bin/python -c "from setuptools_scm import get_version; print(get_version())")
printf "SOURCE_VERSION=%s\n" "$VLLM_VERSION_OVERRIDE"
/home/vikash/oracle/rdna3-wmma-latest/.venv/bin/python setup.py build --build-base /home/vikash/oracle/rdna3-wmma-latest/python-build --build-temp /home/vikash/oracle/rdna3-wmma-latest/native bdist_wheel --dist-dir /home/vikash/oracle/rdna3-wmma-latest/wheel
'
