#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
echo "=== MEMFIT $(date -Is) host=$(hostname) ==="
mountpoint -q /workspace && echo "WORKSPACE_MOUNTED=yes" || { echo "WORKSPACE_MOUNTED=NO -- ABORT"; exit 2; }
free -g; echo
B=/tmp/memfit; mkdir -p $B; cp "${SRC:-/workspace/glm53-rocm-probe/memfit.hip}" $B/memfit.hip || exit 2
echo "source sha256: $(sha256sum $B/memfit.hip)"
echo "--- compile for gfx1151 (FAIL HERE, not at the throw) ---"
hipcc --offload-arch=gfx1151 -O2 -o $B/memfit $B/memfit.hip 2>&1 | tail -20
rc=${PIPESTATUS[0]}
if [ ! -x $B/memfit ]; then echo "COMPILE_FAILED rc=$rc -- ABORT"; exit 3; fi
echo "COMPILE_OK; binary sha256: $(sha256sum $B/memfit)"
echo "--- compiled arch manifest (assert, do not assume) ---"
(roc-obj-ls $B/memfit 2>/dev/null || llvm-objdump --offloading $B/memfit 2>/dev/null || echo "no arch tool") | head -10
echo
echo "--- run ---"
$B/memfit
echo "=== MEMFIT-END rc=$? ==="
free -g
