#!/usr/bin/env bash
# HEADPIN job B (corrected): requirements/build is a DIRECTORY, not build.txt.
# HEADPIN: does e126687a9a (the qwen4_exp landing commit) install and import on
# aarch64 once python3-dev is present?  Every leg reports its own literal rc.
set -u
TARGET=e126687a9a828d513c01a07cd69f025f27d63280
ROOT=/tmp/headpin-e126687-b
rm -rf "$ROOT"; mkdir -p "$ROOT"; cd "$ROOT" || exit 80

echo "=== HEADPIN t=0 $(date -u +%FT%TZ) ==="
echo "ARCH=$(uname -m) HOST=$(hostname) NPROC=$(nproc)"
echo "TARGET_WANTED=$TARGET"
df -h /tmp | sed 's/^/DF_BEFORE /'
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) t=${SECONDS}s"; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

echo "--- python3 baseline ---"
python3 --version 2>&1 | sed 's/^/PY /'
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)
echo "PYINC=$PYINC"
if [ -f "$PYINC/Python.h" ]; then echo "PYTHON_H_BEFORE=present"; else echo "PYTHON_H_BEFORE=absent"; fi

echo "--- APTDEV t=${SECONDS}s ---"
apt-get update -qq > apt.log 2>&1; APTUPD_RC=$?
echo "APTUPD_RC=$APTUPD_RC"
apt-get install -y -qq python3-dev python3-venv build-essential >> apt.log 2>&1; APTDEV_RC=$?
echo "APTDEV_RC=$APTDEV_RC"
tail -4 apt.log | sed 's/^/APT /'
if [ -f "$PYINC/Python.h" ]; then echo "PYTHON_H_AFTER=present"; else echo "PYTHON_H_AFTER=absent"; fi

echo "--- CLONE t=${SECONDS}s ---"
GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c http.version=HTTP/1.1 \
  clone -q https://github.com/vllm-project/vllm.git src > clone.log 2>&1
CLONE_RC=$?
echo "CLONE_RC=$CLONE_RC"
tail -3 clone.log | sed 's/^/CLONE /'
[ $CLONE_RC -eq 0 ] || { echo "FATAL: clone failed"; exit 81; }
cd "$ROOT/src" || exit 82
git checkout -q "$TARGET" > co.log 2>&1; CO_RC=$?
echo "CHECKOUT_RC=$CO_RC"; tail -3 co.log | sed 's/^/CO /'
HEAD_SHA=$(git rev-parse HEAD)
echo "HEAD_SHA=$HEAD_SHA"
[ "$HEAD_SHA" = "$TARGET" ] || { echo "TARGET MISMATCH want=$TARGET -- ABORT"; exit 9; }
echo "TARGET CONFIRMED $TARGET"
echo "SHALLOW=$(git rev-parse --is-shallow-repository)"
echo "REVCOUNT=$(git rev-list --count HEAD)"
echo "GIT_DESCRIBE=$(git describe --tags 2>&1)"
echo "INSTANTTENSOR_IN_RUNTIME=$(grep -c instanttensor requirements/cuda.txt)"
grep -n instanttensor requirements/cuda.txt | sed 's/^/ITLINE /'
echo "QWEN4EXP_REGISTRY_LINES=$(grep -c Qwen4Exp vllm/model_executor/models/registry.py)"
echo "QWEN4EXP_FILES=$(git ls-files 'vllm/models/qwen4_exp/*' | wc -l)"

echo "--- VENV t=${SECONDS}s ---"
python3 -m venv "$ROOT/venv" > venv.log 2>&1; VENV_RC=$?
echo "VENV_RC=$VENV_RC"; tail -3 venv.log | sed 's/^/VENV /'
[ $VENV_RC -eq 0 ] || { echo "FATAL: venv"; exit 83; }
# shellcheck disable=SC1091
. "$ROOT/venv/bin/activate"
echo "VENV_PY=$(command -v python)"
python -m pip install -q -U pip setuptools wheel setuptools_scm > pipup.log 2>&1; PIPUP_RC=$?
echo "PIPUP_RC=$PIPUP_RC"; tail -3 pipup.log | sed 's/^/PIPUP /'

echo "--- BUILDDEPS t=${SECONDS}s ---"
python -m pip install -r requirements/build/cuda.txt > builddeps.log 2>&1; BUILDDEPS_RC=$?
echo "BUILDDEPS_RC=$BUILDDEPS_RC"
tail -6 builddeps.log | sed 's/^/BUILDDEPS /'

echo "--- RUNDEPS t=${SECONDS}s ---"
python -m pip install -r requirements/cuda.txt > rundeps.log 2>&1; RUNDEPS_RC=$?
echo "RUNDEPS_RC=$RUNDEPS_RC"
tail -25 rundeps.log | sed 's/^/RUNDEPS /'
echo "--- instanttensor as installed ---"
python -m pip show instanttensor 2>&1 | head -4 | sed 's/^/ITSHOW /'
grep -iE 'instanttensor' rundeps.log | tail -8 | sed 's/^/ITLOG /'
python -m pip list 2>/dev/null | grep -iE 'torch|transformers|flashinfer|cutlass|quack|humming|tilelang|tvm-ffi|fastsafetensors|instanttensor|nvcc' | sed 's/^/PIPLIST /'

echo "--- NVCC FROM PIP (resolves #2589 7.4, which guessed a path) t=${SECONDS}s ---"
NVCC_PIP=$(find "$ROOT/venv" -type f -name nvcc 2>/dev/null | head -1)
echo "NVCC_FROM_PIP=${NVCC_PIP:-absent}"
[ -n "$NVCC_PIP" ] && "$NVCC_PIP" --version 2>&1 | tail -2 | sed 's/^/NVCCV /'

echo "--- BUILD t=${SECONDS}s ---"
export VLLM_USE_PRECOMPILED=1
python -m pip install -e . --no-build-isolation > build.log 2>&1; BUILD_RC=$?
echo "BUILD_RC=$BUILD_RC"
tail -25 build.log | sed 's/^/BUILD /'
echo "SCM_VERSION=$(python -c 'import setuptools_scm;print(setuptools_scm.get_version())' 2>&1 | tail -1)"

echo "--- IMPORT t=${SECONDS}s (from / so it reads the installed package) ---"
cd / || exit 84
python -c 'import vllm; print("VLLM_VERSION="+vllm.__version__)' > import.log 2>&1; IMPORT_RC=$?
echo "IMPORT_RC=$IMPORT_RC"
cat import.log | sed 's/^/IMPORT /'
python -c 'import importlib.metadata as m; print("DIST_VERSION="+m.version("vllm"))' 2>&1 | sed 's/^/DIST /'

echo "--- EXT t=${SECONDS}s ---"
python -c 'import vllm._C; print("EXT ok")' > ext.log 2>&1; EXT_RC=$?
echo "EXT_RC=$EXT_RC"
tail -3 ext.log | sed 's/^/EXT /'

echo "--- QWEN4EXP REGISTRATION t=${SECONDS}s ---"
python - > reg.log 2>&1 <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
print("SUPPORTED_ARCH_COUNT=%d" % len(archs))
for a in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP"):
    print("REG_%s=%s" % (a, a in archs))
PY
REG_RC=$?
echo "REG_RC=$REG_RC"
cat reg.log | sed 's/^/REG /'

echo "--- QWEN4EXP MODULE IMPORT t=${SECONDS}s ---"
python - > mod.log 2>&1 <<'PY'
import importlib
m = importlib.import_module("vllm.models.qwen4_exp")
print("MODULE_FILE=%s" % m.__file__)
for a in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP"):
    print("CLASS_%s=%s" % (a, getattr(m, a, None)))
PY
MODIMPORT_RC=$?
echo "MODIMPORT_RC=$MODIMPORT_RC"
tail -30 mod.log | sed 's/^/MOD /'

echo "--- DEVICE (records what this box can never answer) t=${SECONDS}s ---"
python -c 'import torch;print("CUDA_AVAIL=%s"%torch.cuda.is_available());print("TORCH=%s"%torch.__version__)' 2>&1 | tail -3 | sed 's/^/DEV /'
nvidia-smi --query-gpu=name,driver_version --format=csv > smi.log 2>&1; echo "SMI_RC=$?"
head -3 smi.log | sed 's/^/SMI /'

echo "--- SUMMARY ---"
echo "SUM APTDEV_RC=$APTDEV_RC BUILDDEPS_RC=$BUILDDEPS_RC RUNDEPS_RC=$RUNDEPS_RC BUILD_RC=$BUILD_RC IMPORT_RC=$IMPORT_RC EXT_RC=$EXT_RC"
df -h /tmp | sed 's/^/DF_AFTER /'
echo "=== HEADPIN DONE t=${SECONDS}s $(date -u +%FT%TZ) ==="
