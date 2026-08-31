#!/usr/bin/env bash
# MODEL-TEXT-GLM-MOE-DSA -- drive the FIRST LOAD and the FIRST TOKEN of
# GLM-5.3 (`glm-dsa`, non-Flash) against the real 201.83 GiB UD-IQ1_S artifact,
# WITH EXPERT STREAMING. Spec `.agents/specs/glm-dsa-latest-deepseek.md`
# §3.3/§3.4/§3.6/§3.7 W7+W9, owed items O7, O9, O14, O15, O17, O23, O29.
# Issue https://github.com/mudler/vllm.cpp/issues/2214.
#
# NOT `set -e`. A refusal is a RESULT here; losing the log to an early exit is
# how a lease gets spent for nothing. Every step captures its own rc, and never
# `$?` after a pipe.
set -u

W=/workspace/glm53-firstload
CKPT=${CKPT:-/workspace/ckpt/GLM-5.3-UD-IQ1_S}
SHARD1=$CKPT/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf
SRC=/tmp/glm53fl/src
BUILD=/tmp/glm53fl/build-cuda
BOX=$(hostname)
OUT=$W/out/$BOX
PROMPT=${PROMPT:-The capital of France is}
MAX_TOKENS=${MAX_TOKENS:-1}
SLOTS=${SLOTS:-4096}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

say(){ printf '\n\n========== %s ==========  %s\n' "$1" "$(date -u +%FT%TZ)"; }
note(){ echo "### RC $1=$2"; }
stamp(){ [ -f "$OUT/stamp.$1" ]; }
mark(){ : > "$OUT/stamp.$1"; }

say "IDENTITY -- WHICH BOX EVERY NUMBER BELOW CAME FROM"
hostname; id -un; uname -m
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv 2>&1 | head -4
echo "cores: $(grep -c ^processor /proc/cpuinfo)"
free -g 2>&1 | head -2

say "WORKSPACE MOUNT GUARD -- a failed CIFS mount is an EMPTY LOCAL DIR and this job would 'succeed' writing nowhere"
if [ ! -s "$W/SENTINEL" ]; then
  echo "FATAL: /workspace is not the NAS -- no $W/SENTINEL"; exit 90
fi
sha256sum "$W/SENTINEL"
echo "EXPECTED_SENTINEL 0043989e9b51063ca0f110887f69bbd7005381ea66d30b22a1c3f33ed5f3becd (informational; the -s test above is the guard)"
df -h /workspace 2>&1 | tail -2

say "DISK BEFORE"
df -h / /tmp /workspace 2>&1 | tail -6

say "THE ARTIFACT -- present, complete, and the exact byte count"
TOTAL_EXPECT=216715365893
if [ ! -d "$CKPT" ]; then echo "FATAL: no checkpoint directory at $CKPT"; exit 91; fi
ls -la "$CKPT"
n=$(ls -1 "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf 2>/dev/null | wc -l)
echo "shards found: $n"
if [ "$n" -ne 6 ]; then echo "FATAL: $n of 6 shards; a partial set is not a model"; exit 91; fi
total=$(du -bc "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf | tail -1 | cut -f1)
echo "total bytes: $total (expected $TOTAL_EXPECT)"
if [ "$total" -ne "$TOTAL_EXPECT" ]; then
  echo "FATAL: byte count disagrees; a short shard reads as a corrupt model"; exit 91
fi
echo "### ARTIFACT OK"
# The six sha256 values were recorded by the fetch script as each shard landed
# (FETCH_FAIL=0) and are re-verified OUTSIDE this lease over the same CIFS
# mount; re-reading 201.83 GiB here would spend the lease on IO, so they are
# TRANSCRIBED and the byte count above is what this job checks.
cat <<'HASHES'
### RECORDED SHARD SHA256 (unsloth/GLM-5.3-GGUF @ 346b3591c7f28d1a23716f97a065ecf12ec14771, UD-IQ1_S)
00001 ff3adab0853dfb00bdf3889ec3f5556196f56b65783115720d57767bbd760dd9
00002 659d04cf4fc0b6026944f34c0b590a635803bff06c1775361e28490db7b168f8
00003 433302bac0e2d54da64c7c2f28509fa1b235aeccdf5b215a8a446ebaad1b5b27
00004 d0a6f19452d5b5cd498e1eb8fbe856e00aed7da1f80c27c095301eabe81e9bc1
00005 2ea1537ffab40fa8b8584a8647ec10fbaa6199dfed45e4019b822da2b319db37
00006 42a76ef04ffc5e321e1240f4e572b6fa6fc3315da5bea22fb598d7460db210fe
HASHES

say "TOOLCHAIN -- unconditional, the container is reused between jobs"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg git cmake ninja-build binutils >/dev/null 2>&1
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
  dpkg -i /tmp/ck.deb >/dev/null 2>&1
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-13-0
fi
export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; exit 90; }
nvcc --version | tail -2

say "SOURCE -- from the staged archive; no network dependency, no --prefix"
if [ ! -s "$W/src.tar.gz" ]; then echo "FATAL: no $W/src.tar.gz"; exit 92; fi
sha256sum "$W/src.tar.gz"
echo "EXPECTED_SRC_SHA256 $(cat "$W/src.tar.gz.sha256" 2>/dev/null)"
if ! stamp src; then
  rm -rf "$SRC"; mkdir -p "$SRC"
  tar -xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: extract failed"; exit 92; }
  test -f "$SRC/CMakeLists.txt" || { echo "FATAL: no CMakeLists.txt at the archive root -- it was made with a --prefix"; exit 92; }
  mark src
fi
echo "BASE_SHA (recorded at archive time): $(cat "$W/BASE_SHA" 2>/dev/null)"

say "CUDA ARCH -- READ off the device, never assumed"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
case "$CC" in
  121) ARCH=121a ;;   # GB10 (dgx)
  110) ARCH=110a ;;   # thor
  "")  ARCH=121a; echo "### WARNING: compute_cap unreadable, defaulting to $ARCH" ;;
  *)   ARCH=$CC ;;
esac
echo "### DEVICE compute_cap=$CC -> CUDA arch $ARCH"

say "CONFIGURE"
if ! stamp cfg; then
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
        -DVLLM_CPP_TRITON=OFF > "$OUT/cmake.log" 2>&1
  rc=$?; note CONFIGURE $rc
  tail -20 "$OUT/cmake.log"
  [ "$rc" -ne 0 ] && { echo "FATAL: configure failed"; exit 93; }
  mark cfg
fi
grep -aE '^(VLLM_CPP_CUDA|VLLM_CPP_CUDA_ARCHITECTURES|VLLM_CPP_TRITON|CMAKE_BUILD_TYPE):' "$BUILD/CMakeCache.txt"

say "BUILD vllm-cli -- -j 4, because unconstrained parallelism has OOM-REBOOTED this box"
if ! stamp build; then
  cmake --build "$BUILD" -j 4 --target vllm-cli > "$OUT/build.log" 2>&1
  rc=$?; note BUILD $rc
  if [ "$rc" -ne 0 ]; then tail -60 "$OUT/build.log"; echo "FATAL: build failed"; exit 94; fi
  mark build
fi
tail -3 "$OUT/build.log"
df -h / /tmp | tail -3

say "BINARY IDENTITY GUARD -- the executable AND every .so beside it"
CLI=$(find "$BUILD" -maxdepth 3 -name vllm-cli -type f -perm -u+x | head -1)
[ -n "$CLI" ] || { echo "FATAL: no vllm-cli under $BUILD"; exit 95; }
echo "CLI=$CLI"
found=no
# `strings` when binutils is present, `grep -a` on the raw bytes otherwise: a
# MISSING TOOL must not read as a missing symbol (a wrapper that is not there
# makes the wrapped command silently not run).
scan(){ if command -v strings >/dev/null 2>&1; then strings -a "$1" 2>/dev/null; else cat "$1"; fi; }
for obj in "$CLI" "$(dirname "$CLI")"/*.so* "$BUILD"/*.so*; do
  [ -e "$obj" ] || continue
  [ -d "$obj" ] && continue
  if scan "$obj" | grep -a -q 'GlmMoeDsaForCausalLM'; then
    echo "identity OK: GlmMoeDsaForCausalLM present in $(basename "$obj")"; found=yes
  fi
done
[ "$found" = yes ] || { echo "FATAL: GlmMoeDsaForCausalLM is in neither vllm-cli nor any .so beside it"; exit 95; }
sha256sum "$CLI" | tee "$OUT/binary.sha256"
find "$BUILD" -maxdepth 3 -name '*.so*' -type f -print0 | sort -z | xargs -0 -r sha256sum | tee -a "$OUT/binary.sha256"

say "SPEC O17 -- THE PUBLISHED FILE STATES NO INDEXER SCHEDULE; the repair is to the FILE"
# `unsloth/GLM-5.3-GGUF` writes no `glm-dsa.attention.indexer.types`, so the
# loader refuses it BY NAME (spec D3: a hardcoded 78-entry table that happens to
# be right is what silently becomes wrong on GLM-5.4). The schedule is
# TRANSCRIBED from `zai-org/GLM-5.3`'s own config.json, which is committed
# verbatim in-tree; nothing is derived, defaulted or guessed. The result is a
# DERIVED artifact with its own sha256 and is NOT unsloth's shard 1.
DERIVED=$W/derived
if ! stamp derived; then
  mkdir -p "$DERIVED"
  python3 - "$SRC/tests/vllm/models/glm_moe_dsa_config_glm53.inc" "$OUT/GLM-5.3-config.json" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = s.index('R"GLM53(') + len('R"GLM53(')
b = s.index(')GLM53"')
open(sys.argv[2], 'w').write(s[a:b] + "\n")
PY
  echo "config.json extracted: $(wc -c < "$OUT/GLM-5.3-config.json") bytes  sha256=$(sha256sum "$OUT/GLM-5.3-config.json" | cut -d' ' -f1)"
  # Hard-link the five PAYLOAD shards (hardlinks work on this CIFS mount; it
  # holds no symlink). Only the 9.4 MB metadata shard is rewritten.
  for f in "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf; do
    b=$(basename "$f")
    [ "$b" = "GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" ] && continue
    ln -f "$f" "$DERIVED/$b" || { echo "FATAL: cannot hardlink $b into $DERIVED"; exit 96; }
  done
  python3 "$SRC/scripts/glm-dsa-write-indexer-types.py" \
      --shard "$SHARD1" --from-config "$OUT/GLM-5.3-config.json" \
      --out "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" --force > "$OUT/indexer_types.log" 2>&1
  rc=$?; note INDEXER_REPAIR $rc
  cat "$OUT/indexer_types.log"
  [ "$rc" -ne 0 ] && { echo "FATAL: the metadata repair refused"; exit 96; }
  mark derived
fi
ls -la "$DERIVED"
echo "### PUBLISHED shard 1 sha256:"; sha256sum "$SHARD1"
echo "### DERIVED  shard 1 sha256 (NOT unsloth/GLM-5.3-GGUF's shard 1):"; sha256sum "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf"

say "LEG 1 -- THE STREAMED LOAD AND THE FIRST TOKEN, on --device cuda"
# `--device cuda` is REQUIRED for the streaming lane: model_loader.cpp builds it
# only under `needs_weight_staging() && host_memory_is_device_addressable()`,
# which is a GB10 and false on every CPU. On --device cpu the 228 towers would
# be read IN PLACE out of a 201.83 GiB CIFS-backed mmap, which is a page-cache
# number wearing a streaming label (spec §3.3).
export VT_MOE_EXPERT_STREAM=1
export VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS
export VT_MOE_EXPERT_STREAM_STATS_EVERY=1
export VT_KV_ALLOC_LOG=1
echo "VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS VT_MOE_EXPERT_STREAM_STATS_EVERY=1 VT_KV_ALLOC_LOG=1"
echo "prompt: [$PROMPT]  max-tokens: $MAX_TOKENS"
echo "NOTE: the reads are served from CIFS (/workspace is //192.168.68.102/Data), NOT local NVMe."
echo "      Spec §3.7 W7's stop condition binds: any pread number below is a CIFS number (O7)."

( "$CLI" --model "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" \
         --device cuda --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 \
         > "$OUT/load.stdout" 2> "$OUT/load.stderr" ) &
pid=$!
hwm=0; gpumax=0; t0=$(date +%s)
: > "$OUT/mem.samples"
while kill -0 "$pid" 2>/dev/null; do
  v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
  [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "${g:-}" ] && [ "$g" -gt "$gpumax" ] 2>/dev/null && gpumax=$g
  printf '%s vmhwm_kb=%s gpu_used_mib=%s\n' "$(( $(date +%s) - t0 ))" "${hwm}" "${g:-NA}" >> "$OUT/mem.samples"
  sleep 15
done
wait "$pid"; rc=$?
elapsed=$(( $(date +%s) - t0 ))
note LOAD $rc

say "RESULT"
echo "### LOAD_RC=$rc   wall=${elapsed}s"
awk -v k="$hwm" 'BEGIN{printf "### VmHWM peak = %d kB = %.2f GiB\n", k, k/1048576}'
echo "### nvidia-smi memory.used peak = ${gpumax} MiB"
echo "    VmHWM is NOT a residency measurement while the towers are mmap-resident: it tracks"
echo "    page-cache pressure (spec O9). The DEVICE pool is the number that answers O9."
echo "--- stdout (the emitted text, verbatim) ---"
cat "$OUT/load.stdout"
echo "--- stderr (head 80) ---"; head -80 "$OUT/load.stderr"
echo "--- stderr (tail 120) ---"; tail -120 "$OUT/load.stderr"
echo "--- expert-stream lines ---"
grep -a '\[expert-stream\]' "$OUT/load.stderr" | head -5
grep -a '\[expert-stream\]' "$OUT/load.stderr" | tail -20
echo "--- device / residency / fit lines ---"
grep -aiE 'kv-alloc|resident|device pool|staged|GiB|fit|arena|slot' "$OUT/load.stderr" | head -80
echo "--- memory samples (first 5, last 5) ---"
head -5 "$OUT/mem.samples"; echo ...; tail -5 "$OUT/mem.samples"

say "LEG 2 -- THE FOCUSED C++ SUITES, BY HAND, WITH THEIR COUNTS"
# Run AFTER the load so a truncated lease still carries the primary result.
# `test_glm_moe_dsa_forward` needs VT_MOE_EXPERT_STREAM=1: StreamRequested() is
# read ONCE into a function-local static, so without it the binary reports 127
# assertions and one failure instead of 5,258 and green.
verdict(){ grep -aE '^\[doctest\] (assertions|test cases)' "$1" | tail -4; }
SUITES="test_glm_moe_dsa_forward test_glm_moe_dsa_gguf_load test_glm_moe_dsa_config test_glm_moe_dsa_schedule test_glm_moe_dsa_gguf_census test_expert_stream_wiring test_expert_stream_capacity"
if ! stamp suites; then
  cmake --build "$BUILD" -j 4 --target $SUITES > "$OUT/build_suites.log" 2>&1
  brc=$?; note BUILD_SUITES $brc
  [ "$brc" -ne 0 ] && tail -40 "$OUT/build_suites.log"
  mark suites
fi
# The census gate is a statement about the PUBLISHED artifact, so it reads the
# published shard 1, not the derived one. And the slot/stats knobs LEG 1 set are
# unset here: a suite run under a non-default budget measures that budget.
unset VT_MOE_EXPERT_STREAM_SLOTS VT_MOE_EXPERT_STREAM_STATS_EVERY
export VT_GLM_DSA_GGUF="$SHARD1"
for t in $SUITES; do
  bin=$(find "$BUILD" -maxdepth 3 -name "$t" -type f -perm -u+x | head -1)
  if [ -z "$bin" ]; then echo "### SUITE $t: NOT BUILT"; continue; fi
  ( cd "$BUILD" && "$bin" ) > "$OUT/$t.log" 2>&1
  trc=$?; echo "### SUITE_RC($t)=$trc"; verdict "$OUT/$t.log"
done

say "DISK AFTER"
df -h / /tmp /workspace | tail -6

{ echo "box=$BOX"; echo "rc=$rc"; echo "wall_s=$elapsed"; echo "vmhwm_kb=$hwm";
  echo "gpu_used_mib_peak=$gpumax"; echo "slots=$SLOTS"; echo "max_tokens=$MAX_TOKENS";
  echo "arch=$ARCH"; } > "$OUT/result.env"
cat "$OUT/result.env"

if [ "$rc" -ne 0 ]; then
  say "THE LOAD DID NOT COMPLETE -- the message above is the FINDING and is recorded verbatim"
fi
exit "$rc"
