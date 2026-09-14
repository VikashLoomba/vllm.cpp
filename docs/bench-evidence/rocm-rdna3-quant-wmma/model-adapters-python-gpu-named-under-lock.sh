#!/usr/bin/env bash
# Trusted local worker_memory instrumentation; root must hold /home/vikash/gpu.lock.
set -euo pipefail
prep=/home/vikash/oracle/rdna3-wmma-latest
container_name=${1:?pass a unique container name followed by Python argv}
shift
exec docker run --rm --pull=never --network=none --user 1000:1000 --name "$container_name" \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 \
  --entrypoint "$prep/.venv/bin/python" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e USER=vikash -e LOGNAME=vikash -e TORCHINDUCTOR_CACHE_DIR="$prep/cache/torchinductor" \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -e VLLM_PLUGINS=gguf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 -e VLLM_NO_USAGE_STATS=1 \
  -e XDG_CONFIG_HOME="$prep/config" \
  -e XDG_CACHE_HOME="$prep/cache" -e TMPDIR="$prep/tmp" \
  -e VLLM_CACHE_ROOT="$prep/cache/vllm" -e TRITON_CACHE_DIR="$prep/cache/triton" \
  -v "$prep:$prep" \
  -v /home/vikash/models:/home/vikash/models:ro \
  -v /home/vikash/vllm.cpp:/home/vikash/vllm.cpp:ro \
  -v /home/vikash/vllm.cpp-rdna3-wmma:/home/vikash/vllm.cpp-rdna3-wmma:ro \
  -w "$prep" \
  sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc "$@"
