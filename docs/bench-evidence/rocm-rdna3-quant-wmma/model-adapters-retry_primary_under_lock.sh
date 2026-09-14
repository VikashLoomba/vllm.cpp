#!/usr/bin/env bash
# The root coordinator invokes this only while holding /home/vikash/gpu.lock.
# New output names retain the first primary-model.json failure unchanged.
set -euo pipefail
prep=/home/vikash/oracle/rdna3-wmma-latest
exec /home/vikash/.venv/bin/python "$prep/adapters/monitor_command.py" \
  --output-jsonl "$prep/adapters/primary-model-retry-monitor.jsonl" \
  --timeout-seconds 600 --docker-name rdna3-wmma-primary-retry \
  -- "$prep/adapters/python-gpu-named-under-lock.sh" rdna3-wmma-primary-retry \
  "$prep/adapters/capture_primary.py" \
  --output "$prep/adapters/primary-model-retry.json" \
  --prompt-ids-output "$prep/adapters/prompt-ids-retry.json"
