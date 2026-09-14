#!/usr/bin/env python3
"""Seal preparation artifacts; exclude coordinating task's active GPU capture outputs."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

PREP = Path('/home/vikash/oracle/rdna3-wmma-latest')
ADAPTERS = PREP / 'adapters'


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {'argv': argv, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


files = []
for pattern in ('*.py', '*.cpp', '*.sh'):
    files.extend(ADAPTERS.glob(pattern))
files += [ADAPTERS / name for name in (
    'capture_llama', 'fake_native_bench', 'HANDOFF.md',
    'llama-adapter-build.log', 'llama-adapter-build.exit',
    'llama-input-error-check.json', 'llama-input-error-check.log', 'invalid-prompt-ids.json',
    'tokenization-check.json', 'tokenization-check.prompt-ids.json', 'tokenization-check.log',
    'monitor-cpu-check.jsonl', 'monitor-timeout-check.jsonl', 'monitor-docker-cpu-check.jsonl',
    'aggregate-fixture-primary.json', 'aggregate-fixture-llama.json',
    'aggregate-fixture-mismatch-primary.json', 'aggregate-cpu-check.json',
    'primary-model.json', 'prompt-ids.json')]
for folder in ('native-ab-cpu-check', 'native-ab-mismatch-check',
               'native-ab-oracle-mismatch-check', 'native-ab-busy-check',
               'native-trace-on-preparation', 'native-trace-off-preparation',
               'fake-sysfs', 'busy-sysfs'):
    files.extend(path for path in (ADAPTERS / folder).rglob('*') if path.is_file())
files += [PREP / name for name in (
    'python-gpu-under-lock.sh', 'python-no-gpu.sh',
    'logs/build-provenance.json', 'logs/import-provenance.json',
    'logs/plugin-source-artifact-verification.json',
    'qwen35-4b-text/config.json', 'qwen35-4b-text/tokenizer.json',
    'qwen35-4b-text/tokenizer_config.json', 'qwen35-4b-text/workload.json')]
manifest = {
    'status': 'COMPLETE_PREPARATION',
    'sealed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'external scratch measurement adapters and completed CPU checks',
    'gpu_workloads_executed_by_adapter_author': False,
    'first_primary_failure': 'coordinator execution; retained separately; no generated tokens',
    'active_gpu_outputs_excluded': ['primary-model-retry*', 'llama-model*', 'native-ab-model*', 'oracle-whole-workload.json'],
    'files': [identity(path) for path in sorted(set(files))],
    'upstream_worktrees': {},
    'llama_adapter_compiler': command(['/opt/rocm/llvm/bin/clang++', '--version']),
    'llama_adapter_dynamic_dependencies': command(['ldd', str(ADAPTERS / 'capture_llama')]),
    'fake_benchmark_compile_argv': ['/usr/bin/c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                  str(ADAPTERS / 'fake_native_bench.cpp'), '-o', str(ADAPTERS / 'fake_native_bench')],
    'check_expectations': {
        'tokenization-check.json': 'TOKENIZED_ONLY; prompt counts [183,174]',
        'llama-adapter-build.exit': '0',
        'llama-input-error-check.json': 'ERROR input_validation; noninteger prompt refused before backend initialization',
        'monitor-cpu-check.jsonl': 'COMPLETE; process RSS/PSS sampled using synthetic GPU sysfs files',
        'monitor-timeout-check.jsonl': 'TIMEOUT; own CPU child removed; monitor exit 124',
        'monitor-docker-cpu-check.jsonl': 'COMPLETE; actual container host PID distinct from CLI PID; no devices exposed',
        'native-ab-cpu-check/summary.json': 'COMPLETE; six same-output CPU processes in fixed order',
        'native-ab-mismatch-check/summary.json': 'TOKEN_MISMATCH; off-arm mutation detected; exit 1',
        'native-ab-busy-check/summary.json': 'ERROR; synthetic busy=3 rejected before any process launch; exit 1',
        'native-ab-oracle-mismatch-check/summary.json': 'TOKEN_MISMATCH; same-binary PASS and numeric ratios retained; cross-oracle FAIL; exit 1',
        'aggregate-cpu-check.json': 'COMPLETE; independent expected duration 10s, 8 requests, 1428 inputs, 128 outputs, 142.8 input tok/s and 12.8 output tok/s',
        'native-trace-on-preparation/recipe-and-result.json': 'PREPARED; exact rocprofv3 on argv; no GPU workload launched',
        'native-trace-off-preparation/recipe-and-result.json': 'PREPARED; exact rocprofv3 off argv; no GPU workload launched',
    },
}
for folder in ('vllm-source', 'llama-source', 'plugin-source'):
    manifest['upstream_worktrees'][folder] = {
        'revision': command(['git', '-C', str(PREP / folder), 'rev-parse', 'HEAD']),
        'status': command(['git', '-C', str(PREP / folder), 'status', '--short'])}
output = ADAPTERS / 'adapter-manifest.json'
output.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(identity(output)))
