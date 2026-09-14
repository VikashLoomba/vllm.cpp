#!/usr/bin/env python3
"""Run the fixed six-process WMMA A/B under the caller's GPU mutex."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

PREP = Path('/home/vikash/oracle/rdna3-wmma-latest')
REPO = Path('/home/vikash/vllm.cpp-rdna3-wmma')
ORDER = ('on1', 'off1', 'off2', 'on2', 'on3', 'off3')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def stat_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def file_identity(path):
    path = Path(path).absolute()
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'sha256': sha256(path), 'stat': stat_identity(path)}


def write_json(path, report):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def command_output(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=30)
    return {'argv': argv, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def linked_libraries(binary):
    result = command_output(['ldd', str(binary)])
    if 'not found' in result['stdout']:
        raise RuntimeError('native benchmark has unresolved shared libraries: ' + result['stdout'])
    if result['exit_code'] and 'not a dynamic executable' not in result['stderr'] + result['stdout']:
        raise RuntimeError('ldd failed: ' + result['stderr'])
    paths = sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)\s+\(', result['stdout'], re.MULTILINE)))
    return {'ldd': result, 'files': [file_identity(path) for path in paths],
            'scope': 'ELF dependencies resolved by ldd; runtime-loaded kernel objects remain in runtime trace evidence'}


def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        match = re.match(r'^\s*([^:]+):\s+(-?\d+(?:\.\d+)?)\s*$', line)
        if match:
            metrics[match.group(1).strip()] = float(match.group(2))
    for name, expected in (('Successful requests', 8), ('Total input tokens', 1428),
                           ('Total generated tokens', 128), ('Maximum request concurrency', 1),
                           ('Ignore EOS (resolved sampling)', 1)):
        if metrics.get(name) != expected:
            raise RuntimeError(f'native metric {name!r} is {metrics.get(name)!r}; expected {expected}')
    return metrics


def validate_ids(rows):
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError('expected eight generated-ID arrays')
    if any(not isinstance(row, list) or len(row) != 16 or
           any(type(token) is not int or token < 0 for token in row) for row in rows):
        raise RuntimeError('each generated-ID array must contain 16 nonnegative integer tokens')
    return rows


def oracle_comparison(path, kind, reference):
    if not path.exists():
        return {'path': str(path), 'status': 'UNAVAILABLE', 'reason': 'capture file absent'}
    capture = json.loads(path.read_text())
    result = {'path': str(path), 'sha256': sha256(path), 'capture_status': capture.get('status')}
    if capture.get('status') != 'COMPLETE':
        return dict(result, status='UNAVAILABLE', reason='capture did not complete')
    if len(capture.get('legs', [])) != 4:
        raise RuntimeError(f'{kind} complete capture does not have four legs')
    if kind == 'primary':
        rows = [row for leg in capture['legs'] for row in leg['generated_token_ids']]
    else:
        rows = [request['generated_token_ids'] for leg in capture['legs'] for request in leg['requests']]
    rows = validate_ids(rows)
    mismatches = [{'request': index, 'native': left, 'oracle': right}
                  for index, (left, right) in enumerate(zip(reference, rows)) if left != right]
    return dict(result, status='MATCH' if not mismatches else 'MISMATCH',
                generated_token_ids=rows, mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=REPO / 'build-rdna3-wmma-operator/examples/vllm-bench')
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--dataset', type=Path, default=PREP / 'qwen35-4b-text/workload.json')
    parser.add_argument('--model', type=Path, default=PREP / 'qwen35-4b-text')
    parser.add_argument('--sysfs-device', type=Path, default=Path('/sys/class/drm/card1/device'))
    parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--primary-report', type=Path, default=PREP / 'adapters/primary-model-retry.json')
    parser.add_argument('--llama-report', type=Path, default=PREP / 'adapters/llama-model.json')
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 43200:
        parser.error('timeout must be positive and at most 43200 seconds')
    args.output_dir = args.output_dir.absolute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'summary.json'
    report = {'status': 'RUNNING', 'phase': 'preparation', 'argv': sys.argv,
              'adapter_sha256': sha256(__file__), 'order': ORDER, 'runs': [],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    try:
        dataset = json.loads(args.dataset.read_text())
        if not isinstance(dataset, list) or len(dataset) != 2:
            raise RuntimeError('original dataset must contain exactly two records')
        expanded = args.output_dir / 'workload-eight.json'
        write_json(expanded, dataset * 4)
        report['datasets'] = {'original': file_identity(args.dataset),
                              'expanded': file_identity(expanded),
                              'original_record_index_for_request': [0, 1] * 4,
                              'transformation': 'exact original records repeated four times in original order'}
        binary = args.binary.absolute()
        if binary.read_bytes()[:4] != b'\x7fELF':
            raise RuntimeError('benchmark must be an ELF executable')
        report['binary'] = file_identity(binary)
        report['shared_libraries'] = linked_libraries(binary)
        report['revision'] = command_output(['git', 'rev-parse', 'HEAD'], args.repo)
        report['worktree_status'] = command_output(['git', 'status', '--short'], args.repo)
        report['model_path'] = str(args.model.absolute())
        if args.model.is_dir():
            model_inputs = [args.model / name for name in
                            ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.gguf')]
        else:
            model_inputs = [args.model]
        report['model_inputs'] = [file_identity(path) for path in model_inputs if path.exists()]
        report['inherited_runtime_environment'] = {
            key: value for key, value in os.environ.items()
            if key.startswith(('VT_', 'HIP_', 'ROCR_', 'HSA_', 'ROCBLAS_', 'HIPBLASLT_', 'OMP_'))
            or key in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
        report['timing_limits'] = [
            'Each process loads one model and benchmarks eight requests: the original two prompts repeated four times.',
            'Each process includes its own initialization and first-use effects; no warm-only native leg is inferred.',
            'Native metrics aggregate all eight requests. Oracle comparison uses all four two-request legs; retain their cold/warm breakdown.',
            'External process wall time includes monitor polling and process startup/teardown. Subtracting benchmark duration is not an isolated startup-latency measurement.',
            'CPU ps percentages are process-lifetime averages. The operator checks contention; the driver records these snapshots.',
            'Same-binary A/B numerical ratios do not establish a cross-oracle performance floor or trace parity.']
        monitored_files = [report['binary'], report['datasets']['original'], report['datasets']['expanded']]
        monitored_files += report['shared_libraries']['files'] + report['model_inputs']
        monitor = PREP / 'adapters/monitor_command.py'
        report['monitor'] = file_identity(monitor)
        write_json(report_path, report)
        reference = None
        for label in ORDER:
            report['phase'] = label
            for identity in monitored_files:
                if stat_identity(identity['path']) != identity['stat']:
                    raise RuntimeError('input or binary changed during A/B: ' + identity['path'])
            if sha256(binary) != report['binary']['sha256']:
                raise RuntimeError('native binary hash changed before ' + label)
            busy = int((args.sysfs_device / 'gpu_busy_percent').read_text().strip())
            if busy < 0 or busy > 2:
                raise RuntimeError(f'GPU idle gate failed before {label}: gpu_busy_percent={busy}, limit=2')
            ps = command_output(['ps', '-eo', 'pid,ppid,stat,psr,pcpu,pmem,comm', '--sort=-pcpu'])
            ps['stdout'] = '\n'.join(ps['stdout'].splitlines()[:31]) + '\n'
            ps_path = args.output_dir / f'{label}.cpu-contention.json'
            write_json(ps_path, {'ps': ps, 'loadavg': Path('/proc/loadavg').read_text().strip(),
                                 'sampled_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            ids_path = args.output_dir / f'{label}.output-token-ids.json'
            monitor_path = args.output_dir / f'{label}.monitor.jsonl'
            native_argv = [str(binary), '--model', str(args.model.absolute()),
                           '--dataset-path', str(expanded), '--num-prompts', '8',
                           '--input-len', '256', '--output-len', '16', '--concurrency', '1',
                           '--max-num-batched-tokens', '512', '--num-blocks', '64',
                           '--temperature', '0', '--seed', '0', '--skip-chat-template',
                           '--ignore-eos', '--output-token-ids', str(ids_path)]
            arm = 'on' if label.startswith('on') else 'off'
            toggle_argv = ['env', '-u', 'VT_ROCM_QUANT_WMMA'] if arm == 'on' else ['env', 'VT_ROCM_QUANT_WMMA=0']
            argv = [sys.executable, str(monitor), '--output-jsonl', str(monitor_path),
                    '--repo', str(args.repo), '--sysfs-device', str(args.sysfs_device),
                    '--interval-seconds', '0.05', '--timeout-seconds', str(args.timeout_seconds),
                    '--'] + toggle_argv + native_argv
            row = {'label': label, 'arm': arm, 'argv': argv, 'native_argv': native_argv,
                   'wmma_environment': None if arm == 'on' else '0', 'gpu_busy_percent_before': busy,
                   'cpu_contention': file_identity(ps_path), 'monitor_path': str(monitor_path)}
            report['runs'].append(row)
            write_json(report_path, report)
            started = time.monotonic()
            completed = subprocess.run(argv, check=False)
            row['external_process_wall_seconds_including_monitor'] = time.monotonic() - started
            row['exit_code'] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f'{label} monitor/benchmark exited {completed.returncode}')
            row['output_token_ids_file'] = file_identity(ids_path)
            row['generated_token_ids'] = validate_ids(json.loads(ids_path.read_text()))
            text = monitor_path.with_suffix('.stdout.log').read_text()
            row['metrics'] = parse_metrics(text)
            row['external_minus_benchmark_duration_seconds'] = (
                row['external_process_wall_seconds_including_monitor'] - row['metrics']['Benchmark duration (s)'])
            if reference is None:
                reference = row['generated_token_ids']
            row['matches_first_process_exactly'] = row['generated_token_ids'] == reference
            write_json(report_path, report)
        report['phase'] = 'verification'
        for identity in monitored_files:
            if stat_identity(identity['path']) != identity['stat'] or sha256(identity['path']) != identity['sha256']:
                raise RuntimeError('input or linked code changed across A/B: ' + identity['path'])
        report['identities_unchanged_after_all_runs'] = True
        report['all_native_output_ids_exact'] = all(row['matches_first_process_exactly'] for row in report['runs'])
        report['oracle_comparisons'] = {
            'primary': oracle_comparison(args.primary_report, 'primary', reference),
            'llama': oracle_comparison(args.llama_report, 'llama', reference)}
        mismatch = (not report['all_native_output_ids_exact'] or
                    any(row['status'] == 'MISMATCH' for row in report['oracle_comparisons'].values()))
        report['same_binary_output_gate'] = 'PASS' if report['all_native_output_ids_exact'] else 'FAIL'
        oracle_states = [row['status'] for row in report['oracle_comparisons'].values()]
        report['cross_oracle_output_gate'] = ('FAIL' if 'MISMATCH' in oracle_states else
                                             'PENDING' if 'UNAVAILABLE' in oracle_states else 'PASS')
        if report['all_native_output_ids_exact']:
            report['same_binary_comparison_scope'] = (
                'All six native generated-ID arrays match exactly; these A/B values remain descriptive when '
                'cross-oracle output parity fails or remains pending. They do not establish cross-oracle correctness.')
            names = sorted(set.intersection(*(set(row['metrics']) for row in report['runs'])))
            report['same_binary_numeric_comparison'] = {}
            for name in names:
                on = statistics.median(row['metrics'][name] for row in report['runs'] if row['arm'] == 'on')
                off = statistics.median(row['metrics'][name] for row in report['runs'] if row['arm'] == 'off')
                report['same_binary_numeric_comparison'][name] = {
                    'on_median': on, 'off_median': off, 'on_over_off': on / off if off else None}
        report.update(status='TOKEN_MISMATCH' if mismatch else 'COMPLETE', phase='complete')
        write_json(report_path, report)
        print(json.dumps({'status': report['status'], 'summary': str(report_path)}))
        return 1 if mismatch else 0
    except Exception as exc:
        report.update(status='ERROR', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        write_json(report_path, report)
        print(json.dumps({'status': 'ERROR', 'phase': report['phase'], 'error': str(exc),
                          'summary': str(report_path)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
