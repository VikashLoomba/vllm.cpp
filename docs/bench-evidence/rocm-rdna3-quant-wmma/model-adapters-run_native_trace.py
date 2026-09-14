#!/usr/bin/env python3
"""Prepare or execute one bounded native rocprofv3 trace under the caller's GPU mutex."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_native_ab import PREP, REPO, file_identity, command_output, parse_metrics, validate_ids, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('on', 'off'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='write exact argv and input identities without reading GPU activity or starting the trace')
    args = parser.parse_args()
    args.output_dir = args.output_dir.absolute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'recipe-and-result.json'
    report = {'status': 'PREPARING', 'arm': args.arm, 'argv': sys.argv,
              'adapter': file_identity(Path(__file__)), 'phase': 'preparation'}
    try:
        source_dataset = PREP / 'qwen35-4b-text/workload.json'
        records = json.loads(source_dataset.read_text())
        if not isinstance(records, list) or len(records) != 2:
            raise ValueError('original workload must contain exactly two records')
        expanded = args.output_dir / 'workload-eight.json'
        write_json(expanded, records * 4)
        binary = REPO / 'build-rdna3-wmma-operator/examples/vllm-bench'
        model = Path('/home/vikash/models/Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf')
        model_identity = file_identity(model)
        if model_identity['sha256'] != '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4':
            raise RuntimeError('retained GGUF hash mismatch')
        profiler = Path('/opt/rocm/bin/rocprofv3')
        trace_tmp = PREP / 'cache/rocprof' / args.output_dir.name
        trace_tmp.mkdir(parents=True, exist_ok=True)
        if str(trace_tmp).startswith('/dev/shm') or not os.access(trace_tmp, os.W_OK):
            raise RuntimeError('profiler temporary directory must be writable task storage')
        profiler_outputs = args.output_dir / 'rocprof'
        profiler_outputs.mkdir()
        ids_path = args.output_dir / 'output-token-ids.json'
        toggle = ['env', '-u', 'VT_ROCM_QUANT_WMMA'] if args.arm == 'on' else ['env', 'VT_ROCM_QUANT_WMMA=0']
        traced_argv = toggle + ['HIP_VISIBLE_DEVICES=0', 'ROCR_VISIBLE_DEVICES=0',
            'ROCPROF_TMPDIR=' + str(trace_tmp), 'TMPDIR=' + str(trace_tmp), str(profiler),
            '--hip-trace', '--kernel-trace', '--memory-copy-trace', '--memory-allocation-trace',
            '--marker-trace', '--stats', '--output-config', '--output-format', 'csv', 'json',
            '--output-directory', str(profiler_outputs), '--output-file', 'native-' + args.arm,
            '--', str(binary), '--model', str(model),
            '--dataset-path', str(expanded), '--num-prompts', '8', '--input-len', '256',
            '--output-len', '16', '--concurrency', '1', '--max-num-batched-tokens', '512',
            '--num-blocks', '64', '--temperature', '0', '--seed', '0', '--skip-chat-template',
            '--ignore-eos', '--output-token-ids', str(ids_path)]
        monitor_path = args.output_dir / 'monitor.jsonl'
        monitor_argv = [sys.executable, str(PREP / 'adapters/monitor_command.py'),
                        '--output-jsonl', str(monitor_path), '--interval-seconds', '0.05',
                        '--timeout-seconds', '600', '--'] + traced_argv
        report.update(
            binary=file_identity(binary), model=model_identity, profiler=file_identity(profiler),
            profiler_version=command_output([str(profiler), '--version']),
            revision=command_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']),
            original_dataset=file_identity(source_dataset), expanded_dataset=file_identity(expanded),
            original_record_index_for_request=[0, 1] * 4,
            rocprof_tmpdir=str(trace_tmp), traced_argv=traced_argv, monitor_argv=monitor_argv,
            trace_scope='unfiltered HIP API, kernel dispatch, memory copy/allocation, and ROCTx traces; no hardware counter or ATT collection',
            interpretation='trace overhead changes timings; use these runs to identify executing paths, not as the unprofiled performance denominator')
        report['status'] = 'PREPARED'
        write_json(report_path, report)
        if args.prepare_only:
            print(json.dumps({'status': 'PREPARED', 'recipe': str(report_path), 'gpu_executed': False}))
            return 0
        report['phase'] = 'idle_gate'
        busy = int(Path('/sys/class/drm/card1/device/gpu_busy_percent').read_text().strip())
        if not 0 <= busy <= 2:
            raise RuntimeError(f'GPU idle gate rejected busy={busy}; limit=2')
        report['gpu_busy_percent_before'] = busy
        report['phase'] = 'trace'
        report['status'] = 'RUNNING'
        write_json(report_path, report)
        result = subprocess.run(monitor_argv, check=False)
        report['exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'trace monitor/application exited {result.returncode}')
        report['generated_token_ids'] = validate_ids(json.loads(ids_path.read_text()))
        report['metrics_with_profiler_overhead'] = parse_metrics(monitor_path.with_suffix('.stdout.log').read_text())
        report['trace_files'] = [file_identity(path) for path in sorted(profiler_outputs.rglob('*')) if path.is_file()]
        if not report['trace_files']:
            raise RuntimeError('rocprofv3 produced no trace files')
        if file_identity(binary) != report['binary'] or file_identity(expanded) != report['expanded_dataset']:
            raise RuntimeError('binary or dataset changed across the trace')
        if file_identity(model) != report['model']:
            raise RuntimeError('retained GGUF changed across the trace')
        report.update(status='COMPLETE', phase='complete')
        write_json(report_path, report)
        print(json.dumps({'status': 'COMPLETE', 'report': str(report_path)}))
        return 0
    except Exception as exc:
        report.update(status='ERROR', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        write_json(report_path, report)
        print(json.dumps({'status': 'ERROR', 'error': str(exc), 'report': str(report_path)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
