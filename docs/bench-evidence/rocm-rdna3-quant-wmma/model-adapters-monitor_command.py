#!/usr/bin/env python3
"""Run one bounded command and sample its process tree and the designated GPU sysfs files."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_snapshot(directory):
    result = {}
    for name in ('mem_info_vram_used', 'mem_info_vram_total', 'gpu_busy_percent', 'pp_dpm_sclk'):
        try:
            text = (directory / name).read_text().strip()
            result[name] = text if name == 'pp_dpm_sclk' else int(text)
        except (OSError, ValueError) as exc:
            result[name] = None
            result.setdefault('read_errors', {})[name] = str(exc)
    result['scope'] = 'whole device; VRAM includes allocations by other processes'
    return result


def inspect_container(name):
    result = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True, timeout=5)
    if result.returncode:
        return None
    rows = json.loads(result.stdout)
    if len(rows) != 1:
        raise RuntimeError('Docker container lookup did not return exactly one container')
    row = rows[0]
    return {'id': row['Id'], 'pid': int(row['State']['Pid']),
            'running': bool(row['State']['Running']), 'created': row['Created']}


def process_snapshot(root_pid, helper):
    if root_pid is None or root_pid <= 0:
        return {'root_pid': root_pid, 'pids': [], 'rss_bytes': None, 'pss_bytes': None,
                'status': 'process_not_available'}
    pids = helper.process_tree(root_pid)
    readings, missing = [], []
    for pid in pids:
        try:
            values = helper.read_process_memory(pid)
            readings.append({'pid': pid, 'rss_bytes': values['rss_kib'] * 1024,
                             'pss_bytes': values['pss_kib'] * 1024})
        except (OSError, ValueError, helper.HarnessError) as exc:
            missing.append({'pid': pid, 'error': str(exc)})
    complete = bool(readings) and not missing
    return {'root_pid': root_pid, 'pids': pids, 'readings': readings, 'missing': missing,
            'rss_bytes': sum(row['rss_bytes'] for row in readings) if complete else None,
            'pss_bytes': sum(row['pss_bytes'] for row in readings) if complete else None,
            'status': 'complete' if complete else 'partial_or_exited',
            'scope': 'sum of repository process_tree descendants; shared mappings can be counted more than once in RSS'}


def stop_owned(process, container):
    if container is not None:
        # This ID was first observed after the command started and its name was absent before launch.
        subprocess.run(['docker', 'stop', '--time', '10', container['id']],
                       text=True, capture_output=True, timeout=20, check=False)
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--stdout-log', type=Path)
    parser.add_argument('--stderr-log', type=Path)
    parser.add_argument('--repo', type=Path, default=Path('/home/vikash/vllm.cpp-rdna3-wmma'))
    parser.add_argument('--sysfs-device', type=Path, default=Path('/sys/class/drm/card1/device'))
    parser.add_argument('--interval-seconds', type=float, default=0.25)
    parser.add_argument('--timeout-seconds', type=float, default=900.0)
    parser.add_argument('--docker-name', help='the unique --name given to the docker run command; samples container host PID')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command or not 0.05 <= args.interval_seconds <= 60 or not 0 < args.timeout_seconds <= 43200:
        parser.error('provide command argv, interval in [0.05,60], and timeout in (0,43200]')
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = args.stdout_log or args.output_jsonl.with_suffix('.stdout.log')
    stderr_path = args.stderr_log or args.output_jsonl.with_suffix('.stderr.log')
    for path in (args.output_jsonl, stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f'refusing to overwrite existing evidence: {path}')
    sys.path.insert(0, str(args.repo))
    from tools.bench import sample_process_memory as helper
    if args.docker_name and inspect_container(args.docker_name) is not None:
        raise RuntimeError(f'container name already exists: {args.docker_name}')
    started = time.monotonic()
    process = None
    container = None
    deadline = started + args.timeout_seconds
    sample_count = 0
    peaks = {'rss_bytes': None, 'pss_bytes': None, 'device_vram_used_bytes': None}
    with args.output_jsonl.open('x') as events, stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        def emit(event):
            event.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         elapsed_seconds=time.monotonic() - started)
            events.write(json.dumps(event, allow_nan=False) + '\n')
            events.flush()
        try:
            emit({'event': 'start', 'argv': args.command, 'monitor_argv': sys.argv,
                  'monitor_sha256': digest(__file__),
                  'helper_sha256': digest(helper.__file__),
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'sysfs_device': str(args.sysfs_device),
                  'resolved_device': str(args.sysfs_device.resolve()),
                  'interval_seconds': args.interval_seconds,
                  'timeout_seconds': args.timeout_seconds,
                  'docker_name': args.docker_name,
                  'device_baseline': device_snapshot(args.sysfs_device),
                  'stdout_log': str(stdout_path), 'stderr_log': str(stderr_path)})
            process = subprocess.Popen(args.command, stdout=stdout, stderr=stderr, start_new_session=True)
            while True:
                now = time.monotonic()
                timed_out = now >= deadline
                if args.docker_name:
                    observed = inspect_container(container['id'] if container else args.docker_name)
                    if observed is not None and (container is None or observed != container):
                        emit({'event': 'container_identity', 'container': observed})
                    if observed is not None:
                        container = observed
                root_pid = container['pid'] if args.docker_name and container else None
                if not args.docker_name:
                    root_pid = process.pid
                process_data = process_snapshot(root_pid, helper)
                device_data = device_snapshot(args.sysfs_device)
                emit({'event': 'sample', 'sample_index': sample_count,
                      'command_pid': process.pid, 'process_tree': process_data, 'device': device_data,
                      'host_mem_available_bytes': helper.read_mem_available() * 1024})
                sample_count += 1
                for key, value in (('rss_bytes', process_data['rss_bytes']),
                                   ('pss_bytes', process_data['pss_bytes']),
                                   ('device_vram_used_bytes', device_data['mem_info_vram_used'])):
                    if value is not None:
                        peaks[key] = value if peaks[key] is None else max(peaks[key], value)
                if timed_out:
                    stop_owned(process, container)
                    emit({'event': 'end', 'status': 'TIMEOUT', 'exit_code': process.returncode,
                          'samples': sample_count, 'sampled_peaks': peaks})
                    return 124
                result = process.poll()
                if result is not None:
                    if args.docker_name and container is None:
                        emit({'event': 'end', 'status': 'MONITOR_INCOMPLETE', 'exit_code': result,
                              'error': 'named container was never observed; no Docker process memory was measured',
                              'samples': sample_count, 'sampled_peaks': peaks})
                        return result if result else 2
                    emit({'event': 'end', 'status': 'COMPLETE' if result == 0 else 'COMMAND_FAILED',
                          'exit_code': result, 'samples': sample_count, 'sampled_peaks': peaks,
                          'peak_scope': 'maximum observed samples, not a continuous or process-GPU peak',
                          'command_stdout_sha256': digest(stdout_path),
                          'command_stderr_sha256': digest(stderr_path)})
                    return result
                time.sleep(min(args.interval_seconds, max(0.0, deadline - time.monotonic())))
        except BaseException as exc:
            stop_owned(process, container)
            emit({'event': 'end', 'status': 'MONITOR_ERROR', 'error': str(exc),
                  'error_type': type(exc).__name__, 'traceback': traceback.format_exc(),
                  'samples': sample_count, 'sampled_peaks': peaks})
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
