#!/usr/bin/env python3
"""Aggregate the four retained two-request capture legs without rerunning inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import traceback

PREP = Path('/home/vikash/oracle/rdna3-wmma-latest')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_capture(path):
    data = json.loads(path.read_text())
    if data.get('status') != 'COMPLETE' or len(data.get('legs', [])) != 4:
        raise ValueError(f'capture is not complete with four legs: {path}')
    return data


def primary_aggregate(data, helper):
    rows = [row for leg in data['legs'] for row in leg['raw_closed_loop_records']]
    prompts = [ids for leg in data['legs'] for ids in leg['prompt_token_ids']]
    duration = sum(leg['metrics']['duration_s'] for leg in data['legs'])
    if len(rows) != 8 or sum(map(len, prompts)) != 1428:
        raise ValueError('primary capture must contain eight requests and 1428 input tokens')
    metrics, ids = helper.derive_metrics(rows, prompts, duration, output_len=16,
                                        max_concurrency=1, async_scheduling='default')
    return {'metrics': metrics, 'generated_token_ids': ids,
            'original_legs': data['legs'],
            'duration_scope': 'sum of four closed-loop leg durations; excludes model initialization and inter-leg memory RPCs',
            'initialization_seconds': data['engine_initialization_seconds'],
            'cold_label_limit': 'first captured request leg after normal engine initialization and its built-in warmup; inspect JIT-cache history'}


def llama_aggregate(data, helper):
    rows = [row for leg in data['legs'] for row in leg['requests']]
    prompts = [row['prompt_token_ids'] for row in rows]
    ids = [row['generated_token_ids'] for row in rows]
    if len(rows) != 8 or sum(map(len, prompts)) != 1428 or any(len(row) != 16 for row in ids):
        raise ValueError('llama capture must contain eight requests, 1428 input tokens, and 128 generated tokens')
    duration = sum(leg['capture_leg_elapsed_seconds'] for leg in data['legs'])
    ttfts = [row['client_ttft_ms_including_capture'] for row in rows]
    e2es = [row['client_e2e_ms_including_capture'] for row in rows]
    tpots = [(end - first) / 15 for end, first in zip(e2es, ttfts)]
    itls = [right - left for row in rows
            for left, right in zip(row['token_emit_elapsed_ms'], row['token_emit_elapsed_ms'][1:])]
    metrics = {'duration_s': duration, 'successful_requests': 8, 'maximum_request_concurrency': 1,
               'input_tokens': 1428, 'output_tokens': 128, 'request_throughput': 8 / duration,
               'input_token_throughput': 1428 / duration, 'output_token_throughput': 128 / duration,
               'total_token_throughput': 1556 / duration,
               'ttft_ms': helper.summarize(ttfts), 'tpot_ms': helper.summarize(tpots),
               'e2el_ms': helper.summarize(e2es), 'itl_ms': helper.summarize(itls),
               'mean_per_stream_decode_rate': 1000 / statistics.fmean(tpots)}
    prefill_seconds = sum(row['prefill_decode_call_ms'] for row in rows) / 1000
    decode_seconds = sum(row['decode_call_ms'] for row in rows) / 1000
    return {'metrics_including_capture': metrics, 'generated_token_ids': ids,
            'original_legs': data['legs'],
            'decode_call_only_metrics': {'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
                                         'prefill_tokens_per_second': 1428 / prefill_seconds,
                                         'decode_tokens_per_second': 120 / decode_seconds},
            'duration_scope': 'sum of four capture-leg durations, including memory clear, finite checks, logits copies, and request memory probes; excludes logit file writes and inter-leg work',
            'initialization_seconds': data['model_and_context_initialization_seconds'],
            'comparison_limit': 'whole-workload request/token counts match; client timing includes additional llama correctness instrumentation and is not an equal-overhead latency denominator'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path)
    parser.add_argument('--llama', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics-helper', type=Path,
                        default=Path('/home/vikash/vllm.cpp-rdna3-wmma/tools/bench/vllm_closed_loop_metrics.py'))
    args = parser.parse_args()
    if not args.primary and not args.llama:
        parser.error('provide at least one completed capture')
    if args.output.exists():
        parser.error('refusing to overwrite existing aggregate evidence')
    report = {'status': 'RUNNING', 'argv': sys.argv,
              'adapter_sha256': digest(Path(__file__)), 'inputs': {}}
    try:
        spec = importlib.util.spec_from_file_location('closed_loop_metrics', args.metrics_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        report['metrics_helper'] = {'path': str(args.metrics_helper), 'sha256': digest(args.metrics_helper)}
        for kind, path, aggregate in (('primary', args.primary, primary_aggregate),
                                       ('llama', args.llama, llama_aggregate)):
            if path:
                report['inputs'][kind] = {'path': str(path), 'sha256': digest(path)}
                report[kind] = aggregate(load_capture(path), helper)
        if args.primary and args.llama:
            report['primary_llama_output_ids_exact'] = (report['primary']['generated_token_ids'] ==
                                                       report['llama']['generated_token_ids'])
        mismatch = report.get('primary_llama_output_ids_exact') is False
        report['status'] = 'TOKEN_MISMATCH' if mismatch else 'COMPLETE'
        result = 1 if mismatch else 0
    except Exception as exc:
        report.update(status='ERROR', error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        result = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
