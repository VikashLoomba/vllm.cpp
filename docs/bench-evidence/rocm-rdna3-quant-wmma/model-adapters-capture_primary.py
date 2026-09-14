#!/usr/bin/env python3
"""Capture the specified Qwen3.5 GGUF workload with the latest Python engine."""
from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from collections.abc import Mapping

PREP = Path('/home/vikash/oracle/rdna3-wmma-latest')
MODEL_SHA256 = '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
OUTPUT_LEN = 16


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, enum.Enum):
        return {'name': value.name, 'value': jsonable(value.value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, '__struct_fields__'):
        return {key: jsonable(getattr(value, key)) for key in value.__struct_fields__}
    if hasattr(value, 'to_dict'):
        return jsonable(value.to_dict())
    return repr(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker_memory(self, reset_peak=False):
    """Public collective_rpc callable; executes inside each actual model worker."""
    import os
    import resource
    import torch
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        'pid': os.getpid(),
        'device_index': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(),
        'allocated_bytes': torch.cuda.memory_allocated(),
        'reserved_bytes': torch.cuda.memory_reserved(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'device_free_bytes': free,
        'device_total_bytes': total,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'device_scope': 'whole device, including other processes',
        'allocator_scope': 'this model worker only',
    }


def load_metrics_module(path):
    spec = importlib.util.spec_from_file_location('wmma_closed_loop_metrics', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import metrics helper: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-ids-output', type=Path)
    parser.add_argument('--flat-dir', type=Path, default=PREP / 'qwen35-4b-text')
    parser.add_argument('--metrics-helper', type=Path,
                        default=Path('/home/vikash/vllm.cpp-rdna3-wmma/tools/bench/vllm_closed_loop_metrics.py'))
    parser.add_argument('--tokenize-only', action='store_true',
                        help='validate metadata and emit prompt IDs without loading the model or using a GPU')
    args = parser.parse_args()
    report = {'status': 'RUNNING', 'phase': 'inputs', 'argv': sys.argv,
              'adapter_sha256': sha256(__file__), 'legs': []}
    try:
        helper = load_metrics_module(args.metrics_helper)
        flat = args.flat_dir.absolute()
        config = json.loads((flat / 'config.json').read_text())
        if config['rope_parameters']['partial_rotary_factor'] != 0.25:
            raise ValueError('the retained partial_rotary_factor must equal 0.25')
        texts = helper.load_prompt_text(flat / 'workload.json', 2)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(flat, local_files_only=True)
        prompt_ids = [tokenizer.encode(text) for text in texts]
        if len(prompt_ids) != 2 or sum(map(len, prompt_ids)) != 357:
            raise ValueError(f'expected two prompts totaling 357 tokens, got {list(map(len, prompt_ids))}')
        if any(not ids or len(ids) + OUTPUT_LEN > 2048 for ids in prompt_ids):
            raise ValueError('prompt exceeds the requested context')
        report.update({
            'input_files': {str(flat / name): sha256(flat / name)
                            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'workload.json')},
            'metrics_helper': {'path': str(args.metrics_helper), 'sha256': sha256(args.metrics_helper)},
            'prompt_texts': texts,
            'prompt_token_ids': prompt_ids,
            'prompt_token_counts': list(map(len, prompt_ids)),
            'tokenization': 'raw tokenizer.encode(text), no chat template and no manually inserted tokens',
            'config_adaptation': {
                'upstream': 'Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                'flat_text_config': config,
                'reason': 'retained GGUF contains the text model only',
                'model_class_overrides': {
                    'Qwen3_5ForConditionalGeneration': 'vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM'},
            },
        })
        ids_path = args.prompt_ids_output or args.output.with_suffix('.prompt-ids.json')
        write_json(ids_path, {'prompt_token_ids': prompt_ids, 'output_len': OUTPUT_LEN,
                             'prompt_token_counts': list(map(len, prompt_ids)),
                             'tokenization': report['tokenization'],
                             'config_sha256': report['input_files'][str(flat / 'config.json')]})
        report['prompt_ids_output'] = str(ids_path)
        if args.tokenize_only:
            report.update(status='TOKENIZED_ONLY', phase='complete', gpu_executed=False)
            write_json(args.output, report)
            print(json.dumps({'status': report['status'], 'output': str(args.output),
                              'prompt_token_counts': report['prompt_token_counts']}))
            return 0

        report['phase'] = 'verify_weights'
        model = flat / 'model.gguf'
        actual_model_hash = sha256(model)
        report['model'] = {'path': str(model), 'resolved_path': str(model.resolve()),
                           'bytes': model.stat().st_size, 'sha256': actual_model_hash}
        if actual_model_hash != MODEL_SHA256:
            raise ValueError('retained model SHA256 does not match the committed experiment')
        from vllm import LLM, SamplingParams, TokensPrompt
        import importlib.metadata
        import vllm
        if vllm.__version__ != '0.28.1rc1.dev812+g39545e475':
            raise RuntimeError(f'wrong vLLM runtime: {vllm.__version__} from {vllm.__file__}')
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('vllm', 'vllm-gguf-plugin', 'torch', 'triton', 'gguf', 'transformers')}
        llm_args = {
            'model': str(model), 'hf_config_path': str(flat), 'tokenizer': str(flat),
            'config_format': 'gguf', 'load_format': 'gguf', 'quantization': 'gguf',
            'model_class_overrides': report['config_adaptation']['model_class_overrides'],
            'dtype': 'auto', 'max_model_len': 2048, 'max_num_seqs': 1,
            'max_num_batched_tokens': 512, 'block_size': 32,
            'enable_prefix_caching': False, 'generation_config': 'vllm',
            'seed': 0, 'disable_log_stats': False,
        }
        report['requested_engine_arguments'] = llm_args
        report['production_defaults'] = {
            'compilation': 'unmodified upstream default',
            'cuda_graphs': 'unmodified upstream default',
            'enforce_eager': 'argument omitted',
            'async_scheduling': 'argument omitted',
            'gpu_memory_utilization': 'argument omitted; inspect resolved configuration',
            'cold_leg': 'first captured request leg after normal engine initialization and its built-in warmup',
            'memory': 'resolved hybrid cache allocation is retained; no comparison to native block counts is inferred',
        }
        write_json(args.output, report)
        report['phase'] = 'engine_initialization'
        start = time.perf_counter()
        llm = LLM(**llm_args)
        report['engine_initialization_seconds'] = time.perf_counter() - start
        actual_ids = [llm.get_tokenizer().encode(text) for text in texts]
        if actual_ids != prompt_ids:
            raise RuntimeError('loaded engine tokenizer differs from exported prompt IDs')
        report['resolved_config'] = jsonable(llm.llm_engine.vllm_config)
        sampling = SamplingParams(
            seed=0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
            presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
            max_tokens=OUTPUT_LEN, min_tokens=0, ignore_eos=True,
            detokenize=False, stop=None, stop_token_ids=None,
        )
        report['resolved_sampling_params'] = jsonable(sampling)
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
        for leg_index in range(4):
            report['phase'] = f'leg_{leg_index}'
            before = llm.collective_rpc(worker_memory, args=(True,))
            duration, records = helper.run_closed_loop(llm, prompts, sampling, 1, 1000 * leg_index)
            after = llm.collective_rpc(worker_memory, args=(False,))
            metrics, ids = helper.derive_metrics(
                records, prompt_ids, duration, output_len=OUTPUT_LEN,
                max_concurrency=1, async_scheduling='default')
            report['legs'].append({
                'leg_index': leg_index, 'kind': 'cold' if leg_index == 0 else 'warm',
                'metrics': metrics, 'prompt_token_ids': prompt_ids,
                'generated_token_ids': ids, 'raw_closed_loop_records': records,
                'memory_before_reset_peak': before, 'memory_after': after,
            })
            write_json(args.output, report)
        report.update(status='COMPLETE', phase='complete', gpu_executed=True,
                      single_model_load=True, captured_legs=4)
        write_json(args.output, report)
        print(json.dumps({'status': report['status'], 'output': str(args.output),
                          'captured_legs': len(report['legs'])}))
        return 0
    except Exception as exc:
        report.update(status='ERROR', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        write_json(args.output, report)
        print(json.dumps({'status': 'ERROR', 'phase': report['phase'],
                          'error': str(exc), 'output': str(args.output)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
