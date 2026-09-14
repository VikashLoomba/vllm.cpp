#!/usr/bin/env python3
"""Run and seal a CPU-only native prompt identity measurement."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

out = Path(__file__).resolve().parent
source = Path('/home/vikash/vllm.cpp-rdna3-wmma')
build = source / 'build-rdna3-wmma-operator'
oracle = Path('/home/vikash/oracle/rdna3-wmma-latest')
model = Path('/home/vikash/models/Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf')
primary = oracle / 'adapters/prompt-ids-retry.json'
raw = oracle / 'qwen35-4b-text/workload.json'
expanded = oracle / 'adapters/native-ab-model-gguf/workload-eight.json'
commands = []

def run(argv, label, env=None, expected=0):
    result = subprocess.run(list(map(str, argv)), cwd=out, env=env, capture_output=True)
    (out / f'{label}.stdout').write_bytes(result.stdout)
    (out / f'{label}.stderr').write_bytes(result.stderr)
    commands.append({'argv': list(map(str, argv)), 'cwd': str(out),
                     'environment_overrides': {} if env is None else {k: env[k] for k in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']},
                     'exit_code': result.returncode, 'expected_exit_code': expected,
                     'stdout': f'{label}.stdout', 'stderr': f'{label}.stderr'})
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if result.returncode != expected:
        raise RuntimeError(f'{label}: exit {result.returncode}, expected {expected}: {result.stderr.decode(errors="replace")[:3000]}')
    return result.stdout

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

source_paths = [
    'src/vllm/entrypoints/model_loader.cpp', 'examples/bench/bench_core.h',
    'src/vllm/model_executor/model_loader/gguf_reader.cpp',
    'include/vllm/model_executor/model_loader/gguf_reader.h',
    'src/vllm/model_executor/model_loader/read_only_file_mapping.cpp',
    'src/vllm/tokenizer/tokenizer.cpp', 'include/vllm/tokenizer/tokenizer.h',
    'src/vllm/tokenizer/bpe.cpp', 'include/vllm/tokenizer/bpe.h',
    'src/vllm/tokenizer/pretokenizer.cpp', 'include/vllm/tokenizer/pretokenizer.h',
    'src/vllm/tokenizer/unicode_data.cpp',
    'include/vllm/tokenizer/unicode_data.h', 'third_party/nlohmann/json.hpp',
    'cmake/vllm_export.map',
]
inputs = [model, primary, raw, expanded, build / 'libvllm.a',
          build / 'libvllm.so.0.0.3', build / 'examples/vllm-bench',
          build / 'build.ninja', build / 'CMakeFiles/rules.ninja']
inputs.extend(source / path for path in source_paths)
before = {str(path): {'sha256': sha(path), 'size': path.stat().st_size} for path in inputs}
(out / 'inputs-before.json').write_text(json.dumps(before, indent=2) + '\n')
if before[str(model)]['sha256'] != '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4':
    raise RuntimeError('Actual model SHA differs from the authorized benchmark artifact')

head = run(['git', '-C', source, 'rev-parse', 'HEAD'], 'source-head').decode().strip()
run(['git', '-C', source, 'diff', '--exit-code',
     'c3fe98ba6c55ce71e75746e1b944a27640464e0f', head, '--'] + source_paths,
    'product-source-equality')
run(['/usr/bin/c++', '--version'], 'compiler-version')
run(['/opt/rocm/lib/llvm/bin/clang++', '--version'], 'linker-version')
run(['/usr/bin/c++', '-std=c++20', '-O3', '-DNDEBUG', '-ffp-contract=off',
     '-Wall', '-Wextra', '-Werror', '-I' + str(source / 'include'),
     '-isystem', source / 'third_party', '-c', out / 'native_prompt_ids.cpp',
     '-o', out / 'native_prompt_ids.o'], 'compile')
# Reuse the benchmark archive and its dependencies. Omit whole-archive so
# unrelated backend and model registration objects are not initialized.
run(['/opt/rocm/lib/llvm/bin/clang++', out / 'native_prompt_ids.o', '-o', out / 'native_prompt_ids',
     '-Wl,-Map=' + str(out / 'link.map'), build / 'libvllm.a', build / 'libblake3_vendored.a',
     '/opt/rocm/lib/libamdhip64.so', '/opt/rocm/lib/libhipblas.so', '/opt/rocm/lib/libhipblaslt.so',
     '/usr/lib/x86_64-linux-gnu/libssl.so', '/usr/lib/x86_64-linux-gnu/libcrypto.so',
     '/opt/rocm/lib/libamdhip64.so.7.15.26333-0000000', '-lgcc'], 'link')

cpu_env = dict(os.environ, HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
run(['strace', '-f', '-e', 'trace=%file,ioctl', '-o', out / 'cpu-syscalls.txt',
     out / 'native_prompt_ids', model, expanded], 'native-ids', env=cpu_env)
native_path = out / 'native-ids.stdout'
native = json.loads(native_path.read_text())
mutated = copy.deepcopy(native)
mutated['prompt_token_ids'][0][0] += 1
(out / 'mutated-same-count.json').write_text(json.dumps(mutated, indent=2) + '\n')
compare = ['python3', out / 'compare_ids.py']
run(compare + [out / 'mutated-same-count.json', primary, raw, expanded], 'red-same-count-id-change', expected=1)
run(compare + [native_path, primary, raw, expanded], 'green-exact-id-match')

trace = (out / 'cpu-syscalls.txt').read_text()
gpu_lines = [line for line in trace.splitlines() if any(device in line for device in ['/dev/kfd', '/dev/dri/', '/dev/nvidia'])]
if gpu_lines:
    raise RuntimeError('CPU-only adapter accessed GPU device paths: ' + '\n'.join(gpu_lines))
after = {str(path): {'sha256': sha(path), 'size': path.stat().st_size} for path in inputs}
(out / 'inputs-after.json').write_text(json.dumps(after, indent=2) + '\n')
if after != before:
    raise RuntimeError('An input changed during the measurement')

object_matches = {}
for obj_rel in [
    'src/vllm/tokenizer/tokenizer.cpp.o', 'src/vllm/tokenizer/pretokenizer.cpp.o',
    'src/vllm/tokenizer/bpe.cpp.o', 'src/vllm/model_executor/model_loader/gguf_reader.cpp.o',
    'src/vllm/tokenizer/unicode_data.cpp.o',
    'src/vllm/model_executor/model_loader/read_only_file_mapping.cpp.o',
]:
    obj = build / 'CMakeFiles/vllm.dir' / obj_rel
    member = subprocess.run(['ar', 'p', str(build / 'libvllm.a'), obj.name], capture_output=True, check=True).stdout
    archived_sha = hashlib.sha256(member).hexdigest()
    object_matches[obj_rel] = {'archive_member_sha256': archived_sha, 'build_object_sha256': sha(obj), 'equal': archived_sha == sha(obj)}
if not all(entry['equal'] for entry in object_matches.values()):
    raise RuntimeError('Archive members differ from the existing build objects')

result = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'verdict': 'PASS', 'scope': 'Exact native prompt IDs for the eight-prompt benchmark workload',
    'source_head': head, 'product_immutable_head': 'c3fe98ba6c55ce71e75746e1b944a27640464e0f',
    'counts': native['prompt_token_counts'], 'total_prompt_tokens': native['total_prompt_tokens'],
    'all_eight_arrays_equal_primary_two_arrays_repeated_four_times': True,
    'same_total_count_mutation_rejected': True,
    'cpu_only': True, 'gpu_device_paths_accessed': gpu_lines,
    'model_loading_or_backend_initialization_called': False,
    'inputs_unchanged': before == after, 'archive_members_match_build_objects': object_matches,
    'tokenizer_and_production_source_unchanged_since_product_head': True,
    'linked_archive_members': sorted(set(re.findall(r'libvllm\.a\(([^)]+)\)', (out / 'link.map').read_text()))),
    'source_chain_anchors': [
        'examples/bench/bench_core.h:561 LoadShareGptPrompts extracts the raw first conversation value',
        'src/vllm/entrypoints/model_loader.cpp:3088 production tokenizer uses Tokenizer::FromGguf(gguf)',
        'src/vllm/model_executor/model_loader/gguf_reader.cpp:614 GgufFile::Open delegates unsplit artifacts to OpenOne',
        'src/vllm/model_executor/model_loader/gguf_reader.cpp:419 OpenOne uses ReadOnlyFileMapping::Open',
        'src/vllm/tokenizer/tokenizer.cpp:887 FromGguf loads the actual artifact metadata',
        'src/vllm/tokenizer/tokenizer.cpp:1021 FromGguf loads token strings and token types',
        'src/vllm/tokenizer/tokenizer.cpp:1065 FromGguf loads merge ranks',
        'examples/bench/bench_core.h:324 pretokenized benchmark uses EncodeWithSpecialTokens',
        'src/vllm/tokenizer/tokenizer.cpp:1324 EncodeWithSpecialTokens applies the template and calls Encode',
        'src/vllm/tokenizer/tokenizer.cpp:1099 EncodePlain calls Pretokenize, byte mapping, and BpeSplit',
    ],
    'link_adaptation': 'The shared library hides all C++ internals. The adapter uses the same existing libvllm.a and dependencies as vllm-bench, with selective archive extraction instead of whole-archive to omit backend registration.',
    'limitations': 'This measures the tokenizer construction and encoding used by production. It does not rerun model generation or independently establish output token parity.',
}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
artifacts = {path.name: {'sha256': sha(path), 'size': path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file() and path.name != 'manifest.json'}
(out / 'manifest.json').write_text(json.dumps(artifacts, indent=2) + '\n')
print(json.dumps(result, indent=2))
