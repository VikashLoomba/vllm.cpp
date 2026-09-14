#!/usr/bin/env python3
"""Seal the trace-model correction while preserving the original adapter manifest."""
import datetime
import json
from pathlib import Path

from run_native_ab import sha256

ADAPTERS = Path('/home/vikash/oracle/rdna3-wmma-latest/adapters')


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


original_path = ADAPTERS / 'adapter-manifest.json'
original = json.loads(original_path.read_text())
original_identity = identity(original_path)
assert original_identity['sha256'] == 'c89ed2148cc9d1c1ccfde308f861fbdcdb416c6e6e39d342fa5e83b0b936f254'
changed_paths = {str(ADAPTERS / name) for name in ('run_native_trace.py', 'HANDOFF.md')}
files = {entry['path']: entry for entry in original['files']}
for path in changed_paths:
    files[path] = identity(Path(path))
additions = [original_path, Path(__file__), ADAPTERS / 'trace-gguf-model-check.json']
for folder in ('history/trace-gguf-fix-before', 'native-trace-on-gguf-preparation',
               'native-trace-off-gguf-preparation'):
    additions.extend(path for path in (ADAPTERS / folder).rglob('*') if path.is_file())
for path in additions:
    files[str(path)] = identity(path)
recipes = [json.loads((ADAPTERS / f'native-trace-{arm}-gguf-preparation/recipe-and-result.json').read_text())
           for arm in ('on', 'off')]
assert recipes[0]['model'] == recipes[1]['model']
model = recipes[0]['model']
assert model['path'] == '/home/vikash/models/Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf'
assert model['sha256'] == '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
files[model['path']] = {'path': model['path'], 'bytes': model['stat'][2], 'sha256': model['sha256']}
manifest = dict(original)
manifest.update(
    sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='external trace-model path correction and original preparation artifacts',
    prior_manifest=original_identity,
    correction={'changed_paths': sorted(changed_paths),
                'reason': 'native model-directory argument selected the safetensors loader without shards; use the retained GGUF file',
                'unchanged_running_driver': files[str(ADAPTERS / 'run_native_ab.py')],
                'preserved_prior_bytes': str(ADAPTERS / 'history/trace-gguf-fix-before/archive-map.json'),
                'gpu_executed_by_helper': False},
    files=[files[path] for path in sorted(files)])
manifest['check_expectations'] = dict(original['check_expectations'])
manifest['check_expectations']['trace-gguf-model-check.json'] = 'PASS; syntax, exact model argv/hash, eight-request dataset, trace bounds, and preserved original evidence'
for arm in ('on', 'off'):
    manifest['check_expectations'][f'native-trace-{arm}-gguf-preparation/recipe-and-result.json'] = (
        'PREPARED; exact retained GGUF argv and verified model SHA256; eight requests; no GPU workload launched')
output = ADAPTERS / 'adapter-manifest-gguf-trace.json'
output.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(identity(output)))
