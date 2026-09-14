#!/usr/bin/env python3
"""Compare every native ID with the primary two-prompt receipt repeated four times."""
import json
import sys
from pathlib import Path

native_path, primary_path, raw_path, expanded_path = map(Path, sys.argv[1:])
native = json.loads(native_path.read_text())
primary = json.loads(primary_path.read_text())
raw = json.loads(raw_path.read_text())
expanded = json.loads(expanded_path.read_text())
expected_ids = primary['prompt_token_ids'] * 4
expected_prompts = [entry['conversations'][0]['value'] for entry in raw] * 4
actual_ids = native['prompt_token_ids']
differences = []
for prompt_index, (actual, expected) in enumerate(zip(actual_ids, expected_ids)):
    if len(actual) != len(expected):
        differences.append({'prompt': prompt_index, 'lengths': [len(actual), len(expected)]})
    for token_index, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            differences.append({'prompt': prompt_index, 'position': token_index, 'native': a, 'primary': e})
checks = {
    'primary_has_two_prompts': len(primary['prompt_token_ids']) == len(raw) == 2,
    'expanded_workload_equals_original_repeated_four_times': expanded == raw * 4,
    'native_raw_prompts_equal_original_repeated_four_times': native['prompts'] == expected_prompts,
    'all_eight_complete_id_arrays_equal_primary_repeated_four_times': actual_ids == expected_ids,
    'native_count_receipt_matches_arrays': native['prompt_token_counts'] == list(map(len, actual_ids)),
    'primary_count_receipt_matches_arrays': primary['prompt_token_counts'] == list(map(len, primary['prompt_token_ids'])),
    'native_total_receipt_matches_arrays': native['total_prompt_tokens'] == sum(map(len, actual_ids)),
}
result = {
    'comparison': f'Every ID from native {native_path} against primary {primary_path}, repeated in original order four times',
    'checks': checks,
    'native_counts': list(map(len, actual_ids)),
    'primary_counts_repeated': list(map(len, expected_ids)),
    'native_total': sum(map(len, actual_ids)),
    'primary_total_repeated': sum(map(len, expected_ids)),
    'same_total_count': sum(map(len, actual_ids)) == sum(map(len, expected_ids)),
    'differences': differences,
    'verdict': 'PASS' if all(checks.values()) else 'FAIL',
}
print(json.dumps(result, indent=2))
sys.exit(0 if all(checks.values()) else 1)
