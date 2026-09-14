# Native benchmark prompt identity

PASS on 14 September 2026, measured entirely on the CPU.

The native tokenizer produced eight complete ID arrays equal to the primary
receipt's two arrays repeated in order four times. Counts were
`[183, 174, 183, 174, 183, 174, 183, 174]`, totaling 1,428 IDs.
The raw eight-prompt workload also equals the original two-prompt workload
repeated four times.

The negative control changed prompt 0, position 0, from ID 760 to 761.
All prompt counts and the total remained unchanged. The comparison exited 1
and identified that exact mismatch. The intact comparison exited 0.

The adapter calls `GgufFile::Open`, `Tokenizer::FromGguf`, and
`EncodeWithSpecialTokens`, using the actual benchmark GGUF. Its SHA256 is
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
The adapter links the exact existing benchmark `libvllm.a` because the shared
library hides the C++ symbols. Selective archive extraction omits unrelated
backend registration. Six archived tokenizer and file-reader objects match
the corresponding build objects byte for byte.

The adapter calls no model loader or backend initializer. The syscall trace
records no GPU device paths. GPU visibility variables were empty.
All recorded input, source, archive, benchmark, and shared-library hashes
remained unchanged during measurement. The measured tokenizer and production
call-site sources are unchanged between product commit `c3fe98ba6` and receipt
head `6e65c6fad`.

## Artifacts

- `native-ids.stdout`: actual raw prompts and all eight native ID arrays.
- `green-exact-id-match.stdout`: complete comparison result, exit 0.
- `red-same-count-id-change.stdout`: changed-ID comparison result, exit 1.
- `result.json`: summary, source chain anchors, and archive object hashes.
- `commands.json`: exact command arguments, exit codes, and output files.
- `inputs-before.json` and `inputs-after.json`: immutable inputs and SHA256 values.
- `cpu-syscalls.txt`: CPU-only file and ioctl trace.
- `link.map`: objects selected from the existing archive.
- `manifest.json`: output artifact hashes.
- `native_prompt_ids.cpp`, `compare_ids.py`, and `run_receipt.py`: external adapters.

## Reproduce

Run `python3 run_receipt.py` in this directory. It verifies the actual GGUF
hash, rebuilds only the external adapter, extracts IDs, proves the negative
control, compares complete arrays, and seals inputs and outputs.

For an independent comparison without compilation or GPU work:

```sh
python3 compare_ids.py native-ids.stdout ../prompt-ids-retry.json ../../qwen35-4b-text/workload.json ../native-ab-model-gguf/workload-eight.json
```

This receipt proves native input identity. It does not rerun model generation
or establish output-token parity.
