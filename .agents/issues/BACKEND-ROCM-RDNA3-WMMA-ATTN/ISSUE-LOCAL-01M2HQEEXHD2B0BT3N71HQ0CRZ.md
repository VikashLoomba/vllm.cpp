ID: ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ
Title: Resolve Gemma 3 4B differences on the expanded primary token gate
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-15
Closed: -

## Problem

Pinned vLLM e126687a9 reproduces all eight distinct 518/1207-token prompts. After linear RoPE and BF16 head corrections, gfx1100 scalar attention differs on 3 of 8 outputs and SharedK WMMA differs on 4 of 8. The original 96-token workload passes both. The broader failures include non-tied primary logits. Retain the full 256-token gate, locate the first intermediate divergence, and keep gfx1100 WMMA opt-in until parity passes. Do not accept this workload as a performance comparison.

## Resolution

15 September 2026, branch evidence before landing: default gfx1100 WMMA
passes the original 96-token and expanded 256-token gates with block sizes
16 and 32. Compiled Gemma boundaries, device RoPE caches, gfx11 packed
operands, and decode accumulation account for the repaired differences.
The attention kernel has zero spills. Three alternating pairs measure a
1.149x median model prefill improvement over scalar at block size 32.

[Measured report](../../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).
PR #3195 carries the implementation for independent human review. Keep this
issue open until the work lands. Expanded scalar fallback token parity and
full-model decode/latency/host-memory performance parity remain open axes;
they do not require the now-correct gfx1100 prefill path to remain opt-in.
