ID: ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ
Title: Enable and validate rocWMMA attention prefill on gfx1100
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-15
Closed: -

## Problem

The production BF16 SharedK attention prefill kernel uses portable rocWMMA fragments, but both device compilation and host dispatch exclude gfx1100. Enable only physical gfx1100 after red-first correctness, emitted-code inspection, same-binary performance checks, and public end-to-end validation. The developer requests a single-agent implementation and an MR-ready result.

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
