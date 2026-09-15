ID: ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ
Title: Enable and validate rocWMMA attention prefill on gfx1100
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: -

## Problem

The production BF16 SharedK attention prefill kernel uses portable rocWMMA fragments, but both device compilation and host dispatch exclude gfx1100. Enable only physical gfx1100 after red-first correctness, emitted-code inspection, same-binary performance checks, and public end-to-end validation. The developer requests a single-agent implementation and an MR-ready result.

## Resolution

-
