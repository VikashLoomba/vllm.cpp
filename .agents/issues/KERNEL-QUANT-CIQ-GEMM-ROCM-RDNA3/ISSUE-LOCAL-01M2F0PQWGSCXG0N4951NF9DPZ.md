ID: ISSUE-LOCAL-01M2F0PQWGSCXG0N4951NF9DPZ
Title: Enable the existing quantized WMMA prefill kernels on gfx1100
Row: KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Current main compiles and dispatches its generic rocWMMA Q4_K and Q6_K prefill kernels only on gfx1200/gfx1201. The installed rocWMMA 2.2.1 implements the same 16x16x16 signed-int8 operation on gfx1100. Verify admission-only reuse on physical gfx1100, preserve the attention architecture guard, and prove production reachability, numerical correctness, and same-binary prefill performance before accepting the default. The user selects RDNA3 WMMA as the next work and explicitly deprioritizes the unrelated #2773 characterization campaign.

## Resolution

-
