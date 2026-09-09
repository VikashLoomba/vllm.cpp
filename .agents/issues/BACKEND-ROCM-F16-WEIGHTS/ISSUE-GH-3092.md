ID: ISSUE-GH-3092
Title: feat(BACKEND-ROCM-F16-WEIGHTS): retain F16 dense and embedding weights on ROCm
Row: BACKEND-ROCM-F16-WEIGHTS
State: OPEN
Kind: UNKNOWN
GitHub: 3092
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-F16-WEIGHTS`
>
> ROCm currently expands GGUF F16 weights because ordinary matrix multiplication and embedding kernels reject F16 storage. This prevents those weights from staying in their checkpoint format on gfx1100.
>
> Implement F16 storage support through ordinary Matmul/MatmulBT and embedding operations, preserving the existing BF16/F32 activation and output contract. Enable loader admission for dense matrix weights and embedding tables only. Keep stacked expert weights on their existing expansion path; this issue does not enable a full F16 activation runtime.
>
> The spec must establish mixed input arithmetic from the pinned upstream sources, cover both ID widths for embeddings, and prove the default public loading and generation path reaches the new providers. Require red-first tests, physical gfx1100 correctness and memory evidence, fresh mutation review, and operator verification.
>
> This work is independent of PR #2782. That PR expands quantized GEMM kernels. F16 RMSNorm activation support remains a distinct concern tracked by #2542. No CI changes are included.
>
> Spec: `.agents/specs/rocm-f16-weights.md` (to be committed before implementation).
>

## Resolution

-
