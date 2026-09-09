ID: ISSUE-GH-3094
Title: feat(BACKEND-ROCM-BF16-MOE): run BF16 grouped experts on ROCm
Row: BACKEND-ROCM-BF16-MOE
State: OPEN
Kind: UNKNOWN
GitHub: 3094
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-BF16-MOE`
>
> ROCm has no providers for kMoeGroupedGemmBf16 or kMoeGroupedGemmBf16GateUpSilu. A BF16 MoE tower therefore cannot reach the shared grouped expert path on a discrete gfx1100 device. The quantized grouped GEMM provider does not supply these operations.
>
> Implement both providers together, using the existing shared MoE and fusion surfaces. Reconcile the pinned vLLM execution chain with the local numeric contract: BF16 result narrowing before activation, routing-weight placement before the down-projection result is stored, and reduction without applying those weights twice. Extend the shared descriptor only where it cannot express that behavior, preserving existing callers unless an explicitly reviewed correction requires otherwise.
>
> Use a small Qwen3MoeForCausalLM fixture through public loading and registered forward execution as the production reachability gate. This proves the shared expert operations; it does not claim complete DeepSeek-V2 or Dots3 support, whose grouped and sigmoid router gaps remain separate.
>
> Require a committed spec, red-first tests, matching pinned-oracle workloads, physical gfx1100 correctness and memory evidence, fresh mutation review, and operator verification. PR #2782 is not a dependency. No CI changes are included.
>
> Spec: `.agents/specs/rocm-bf16-moe.md` (to be committed before implementation).
>
> ## Provenance
>
> This scoped implementation issue follows #1928, which was split from #1870 after identifying that device-fit arithmetic and missing BF16 expert kernels are separate changes. The original record named the parent BACKEND-ROCM row and suggested gfx1200 evidence. The current work uses its own child row and the developer's authorized local gfx1100 hardware. The source and numeric contract are established by the committed spec before implementation.
>
> The authenticated contributor cannot edit #1928 to assign its first-line Row field. This child supplies the matching issue/spec/PR ownership record. The implementation MR will close both issues when the missing providers land.
>

## Resolution

-
