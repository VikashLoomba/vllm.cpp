ID: ISSUE-LOCAL-01M2HPT4SGDCYZNAYH4VKBZJK4
Title: Project Gemma 3 logits in the resolved BF16 model dtype
Row: MODEL-GEMMA3-HEAD-DTYPE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: -

## Problem

The final Gemma 3 projection writes FP32 directly. Pinned vLLM e126687a9 uses model-dtype BF16 and widens only in the sampler. Real 4B prompts produce different greedy IDs at close logits. Match the projection dtype, retain FP32 at the existing runner output seam, and rerun the unmodified primary token workloads.

## Resolution

-
