# QUANT-CT-MXFP4-FLASH-PTXAS — arbitrate the ptxas lineage behind vLLM's faster flash decode SASS

<!-- spec-status: ACTIVE -->
Row: `QUANT-CT-MXFP4-FLASH-PTXAS` (helper, `row/QUANT-CT-MXFP4-FLASH-PTXAS`).
Base: `origin/main` `362a3c99`. Vehicle: `Yi30/Qwen3-8B-MXFP4` (dense
`Qwen3ForCausalLM`, W4A16 Marlin keep-quant); oracle arm
`VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`, vLLM 0.25.0-stage. GB10 sm_121a.

## Why this row exists

#75 closed the codegen lens to a MEASURED verdict: same `flash_fwd_splitkv` source
(vendored from vLLM's pinned `2c839c33`), same grid (1×3×64), same 8.33% occupancy
(both smem-limited to 1 CTA/SM by 81.92 KB), same L2 ~1%, and — with
`compute_80 + -use_fast_math` (Build C) — matched vLLM's register (241) AND
instruction (17,008 vs 17,020) counts EXACTLY. Yet Build C is ~167 us/call, vLLM
is ~157 us/call. #75 attributed the residual to "vLLM's wheel `ptxas`
SASS-scheduling quality (older CUDA 12.x lineage)". This row arbitrates that
attribution to a MEASURED yes/no: get the wheel's exact ptxas lineage, re-assemble
our flash PTX with it, and A/B the kernel.

## W1 — arbitrate the ptxas lineage
(a) Identify the wheel's toolkit lineage: `cuobjdump` the fa2 `.so`.
(b) Obtain that ptxas (venv `nvidia-cuda-nvcc-cu12` wheels or a scratch pip download).
(c) Compile our flash decode TU to PTX (compute_80 + fast-math per #75 Build C),
assemble with the candidate ptxas lineages → cubin, load via `cuModule` in a
microbench with the c8 decode params, and A/B (CUDA-event medians + ncu).
THE ARBITER: does a different-lineage-ptxas cubin hit ~156-157 us? NO ⇒ refutation,
close flash as measured-irreducible-for-us, STOP.

## W2 — vendor (only on a YES)
Mirror the GDN Triton-AOT precedent: commit the cubin + exact regen recipe under the
vendored-kernels tree; a load path routing the sm_121a decode flash launch through
the vendored cubin, gated `VT_FA2_VENDORED_CUBIN`, default per parity-enablers ONLY
IF the full battery is green (near-tie razor for the fast-math numerics; #44 smoke;
SACRED 0.6B/4B + 32B strict; async; memcheck; eager+graphed; both GQA ratios).
hd256 (27B/35B) cross-model projection measured, not flipped, as a follow-up.

## W3 — binding + verdict
c1..c8 ×3 production defaults vs 1.020/0.962/0.966/0.969. THE PARITY VERDICT:
≥1.0 every axis ⇒ MXFP4 parity DONE; short ⇒ honest terminal map with the fresh
decomposition (flash-after-arbitration + glue ~18% + marlin/host remainder).

## Gates
Byte-exact razor: #44 MXFP4-8B smoke 3/3 + coherent. If fast-math shifts the
reduction order: SACRED 0.6B/4B distributional + 32B strict, async, memcheck,
eager+graphed. Box safety: BOTH flock locks, free -g ≥ 90, worker STOPPED, tmux +
done-markers, sequential arms, single-load steady-state, disk floor 15G.
