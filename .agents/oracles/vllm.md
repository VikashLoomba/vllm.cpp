# vLLM — the primary oracle

The mirror source and the only oracle that outranks the others. Where vLLM
implements a behavior, it defines it; a secondary oracle disagreeing with vLLM
loses.

**The pin lives in [`../upstream-sync.md`](../upstream-sync.md)**, in its
` ```parity-pin ` block, together with the runtime and distribution version
strings a live oracle reports about itself and the `+g<sha>` constraint
`assert_oracle_commit` enforces. It is restated below only as identity — the
sync cycle advances the block over there, and `tools/bench/` reads that block,
not this file.

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = every behavior vLLM implements — defaults, modes, errors, edge cases, and both correctness and speed gates
pin = 5559679229bc961848b121ccdeaa8fa5d79bec98
pin_label = 0.26.0.dev0
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/upstream-sync.md
```

## The candidate `e126687a9a`, and what its run half established

**The pin above does not move**, and nothing in this section is a reason to move
it. This records one measured property of a candidate revision so that the next
sync cycle does not re-measure it.

`e126687a9a828d513c01a07cd69f025f27d63280` (2026-08-31, the revision that
registers `Qwen4ExpForCausalLM`) **builds from source and runs a model** on
`thor:gpu0`. Measured 2026-09-03, aarch64, NVIDIA Thor, compute capability 11.0,
driver 595.78:

```console
SRCBUILD_RC=0   94 min, MAX_JOBS=4, TORCH_CUDA_ARCH_LIST=11.0, CUDA 13.0.88
EXT_PRESENT=True  seven compiled extensions; vllm._custom_ops.rms_norm EXECUTED on device
IMPORT_RC=0     vllm.__version__ = 0.28.1rc1.dev132+ge126687a9, read from cd /
RUN_RC=0        facebook/opt-125m, greedy, FLASH_ATTN/FA2, eager AND compiled
```

Evidence: [`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md),
issue [#2611](https://github.com/mudler/vllm.cpp/issues/2611).

**What that is worth, and the limit is first because it is the larger fact.**
`qwen4_exp` — the model this candidate is wanted for — **does not run on
`thor:gpu0`** at this revision: its QSA indexer's `cooperative_topk` refuses to
launch with a cluster misconfiguration
([#2626](https://github.com/mudler/vllm.cpp/issues/2626)), and its published
safetensors arms exceed the largest fleet box. So a pin here would carry a
registered, importable, executable vLLM that still cannot serve
`MODEL-MM-QWEN4-EXP`'s own model on this fleet. What was demonstrated is that
**some** model builds and runs, on one device; whether AGENTS.md's "builds and
runs the model" is satisfied by that, or requires the model in question, is a
reading this record does not settle. It is **not** a pin advance and **not** a
`gateable = yes` for this revision. Beyond the question above, a pin advance
additionally owes these four:

1. The **290-entry PORT-NOW queue** for `5559679229..e126687a9a`, unworked
   ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
2. A **declared token-exact gate**. The run above is six prompts at 16 tokens
   and gates nothing; it agreed with `tests/parity/goldens/opt_greedy` exactly,
   which is informative and not a result.
3. **Step 6 re-measurement** of every benchmark baselined against FlashInfer
   0.6.15.post1, CUTLASS DSL 4.6.0, the `transformers` floor and the
   `VLLM_ALLREDUCE_USE_FLASHINFER` default, all of which move under this
   candidate.
4. A reading on **`dgx:gpu0`**. Only `thor:gpu0` was measured.

**Evidence for the paragraph above.**
[#2626](https://github.com/mudler/vllm.cpp/issues/2626) is owned by
`MODEL-MM-QWEN4-EXP` and listed under `## Owed` in
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md); the
measurement and its explicit non-claims are in
[`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)
§6.

## Device-scoped gateability

`gateable = yes` above is a property of the oracle, not a promise about every
board. Where a device has been MEASURED to build and run a gate model, it is
recorded here with the evidence; absence from this table means unmeasured, never
unsupported. One row per measurement, appended by the change that made it.

| device | arch | measured | builds | runs a gate model | evidence |
|---|---|---|---|---|---|
| `strix:gpu0` | `gfx1151` (RDNA 3.5, Radeon 8060S, ROCm 7.2.4) | 2026-09-03 | yes | yes — Qwen3.8-27B Q4_K_M GGUF, 6 prompts x 48 greedy tokens, reproducible | [`oracle-vllm-gfx1151-20260903.md`](../../docs/bench-evidence/oracle-vllm-gfx1151-20260903.md) |

**gfx1151 needs five packages a bare ROCm image does not carry**, and each of
their absences presents as a device failure rather than as a provisioning gap:
`python3-dev`, `rocm-libs`, `libdrm-dev`, the ROCm `torchvision`, and `amdsmi`.
`amdsmi` is the one that matters most: `vllm/platforms/__init__.py:110-128`
decides whether the platform is ROCm by importing it, so without it vLLM falls
back to `UnspecifiedPlatform` on a fully working ROCm box. The evidence file
carries the exact message each one produces.

**No `HSA_OVERRIDE_GFX_VERSION` was set for that measurement**, and none may be
set for another. That knob makes the runtime report a different device, and a
pin taken under it is a pin on a fiction.
