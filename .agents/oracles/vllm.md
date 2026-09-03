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
pin = e126687a9a828d513c01a07cd69f025f27d63280
pin_label = 0.28.1rc1.dev132
pinned_on = 2026-09-03
gateable = yes
evidence = .agents/sync/2026-09-03-e126687-runhalf.md
```

## What this pin establishes, and what it does NOT

**`gateable = yes` above says the oracle builds and runs. It does NOT say any
gate in this tree has been run against it.** Those are different statements and
this section keeps them apart, because the pin advanced
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)) on a developer ruling
that put step 6 after step 7, so the advance carries obligations it has not
discharged.

### Established, and this is the whole of it

`e126687a9a828d513c01a07cd69f025f27d63280` (2026-08-31, vllm#53896, the revision
that registers `Qwen4ExpForCausalLM`) **builds from source and runs a model** on
`thor:gpu0`. Measured 2026-09-03, aarch64, NVIDIA Thor, compute capability 11.0,
driver 595.78:

```console
SRCBUILD_RC=0   94 min, MAX_JOBS=4, TORCH_CUDA_ARCH_LIST=11.0, CUDA 13.0.88
EXT_PRESENT=True  seven compiled extensions; vllm._custom_ops.rms_norm EXECUTED on device
IMPORT_RC=0     vllm.__version__ = 0.28.1rc1.dev132+ge126687a9, read from cd /
RUN_RC=0        facebook/opt-125m, greedy, FLASH_ATTN/FA2, eager AND compiled
```

Evidence: [`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md),
issue [#2611](https://github.com/mudler/vllm.cpp/issues/2611), and the install
half [`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md)
([#2593](https://github.com/mudler/vllm.cpp/issues/2593)).

### NOT established, and the pin's movement makes none of it true

1. **`qwen4_exp` does not run on this fleet at this revision.** The model this
   pin was taken for: its QSA indexer's `cooperative_topk` refuses to launch with
   a cluster misconfiguration
   ([#2626](https://github.com/mudler/vllm.cpp/issues/2626), cause
   unestablished), and its published safetensors arms exceed the largest fleet
   box. So `MODEL-MM-QWEN4-EXP` has a registered, importable, executable vLLM
   that still cannot serve its own model here. That is strictly better than the
   prior pin, where the architecture was not registered at all, and it is not a
   runnable oracle for the row.
2. **No gate model, and no gate.** `facebook/opt-125m` at six prompts and sixteen
   tokens is a smoke test that agreed with `tests/parity/goldens/opt_greedy`.
   **Every committed golden in this tree was captured against `555967922` or
   earlier and none has been re-validated here.** The declared token-exact gate
   at this pin is specified by
   [`../specs/upstream-sync-headpin-tokengate.md`](../specs/upstream-sync-headpin-tokengate.md)
   ([#2794](https://github.com/mudler/vllm.cpp/issues/2794)) and must be captured
   on `dgx:gpu0` (GB10, `sm_121a`), the device the committed golden came from,
   with `--runs 5`. The token path reads no pin constant, so it is NOT subject to
   the harness refusal below and can be captured at any revision that imports.
   That capture also answers the OPT golden that was never re-validated at the
   PRIOR pin either.
3. **Step 6 is owed POST-HOC**
   ([#2818](https://github.com/mudler/vllm.cpp/issues/2818)). Narrowed by
   [`../sync/2026-09-03-e126687-step6.md`](../sync/2026-09-03-e126687-step6.md)
   ([#2771](https://github.com/mudler/vllm.cpp/issues/2771)) to **FlashInfer
   `0.6.15.post1` to `0.6.18` on two gates**: `vllm-online-serving` (three
   throughput rows plus the startup row), where FlashInfer is the NVFP4 GEMM
   under the denominator and the CUTLASS source tree our own arm compiles
   against, and `speculative-decoding` (two rows), where it is the oracle's
   attention backend. **`nvidia-cutlass-dsl` `4.6.0` to `4.6.2` is NOT
   discharged**: a fresh review of
   [#2783](https://github.com/mudler/vllm.cpp/pull/2783) found a warmup path
   gated on `has_device_capability(90)`, which is `>=` and admits `sm_121`, whose
   `compile(...)` is unguarded — it cannot move the steady-state math, but it can
   abort engine start and it sits inside the startup ratio
   `docs/benchmarks/vllm-online-serving.md:73` publishes.
   The `transformers` floor (`>= 5.5.3` to `>= 5.10.4`, resolved `5.14.1`, above
   both) and the `VLLM_ALLREDUCE_USE_FLASHINFER` default flip (inert at
   `tensor_parallel_size == 1`, which every committed gate runs) were discharged
   at the PRIOR pin and **still hold at this one**, because neither argument
   depends on which revision is pinned. **A red on the re-measurement requires
   reverting this pin**, not holding it; that is part of the developer ruling and
   is recorded in [`../upstream-sync.md`](../upstream-sync.md).
4. **No reading on `dgx:gpu0`.** Only `thor:gpu0` was measured, on one day. Every
   binding benchmark number in this tree was taken on GB10, and nothing here
   transfers to it.
5. **Step 5 did not run.** The 290-entry PORT-NOW queue for
   `5559679229..e126687a9a` is classified and reconciled but unworked, so 90
   files whose `Ported from:` header names `55596792` now name a revision BELOW
   the pin. Owed by [#2611](https://github.com/mudler/vllm.cpp/issues/2611).

**Evidence for the limits above.**
[#2626](https://github.com/mudler/vllm.cpp/issues/2626) is owned by
`MODEL-MM-QWEN4-EXP` and listed under `## Owed` in
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md); the
measurement and its explicit non-claims are in
[`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)
§6 and §7.

### The prior pin, for anyone reading a number taken under it

`5559679229bc961848b121ccdeaa8fa5d79bec98`, `0.26.0.dev0`, pinned 2026-07-26,
FlashInfer `0.6.15.post1`, CUTLASS DSL `4.6.0`, transformers `5.14.1`. Every
binding ratio published in `docs/benchmarks/` was measured against it. That
advance re-captured goldens on its own oracle and recorded zero real drift; this
one did not, which is the difference §"NOT established" item 2 is about.

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
