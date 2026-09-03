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
   which is informative and not a result. Specified 2026-09-03 by
   [`../specs/upstream-sync-headpin-tokengate.md`](../specs/upstream-sync-headpin-tokengate.md)
   ([#2794](https://github.com/mudler/vllm.cpp/issues/2794)), which establishes
   three things and closes none of them. **The token path is NOT subject to the
   §3 harness refusal** — `scripts/opt-oracle-capture.py`,
   `scripts/opt-dgx-gate.sh` and `tests/vllm/models/test_opt_paged_engine.cpp`
   read no pin constant, so the gate can be captured at any revision that
   imports. **The gate must be captured on `dgx:gpu0` (GB10, `sm_121a`)**, the
   device the committed golden came from, against the bf16-materialized
   checkpoint and with `--runs 5`, because K selects the STRICT bar; a Thor
   capture moves the silicon and the rounding path alongside the revision, which
   is what makes the run above informative rather than a gate. **And the OPT
   golden was never re-validated at the ACTIVE pin either** — it was captured
   once at `b8358a5b9` against vLLM 0.25.0, and the `5559679229` advance's W3b
   table discharged it in a row for already-ratified near-tie-robust gates, a
   class OPT's strict no-band gate is not in. The capture that answers this
   obligation answers that one too.
3. **Step 6 re-measurement.** Narrowed 2026-09-03 by
   [`../sync/2026-09-03-e126687-step6.md`](../sync/2026-09-03-e126687-step6.md)
   ([#2771](https://github.com/mudler/vllm.cpp/issues/2771)) from "four
   denominators" to **FlashInfer 0.6.15.post1 to 0.6.18, on two gates**:
   `vllm-online-serving` (three rows), where it is the NVFP4 GEMM under the
   denominator and the CUTLASS source tree our own arm is compiled from, and
   `speculative-decoding` (two rows), where it is the oracle's attention
   backend. The `transformers` floor and the `VLLM_ALLREDUCE_USE_FLASHINFER`
   default are **discharged** as reaching no committed gate, each with its scope
   limit recorded there. **`nvidia-cutlass-dsl` is NOT discharged**: a fresh
   review of [#2783](https://github.com/mudler/vllm.cpp/pull/2783) falsified the
   first pass's claim by finding a warmup path gated on
   `has_device_capability(90)` rather than on capability family 100, which
   compiles CuteDSL at engine start on GB10. It cannot move the steady-state
   math, but it can abort engine start and it sits inside the startup ratio
   `docs/benchmarks/vllm-online-serving.md:73` publishes. **The re-measurement itself is still owed**; that report
   ran no job, and §6 records that the committed harness **structurally refuses
   to measure at any revision but the pinned one** — `online_gate.py:3529-3540`
   checks the distribution and runtime versions and `:3542` the commit, all
   before the FlashInfer gate at `:3552-3560`, and all read from the same
   `parity-pin` block. So the block must move before ANY pin advance can be
   measured through this harness, which puts this obligation and that edit in an
   order nothing states.
4. A reading on **`dgx:gpu0`**. Only `thor:gpu0` was measured.

**Evidence for the paragraph above.**
[#2626](https://github.com/mudler/vllm.cpp/issues/2626) is owned by
`MODEL-MM-QWEN4-EXP` and listed under `## Owed` in
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md); the
measurement and its explicit non-claims are in
[`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)
§6.
