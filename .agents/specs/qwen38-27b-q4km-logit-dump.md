# The logit dump: measure our per-step delta against the oracle instead of ranking terms by argument

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Predecessor: [`qwen38-27b-q4km-token-exactness.md`](qwen38-27b-q4km-token-exactness.md),
landed as `15f750fb9`.

## Why this exists

The predecessor row established what the Q4_K_M token gate can and cannot mean,
and it killed four candidate causes. Every one of them was the best-motivated
candidate on its list at the time:

| candidate | how it died |
|---|---|
| `QuantizeQ8KK` / the Q8_K activation quantizer | byte-identical to `quantize_row_q8_K_ref`, 0 of 4.6 M quants |
| a Q6_K port defect | bit-identical to the generic body |
| a Q4_K/Q5_K port defect | differs by the same magnitude ggml differs from itself |
| the bf16 RESIDUAL, two roundings per layer on the 64-layer accumulator | `VT_BF16_RESIDUAL=0` measured: 5 of 6 either way, two prompts WORSE |

The last one was the leading hypothesis of both the implementer and the
operator. **Ranking hypotheses by per-store magnitude is not working**, and it
has now cost four dispatches. Every remaining hypothesis is ranked by argument
because our tree exposes no logit vector on any production path. The 2026-08-23
evidence said that instrument was owed; it is still owed; this row builds it.

## What exists already, and what is missing

The ORACLE half is committed and was exercised on `thor:gpu0`:
`/mnt/nas_share/rc/q4ktok-thor/job/oracle_logits.cpp` teacher-forces stock
`llama.cpp` `b10451` along a supplied id sequence and dumps
`n_predict x n_vocab` f32; `cmp_logits.c` diffs two dumps elementwise. Together
they produced the band this row measures against.

Missing is OUR half: one env-gated dump of the final logit vector per decode
step, reachable from `vllm-bench` through the same seam `--output-token-ids`
already uses.

## The denominator this measures against

Measured 2026-09-02, rc job `deb6322d`, `thor:gpu0`: the oracle against ITSELF,
one artifact and one recipe, differing only in which of ggml's own Q4_K kernels
ran, teacher-forced along one sequence so the vectors are comparable.

```text
GLOBAL max_abs=1.365718e+00 rms=7.886647e-02 n=71516160 argmax_flips=1
```

| per step, 288 steps | min | median | max |
|---|---:|---:|---:|
| max abs logit delta | 0.2020 | 0.3790 | 1.3657 |
| rms logit delta | 0.0412 | 0.0729 | 0.1856 |

That is the noise floor of "which kernel ran" on this model. Our delta gets
placed against it.

## PRE-REGISTERED OUTCOMES

**Written before the instrument exists and before any of our logits have been
seen.** Recorded in advance because this row's whole purpose is to stop
conclusions being assembled after the fact, and because four hypotheses have
already died on this row after sounding decisive beforehand. Whichever branch the
data selects, it selects it here and not in a later reading.

The statistic is the per-step delta between our final logit vector and the
oracle's, teacher-forced along the SAME token sequence so the vectors describe
the same context, summarised the same way `cmp_logits.c` already summarises the
oracle against itself: per-step `max_abs` and `rms`, plus the global figures.

1. **INSIDE the band** -- our per-step `max_abs` distribution overlaps
   0.2020 to 1.3657 and our median `rms` is at or below ~0.0729.
   Then we are AT the noise floor: the 5-of-6 rate is near-tie luck rather than
   an error of ours, no further precision term is worth chasing, and the gate
   needs the kernel-path pin the oracle record now carries and little else. The
   row would then close by tightening the gate's definition, not by changing
   arithmetic.
2. **A FEW TIMES the band** -- our per-step `max_abs` runs roughly 2x to 20x the
   oracle's, i.e. medians in the ~0.8 to ~8 range.
   Then ONE localisable term remains, and the next dispatch is the per-layer
   hidden-state bisect the 2026-08-23 evidence listed third and nobody has run:
   one prompt, one position, our hidden state against llama.cpp's at each of the
   64 layers, with the GDN (`ssm_*`) layers and the 16 full-attention layers
   separated because they are different code.
3. **ORDERS above the band** -- our per-step `max_abs` is 100x or more, i.e.
   medians above ~40 against absolute logits of 15.9 to 22.6.
   Then the precision framing is WRONG and the 282-of-288 rank-1 evidence needs
   re-reading: a difference that large cannot leave the vocabulary ordering
   almost everywhere intact, so either the comparison is mis-aligned (a harness
   defect, and the first suspect is teacher-forcing alignment) or something
   structural survives that the rank statistic hid.

**A fourth result is admissible and must be reported if it occurs:** the deltas
are not stationary across the 48 steps -- for instance small early and growing,
or spiking only at the six contested steps. That shape would say the error
ACCUMULATES rather than being a fixed per-step offset, which none of the three
branches above assumes, and it would redirect the bisect from "which layer" to
"which step".

## Scope

1. One env-gated final-logit dump, written where the logits actually live,
   reachable from `vllm-bench` on its ordinary configuration.
2. Teacher forcing on our side is NOT in scope: the comparison forces the ORACLE
   along OUR ids, which is what the existing harness already does and what the
   recorded margins used.
3. One `rc` job: our run dumping logits, the oracle run teacher-forced along our
   ids dumping logits, then `cmp_logits.c`.
4. Report against the pre-registered branches above. No new hypothesis is ranked
   by argument in this row.

Out of scope: fixing whatever the measurement finds, the incomplete `VT_ACT_F32`
conversion, [#2548](https://github.com/mudler/vllm.cpp/issues/2548), and any
throughput number.

## Risks

- **The dump must not change what it measures.** It is a read of a buffer that
  already exists, env-gated off by default, and the gate is that the token ids
  produced with the dump ON are byte-identical to the same run with it OFF.
  Without that control the instrument could be reporting its own perturbation.
- **Alignment is the first suspect on any surprising result.** Step `i` of our
  dump must be the same context as step `i` of the oracle's. The control is that
  our argmax at each dumped step equals the id already recorded in
  `--output-token-ids` for that step.
- Size: 248320 x 4 bytes x 48 steps x 6 prompts is ~286 MB per side. Written to
  worker-local `/tmp`, never to CIFS mid-run.

## Gates

This row produces a MEASUREMENT and no gate verdict. `TOKEN_GATE` stays as the
predecessor left it: `FAIL`, 5 of 6, and no speed or memory axis is admissible.

## Evidence required

- The rc job ids, the device, the raw log paths.
- The identity control: ids with the dump on == ids with the dump off.
- The alignment control: our dumped argmax == our recorded `--output-token-ids`.
- The per-step delta table in the same shape as the oracle's, and the branch it
  selects, named against the pre-registration above.

## Stop conditions

- Do not weaken the token gate; this row does not touch it.
- If the deltas select branch 3, treat harness alignment as the first suspect and
  prove alignment before concluding anything about the engine.
- Report an unstationary delta shape if it appears, even though no pre-registered
  branch predicts it.

## Now

`ACTIVE`.
