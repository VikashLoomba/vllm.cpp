# The ratified distributional gate, scored on the Qwen3.8-27B Q4_K_M arm

Row `QUANT-QWEN38-27B-GGUF-ARM`, under
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). This spec adds no
product code and changes no gate. It locates the ratified near-tie distributional
gate, states its exact terms, pre-registers the pass and fail rule with every
outcome, and then scores this arm against it in the next commit.

## 1. Scope

In scope:

- Locate the ratified form and quote its binding terms from the authoritative
  text and from the code that implements it.
- Pre-register the rule and every outcome, before the score.
- Score both limbs separately, from data already committed.
- Report the count of steps where our token is not in the oracle's distribution.

Out of scope:

- Any change to engine numerics.
- Any speed, latency or memory number. The arm's declared token gate is `FAIL`,
  and `AGENTS.md` §Gates admits no performance result from it.
- Any change to the declared token gate. The row's own spec forbids it.
- A new `rc` job. Every input the ratified gate needs is already committed, and
  no GPU work can change a verdict that is already determined by them.

## 2. The ratified form

### 2.1 Two artefacts, and which one is the gate

The tree holds two near-tie artefacts, and they are not interchangeable.

**The band** is a component. `constexpr int32_t kNearTieMnats` appears in 24 test
files and every one of them gives it the value 500. It admits a position whose
teacher-forced gap is at most 0.500 nats.

**The ratified near-tie DISTRIBUTIONAL gate** is the complete instrument, and it
is what a request for a distributional gate means here. It was approved by the
developer on 2026-07-27 under `CLAIM-MM-SPEED-DECODE-KERN-ADOPT`. Its
authoritative text is §12.2 of
[`multimodal-speed.md`](multimodal-speed.md). Its implementation is
[`../../scripts/mm/a3_voxtral_neartie_gate.py`](../../scripts/mm/a3_voxtral_neartie_gate.py)
and
[`../../tests/vllm/multimodal/test_voxtral_e2e.cpp`](../../tests/vllm/multimodal/test_voxtral_e2e.cpp),
with the fixture `voxtral_neartie.json`. The
[`../feature-matrix.md`](../feature-matrix.md) `MODEL-MM` row records the
adoption.

The band alone is not the gate. §12.2 and the test add a second conjunct
specifically so that it cannot be.

### 2.2 The binding limb, quoted

§12.2:

> **BINDING CORRECTNESS = the teacher-force PASS** (`result==PASS` +
> `n_divergent==0` + `over_band_failures==0` + `worst_gap <= 0.5`).
> Kernel-INDEPENDENT: both the scalar AND the FA2 branch teacher-force PASS. This
> is the pass/fail verdict.

Four conjuncts, all required. `test_voxtral_e2e.cpp` asserts each of them:

```cpp
CHECK(nt["result"].get<std::string>() == "PASS");
CHECK(fixture_n_divergent == 0);  // the validated sequence IS vLLM's argmax throughout
CHECK(fixture_over_band == 0);    // no divergence exceeded the 0.5-nat near-tie band
CHECK(fixture_worst_gap <= 0.5);
```

**`n_divergent` is the conjunct that decides this arm, so its definition is taken
from the code rather than from prose.** In
`scripts/mm/a3_voxtral_neartie_gate.py`:

```python
gap = top_lp - our_lp
is_argmax = gap <= 1e-9
if not is_argmax:
    n_div += 1
```

`n_divergent` counts every position where our token is **not** the oracle's
teacher-forced argmax, at a tolerance of 1e-9 nats. It is not a count of
over-band positions; the harness counts those separately into
`over_band_failures`, and the harness's own `result` field is computed from those
alone. §12.2 requires `n_divergent == 0` **on top of** `result == PASS`, which is
exactly why the band cannot carry the verdict by itself.

So the binding limb is not "within 0.5 nats". It is: **our token IS the oracle's
teacher-forced argmax at every position**, and the 0.500 band bounds how far a
position may sit from that when it is an exact tie resolved the other way.

**Where the verdict is computed, stated because the test says so itself.** The
teacher-force runs offline in Python against the live oracle, and its verdict is
committed into `voxtral_neartie.json`. The C++ test does not hold an oracle
in-process, so its four `CHECK`s pin the fixture rather than recompute it. Its
own comment says this, and says why: PR #439 finding F7 corrected an earlier
"BINDING CORRECTNESS" label on those constants, because "the FA-2 arm prints
`divergent=0 worst_gap_nats=0` while FAILING, precisely because those two came
out of the file rather than out of `got`". The verdict is therefore the Python
harness's output, and §5 scores this arm against that computation, using the
teacher-forced ranks and gaps this row's evidence documents already record.

The Voxtral precedent passed at `worst_gap 0.0000`. §12.2 records it in those
terms: "every one of the 48 FA2 tokens IS vLLM's teacher-forced argmax", and the
two positions where it leaves the strict golden are "both exact bf16 ties
(gap 0.000)". An exact tie has `gap` 0 and therefore does not increment
`n_divergent`. A strictly positive gap does.

### 2.3 The second limb, quoted

§12.2:

> **Strict prefix** -- token-exact vs vLLM greedy up to the first genuine bf16
> exact tie.

Asserted as `CHECK(strict_prefix >= 18)` in the Voxtral case, where 18 is the
position of a two-way EXACT tie: "FA2 tok 24466 vs golden 1584, IDENTICAL logprob
-1.9875, gap 0.000".

The limb is not satisfied by any prefix of any length. The prefix must **end at a
genuine exact tie**, because that is what makes both continuations valid. A
prefix that ends where the oracle's logits do separate the two tokens is a
divergence, not a tie.

### 2.4 The determinism anchor, which is not the bar

§12.2 names a third element and explicitly excludes it from the verdict:

> **Determinism anchor** (NOT the correctness bar) -- `got == nt_tokens`: the
> build reproduces the offline-teacher-force-validated ... sequence, guarding
> against silent decode regressions.

It is recorded here so nobody mistakes a reproducibility result for a correctness
result.

### 2.5 What "kernel-independent" means, read from the text

§12.2's "Kernel-INDEPENDENT: both the scalar AND the FA2 branch teacher-force
PASS" is about **our** two decode kernels, not the oracle's. The verdict must not
depend on which of our branches executed. This spec reads it that way because
that is what the sentence says, and it records the reading explicitly because the
other reading is available and would change what §3.1 demands.

## 3. The pre-registered rule

Fixed before the score. The score is in the next commit on this branch, so Git
order records that the rule was written down first.

**This is not a blind pre-registration and must not be read as one.** Every
figure the next commit reports was already committed to this tree, and nobody
could write this rule without first reading the evidence documents that carry
them. What §3 fixes is the rule, before it is applied and before any
re-measurement.

### 3.1 The denominator, fixed in advance

The oracle is llama.cpp at pin `10bf611e533d81f739128304991c5e133c6aebd8`
(tag `b10451`), **stock default configuration**, with `use_extra_bufts`,
`n_ubatch`, the RESOLVED flash-attention type, `system_info`, host architecture
and thread count asserted and recorded per run. This restates the rule already
committed at `8e499f68b`.

**Exactly one configuration is the denominator.** No second configuration may
enter it, at any step, for any reason. A denominator chosen per step is not a
denominator.

Under §2.5's reading, the kernel-independence requirement applies to **our**
tiers: a verdict is reported per tier, and a verdict that differs between our CPU
and ROCm tiers is reported as differing rather than as one number.

### 3.2 K

K is not a parameter. The gate teacher-forces the oracle once along our ids and
reads its logits. It does not sample the oracle K times. Any document reporting a
K for this gate is reporting something else.

### 3.3 The outcomes

| # | Condition | Verdict |
|---|---|---|
| O1 | `n_divergent == 0` and `over_band_failures == 0` and `worst_gap <= 0.5` and the strict prefix ends at an exact tie | `DIST_GATE=PASS`, and it still needs the maintainer's explicit ratification before it may be quoted for this arm |
| O2 | `n_divergent > 0`, every gap inside the band | `DIST_GATE=FAIL` on the binding limb. Report the band result separately and never as the verdict |
| O3 | `over_band_failures > 0` or `worst_gap > 0.5` | `DIST_GATE=FAIL`, forward divergence |
| O4 | our token outside the oracle's top-K anywhere | `DIST_GATE=FAIL`, forward divergence |
| O5 | any conjunct holds only because the denominator gained a second configuration | **Not scoreable.** Refuse and escalate |
| O6 | the two tiers disagree | Report per tier. One tier's result is never the arm's result |

O2 is the outcome this rule exists to make unmissable: a band pass with
`n_divergent > 0` is a **FAIL**, because the band is one conjunct of four.

O5 is the search trap. Membership that appears after the denominator grows is
search, not membership.

### 3.4 The two counts to report, and which is the gate's

The brief asks for the count of steps where our token is not in the oracle's
distribution. Two readings exist and they give different numbers, so both are
reported and the gate's own reading is named:

- **The gate's reading**, and the one that decides: `n_divergent`, the count of
  positions where our token is not the oracle's teacher-forced argmax.
- **A top-K membership reading**: the count of positions where our token is
  absent from the oracle's top-K. This is not the gate's conjunct. It is reported
  because the brief asks for it, and because a zero there is uninformative when
  `n_divergent` is not zero.

### 3.5 What a result may never do

A distributional-gate result never replaces `TOKEN_GATE`. `TOKEN_GATE` stays
exact and stays `FAIL` until 6 of 6 prompts agree with the stock default. No
result here unblocks a speed, latency or memory number.

## 4. The prior refusals, which stand

Five landed statements refuse a band for this arm, three of them written by the
sessions that took the 2026-09-02 measurements.

1. **The row's own spec §Gates**, in
   [`qwen38-27b-q4km-token-exactness.md`](qwen38-27b-q4km-token-exactness.md):
   "The near-tie band is not available and is not reached for." Its stop
   conditions add "Do not widen, weaken or delete the gate."
2. **The self-consistency evidence**: "The tolerance stays exact and no band is
   reached for", and "It does not clear the arm."
3. **The oracle pin**, [`../oracles/llama-cpp.md`](../oracles/llama-cpp.md):
   "This is not a licence to excuse a divergence against this oracle. Being
   inside the oracle's noise band per divergence is not the same as being at its
   noise floor by rate."
4. **The parent spec §Gates**, in
   [`qwen38-27b-quant-arms.md`](qwen38-27b-quant-arms.md): "The ratified
   near-tie band applies only where the oracle's own greedy decode is
   non-deterministic".
5. **The path-pin evidence** enumerated a set-membership form as its option B,
   recorded "This is weaker and it is not adopted here", and escalated the
   question as "a product decision and is not the implementer's to take".

This spec does not overturn any of them. It scores the ratified gate because the
brief asked what that gate says, and §5 reports what it says.

## 5. The score is not in this commit

Sections 5 and 6 are added by the NEXT commit on this branch, so that Git order
records the rule as fixed before the document read a number back. §3 states what
that ordering does and does not prove. Sections 9, 12 and 15 name §5 and §6
already; they describe the finished document, and this commit is the half that
must exist first.

## 6. Limb 2 and the bigger-model strict limb

The 2026-07-20 methodology that introduced the near-tie doctrine carries a
further requirement, recorded in the developer's ratification note as item 3:

> ALSO verify the forward on a BIGGER dense model ... where vLLM IS
> deterministic, for a clean STRICT token-exact pass.

The §12.2 gate's own second limb is the strict PREFIX of §2.3, which is a
different obligation. Both are reported in §5, separately, and neither is treated
as satisfying the other.

**No vehicle exists for the bigger-model strict limb.** The search covered
[`../quantization-matrix.md`](../quantization-matrix.md),
[`../model-matrix.md`](../model-matrix.md),
[`../parity-ledger.md`](../parity-ledger.md), `docs/bench-evidence/`, `tests/`
and `scripts/`. Three candidates were found and each is disqualified.

**Candidate 1, the APEX GGUF gate.** The tree's only strict token-exact GGUF gate
against a llama.cpp oracle:
[`../../tests/parity/test_qwen36_gguf_engine.cpp`](../../tests/parity/test_qwen36_gguf_engine.cpp)
asserts `got == want_greedy_ids` over 2 prompts and 16 tokens. Four
disqualifications, each from the gate's own committed manifest
`tests/parity/goldens/qwen36_gguf_35b/manifest.json`:

1. **The oracle is not pinned.** The manifest names "llama.cpp fork (dgx.casa
   `~/llama-phase93-qwen3next-gqa-bcast`, arch `qwen35moe`)" -- a
   developer-local fork with no commit recorded anywhere in this tree, not stock
   `b10451`. `AGENTS.md` requires every oracle to be pinned, so this gate
   certifies nothing about the pinned oracle this row uses.
2. **The forward recipe differs.** The manifest states "ours: dequant->bf16 +
   bf16 GEMMs". This arm's production default is the keep-quant path.
3. **The quant mix differs**: `F32+Q3_K+Q4_K+Q6_K` and `F32+Q8_0+Q5_K+Q6_K`, not
   a Q4_K-dominant file.
4. **Its prompts were selected away from near-ties.** The manifest records oracle
   top-2 margins of 0.15, 0.28 and 1.91 nats, and discloses that a third prompt
   was excluded because the oracle's own top-2 there are a 0.040-nat near-tie.
   Selecting for separation is the right property for this limb; it is recorded
   so nobody reads the exclusion as a defect.

**Candidate 2, the Qwen3.8-27B bf16 arm.** Gated against vLLM, and not a strict
pass: [`qwen38-27b-bf16-gate.md`](qwen38-27b-bf16-gate.md) records 4 of 7 prompts
strict with the other 3 adjudicated as exact ties inside the band. It also loads
safetensors and never enters the GGUF dequantization path.

**Candidate 3, a smaller Q4_K_M model.** Not found.

**No harness exists either.** The ratified instrument reads vLLM
`prompt_logprobs`. No `scripts/*neartie*` harness in the tree takes a llama.cpp
oracle, and the two in-tree uses of `kNearTieMnats` beside a GGUF target are
self-referential: `tests/parity/test_qwen35_gguf_spec_decode.cpp` states "There is
no vLLM oracle for a GGUF target" and applies its band to a CPU-against-GPU delta
inside our own engine, and `tests/vllm/models/test_deepseek_v4_gguf_load.cpp`
compares our keep-quant forward against our own dequantizing build.

## 7. Ratification is the maintainer's, not this spec's

Whether the ratified distributional gate may be applied to this arm at all is a
product decision. The path-pin evidence already escalated it under #2534 and
recorded that it "is not the implementer's to take".

**This spec establishes what the gate WOULD say. It does not establish that the
gate applies here.** Nothing in §5 may be quoted as a ratified result for this
arm, in either direction, without that decision.

## 8. Design

No code. §3 is the part with future value: the rule is fixed, so a later
ratification does not get to choose a rule after seeing a number.

## 9. Risks

- **§5 gets quoted without its conjunct structure.** A band figure read alone
  reverses the verdict. §3.3 O2 exists for this, and §5 states the four conjuncts
  beside their results.
- **The two counts of §3.4 get conflated.** They differ, and only one is the
  gate's.

## 10. Tests

None. No product code changed and no gate changed. Every figure in §5 names the
committed document it came from.

## 11. Gates

`TOKEN_GATE` is unchanged and stays `FAIL`. This spec adds no gate and relaxes
none.

## 12. Evidence

The gate's terms are quoted from `multimodal-speed.md` §12.2,
`scripts/mm/a3_voxtral_neartie_gate.py` and
`tests/vllm/multimodal/test_voxtral_e2e.cpp`. Every figure in §5 is transcribed
from a committed evidence document named beside it.

## 13. Stop conditions

- Do not widen, weaken or delete the declared token gate.
- Do not report a band result as the distributional gate's verdict.
- Do not add a second configuration to the denominator.

## 14. Now

`QUANT-QWEN38-27B-GGUF-ARM` does not change lifecycle state. `TOKEN_GATE` stays
`FAIL`.

## 15. Owed

- The ratification decision under
  [#2534](https://github.com/mudler/vllm.cpp/issues/2534). §5 adds an input to it.
- The unpinned oracle behind the APEX GGUF gate, found by §6's search.
  `AGENTS.md` requires every oracle to be pinned and that one names a
  developer-local fork branch with no recorded commit. Reported here rather than
  fixed, because it belongs to the rows that gate on it.
- The ROCm-local numerics term at `p1/45`, `p3/45` and `p4/14`, owned by
  `BACKEND-ROCM`.
- The aarch64 `ggml_gemv_q4_K_8x8_q8_K` port, unchanged and still owed.
