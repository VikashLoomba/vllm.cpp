# ENG-ATTENTION-WINDOW — one owner for the sliding-window rule

Row: `ENG-ATTENTION-WINDOW`. Issue:
[#2388](https://github.com/mudler/vllm.cpp/issues/2388).

## Scope

| | |
|---|---|
| In | Route the existing per-model sliding-window derivations through `ResolveAttentionWindow`, which already implements the rule and is reached by nothing. Reconcile what that exposes. |
| Out | Changing any model's ATTENTION MATH. The decoder result is identical and must stay identical; this row is about who owns the rule, not what it computes. |
| Out | The encoder-only symmetric case. The resolver implements it, no model in this tree asks for it, and inventing a caller would be worse than leaving it unreached. |

## What is actually true, measured rather than assumed

`ResolveAttentionWindow` (`src/vllm/model_executor/layers/attention/attention.cpp:12`)
returns, for the decoder path, exactly `{window - 1, 0}` — **byte-identical to
what all seven call sites already inline**. The refactor is therefore
behaviour-preserving on the value, and that is the premise the whole row rests
on. It is not a rewrite of attention.

It differs from the inline copies in three ways, all additive:

1. a `[1, INT32_MAX]` bound that THROWS outside the range;
2. the encoder-only symmetric `{radius, radius}` case;
3. per-layer window taking precedence over the model window, with a
   model-level disable.

**The lower bound is already enforced everywhere, by hand.** Every one of the
five model sites guards `*sliding_window > 0` before constructing the window:
`gemma2.cpp:197`, `gemma3.cpp:192`, `gemma4.cpp:335`, `olmo2.cpp:195`,
`muse_glimmer.cpp:229`. So adopting the resolver cannot start throwing where the
tree previously computed something — which is the risk that would have made this
row dangerous, and it is absent.

The UPPER bound is enforced nowhere today. No realistic window approaches
`INT32_MAX`, so adopting it is a refusal that should never fire; it is worth
having for the same reason the lower one is.

## The correction this row exists to record

An earlier reading of #2388 stated that the resolver's
`disable_model_sliding_window` parameter was "doubly unreachable" because no such
flag exists in the tree. **That is wrong.** It exists, under a different name and
twice:

- `gemma2.cpp:66` `SlidingWindowEnabled()` reading `VT_GEMMA2_SLIDING`
- `gemma3.cpp:71` `SlidingWindowEnabled()` reading `VT_GEMMA3_SLIDING`

Two identical functions, two environment variables, one behaviour. The grep that
missed them looked for `disable_sliding_window`, the upstream spelling, and a
failed grep is not proof of absence.

## What routing exposes, which is the real finding

The kill switch exists for **two of five** models. `gemma4`, `olmo2` and
`muse_glimmer` have no way to turn sliding-window attention off, so the same
debugging step that works on Gemma-3 is unavailable on Gemma-4. That asymmetry is
invisible while the rule is copied per model and becomes obvious the moment one
function owns it.

Both variables are on `scripts/env-doc-allowlist.txt` rather than in
`docs/ENVIRONMENT.md`, so neither is user-facing today.

## Work breakdown

**W1 — adopt the resolver at the five model sites.** Each site keeps its own
extra predicate (`use_rope` for muse_glimmer, the routing decision for gemma3's
per-layer pattern); those are model shape, not the window rule, and must not move
into a shared function. The value each site computes must be unchanged, and a
test asserts that per site rather than asserting the resolver in isolation.

**W2 — the two shared-path sites**
(`v1/attention/backend.cpp:323`, `mla_chunked_context.h:363`) are NOT model code
and take an already-resolved `sliding_window`. They are listed for completeness
and deliberately deferred: they sit under paths this row cannot exercise on CPU,
and moving them without a device gate would be the same unverified change this
campaign has been closing, not opening.

**W3 — the kill-switch asymmetry.** A decision, not a refactor: either every
model gets one under a single spelling, or the two that have one lose it. Owed,
not chosen here, because it changes a user-visible debugging surface.

## Gates

- Per-site equality: for each of the five, the resolver's output equals the
  inline expression it replaces, over a table of windows including 1 (radius 0)
  and the largest plausible value.
- The `[1, INT32_MAX]` refusal fires, and fires by name.
- Mutation: break the resolver's `radius = window - 1` and require the per-site
  equality tests RED on a mutant that COMPILES.

## Risks

**The one that matters.** If any site's guard differs from `> 0` in a way this
spec missed, adopting the resolver changes behaviour silently, because the value
is only wrong for inputs the old guard excluded and the new one does not. The
per-site equality test is written against the ORIGINAL expression for that
reason, rather than against a shared expectation.

**Not CPU-testable end to end.** These sites feed paged attention; a value test
pins the window, not the attention. That is stated rather than implied.

## Now

W1 in progress. W2 and W3 owed.
