# LTX25-ANCHOR-REPAIR — resolve the campaign's `path:line` anchors against the tree that is here now

Issue: [#1230](https://github.com/mudler/vllm.cpp/issues/1230) (the tree-wide
off-by-N sweep this row takes) — state `REMOTE_UNVERIFIED`, see §7.

## Scope

Twelve `## Owed` items across `.agents/specs/ltx25-*.md` name something they
call an anchor. This row resolves every **`path:line` citation** in those specs
against the current tree, corrects what is correctable, and says plainly what is
not.

IN SCOPE:

1. Census every `path:line` citation in `.agents/specs/ltx25-*.md`, resolve each
   against the tree, and classify it.
2. Take the one sweep that is mechanical, uniform and re-derivable: #1230's two
   off-by-N upstream anchors, re-derived at the `ltx-2` pin rather than
   inherited from the citing prose.
3. Repair the corrections the owed items **wrote down and that have themselves
   since rotted**, in a form that cannot rot again.
4. Close the owed items the tree has falsified, on that evidence.

OUT OF SCOPE, and each says why:

- **Extending `scripts/check-symbol-anchors.py` to the `path:line` form.**
  `ltx25-text-linear-mem.md` `## Owed` asks for it and says it needs its own
  spec and its own red-first mutation. It does. A checker that resolves a bare
  line number has no expectation to check it against — that is the #911
  tautology, which reported 27/27 FRESH while five anchors pointed at unrelated
  code — so the design question is what the expectation IS, and answering it is
  a row, not a follow-up line. `NEEDS_DECISION` is returned rather than widened
  into.
- **The repo-wide sweep of #911.** `ltx25-prompt-adaln.md` `## Owed` names
  `ENG-RECORD-ANCHOR-RATCHET` (#632) as the row that should absorb it. This row
  pays one class of it and does not claim the rest.
- **`.agents/completed/`.** It is the frozen archive. AGENTS.md `## Records`
  keeps its provenance, and `check-symbol-anchors.py`'s `FROZEN_PREFIXES` agrees.
  Rewriting an archived citation would forge what a past session wrote.
- **Everything the word "anchor" also means in this campaign.** See §2.

## 1. The census

`.agents/specs/ltx25-*.md`, 46 files at `ed5ecea2e`, matched with the
line-anchor form of `check-symbol-anchors.py`'s own `CITATION_RE`
(`scripts/check-symbol-anchors.py:87-90`, an anchor this row re-resolved and
found FRESH). Every count below is stamped with that tree, because a bare count
in a document is a drift lock:

| Class | Count |
|---|---:|
| Total `path:line` citations | 1,668 |
| Resolvable in this tree (full path, or a basename unique among tracked files) | 649 |
| Not a repo path — the pinned `ltx-2`, `vllm-omni` and `diffusers` oracles, or an external file | 1,013 |
| Basename ambiguous across tracked files | 6 |
| Resolvable and **out of range** — the cited file is shorter than the cited line | 10, of which 1 is a deliberate illustration |

Out-of-range is the weakest available test and it is the only *mechanical* one:
a line number carries no claim, so "line 504 exists" is true of any file with
504 lines. Every verdict below the out-of-range set was taken by reading the
citing prose for what it claims and then reading the cited span, which is the
same two-sided rule `check-symbol-anchors.py` states in its own header — the
expectation comes from the CITING file and the evidence from the cited one.

## 2. "Anchor" means three different things here, and only one of them is a citation

This is the first finding, because it changes what the twelve items are:

| Sense | Items | In this row |
|---|---|---|
| A `path:line` **citation** into a source file | `ltx25-oracle-absolute`, `ltx25-text-linear-mem` (x3), `ltx25-dit-attn-flash`, `ltx25-keyframe-interp` #1230, `ltx25-prompt-adaln` #911 | YES |
| A **schedule anchor** — the sigma-schedule attachment point of a phase | `ltx25-keyframe-interp` #1150 and #1220, `ltx25-ti2vid-recipe` #1150 | NO |
| A **phase-log scope anchor** — where a `RenderPhaseLog` scope opens and closes | `ltx25-device-residency`, `ltx25-test-determinism` | NO |

The second and third senses are engine behaviour with their own gates and their
own owning rows. Reading them as citation staleness — which the word invites —
would have produced a change that repaired nothing and touched a landed
measurement. They are reported, not repaired.

## 3. #1230 — the two off-by-N upstream anchors, re-derived at the pin

Re-derived by reading the pinned files, not by inheriting the citation.
`~/_git/LTX-2` at `HEAD = fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, the
revision `.agents/oracles/ltx-2.md` records, read with `git show <pin>:<path>`
so the working tree cannot substitute another revision:

| Cited | What is actually there at `fd4ded7f` | Correct |
|---|---|---|
| `latent_cond.py:38-39` | `:38` is `latent_state = latent_state.clone()`, `:39` is blank | `:40-41`, the two writes to `clean_latent` and `denoise_mask` |
| `schedulers.py:31` | `:31` is the return annotation `) -> torch.FloatTensor:` | `:32`, the `tokens = math.prod(latent.shape[2:])` read |

The sweep is uniform and takes every live site in one commit. A PARTIAL
correction is strictly worse than none — `ltx25-keyframe-interp.md` `## Owed`
records why: a reverted half-correction left `ltx2_video.cpp` citing `:38-39` in
one comment and `:40-41` a hundred lines later, with nothing telling a reader
which to believe, and a uniformly wrong anchor is fixable by one grep while a
mixed one is not.

Two obstacles that spec named are now gone, and that is what makes the sweep
takeable here rather than owed on:

- `scripts/check-doc-checkpoint.py` — whose `USER_USAGE_PREFIXES` made a
  comment-only edit in `include/vllm/` demand a `docs/USAGE.md` edit the change
  did not owe — **no longer exists in this tree.**
- The sweep touches only comments and spec prose. No expression, declaration or
  test assertion moves.

## 4. The corrections that have themselves rotted

Three owed items recorded a correction in prose because the record they were
correcting could not be edited. Two of those corrections are now stale:

| Owed item | It recorded | At `origin/main` today |
|---|---|---|
| `ltx25-text-linear-mem` | `cpu_ops.cpp:125` should read `:130` for `static thread_local std::vector<float> af` | `af` is at `:225`. `:130` is not it |
| `ltx25-oracle-absolute` | `examples/ltx2_gen/main.cpp:306-451` should read `:318-476` | The flag loop reaches past `:490`. `:318-476` no longer covers it |
| `ltx25-oracle-absolute` | `tools/oracle/ltx2_oracle.py:88` should read `:89` | HOLDS. `:89` is `NUM_INFERENCE_STEPS = 8` |

The repair is not a third line number. It is the anchor form AGENTS.md and
`check-symbol-anchors.py` both ask for: name the SYMBOL, which survives every
edit that does not rename it, and which the existing gate already checks. A
correction written as a line number is a correction with an expiry date, and
these two expired inside two weeks.

## 5. What the tree has falsified

- **`ltx25-dit-attn-flash`: "`scripts/attention-rung-allowlist.txt` will carry
  two STALE stems."** It does not. The file's IN FLIGHT section is EMPTY, and
  its own header records `ltx2` and `ltx2_device` as deleted under #1663 by the
  preflight run the handoff named. The item's stated close condition —
  "Owner: this row until it is deleted" — is met.
- **`.agents/issue-index.md` no longer exists.** It is retired to
  `.agents/completed/issue-index.md` and the index is DERIVED
  (`scripts/agent-issue-index.py --refresh`). Both items that pleaded
  append-only un-correctability are therefore falsified IN THEIR REASONING: the
  rows are unreachable now because they are archived, which is a different rule
  with a different scope. The conclusion is unchanged and the reason is not, so
  the reason is corrected rather than the bullet deleted.

## 6. What stays owed, and why it cannot be repaired

- **`runguard.py:236-237,260`.** The file is outside this repository on a
  mutable NAS path and was rewritten after the citing commit. No revision of it
  is pinned, so there is no text to resolve the citation against — this is not a
  number that needs correcting, it is a citation with no denominator. The claim
  it supports does not rest on it, and `ltx25-text-linear-mem.md` already gives
  the substitute evidence from the retained `memguard.tsv`.
- **`docs/BENCHMARKS.md:465` and `:494`.** `docs/BENCHMARKS.md` is 18 lines: it
  is now an index of `docs/benchmarks/<benchmark-id>.md` detail files. Neither
  line exists and neither has a successor line — the content moved to another
  file. Repointed by file and section name, which is what survives the split.
- **The five `examples/ltx2_gen/main.cpp` anchors inside the archived index.**
  Frozen. The correction lives in the citing spec, in the rot-proof form of §4.

## 7. `REMOTE_UNVERIFIED` on every issue number this row cites

`gh` on this host authenticates as `localai-org-maint-bot` and sees 210 of the
repository's ~2,477 issue numbers in any state. `gh issue list` works — it
returns #41 and #2477 — while `gh api repos/mudler/vllm.cpp/issues/<n>` returns
404 for #911, #1230, #1663 and #2130 alike, and #2130 is cited by a landed
record so it demonstrably exists. A number absent from a listing this partial is
`REMOTE_UNVERIFIED`, never absence. No issue is opened, closed or edited from
this row, and no claim is made about any issue's state.

## 8. Gates

```sh
python3 scripts/check-symbol-anchors.py
python3 scripts/check-symbol-anchors.py --self-test
scripts/agent-preflight.sh --staged
```

The symbol anchors this row introduces are gated by the first command, which is
the point of writing them in that form. The line anchors it corrects are gated
by nothing, which is the debt §"Scope" hands back as `NEEDS_DECISION`.

## 9. Risks

- **A blind sweep of a stale string rewrites the prose that DESCRIBES it.**
  `ltx25-keyframe-interp.md` quotes `latent_cond.py:38` in order to say what is
  at that line. Those occurrences are history and are excluded by hand, not by
  pattern.
- **An anchor goes stale inside its own pull request.** Every anchor this row
  writes is re-resolved against the branch head after the last commit, not
  against the tree it was written on.

## Now

`DONE`.

## Owed

- **The `path:line` class is still gated by nothing.** 1,668 of them in the
  `ltx25-*` specs alone. Extending `check-symbol-anchors.py` is
  `NEEDS_DECISION`, returned rather than taken: see §"Scope".
- **The 643 resolvable, in-range `path:line` citations are not individually
  verified.** Out-of-range is mechanical; in-range is a reading. This row read
  the ones the twelve owed items name and the ones a corroboration sweep flagged,
  and it does not claim the remainder are fresh — it claims nothing about them.
- **`runguard.py` and `docs/BENCHMARKS.md:465`/`:494`** stay uncorrectable for
  the reasons in §6. Repointing, not repair, is what was available.
