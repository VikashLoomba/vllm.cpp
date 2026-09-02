# Fuse N LoRA adapters, because upstream's aggregator takes a list and its second product rounds differently from its first

Row: `LTX25-LORA-FUSION`.
Issue: [#932](https://github.com/mudler/vllm.cpp/issues/932) — the `## Owed`
entry "N-adapter fusion, which additionally needs upstream's SECOND rounding
pattern (`addmm_` with `alpha`, `fuse_loras.py:115`) that this row refuses
rather than guesses" in
[`ltx25-ic-lora.md`](ltx25-ic-lora.md). Order 8, item **A17** of
[`ltx25-completion-scope.md`](ltx25-completion-scope.md).
Predecessor: [`ltx25-ic-lora.md`](ltx25-ic-lora.md) (the one-adapter arm) and
[`ltx25-lora-fuse-seam.md`](ltx25-lora-fuse-seam.md) (the `vt::Matmul` routing).

**`gh` cannot read #932.** `gh api repos/mudler/vllm.cpp/issues/932` returns
`404` in this session, as do #930 and #975, while #2585 resolves — the scattered
blindness #435/#644/#1854/#2295 record. `REMOTE_UNVERIFIED` is not absence, and
#930 is cited by the predecessor spec as landed in `c7cb59fbb`, so the low range
exists. The issue is referenced and not re-filed.

## 1. The refusal this row lifts, and whether its stated reason was true

`Ltx2ResolveLoraReferenceFactors` refuses a second adapter
(`src/vllm/model_executor/models/ltx2_lora.cpp`, the `adapters.size() > 1`
branch), and `Ltx2FuseLoraIntoTensor` carries a second refusal for the case that
one cannot reach. Two sibling rows in this campaign found their refusal's stated
reason FALSE, so it was checked line by line against the pin before anything was
written.

**It was true.** Every clause holds at Lightricks/LTX-2 `fd4ded7f`:

| the refusal says | read at the pin | verdict |
|---|---|---|
| "Upstream's own dubit.py enforces the same (dubit.py:364-365)" | `:364` `if not args.lora or len(args.lora) != 1:`, `:365` `raise ValueError("Dub-It requires exactly one --lora (the Dub-It IC-LoRA).")` | TRUE |
| "hdr_ic_lora.py takes exactly one (hdr_ic_lora.py:271-272)" | `:271` `lora_path = str(Path(hdr_lora).resolve())`, `:272` `loras = (LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)` — a one-tuple built from a scalar parameter | TRUE |
| "ic_lora.py does accept a list" | `ic_lora.py:75` `loras: list[LoraPathStrengthAndSDOps]` | TRUE |
| "Upstream aggregates them with a SECOND rounding pattern (addmm_ with alpha, fuse_loras.py:115)" | `:115` `aggregated.addmm_(product.b, product.a, alpha=product.strength)` against `:113` `torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)` | TRUE |

So the refusal was an honest statement of a real gap, and this row closes the
gap rather than correcting a false claim. What it did NOT say, and what makes
the gap bigger than "one pipeline entry point wants a list", is in §2.

**The `--lora` CLI is the fourth reader and it has always been N.**
`utils/args.py:600-611` declares `--lora` with `action=LoraAction`, `nargs="+"`,
`default=[]` and the help text "Can be specified multiple times. Example:
`--lora path/to/lora1.safetensors 0.8 --lora path/to/lora2.safetensors`". Every
pipeline that reaches `DiffusionStage.from_checkpoint` through that flag can
therefore be handed N, and `dubit.py` and `hdr_ic_lora.py` are the two that
narrow it back to one. The arity cap mirrored the two narrowest readers and not
the flag.

## 2. The composition upstream needs and this tree could not express

`dfr_pipeline.py:212` builds `stage_loras = (*self._user_loras,
*self._distilled_lora)` and hands that tuple to the single `DiffusionStage` it
builds at `:213`, whose `loras=` argument is `:217`. That is TWO adapters on one
resident DiT whenever a user adapter is supplied alongside the required
distilled one.

This tree routes both through the SAME `lora_path` load extra
(`ltx2_video.cpp`, the `requires_distilled_lora` refusal names
`kLtx2LoraPathExtra` as where the distilled adapter goes), and that extra held
exactly one value. So the DFR and HQ arms could carry the distilled adapter OR a
detailing IC-LoRA and never both, and the arity cap was the reason. The cap is
therefore not only a missing capability on `ICLoraPipeline`; it is the reason a
shipped recipe cannot be given what upstream gives it.

## 3. The arithmetic, measured rather than transcribed

`aggregate_lora_products` (`fuse_loras.py:99-116`) has two forms:

```python
if aggregated is None:
    aggregated = torch.matmul(product.b * product.strength, product.a).to(dtype=dtype)
else:
    aggregated.addmm_(product.b, product.a, alpha=product.strength)
```

The first is landed and gated. The second was NOT transcribed: three candidate
models of `addmm_` on a bf16 aggregator were run against the pinned module
itself, executed at `fd4ded7f` with `torch 2.11.0+cu130`:

| model | rule | matches upstream |
|---|---|---|
| **A** | `bf16( f32(agg) + strength * f32_accumulated(B @ A) )` | **YES** |
| B | the product ROUNDED to bf16 before the strength is applied | no — differed on 3 of 20 elements |
| C | `strength * product` rounded to bf16 before the add | no — differed on 2 of 20 elements |

Model A held byte-for-byte over 40 randomized trials with 2 to 4 adapters and
`M`, `K`, `N` drawn from 1 to 8, and over `K` in {16, 64, 256, 1024} at
`M = N = 3`. **The distinguishing fact is that no bf16 rounding sits between the
GEMM and the strength.** The first form rounds `B * strength` to bf16 BEFORE its
matmul; the second applies `alpha` to an f32 accumulation and rounds ONCE, at
the store. A port that reused the first form's pre-scaling for the second would
be wrong in a way no tolerance-based gate would report, which is why B and C are
recorded here with the element counts that separate them.

`vt::Matmul`'s contract already admits the shape this needs: "a/b float dtypes
(f32/f16/bf16), out **f32** or bf16, f32 accumulation" (`vt/ops.h`). So the
second form is the same shared seam with an f32 output tensor, and no new op is
added.

## 4. Scope

**In.**

- `Ltx2FuseLoraIntoTensor`: the second product form, and the loop over N
  adapters in the order the load supplied them (`_products_for_sd_key`
  iterates `lora_sd_and_strengths` in list order).
- `Ltx2ResolveLoraReferenceFactors`: drop the arity refusal. Its conflict
  branches (`ic_lora.py:155-173`) become reachable for the first time.
- The load extras: `lora_path_2`, `lora_strength_2`, ... `_N` beside the
  existing `lora_path` / `lora_strength`, which keep their meaning for the
  first adapter. `ltx2_gen --lora` accumulates across repetitions, mirroring
  `args.py:600-611`.
- The header and comment statements that assert the cap: `ltx2_lora.h`,
  `ltx2_loader.h`, `ltx2_pipeline.h`, `ltx2_loader.cpp`, `ltx2_pipeline.cpp`.
- `docs/FEATURES.md`, one row.

**Out.**

- **A15** (`ICLoraPipeline`, reference-image and reference-video conditioning)
  and **A16** (`conditioning_attention_strength < 1.0`), the two siblings in
  order 8. Neither is a dependency: N-adapter fusion is complete without a
  reference conditioning path, and the reference refusal
  (`ltx2_video.cpp`) names causes this row does not touch.
- Per-phase adapter STRENGTH (`ti2vid_two_stages_hq.py:92-101`), owed by #1144.
  Lifting the arity cap does not supply it, and `Ltx2PhaseLoraScope` stays a
  two-valued enum — see §5.
- Re-quantizing a fused FP8/NVFP4 weight. Unchanged from the predecessor row.

### 4.1 Why the transport is INDEXED KEYS and not a delimited list

The extras surface is `std::map<std::string, std::string>` and
`VideoExtrasFromArrays` folds the C ABI's parallel arrays into it, so a repeated
`lora_path` key silently last-wins. N adapters therefore need a spelling.

The tree's own precedent for a `nargs` flag is a comma-separated value
(`ApplyStgBlocksExtra`, mirroring `--*-stg-blocks`). **That precedent does not
transfer, because those values are integers and these are PATHS.** A comma is
legal in a POSIX filename, and the only bytes that are not are `/` and `NUL`,
neither of which can serve as a separator in a C string. A delimited encoding
would therefore split some real path in half and refuse it as two nonexistent
files, or worse, name two files that both happen to exist. Indexed keys cannot
be ambiguous, are byte-compatible with every existing caller, and make a gap
(`lora_path_3` with no `lora_path_2`) a refusal instead of a silent renumber.

This is a transport decision and not a mirrored behaviour: upstream's transport
is `argparse`, which this ABI is not.

## 5. `Ltx2PhaseLoraScope` stays two-valued, and its stated reason changes

`ltx2_pipeline.h` argues that `kAllAdapters` / `kNoAdapters` is the COMPLETE
space because "`Ltx2ResolveLoraReferenceFactors` refuses more than one adapter
by name ... so the powerset of the load's adapters has exactly two members".
**That argument dies with this row and the conclusion survives it.** Read at the
pin, no upstream recipe scopes a PROPER SUBSET of its adapters to a phase:

- `dfr_pipeline.py:212` gives `(*user, *distilled)` to the one stage both
  phases share — all, on both.
- `ic_lora.py:108` against `:119` — all on stage 1, none on stage 2.
- `a2vid_two_stage.py:107` against `:114`, `ti2vid_two_stages.py:140` against
  `:151`, `ti2vid_two_stages_hq.py:154` against `:165` — the same two states.

What `ti2vid_two_stages_hq.py` varies per stage is the STRENGTH (0.25 at
`:92-96`, 0.5 at `:97-101`), not the membership, and that is #1144. So the
comment is rewritten onto the reason that is still true, rather than left
asserting a cap that no longer exists.

## 6. Risks

- **The second form is a branch that was previously unreachable.** It is now
  reachable, and a wrong rounding there is invisible to a token gate and to any
  tolerance. Mitigated by §3's golden, generated by EXECUTING the pinned module,
  and asserted as byte equality on the bf16 bit patterns.
- **The f32 intermediate.** The second form needs a `vt::Matmul` with an f32
  output where the first uses bf16. A reviewer should check that the f32 buffer
  is not carried into the first form, which would silently widen the landed and
  gated arm.
- **Reachability.** Lifting a refusal in `ltx2_lora.cpp` alone would land a
  capability no user can request, because the extras surface held one path. The
  transport change is what makes the row reachable, and §8's mutation is on the
  transport and not on the fuser.
- **The derived reader-anchor list** in `ltx2_video.cpp` moves when lines are
  inserted above it. Re-derived from the gate's own printed replacement.

## 7. Gates

```sh
cmake --build build --target test_ltx2_lora test_ltx2_video test_ltx2_loader -j 4
./build/tests/test_ltx2_lora
./build/tests/test_ltx2_video -tc="*lora*"
./build/tests/test_ltx2_loader
python3 scripts/agent-integration.py --base origin/main
```

## 8. Evidence required

- The three-model probe of §3 against the pinned module, with the element counts
  that separate A from B and C.
- Red-before on the N-adapter golden with the second form unimplemented.
- **The reachability mutation**: delete the `lora_path_2` parse in
  `ltx2_video.cpp` and confirm the new `test_ltx2_video` case REDs while every
  `test_ltx2_lora` case stays green. That difference is the whole claim — a unit
  suite that builds two `Ltx2LoraAdapter`s by hand measures the class.
- What the goldens do NOT cover, stated.

## 9. Stop conditions

- `NEEDS_DECISION` rather than growing into A15 or A16.
- `NEEDS_DECISION` rather than inventing a per-phase adapter SUBSET that no
  upstream recipe asks for.
- Do not use the GPU. This row is CPU-only by construction: the fusion runs on
  the host before the device copy.

## Now

`ACTIVE` — N-adapter fusion implemented, gated against the executed pinned
module, and reachable through the `lora_path_N` load extras and repeated
`ltx2_gen --lora`.
