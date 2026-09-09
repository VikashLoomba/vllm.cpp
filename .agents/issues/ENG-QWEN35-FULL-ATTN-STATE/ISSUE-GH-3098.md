ID: ISSUE-GH-3098
Title: fix(ENG-QWEN35-FULL-ATTN-STATE): validate state only for GDN consumers
Row: ENG-QWEN35-FULL-ATTN-STATE
State: OPEN
Kind: UNKNOWN
GitHub: 3098
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-QWEN35-FULL-ATTN-STATE`
>
> ## Defect
>
> A valid Qwen3.5 GGUF with one full-attention layer loads through the public API, then fails its first completion. The engine reports `GDN state index out of range` despite the model containing no GDN layers.
>
> This blocks the native ROCm quantized-gather regression at the public completion call. The fixture sets `qwen35.full_attention_interval=1`, hidden width 256, four query heads, one KV head, head width 64, and vocabulary size 128.
>
> At base `6db4bef906859e864c82523c01107473f7dcca29`, the executing chain is:
>
> - `MakeQwen3_5KVCacheSpec` publishes a GDN group even when no layer consumes it.
> - The runner allocates no recurrent buffers but constructs request GDN metadata.
> - `CheckDensePagedForward` verifies that both the model's GDN layer count and cache count are zero.
> - It then validates that request metadata against zero state slots and rejects live slot 0.
> - `BuildStepDevInputs` also validates and uploads GDN metadata unconditionally.
>
> The shared GDN validator was introduced in `f344decf4`. Its owning specification requires validation for actual GDN consumers. A zero-slot bypass alone could hide malformed recurrent models and is not an acceptable repair.
>
> ## Reproduction
>
> The operator reproduced the failure on local gfx1100 under the GPU mutex. The public test loads the model and requests four greedy tokens from `[1,0,63,127,63]`. It fails before the gather-provider assertion.
>
> The test-only gather binary has SHA256 `f6a109ef33a133b381a112313591534f560adedb0d4919f0d67567cd597552e1`. It is built from specification commit `670e6d78ddf55231394748e0032939fd53dc56a5` plus the new public test. The result is exit 1 with five passing assertions and one completion failure.
>
> ## Required result
>
> Respect the model's actual GDN consumers when validating and preparing recurrent state. Preserve rejection of missing, malformed, duplicate, and out-of-range state for models that contain GDN layers. Prove public completion for the model without GDN layers, including prefill and decode, and mutation-test both sides of this distinction.
>
> The fix requires a committed specification and independent implementation review. It changes no quantized provider or CI configuration.
>

## Resolution

The scoped implementation passes public completion through both default and
callback sampling arms. The operator reproduced the independent red and reran
all public and focused CPU gates. The implementer completed 30 mutation checks.

Fresh review found missing negative coverage for the first query offset and
internal offset ordering. The test repair covers both graph siblings.
Each independent guard deletion now fails its corresponding new subcase.
The unchanged control and four complete focused CPU suites pass.
The specification records this repair's hashes, commands, and evidence.

The exact GGUF attempt on pinned vLLM refuses the text-only engine configuration
before generating a token. The owning [specification](../../specs/qwen35-full-attn-state.md)
records the source hashes, commands, logs, and pending G4 obligation. Fresh
review and the final operator gate remain pending. Keep this issue open until
qualified work lands.

The full-attention runner regression now requires successful execution and
sampled-token feedback instead of the former GDN refusal. Its focused case
passes 18 assertions and the complete CPU runner suite passes 41 cases and
1914 assertions. Restoring unconditional dense GDN validation fails the new
case at the original refusal. The scoped spec amendment precedes this test-only
repair; runner allocation, group topology, fixtures, and product code stay
unchanged. The specification records the exact red, green, and mutation evidence.
