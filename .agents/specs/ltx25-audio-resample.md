# LTX-2.5 — arbitrary-ratio audio resampling (A19)

Row: `LTX25-AUDIO-RESAMPLE`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#2583](https://github.com/mudler/vllm.cpp/issues/2583). Plan:
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) §8 order **7**, gap
**A19**, itself issue
[#2526](https://github.com/mudler/vllm.cpp/issues/2526).

Upstream pins:

| Reference | Registry id | Revision |
|---|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `ltx-2` | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| torchaudio, the module `ops.py:40` calls | — | `2.11.0+cu130`, the wheel the `ltx-2` pin resolves and the goldens were generated with |

Read from a local checkout at that revision with `git show <sha>:<path>`, never
from a working tree, because a working tree can differ from the pin.

---

## 0. Honesty statement — what this row does and does not claim

This row makes an audio file at **any** sample rate drive an LTX-2.5 render. It
does not claim A3 (`DubItPipeline`), A18 (reference-audio conditioning delivery),
or any IC-LoRA. It unblocks the first two by removing the refusal that stood in
front of both; it does not implement either, and both stay owed by their own
rows.

**Upstream ships no tests at this pin.** `find . -iname '*test*' -not -path
'./.git/*'` over `Lightricks/LTX-2 @ fd4ded7f` returns empty. So "port the
upstream tests in the same change" has nothing to port from the model author, and
the obligation becomes what §6 does instead: **execute** the upstream module at
the pin and emit its outputs as goldens, plus pin each upstream default against a
`file:line`.

**No render on real weights is claimed.** The row is gated on reduced-dimension
fixtures and executed-upstream goldens, exactly as `LTX25-A2V-AUDIO-INPUT` was.

---

## 1. Scope

In:

* Port `torchaudio.functional.resample` at the defaults `ops.py:40` passes, as an
  arbitrary rational-ratio polyphase sinc-hann resampler.
* Call it from `Ltx2WaveformToLogMel`, which is where upstream calls it.
* Drop the two refusals that stand in front of it, and repair the **nine**
  statements across this tree that misname upstream's filter as kaiser.

Out:

* `sinc_interp_kaiser` and the `beta` knob. Upstream passes neither, and porting
  an unreached branch is the shape `## Nothing lands dead` forbids.
* Non-default `lowpass_filter_width`, `rolloff`. Same reason.
* Compressed audio containers, channel mixing, A3, A18. Each is refused or owed
  elsewhere and none of them is this gap.

---

## 2. Our baseline — the gap, on this tree

`Ltx2WaveformToLogMel` refuses any waveform whose rate is not
`config.target_sample_rate`
(`src/vllm/model_executor/models/ltx2_audio_vae.cpp:1051-1060`), and
`Ltx2DecodeAudioWav` refuses the same thing one hop earlier so the message can
name the file (`src/vllm/model_executor/models/ltx2_audio_input.cpp:96-105`).
Between them, **every** audio input at a rate other than the checkpoint's is
turned away, so `a2vid_two_stage` only accepts a take the user resampled with
some other tool first.

**The refusal misnames the filter, in nine places.** Its message and eight
sibling comments call `torchaudio.functional.resample` "an arbitrary-ratio
polyphase **kaiser** resampler". It is not one. `resample`'s signature defaults
`resampling_method="sinc_interp_hann"` (torchaudio 2.11
`functional/functional.py:1441`), and `_get_sinc_resample_kernel` builds a kaiser
window only on the `else` of `if resampling_method == "sinc_interp_hann"`
(`:1384-1391`). `ops.py:40` passes neither `resampling_method` nor `beta`, so the
window is `cos(t*pi/lpw/2)**2` — a **hann** window, the same family this tree
already ports for the vocoder's BWE stage at `ltx2_audio_vae.cpp:440`
(`Ltx2HannSincResampleFilter1d`), which is exactly that kernel specialized to
`orig_freq == 1`.

So what is missing is the **arbitrary rational ratio**, not the window. Sizing
A19 at M was right; the reason written beside it was wrong, and this row corrects
the reason as well as the code.

The nine sites: `ltx2_audio_input.cpp:101`, `ltx2_audio_vae.cpp:1057`,
`ltx2_audio_vae_encoder.h:173`, `ltx2_video.h:315`, `ltx2_audio_input.h:124`,
`test_ltx2_video.cpp:7518`, and `ltx25-a2v-audio-input.md:155` and `:463`.

---

## 3. Upstream chain — read end to end, not inferred

| Hop | Upstream | What it settles |
|---|---|---|
| 1 | `a2vid_two_stage.py:196` | `decode_audio_from_file` runs at the file's **native** rate |
| 2 | `decode.py:290-296` | the window (`start_time`, `max_duration`) is applied in samples at the **native** rate, before any resample |
| 3 | `decode.py:173-176` | the waveform is **float32**, `astype(np.float32)`, normalised to [-1, 1] |
| 4 | `a2vid_two_stage.py:200` -> `audio_vae.py:271` | `encode_audio` -> `audio_processor.waveform_to_mel(audio.to(device=device))`; `.to(device=...)` moves, it does not cast |
| 5 | `ops.py:44-49` | `waveform_to_mel` calls `self.resample_audio(audio)` **first**, then the mel transform |
| 6 | `ops.py:36-42` | `resample_audio` returns `audio` unchanged when the rates are equal; otherwise `torchaudio.functional.resample(waveform, orig, target)` and then `.to(device=..., dtype=waveform.dtype)` |
| 7 | `functional.py:1470-1476` | `orig_freq <= 0 or new_freq <= 0` raises; `orig_freq == new_freq` returns the input; `gcd` reduces the ratio |
| 8 | `functional.py:1305-1402` | the kernel: `base_freq = min(o, n) * rolloff`, `width = ceil(lpw * o / base_freq)`, one row per output phase |
| 9 | `functional.py:1405-1432` | zero-pad `(width, width + o)`, `conv1d` stride `o`, transpose-interleave, truncate to `ceil(n * L / o)` |
| 10 | `a2vid_two_stage.py:301-303` | the returned soundtrack is the caller's **original** `Audio`, at its **native** rate. The resample never reaches the output |

Hop 10 is why nothing downstream of the encoder changes: the file's own rate
stays the render's `sample_rate`, exactly as today.

---

## 4. Port map

| Upstream | Ours |
|---|---|
| `functional.py:1305-1402` `_get_sinc_resample_kernel` | `Ltx2SincResampleKernel` (file-local), `ltx2_audio_vae.cpp` |
| `functional.py:1405-1432` `_apply_sinc_resample_kernel` | `Ltx2ResampleWaveform`, `ltx2_audio_vae.cpp` |
| `functional.py:1435-1490` `resample` | `Ltx2ResampleWaveform`'s guards and early return |
| `ops.py:36-42` `AudioProcessor.resample_audio` | the call at the head of `Ltx2WaveformToLogMel` |

`Ltx2ResampleWaveform` is declared in
`include/vllm/model_executor/models/ltx2_audio_vae_encoder.h` beside the mel
front-end it belongs to, because upstream keeps them in one module (`ops.py`).

### The dtype, and why it is `float` and not `double`

Upstream resamples in **float32**: hop 3 makes the waveform `np.float32`, hop 4
does not cast it, and `_get_sinc_resample_kernel` is called with
`dtype=waveform.dtype`, so **the filter itself is built in float32** — `idx`,
`t`, the clamp, the window, `sin(t)/t` and the scale, all of it
(`functional.py:1376-1397`).

This is not a detail to round up. Building the same filter in `double` and
storing it as `float` is *more accurate than upstream* and therefore **further
from it**: measured against the pinned oracle, a double-built kernel differs by
up to **1.03e-05** where a float-built one differs by **1.79e-07**, a factor of
57. AGENTS.md's *Inherit vLLM defaults* names exactly this failure — a dtype that
is too wide, which no token gate can see. The port mirrors the float32
arithmetic operation for operation, including the association `sinc * (window *
scale)` (`:1397`) and the fact that `scale = base_freq / orig_freq` is computed
in Python `float` (f64) and *then* narrowed (`:1395`).

The one deliberate exception is the convolution **accumulator**, which is
`double`. torch's `conv1d` reduction order over 161 taps is a vectorised
implementation detail that no C++ loop reproduces, so the choice is between two
different roundings; the exact one is chosen and annotated. It is an accumulator,
not a stored dtype: every stored value stays `float`.

### The one shape decision

`Ltx2WaveformToLogMel` takes `[channels, samples]` channel-major. torchaudio
packs the batch and resamples the last axis, so each channel resamples
independently against the same kernel. The port builds the kernel once and walks
the channels, which is that, not an approximation of it.

---

## 5. Reachability — the sentence the records must carry

The production entry point is `vllm_video_generate` -> `VideoEngine::Generate`
-> `Ltx2VideoEngine::Generate` -> `Ltx2DecodeAudioWav`
(`src/vllm/multimodal/ltx2_video.cpp:3063`) -> `Ltx2EncodeAudioToLatent`
(`:3068`) -> `Ltx2WaveformToLogMel`
(`src/vllm/model_executor/models/ltx2_audio_input.cpp:200`) -> the new
`Ltx2ResampleWaveform`. Every hop already exists and is already gated; this row
adds the last one and deletes the two refusals that stood in the chain.

The smallest failing test therefore enters through `LoadVideoEngine` +
`engine->Generate` with a WAV at a rate the fixture's audio VAE does not declare,
per the note at `test_ltx2_video.cpp:7229-7240`. A unit test over
`Ltx2ResampleWaveform` alone would prove the function works and never that a
request can arrive at it.

**Mutation:** delete the `Ltx2ResampleWaveform` call in `Ltx2WaveformToLogMel`
and the video-level case must red. Recorded in `## Outcome`.

---

## 6. Tests to port

Upstream ships none (§0). What this change does instead:

1. **Goldens by execution.** `scripts/gen-ltx2-vae-goldens.py` gains sections
   **8d** and **8e**, importing `ltx_core.model.audio_vae.ops.AudioProcessor`
   from the `--ltx2` checkout and running `resample_audio` and `waveform_to_mel`.
   They ride the generator's existing `kLtx2VaeUpstreamRevision` anchor, which the
   C++ suite already asserts against a pinned SHA, so a regeneration against a
   different upstream fails the gate rather than replacing the oracle.
2. **The rate pairs are chosen to reach distinct arms**, not for coverage
   arithmetic: 16000 -> 24000 (pure upsample, `o = 2`), 44100 -> 24000
   (`gcd = 300`, `o = 147`, `width = 12`, the widest kernel), 48000 -> 24000
   (pure downsample, `o = 2`), and 24000 -> 24000 (the equal-rate early return,
   which must be a **copy** and not a filtered pass).
3. **A lower bound, not only a tolerance.** A resampler that returned zeros, or
   that returned the input untouched, satisfies a `max|diff| < tol` gate against
   the wrong golden and any shape check. The cases assert the output's absmax is
   positive **and** that a resampled take differs from the same samples read as
   if they were already at the target rate — the exact wrong answer the refusal
   was written to prevent.
4. **The end-to-end case**, at `test_ltx2_video.cpp`, replaces the SUBCASE that
   asserted the refusal: a 44.1 kHz WAV now renders, and its conditioning trace
   carries a non-zero audio latent whose digest **differs** from the same file
   written at the fixture's own rate.

---

## 7. Gates

```sh
cmake --build build -j 4 --target test_ltx2_vae test_ltx2_video
./build/tests/test_ltx2_vae -tc='*resampl*'
./build/tests/test_ltx2_video -tc='*audio*'
scripts/agent-preflight.sh --staged
```

Plus regeneration parity:

```sh
python3 scripts/gen-ltx2-vae-goldens.py --ltx2 ~/_git/LTX-2 \
    --out tests/vllm/models/ltx2_vae_goldens.inc
git diff --exit-code tests/vllm/models/ltx2_vae_goldens.inc
```

---

## 8. Dependencies

None. A19 depends on nothing in §8 of the plan; A3 and A18 depend on it.

---

## 9. Work breakdown

1. Spec (this file), committed first.
2. Generator sections 8d/8e and the regenerated `.inc`.
3. `test_ltx2_vae.cpp` cases, red before the port.
4. `Ltx2ResampleWaveform` plus the `Ltx2WaveformToLogMel` call.
5. Delete the two refusals; repair the nine kaiser statements.
6. `test_ltx2_video.cpp`: the refusal SUBCASE becomes the render case.
7. `docs/FEATURES.md` and `docs/USAGE.md` where they state the rate constraint.

---

## 10. Risks/decisions

| # | Risk | Handling |
|---|---|---|
| R1 | A `double` filter reads as "better" and silently widens the model path | §4 measures both against the oracle and pins float32 with the number. The gate tolerance is tight enough (2.5e-07) that a double-built kernel **fails** it |
| R2 | Removing a refusal admits an input that then fails deeper, with a worse message | The window and channel refusals stay; only the rate one goes. The `samples > n_fft/2` check in `Ltx2WaveformToLogMel` now sees the **resampled** length, which is upstream's order (hop 5) |
| R3 | A tolerance-only gate passes a resampler that returns the input | §6.3's lower bound and difference assertions |
| R4 | The nine kaiser statements are repaired in prose but the sizing document keeps the claim | `ltx25-completion-scope.md` is another row's spec and is not edited here; the correction is recorded on #2583 and in §2 |
| D1 | Should `Ltx2DecodeAudioWav` keep `want_sample_rate`? | No. Upstream's decoder has no target rate (hop 1); the parameter existed only to raise. Removed, and the one call site updated |

---

## Now

`ACTIVE`. Implementation in flight on `row/LTX25-AUDIO-RESAMPLE`, issue #2583.

---

## Owed

* **A real-checkpoint A2V render from a 44.1 kHz source.** The gate here is
  reduced-dimension fixtures plus executed-upstream goldens; a full render needs
  the GPU lease and belongs to the campaign's render rows. Inherits
  [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
* **The `sinc_interp_kaiser` arm and the `beta` knob.** Unreached from
  `ops.py:40`; deliberately not ported (§1). Inherits
  [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
* **The audio VAE's f32/bf16 tension.** `ltx2_audio_vae.cpp` argues f32 from
  upstream's vocoder float32 pin (`vocoder.py:575-580`) while upstream builds
  `AudioDecoder` in bf16. This row does not resolve it and does not contradict
  it: the resampler's float32 is argued from `ops.py`'s own chain (§4), not from
  that pin. Inherits [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
