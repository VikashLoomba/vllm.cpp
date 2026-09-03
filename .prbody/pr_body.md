A PCM16 WAV whose `fmt ` chunk named any rate but `audio_config.sampling_rate` was
refused with HTTP 400. It is now resampled and served. This is the row's first
RECORDED DIVERGENCE, and the reason is not that mirroring was hard.

WHY THIS CANNOT BE A MIRROR. Upstream resamples in its data parser
(`MultiModalDataParser(target_sr=..., target_channels=1)`,
vllm/models/dots3_note/common/processor.py:523-525 -> vllm/multimodal/parse.py:695),
and `AudioResampler`'s default method is `"pyav"` (vllm/multimodal/audio.py:283),
which is libswresample. That arm is not bit-identical to ITSELF: on ffmpeg 6.1.1,
one binary and one input differing only in CPU dispatch produce 24691 differing
samples of 32000 at a worst absolute difference of 9.686e-08. A bit-exact gate
against it is impossible in principle, not merely inconvenient. Its option defaults
also come from an unpinned linked binary, and its auto-resolved cutoff is not
readable from outside the source.

WHAT IS IMPLEMENTED INSTEAD IS STILL UPSTREAM. `resample_audio_scipy`
(vllm/multimodal/audio.py:232-250 @ 9035151d6) is another arm of upstream's own
switch, and vLLM already ships it in production for another model
(vllm/model_executor/models/phi4mm.py:580). The `pyav` arm is refused permanently
and the refusal says why. Distance from the real default, re-measured here rather than relayed
and reported WITH ITS PROBE, because the ordering is signal-dependent: on a
0 -> 7500 Hz sweep at 44100 -> 16000, scipy is 51.78 dB from swresample, soxr 44.63
and torchaudio 29.02. That reproduces the numbers in #2828 to about 2 dB, and it
also shows what "band-limited content" has to mean there: content that FILLS the
band up to the new Nyquist. On content well below it all three are good and soxr
wins by 30 dB, because a resampler's transition band cannot matter where there is
no energy in it. Spec section 4.17.2 carries the four-probe table, so nobody quotes
one number as a property of the algorithms.

The non-determinism was reproduced too, not relayed: 19846 of 32000 samples differ
at a worst 2.980e-07 on this slice's own probe, against #2828's 24691 and 9.686e-08
on its.

`Ltx2ResampleWaveform` (#2583) is deliberately NOT reused. It is a genuine polyphase
resampler and it is the tempting reuse. It is 22.8 dB FURTHER from this oracle on the
0 -> 7500 Hz sweep, because torchaudio's defaults are a short Hann kernel against
swr's 32-tap kaiser-9 — #2828's "about 25 dB", measured here. The caveat runs the
other way too: on a low-frequency tone it scores 70.12 dB and BEATS scipy, because
the short kernel's poor transition band never gets exercised. A reviewer who
validates it on the wrong probe will find it excellent. The spec says so, so that
the next reader does not simplify it back.

THE ALGORITHM WAS VERIFIED, NOT TRANSCRIBED. Spec section 4.17.3 writes out
`resample_poly`'s six steps as scipy 1.17.1's own source states them, and the
committed generator re-derives them in plain Python with `i0` as its own power
series. That reimplementation agrees with scipy to 6.1e-15 in double and to 0.0
after narrowing to float32, over all five golden cases.

THE SEAM IS OPTED INTO PER MODEL AND NEVER BLANKET. Five rows here refuse a rate
mismatch and they are not the same policy: Parakeet's refusal is upstream-faithful,
because feature_extraction_parakeet.py raises rather than resampling, while
dots3-note's upstream resamples. audio_processor.cpp's Whisper/Voxtral refusal and
parakeet_audio_processor.cpp are untouched.

THE GATE IS A CONSISTENCY GATE AND SAYS SO. Section 6.4 option B, on #2583's design.
Committed scipy goldens for 44100, 48000, 22050 and 8000 -> 16000, plus a fifth case
at 44100 whose signal carries a tone above the 8 kHz output Nyquist, because that is
the only content that separates a real anti-alias filter from picking samples. A
tolerance alone would gate nothing, so each case also asserts a length, a lower
bound, and a difference against a nearest-sample decimation computed in the test.
Section 4.17.7 states what this establishes and what it does not.

THE SERVED INVERSION EARNED ITS KEEP. `ProcessWaveform` rebound the sample pointer
and the length after resampling and left `sample_rate` at the request's value, so
`WhisperAudioProcessor::ProcessWaveform` -- driven once per chunk and carrying its own
rate refusal for a different row -- threw a bare `runtime_error`. That is HTTP 500 and
not 400, and the served suite read `500 == 400`. A unit test on the resampler could
not have seen it.

ONE DEFECT THE LIFT CREATED IS CLOSED IN THE SAME CHANGE. `mm_hash` is a
cross-request encoder-cache key. Two files carrying identical PCM16 samples at
different declared rates decode to identical float buffers, and must not share an
entry. `HashAudio` gains a rate-aware overload that keys on the resampled waveform.

ONE THING IS REFUSED THAT UPSTREAM DOES NOT REFUSE, and it is labelled as a
divergence in the message itself. `kMaxPolyphaseRate` caps the reduced ratio at
100000, because the anti-alias filter is `20 * max(up, down) + 1` taps and the rate is
named by the request's own WAV header. Every ordinary rate reduces far below it.

THE MEASURED EVIDENCE is in spec section 4.17.11 through 4.17.13. The port
reproduces scipy's own float32 output bit for bit on four of the five golden cases
and to 8.31e-18 on the fifth. Seven mutations were driven, each rebuilt with the
build asserted successful before any result was read, each with its binary sha256
recorded, and the tree restored byte-for-byte and verified at the binary.

TWO OF THOSE MUTATIONS FIRST READ GREEN on the served suite, and that changed the
test rather than the write-up. Its value assertion compared the request against an
offline reference computed with the SAME production code, so a zeroed or aliased
resampler moved both sides together — a shared-helper consistency check wearing a
correctness gate. Two controls now come from outside the resample path: a WAV of
literal silence, which is never resampled at all, and the natively-16 kHz fixture,
which is the same continuous signal from the same closed form. The resampled 44.1
kHz clip lands closer to that native recording (0.00705) than to its own offline
resample (0.00957), because the offline arm pays a PCM16 quantization the served
arm does not.

ONE MUTATION THE SERVED SUITE STILL CANNOT SEE is named rather than hidden. A
one-sample phase shift is caught by the front-end suite at 1.2e-7 and sits inside
what the served suite's native-clip control allows. The values are gated where the
values are; the served suite gates that the capability is reached and is neither
dead nor aliased.

Closes #2828.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:claude-opus-5 [claude-code]
