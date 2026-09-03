// Ported from: transformers WhisperFeatureExtractor
// (feature_extraction_whisper.py `_torch_extract_fbank_features`) — the torch STFT
// log-mel path that runs when torch is installed; audio_utils.mel_filter_bank
// (slaney) dumped as a golden constant; vllm/model_executor/models/whisper.py
// (get_num_audio_tokens:656, _get_prompt_updates:740). @ vLLM e24d1b24 /
// transformers 5.13.1. See audio_processor.h for full provenance.
//
// STFT parity note: the oracle uses torch.stft (FFT). Our direct DFT (only the
// 201 needed bins per frame) differs from FFT only in float summation order, so
// the log-mel matches to a small rel-L2 (measured, gated), NOT bit-exact — this is
// stated + justified. The placeholder ids and mm-hash are bit/byte-exact.
#include "vllm/multimodal/audio_processor.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vllm/multimodal/hasher.h"

namespace vllm::multimodal {

namespace {

// Read a little-endian unsigned integer of `n` bytes from p.
uint32_t ReadLE(const uint8_t* p, int n) {
  uint32_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

constexpr double kPi = 3.14159265358979323846;

}  // namespace

namespace {

// The `fmt `/`data` chunk walk, ONE copy for both entry points below. W7c-1
// (#2813) added the multi-channel arm and did NOT write it a second parser:
// a hand-written parallel path is what AGENTS.md's shared-seam rule forbids,
// and the two decoders differ only in what they do with the frames.
struct RawWavPcm16 {
  const uint8_t* frames = nullptr;  // interleaved: L0 R0 L1 R1 ...
  size_t num_frames = 0;
  int channels = 0;
  int sampling_rate = 0;
};

RawWavPcm16 ParseWavPcm16(const uint8_t* wav, size_t n, const char* who) {
  const std::string me(who);
  if (n < 44 || std::memcmp(wav, "RIFF", 4) != 0 ||
      std::memcmp(wav + 8, "WAVE", 4) != 0) {
    throw std::runtime_error(me + ": not a RIFF/WAVE buffer");
  }
  // Walk chunks after the 12-byte RIFF header.
  size_t pos = 12;
  int channels = 0, bits = 0, rate = 0;
  const uint8_t* data = nullptr;
  size_t data_len = 0;
  bool have_fmt = false;
  while (pos + 8 <= n) {
    const uint8_t* id = wav + pos;
    const uint32_t sz = ReadLE(wav + pos + 4, 4);
    const size_t body = pos + 8;
    if (body + sz > n) break;
    if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
      const uint16_t fmt = static_cast<uint16_t>(ReadLE(wav + body, 2));
      channels = static_cast<int>(ReadLE(wav + body + 2, 2));
      rate = static_cast<int>(ReadLE(wav + body + 4, 4));
      bits = static_cast<int>(ReadLE(wav + body + 14, 2));
      if (fmt != 1) throw std::runtime_error(me + ": not PCM (fmt!=1)");
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      data = wav + body;
      data_len = sz;
    }
    pos = body + sz + (sz & 1);  // chunks are word-aligned
  }
  if (!have_fmt || data == nullptr) {
    throw std::runtime_error(me + ": missing fmt/data chunk");
  }
  if (bits != 16) throw std::runtime_error(me + ": not 16-bit PCM");
  // A zero channel count is a malformed header, not an arm anybody owes.
  if (channels < 1) {
    throw std::runtime_error(me + ": the `fmt ` chunk declares a channel count of " +
                             std::to_string(channels));
  }

  RawWavPcm16 raw;
  raw.frames = data;
  raw.channels = channels;
  raw.sampling_rate = rate;
  // A TRAILING PARTIAL FRAME IS DROPPED, not refused. libsndfile reads whole
  // frames and ignores a short tail (`sf_readf_*` returns whole frames), so
  // refusing here would be STRICTER than the oracle. The mono path has always
  // truncated this way, through `data_len / 2`.
  raw.num_frames = data_len / (2 * static_cast<size_t>(channels));
  return raw;
}

}  // namespace

DecodedAudio DecodeWavPcm16Mono(const uint8_t* wav, size_t n) {
  const RawWavPcm16 raw = ParseWavPcm16(wav, n, "DecodeWavPcm16Mono");
  if (raw.channels != 1) throw std::runtime_error("DecodeWavPcm16Mono: not mono");

  DecodedAudio out;
  out.sampling_rate = raw.sampling_rate;
  const size_t num = raw.num_frames;
  out.samples.resize(num);
  // UNCHANGED, deliberately. `parakeet_transcription.cpp:123`,
  // `chat_mm.cpp:137` and `test_voxtral_e2e.cpp:170` call this, and W7c-1 must
  // not move any of them by a bit. The multi-channel arm is the sibling below.
  for (size_t i = 0; i < num; ++i) {
    const int16_t s = static_cast<int16_t>(ReadLE(raw.frames + 2 * i, 2));
    out.samples[i] = static_cast<float>(s) / 32768.0f;
  }
  return out;
}

// W7c-1 (#2813). Upstream reduces to mono with a plain MEAN over the channel
// axes, and it does so in two independent places:
//
//   * the decode side, `vllm/multimodal/media/audio.py:207-208 @ 9035151d6` --
//     `if mono and y.ndim > 1: y = np.mean(y, axis=tuple(range(y.ndim - 1)))`,
//     reached because `load_audio`'s `mono` default is True (`:220`); the PyAV
//     fallback arm takes the same mean at `:168-169`;
//   * the parser side, `vllm/multimodal/audio.py:150-152 @ 9035151d6`, where
//     `AudioSpec.target_channels` is 1 and `channel_reduction` is
//     `ChannelReduction.MEAN` (`:69-70`, and the MEAN member's own comment
//     reads "default, preserves energy balance"). dots3-note selects that spec
//     at `vllm/models/dots3_note/common/processor.py:523-525 @ 9035151d6`.
//
// THE INTERMEDIATE TYPE IS A DECISION, NOT A TRANSLATION, because upstream
// reduces a float32 array (`soundfile.read(dtype="float32")`) and this port has
// interleaved int16 frames. It accumulates in INT32, divides once in DOUBLE and
// narrows once to FLOAT:
//
//   * the int32 sum CANNOT OVERFLOW: |s| <= 32768 and `channels` is a uint16
//     field, so |sum| <= 32768 * 65535 = 2147450880 < 2^31. It is EXACT, which
//     no float32 accumulator can promise past 256 channels;
//   * there is EXACTLY ONE rounding, at the narrowing store -- nothing is
//     rounded to float and then combined again;
//   * for a POWER-OF-TWO channel count that rounding is not one: with C = 2^k
//     the quotient is `acc * 2^-(15+k)`, a significand of at most 16 + k bits
//     against float's 24, so it is exact -- and upstream's float32 mean is
//     exact for the same reason. The two therefore agree BIT FOR BIT at C = 1
//     (so no mono waveform moves) and at C = 2 (the case this exists to serve).
//     Past a power of two the two may differ by one float ulp and this arm is
//     the more accurate of the pair, because its sum is exact where numpy's is
//     not. Stated in `.agents/specs/dots3-note.md` 4.16.2, gated in
//     `test_dots3_note_audio.cpp`.
DecodedAudio DecodeWavPcm16MeanToMono(const uint8_t* wav, size_t n) {
  const RawWavPcm16 raw = ParseWavPcm16(wav, n, "DecodeWavPcm16MeanToMono");

  DecodedAudio out;
  out.sampling_rate = raw.sampling_rate;
  out.samples.resize(raw.num_frames);
  const size_t c = static_cast<size_t>(raw.channels);
  const double denom = 32768.0 * static_cast<double>(raw.channels);
  for (size_t f = 0; f < raw.num_frames; ++f) {
    int32_t acc = 0;
    for (size_t ch = 0; ch < c; ++ch) {
      acc += static_cast<int16_t>(ReadLE(raw.frames + 2 * (f * c + ch), 2));
    }
    out.samples[f] = static_cast<float>(static_cast<double>(acc) / denom);
  }
  return out;
}

WhisperAudioProcessor::WhisperAudioProcessor(AudioProcessorConfig cfg,
                                             std::vector<float> mel_filters)
    : cfg_(std::move(cfg)), mel_filters_(std::move(mel_filters)) {
  const size_t expect =
      static_cast<size_t>(cfg_.num_freq_bins()) * static_cast<size_t>(cfg_.n_mels);
  if (mel_filters_.size() != expect) {
    throw std::runtime_error("WhisperAudioProcessor: mel_filters size mismatch");
  }
}

AudioKwargs WhisperAudioProcessor::ProcessWaveform(const float* samples,
                                                   int64_t num_samples,
                                                   int sample_rate) const {
  if (sample_rate != cfg_.sampling_rate) {
    // Genuine resample (windowed sinc, à la librosa) is deferred — mirror the
    // image SmartResize/bicubic identity-only guard. The whisper-small fixture is
    // already at cfg.sampling_rate (16 kHz), so this path is never taken here.
    throw std::runtime_error(
        "WhisperAudioProcessor: resample deferred; provide audio at "
        "cfg.sampling_rate (16 kHz)");
  }
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int n_mels = cfg_.n_mels;
  const int n_freq = cfg_.num_freq_bins();
  const int n_pad = cfg_.n_samples_padded();  // 480000

  // ---- pad / truncate the waveform to n_pad (WhisperFeatureExtractor.pad,
  // padding="max_length", max_length=n_samples, truncation=True) ----
  std::vector<double> wav(static_cast<size_t>(n_pad), 0.0);
  const int64_t copy = std::min<int64_t>(num_samples, n_pad);
  for (int64_t i = 0; i < copy; ++i) wav[static_cast<size_t>(i)] = samples[i];

  // ---- torch.stft(center=True) reflect-pads by n_fft/2 each side ----
  const int p = n_fft / 2;  // 200
  const int L = n_pad;
  const int Lp = L + 2 * p;
  std::vector<double> padded(static_cast<size_t>(Lp), 0.0);
  for (int j = 0; j < L; ++j) padded[static_cast<size_t>(p + j)] = wav[static_cast<size_t>(j)];
  // reflect (no edge repeat): left[i]=a[p-i] (i=0..p-1); right[k]=a[L-2-k] (k=0..p-1)
  for (int i = 0; i < p; ++i) padded[static_cast<size_t>(i)] = wav[static_cast<size_t>(p - i)];
  for (int k = 0; k < p; ++k) {
    const int src = L - 2 - k;
    padded[static_cast<size_t>(p + L + k)] =
        (src >= 0) ? wav[static_cast<size_t>(src)] : 0.0;
  }

  // number of STFT frames = 1 + (Lp - n_fft)/hop; drop the last (stft[...,:-1]).
  const int n_frames_full = 1 + (Lp - n_fft) / hop;
  const int n_frames = n_frames_full - 1;

  // ---- periodic Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n / n_fft) ----
  std::vector<double> win(static_cast<size_t>(n_fft));
  for (int nn = 0; nn < n_fft; ++nn) {
    win[static_cast<size_t>(nn)] = 0.5 - 0.5 * std::cos(2.0 * kPi * nn / n_fft);
  }

  // ---- precompute DFT twiddles cos/sin[k*n_fft + j] for k in [0,n_freq), j<n_fft ----
  std::vector<double> cosk(static_cast<size_t>(n_freq) * n_fft);
  std::vector<double> sink(static_cast<size_t>(n_freq) * n_fft);
  for (int k = 0; k < n_freq; ++k) {
    for (int j = 0; j < n_fft; ++j) {
      const double ang = 2.0 * kPi * k * j / n_fft;
      cosk[static_cast<size_t>(k) * n_fft + j] = std::cos(ang);
      sink[static_cast<size_t>(k) * n_fft + j] = std::sin(ang);
    }
  }

  // ---- log-mel spectrogram [n_mels, n_frames] ----
  std::vector<float> feat(static_cast<size_t>(n_mels) * n_frames);
  std::vector<double> mag2(static_cast<size_t>(n_freq));
  std::vector<double> frame(static_cast<size_t>(n_fft));
  double gmax = -1e300;  // running max over log10(mel) for the -8 clamp

  // First pass: fill log10(clamp(mel_spec, 1e-10)) and track the global max.
  for (int f = 0; f < n_frames; ++f) {
    const int start = f * hop;
    for (int j = 0; j < n_fft; ++j) {
      frame[static_cast<size_t>(j)] =
          padded[static_cast<size_t>(start + j)] * win[static_cast<size_t>(j)];
    }
    for (int k = 0; k < n_freq; ++k) {
      double re = 0.0, im = 0.0;
      const double* ck = &cosk[static_cast<size_t>(k) * n_fft];
      const double* sk = &sink[static_cast<size_t>(k) * n_fft];
      for (int j = 0; j < n_fft; ++j) {
        re += frame[static_cast<size_t>(j)] * ck[j];
        im -= frame[static_cast<size_t>(j)] * sk[j];  // e^{-i...}
      }
      mag2[static_cast<size_t>(k)] = re * re + im * im;
    }
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;  // mel_filters.T @ magnitudes: sum_k filt[k,m]*mag2[k]
      for (int k = 0; k < n_freq; ++k) {
        acc += static_cast<double>(mel_filters_[static_cast<size_t>(k) * n_mels + m]) *
               mag2[static_cast<size_t>(k)];
      }
      if (acc < 1e-10) acc = 1e-10;  // torch.clamp(min=1e-10)
      const double lg = std::log10(acc);
      feat[static_cast<size_t>(m) * n_frames + f] = static_cast<float>(lg);
      if (lg > gmax) gmax = lg;
    }
  }

  // Second pass: x=max(x, gmax-8); x=(x+4)/4 (whisper normalization).
  const double floor = gmax - 8.0;
  for (auto& v : feat) {
    double x = v;
    if (x < floor) x = floor;
    x = (x + 4.0) / 4.0;
    v = static_cast<float>(x);
  }

  AudioKwargs out;
  out.n_mels = n_mels;
  out.n_frames = n_frames;
  out.input_features = std::move(feat);
  return out;
}

std::string WhisperAudioProcessor::HashAudio(const float* samples,
                                             int64_t num_samples) const {
  return MultiModalHasher::HashAudioF32(cfg_.model_id, samples, num_samples);
}

std::vector<int32_t> ExpandAudioPlaceholders(
    const std::vector<int32_t>& prompt_ids, int32_t audio_placeholder_id,
    const std::vector<int>& num_audio_tokens_per_item,
    std::vector<std::array<int, 2>>* placeholders) {
  std::vector<int32_t> out;
  out.reserve(prompt_ids.size());
  if (placeholders) placeholders->clear();
  size_t item = 0;
  for (int32_t id : prompt_ids) {
    if (id == audio_placeholder_id && item < num_audio_tokens_per_item.size()) {
      const int n = num_audio_tokens_per_item[item++];
      const int offset = static_cast<int>(out.size());
      for (int i = 0; i < n; ++i) out.push_back(audio_placeholder_id);
      if (placeholders) placeholders->push_back({offset, n});
    } else {
      out.push_back(id);
    }
  }
  return out;
}

}  // namespace vllm::multimodal
