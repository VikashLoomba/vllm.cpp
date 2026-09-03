// dots3-note W7a (#2703) — THE AUDIO TOWER AND ITS FRONT END, against TWO
// INDEPENDENT double-precision references.
//
// WHAT THIS FILE ESTABLISHES, AND WHAT IT DOES NOT. `.agents/specs/
// dots3-note.md` §6.4 records option B: this row has NO oracle and will not get
// one — 576.89 GB bf16 / 298.67 GB fp8 against 119-122 GiB hosts — so
// correctness is argued by a reference written from the upstream Python that
// shares NO helper with the implementation. That is a CONSISTENCY gate: two
// implementations agree, neither is shown to match vLLM, and no performance
// number is claimable on any axis.
//
// THE ROW'S CONVENTION IS TO PROVE THE INDEPENDENCE BY ENUMERATION. W6a listed
// 70 qualified names in its reference namespace, W6b 105 and W6c 45, all
// `std::`. This file has TWO reference namespaces and reports both counts,
// because one reference covering a DFT, a mel filterbank and a 32-layer tower
// with a four-stage temporal mask is too large for a reviewer to hold, and the
// two halves fail in unrelated ways: the front end's hazards are windowing,
// framing and normalisation; the tower's are ordering, masking and bias
// placement.
//
// AND ONE PLACE THERE **IS** A REAL ORACLE, which almost nothing on this row
// gets. `tests/vllm/multimodal/fixtures/voxtral_audio/voxtral_mel_filters_f32.bin`
// is a COMMITTED [201, 128] float32 matrix that
// `scripts/mm/a3_voxtral_oracle_capture.py:141-147` dumped from
// `mistral_common.audio.mel_filter_bank(201, 128, 0.0, 8000.0, 16000)` — a
// third party's implementation of the same formula `nvidia/audio.py:98-106`
// calls. The shared `MelFilterBankSlaney` seam is asserted against it BIT for
// BIT, which settles HTK-versus-Slaney, the `norm` argument and the
// integer-divided-Nyquist detail outright.
//
// Upstream read in `~/_git/vllm` at `9035151d6`. `dots3_note` does not exist at
// the parity pin `5559679229`, so every anchor names that SHA.
#include "vllm/model_executor/models/dots3_note_audio.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "dots3_note_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/multimodal/mel_filter_bank.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using dots3_tiny::TinyCheckpoint;
using dots3_tiny::TinySpec;

namespace {

std::string FixtureDir() { return DOTS3_NOTE_CKPT_FIXTURE_DIR; }
std::string VoxtralFixtureDir() { return VOXTRAL_AUDIO_FIXTURE_DIR; }

TinySpec AudioSpec() {
  TinySpec s;
  s.with_audio = true;
  return s;
}

std::vector<float> ReadF32File(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: " << path);
  f.seekg(0, std::ios::end);
  const std::streamoff bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE 1 — THE FRONT END.
//
// Written from `nvidia/audio.py:96-126` @ `9035151d6` and from transformers
// `audio_utils.mel_filter_bank:453` / `hertz_to_mel:285` / `mel_to_hertz:321`,
// in double precision, sharing NO helper with `src/`. Every qualified name
// below is `std::`, and the enumeration case MEASURES that from this file's
// own bytes rather than trusting the list.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_front {

// Every qualified name used in this namespace, MEASURED by the enumeration case
// below — not transcribed. All 11 are `std::`: this reference calls nothing
// from `vllm::`, nothing from `vt::` and nothing from the file under test.
//
//  1 std::size_t (36)         2 std::vector (19)    3 std::int64_t (4)
//  4 std::log (3)             5 std::cos (2)        6 std::min (2)
//  7 std::exp (1)             8 std::log10 (1)      9 std::max (1)
// 10 std::numeric_limits (1) 11 std::sin (1)
//
// 71 occurrences of 11 distinct names. THIS LIST WAS WRONG WHEN THE FILE WAS
// FIRST WRITTEN, and nothing could see it: it read 22 names asserted against a
// hand-written `kQualifiedNames = 22`, and eleven of them — `std::pow`,
// `std::string`, `std::sqrt`, `std::abs`, `std::int16_t`, `std::to_string`,
// `std::fabs`, `std::floor`, `std::ceil`, `std::round`, `std::llround` — are
// used NOWHERE in this namespace. Two (`std::llround`, `std::int16_t`) appear
// nowhere in this FILE except inside that list. A constant compared with a
// literal is a transcription, and a transcription cannot gate what it
// transcribes; adding a `vllm::` call would not have moved it either.
inline constexpr int kDistinctQualifiedNames = 11;
inline constexpr int kQualifiedNameOccurrences = 71;

constexpr double kPi = 3.14159265358979323846;

// `audio_utils.hertz_to_mel(mel_scale="slaney")` :285-296.
double HzToMel(double hz) {
  if (hz >= 1000.0) return 15.0 + std::log(hz / 1000.0) * (27.0 / std::log(6.4));
  return 3.0 * hz / 200.0;
}
// `mel_to_hertz(mel_scale="slaney")` :321-331.
double MelToHz(double mel) {
  if (mel >= 15.0) return 1000.0 * std::exp((std::log(6.4) / 27.0) * (mel - 15.0));
  return 200.0 * mel / 3.0;
}

// `mel_filter_bank(..., norm="slaney", mel_scale="slaney")` :453, returned as
// [n_freq][n_mels] the way upstream returns it.
std::vector<std::vector<double>> Bank(int n_freq, int n_mels, double f_min,
                                      double f_max, int sr) {
  // :516-519 — centres equally spaced in MEL space, then back to Hz.
  const double m0 = HzToMel(f_min), m1 = HzToMel(f_max);
  std::vector<double> fc(static_cast<std::size_t>(n_mels + 2));
  for (int i = 0; i < n_mels + 2; ++i) {
    const double mel = m0 + (m1 - m0) * static_cast<double>(i) /
                                static_cast<double>(n_mels + 1);
    fc[static_cast<std::size_t>(i)] = MelToHz(mel);
  }
  fc[static_cast<std::size_t>(n_mels + 1)] = MelToHz(m1);
  // :528 — the INTEGER-divided Nyquist.
  std::vector<double> ff(static_cast<std::size_t>(n_freq));
  const double top = static_cast<double>(sr / 2);
  for (int k = 0; k < n_freq; ++k) {
    ff[static_cast<std::size_t>(k)] =
        top * static_cast<double>(k) / static_cast<double>(n_freq - 1);
  }
  std::vector<std::vector<double>> out(
      static_cast<std::size_t>(n_freq),
      std::vector<double>(static_cast<std::size_t>(n_mels), 0.0));
  for (int m = 0; m < n_mels; ++m) {
    const double lo = fc[static_cast<std::size_t>(m)];
    const double mid = fc[static_cast<std::size_t>(m) + 1];
    const double hi = fc[static_cast<std::size_t>(m) + 2];
    const double enorm = 2.0 / (hi - lo);  // :532-535, the slaney area norm
    for (int k = 0; k < n_freq; ++k) {
      const double f = ff[static_cast<std::size_t>(k)];
      const double down = (f - lo) / (mid - lo);
      const double up = (hi - f) / (hi - mid);
      const double tri = std::max(0.0, std::min(down, up));
      out[static_cast<std::size_t>(k)][static_cast<std::size_t>(m)] = tri * enorm;
    }
  }
  return out;
}

// `pad_or_trim` (:84-93) + `log_mel_spectrogram` (:117-126), the whole front
// end, as [n_mels][n_frames].
//
// `torch.stft(center=True)` REFLECT-pads by n_fft/2 on each side; the frame
// count is `1 + (padded - n_fft) / hop` and `stft[..., :-1]` drops the last.
// The window is torch's PERIODIC Hann. The spectrogram is POWER (`.abs()**2`).
// The floor is a GLOBAL max minus 8, not a per-band one.
std::vector<std::vector<double>> LogMel(const std::vector<float>& wav,
                                        int pad_to, int n_fft, int hop,
                                        int n_mels, int sr) {
  std::vector<double> x(static_cast<std::size_t>(pad_to), 0.0);
  const std::int64_t copy =
      std::min<std::int64_t>(static_cast<std::int64_t>(wav.size()), pad_to);
  for (std::int64_t i = 0; i < copy; ++i)
    x[static_cast<std::size_t>(i)] = wav[static_cast<std::size_t>(i)];

  const int p = n_fft / 2;
  const int L = pad_to;
  std::vector<double> padded(static_cast<std::size_t>(L + 2 * p), 0.0);
  for (int i = 0; i < L; ++i)
    padded[static_cast<std::size_t>(p + i)] = x[static_cast<std::size_t>(i)];
  for (int i = 0; i < p; ++i)
    padded[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(p - i)];
  for (int i = 0; i < p; ++i) {
    const int src = L - 2 - i;
    padded[static_cast<std::size_t>(p + L + i)] =
        src >= 0 ? x[static_cast<std::size_t>(src)] : 0.0;
  }
  const int frames = 1 + (L + 2 * p - n_fft) / hop - 1;

  std::vector<double> win(static_cast<std::size_t>(n_fft));
  for (int i = 0; i < n_fft; ++i) {
    win[static_cast<std::size_t>(i)] =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(n_fft));
  }
  const int n_freq = 1 + n_fft / 2;
  const std::vector<std::vector<double>> bank =
      Bank(n_freq, n_mels, 0.0, static_cast<double>(sr) / 2.0, sr);

  std::vector<std::vector<double>> out(
      static_cast<std::size_t>(n_mels),
      std::vector<double>(static_cast<std::size_t>(frames), 0.0));
  double gmax = -std::numeric_limits<double>::infinity();
  for (int f = 0; f < frames; ++f) {
    std::vector<double> mag(static_cast<std::size_t>(n_freq), 0.0);
    for (int k = 0; k < n_freq; ++k) {
      double re = 0.0, im = 0.0;
      for (int j = 0; j < n_fft; ++j) {
        const double v = padded[static_cast<std::size_t>(f * hop + j)] *
                         win[static_cast<std::size_t>(j)];
        const double ang = 2.0 * kPi * static_cast<double>(k) *
                           static_cast<double>(j) / static_cast<double>(n_fft);
        re += v * std::cos(ang);
        im -= v * std::sin(ang);
      }
      mag[static_cast<std::size_t>(k)] = re * re + im * im;
    }
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;
      for (int k = 0; k < n_freq; ++k) {
        acc += bank[static_cast<std::size_t>(k)][static_cast<std::size_t>(m)] *
               mag[static_cast<std::size_t>(k)];
      }
      if (acc < 1e-10) acc = 1e-10;
      const double lg = std::log10(acc);
      out[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)] = lg;
      if (lg > gmax) gmax = lg;
    }
  }
  const double floor_v = gmax - 8.0;
  for (auto& row : out) {
    for (double& v : row) {
      if (v < floor_v) v = floor_v;
      v = (v + 4.0) / 4.0;
    }
  }
  return out;
}

}  // namespace ref_front

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE 2 — THE TOWER.
//
// Written from `nvidia/audio_encoder.py` and `nvidia/audio.py` @ `9035151d6`,
// in double precision, sharing NO helper with `src/` or with `ref_front`. Every
// qualified name is `std::`, measured the same way.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref_tower {

// MEASURED, as above. All 6 are `std::`.
//
//  1 std::size_t (24)   2 std::vector (14)   3 std::int64_t (10)
//  4 std::sqrt (3)      5 std::erf (1)       6 std::exp (1)
//
// 53 occurrences of 6 distinct names. This list read 19 names before it was
// measured; thirteen of them are not used here. That this reference reaches
// only SIX names is itself the point — a 32-layer tower in double precision
// needs `exp`, `erf` and `sqrt` and nothing else, so anything else appearing
// in the measurement is a helper that leaked in from `src/`.
inline constexpr int kDistinctQualifiedNames = 6;
inline constexpr int kQualifiedNameOccurrences = 53;

using Mat = std::vector<std::vector<double>>;

// `RMSNorm.forward` (:36-39): `x * rsqrt(mean(x^2) + eps)` and THEN `weight *`.
// Deliberately upstream's order, not `vt::RmsNorm`'s — see
// `dots3_note_audio.h`'s note on the one formula difference.
Mat RmsNorm(const Mat& x, const std::vector<double>& w, double eps) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    double acc = 0.0;
    for (double v : x[r]) acc += v * v;
    const double inv = 1.0 / std::sqrt(acc / static_cast<double>(x[r].size()) + eps);
    out[r].resize(x[r].size());
    for (std::size_t c = 0; c < x[r].size(); ++c)
      out[r][c] = x[r][c] * inv * w[c];
  }
  return out;
}

// `nn.LayerNorm` — mean-subtracting, BIASED variance, weight AND bias.
Mat LayerNorm(const Mat& x, const std::vector<double>& w,
              const std::vector<double>& b, double eps) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    const std::size_t n = x[r].size();
    double mean = 0.0;
    for (double v : x[r]) mean += v;
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (double v : x[r]) var += (v - mean) * (v - mean);
    var /= static_cast<double>(n);
    const double inv = 1.0 / std::sqrt(var + eps);
    out[r].resize(n);
    for (std::size_t c = 0; c < n; ++c)
      out[r][c] = (x[r][c] - mean) * inv * w[c] + b[c];
  }
  return out;
}

// y[r][n] = sum_k x[r][k] * w[n][k] (+ b[n]) — `F.linear`.
Mat Linear(const Mat& x, const Mat& w, const std::vector<double>* b) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    out[r].assign(w.size(), 0.0);
    for (std::size_t n = 0; n < w.size(); ++n) {
      double acc = b != nullptr ? (*b)[n] : 0.0;
      for (std::size_t k = 0; k < x[r].size(); ++k) acc += x[r][k] * w[n][k];
      out[r][n] = acc;
    }
  }
  return out;
}

// `nn.functional.gelu` with the default `approximate='none'`: the EXACT erf
// form, not the tanh approximation.
double GeluErf(double v) {
  return 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
}

double Silu(double v) { return v / (1.0 + std::exp(-v)); }

// `swiglu` (:42-44): `x1, x2 = x.chunk(2, -1); silu(x1) * x2`. GATE THEN UP.
Mat SwiGlu(const Mat& x) {
  Mat out(x.size());
  for (std::size_t r = 0; r < x.size(); ++r) {
    const std::size_t half = x[r].size() / 2;
    out[r].resize(half);
    for (std::size_t c = 0; c < half; ++c)
      out[r][c] = Silu(x[r][c]) * x[r][c + half];
  }
  return out;
}

// ONE stride-2 padding-1 3x3 Conv2d over `[C][F][T]`, then GELU. Written as the
// four-deep loop upstream's `nn.Conv2d` performs, NOT as an im2col — so the
// implementation's im2col composition is being CHECKED rather than repeated.
std::vector<Mat> Conv2dGelu(const std::vector<Mat>& x, const std::vector<Mat>& w,
                            const std::vector<double>& b, std::size_t out_ch) {
  const std::size_t in_ch = x.size();
  const std::size_t Fi = x[0].size(), Ti = x[0][0].size();
  const std::size_t Fo = (Fi + 2 - 3) / 2 + 1, To = (Ti + 2 - 3) / 2 + 1;
  std::vector<Mat> out(out_ch, Mat(Fo, std::vector<double>(To, 0.0)));
  for (std::size_t co = 0; co < out_ch; ++co) {
    for (std::size_t fo = 0; fo < Fo; ++fo) {
      for (std::size_t to = 0; to < To; ++to) {
        double acc = b[co];
        for (std::size_t ci = 0; ci < in_ch; ++ci) {
          for (std::size_t kf = 0; kf < 3; ++kf) {
            const std::int64_t fi =
                static_cast<std::int64_t>(2 * fo) - 1 + static_cast<std::int64_t>(kf);
            if (fi < 0 || fi >= static_cast<std::int64_t>(Fi)) continue;
            for (std::size_t kt = 0; kt < 3; ++kt) {
              const std::int64_t ti = static_cast<std::int64_t>(2 * to) - 1 +
                                      static_cast<std::int64_t>(kt);
              if (ti < 0 || ti >= static_cast<std::int64_t>(Ti)) continue;
              acc += x[ci][static_cast<std::size_t>(fi)][static_cast<std::size_t>(ti)] *
                     w[co][ci][kf * 3 + kt];
            }
          }
        }
        out[co][fo][to] = GeluErf(acc);
      }
    }
  }
  return out;
}

// `_temporal_mask` (:528-533): zero every position at or past `valid` on the
// TIME axis.
void MaskTime(std::vector<Mat>* x, std::int64_t valid) {
  for (Mat& ch : *x) {
    for (std::vector<double>& row : ch) {
      for (std::size_t t = 0; t < row.size(); ++t) {
        if (static_cast<std::int64_t>(t) >= valid) row[t] = 0.0;
      }
    }
  }
}

}  // namespace ref_tower

// ═══════════════════════════════════════════════════════════════════════════
// THE ENUMERATION INSTRUMENT.
//
// Reads THIS source file at `DOTS3_AUDIO_TEST_SOURCE` (the same arrangement
// `MODELOPT_MIXED_FIXTURE_DIR` uses to hand a test a path), strips comments and
// string/char literals, takes the span of one reference namespace, and counts
// every `scope::name`. Comments must be stripped or the enumeration LIST above
// would count itself and the instrument would agree with any list it was given
// — which is the exact failure this replaces.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

struct RefNames {
  int distinct = 0;
  int occurrences = 0;
  std::set<std::string> scopes;
  std::set<std::string> names;
};

std::string Join(const std::set<std::string>& s) {
  std::string out;
  for (const std::string& v : s) {
    if (!out.empty()) out += ",";
    out += v;
  }
  return out;
}

bool IsIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

// Comments and literals out, everything else through unchanged.
std::string StripCommentsAndLiterals(const std::string& code) {
  std::string out;
  const size_t n = code.size();
  for (size_t i = 0; i < n;) {
    const char c = code[i];
    if (c == '/' && i + 1 < n && code[i + 1] == '/') {
      const size_t j = code.find('\n', i);
      i = (j == std::string::npos) ? n : j;
    } else if (c == '/' && i + 1 < n && code[i + 1] == '*') {
      const size_t j = code.find("*/", i + 2);
      i = (j == std::string::npos) ? n : j + 2;
    } else if (c >= '0' && c <= '9' && (i == 0 || !IsIdentChar(code[i - 1]))) {
      // A pp-number, taken WHOLE. C++14 lets a numeric literal carry `'` digit
      // separators (`16'000`), and treating that `'` as a char-literal
      // delimiter makes the scan below run to the NEXT `'` and drop everything
      // in between. That is a hole in THIS instrument and not a cosmetic one:
      // two separators bracketing a `vt::` call would hide the call from the
      // enumeration, and the independence property would read GREEN while being
      // false. The token must START at a digit that does not continue an
      // identifier, so `u8'a'` is still a char literal and is still stripped.
      size_t j = i;
      while (j < n) {
        if (IsIdentChar(code[j]) || code[j] == '.') {
          ++j;
        } else if (code[j] == '\'' && j + 1 < n && IsIdentChar(code[j + 1])) {
          j += 2;
        } else if ((code[j] == '+' || code[j] == '-') && j > i &&
                   (code[j - 1] == 'e' || code[j - 1] == 'E' ||
                    code[j - 1] == 'p' || code[j - 1] == 'P')) {
          ++j;
        } else {
          break;
        }
      }
      out.append(code, i, j - i);
      i = j;
    } else if (c == '"' || c == '\'') {
      size_t j = i + 1;
      while (j < n && code[j] != c) j += (code[j] == '\\') ? 2 : 1;
      i = j + 1;
    } else {
      out += c;
      ++i;
    }
  }
  return out;
}

RefNames QualifiedNamesIn(const std::string& ns) {
  std::ifstream in(DOTS3_AUDIO_TEST_SOURCE, std::ios::binary);
  REQUIRE_MESSAGE(in.good(),
                  "the enumeration instrument could not open its own source at "
                      << DOTS3_AUDIO_TEST_SOURCE);
  const std::string src((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::string open = "namespace " + ns + " {";
  const std::string close = "}  // namespace " + ns;
  const size_t a = src.find(open);
  const size_t b = src.find(close);
  REQUIRE(a != std::string::npos);
  REQUIRE(b != std::string::npos);
  REQUIRE(b > a);
  const std::string body = StripCommentsAndLiterals(src.substr(a, b - a));

  RefNames r;
  for (size_t i = 0; i + 1 < body.size(); ++i) {
    if (body[i] != ':' || body[i + 1] != ':') continue;
    // the scope to the left
    size_t s = i;
    while (s > 0 && IsIdentChar(body[s - 1])) --s;
    if (s == i) continue;
    // the name to the right
    size_t e = i + 2;
    size_t t = e;
    while (t < body.size() && IsIdentChar(body[t])) ++t;
    if (t == e) continue;
    const std::string scope = body.substr(s, i - s);
    const std::string name = body.substr(e, t - e);
    r.scopes.insert(scope);
    r.names.insert(scope + "::" + name);
    ++r.occurrences;
  }
  r.distinct = static_cast<int>(r.names.size());
  return r;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE ONE REAL ORACLE ON THIS ROW.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the SHARED slaney bank reproduces the committed voxtral oracle BIT for BIT") {
  // `nvidia/audio.py:98-106` @ `9035151d6`:
  //   mel_filter_bank(num_frequency_bins=1 + 400//2, num_mel_filters=128,
  //                   min_frequency=0.0, max_frequency=16000/2,
  //                   sampling_rate=16000, norm="slaney", mel_scale="slaney")
  const std::vector<float> ours =
      vllm::multimodal::MelFilterBankSlaney(201, 128, 0.0, 8000.0, 16000);
  const std::vector<float> oracle =
      ReadF32File(VoxtralFixtureDir() + "/voxtral_mel_filters_f32.bin");

  REQUIRE(oracle.size() == 201u * 128u);
  REQUIRE(ours.size() == oracle.size());

  // BIT-FOR-BIT, not a tolerance. Both sides construct in double and round once
  // on the store, so anything short of exact would mean a formula difference
  // rather than a rounding one — and this is the ONE assertion on this row that
  // can tell those apart, because the other side of it is
  // `mistral_common.audio.mel_filter_bank` and not a second transcription of
  // ours.
  std::size_t mismatched = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < ours.size(); ++i) {
    if (ours[i] != oracle[i]) {
      ++mismatched;
      worst = std::max(worst, static_cast<double>(std::fabs(ours[i] - oracle[i])));
    }
  }
  MESSAGE("mel bank vs voxtral oracle: " << mismatched << " of " << ours.size()
                                         << " values differ, worst |delta| " << worst);
  CHECK(mismatched == 0u);

  // The bank is SPARSE, and saying so keeps a reader from reading "25728 values
  // agree" as 25728 independent facts: at 128 mels over 201 bins most of the
  // low-frequency triangles fall between FFT bins and are entirely zero.
  std::size_t nonzero = 0;
  for (float v : oracle)
    if (v != 0.0f) ++nonzero;
  MESSAGE("the bank is sparse: " << nonzero << " of " << oracle.size()
                                 << " values are nonzero");
  CHECK(nonzero > 0u);

  // The TRANSPOSED accessor is the same numbers in the other order — which is
  // what makes Parakeet byte-identical across the extraction rather than
  // merely close.
  const std::vector<float> t =
      vllm::multimodal::MelFilterBankSlaneyTransposed(201, 128, 0.0, 8000.0, 16000);
  REQUIRE(t.size() == ours.size());
  std::size_t t_bad = 0;
  for (int k = 0; k < 201; ++k)
    for (int m = 0; m < 128; ++m)
      if (t[static_cast<std::size_t>(m) * 201 + k] !=
          ours[static_cast<std::size_t>(k) * 128 + m])
        ++t_bad;
  CHECK(t_bad == 0u);
}

TEST_CASE("dots3-note W7a: the two references share no helper with src/, by enumeration") {
  // The row's convention (W6a 70, W6b 105, W6c 45 qualified names, all `std::`).
  //
  // THIS CASE RE-READS THIS FILE AND COUNTS. It used to compare two
  // hand-written constants with two literals, which measured nothing: the lists
  // beside those constants named 22 and 19 names when the references actually
  // use 11 and 6, and a reference that GREW a `vllm::` call would not have
  // moved either number. The load-bearing assertion below is the SCOPE SET —
  // every qualified name in each reference span resolves through `std::` — and
  // it is computed from the bytes, so one `vllm::` or `vt::` call reddens it.
  const RefNames front = QualifiedNamesIn("ref_front");
  const RefNames tower = QualifiedNamesIn("ref_tower");

  // The instrument must be shown to have READ something. A parse that found an
  // empty span would otherwise report "zero non-std:: names" and pass.
  REQUIRE(front.occurrences > 0);
  REQUIRE(tower.occurrences > 0);

  MESSAGE("ref_front: " << front.distinct << " distinct, " << front.occurrences
                        << " occurrences, scopes=" << Join(front.scopes));
  MESSAGE("ref_tower: " << tower.distinct << " distinct, " << tower.occurrences
                        << " occurrences, scopes=" << Join(tower.scopes));

  // The independence property itself.
  CHECK(Join(front.scopes) == "std");
  CHECK(Join(tower.scopes) == "std");

  // And the enumerated lists, now that they are the measurement's own output.
  CHECK(front.distinct == ref_front::kDistinctQualifiedNames);
  CHECK(front.occurrences == ref_front::kQualifiedNameOccurrences);
  CHECK(tower.distinct == ref_tower::kDistinctQualifiedNames);
  CHECK(tower.occurrences == ref_tower::kQualifiedNameOccurrences);

  // AND THE STRIPPER ITSELF, because the property above is only as true as the
  // scan that measures it. `StripCommentsAndLiterals` used to treat every `'`
  // as a char-literal delimiter, so two C++14 digit separators bracketing a
  // `vt::` call hid that call and this case read GREEN with a live reach
  // inside a reference. Nothing above can see that: the clean file carries no
  // separator, so the counts do not move either way. These four are the only
  // standing gate on the pp-number rule.
  const std::string bracketed =
      "double g = 16'000.0 * vt::Scale(vllm::kOne) / 1'280.0;";
  CHECK(StripCommentsAndLiterals(bracketed).find("vt::Scale") !=
        std::string::npos);
  CHECK(StripCommentsAndLiterals(bracketed).find("vllm::kOne") !=
        std::string::npos);
  // A PREFIXED char literal is still a literal, which is what the shorter
  // "ignore a `'` whose previous character is alphanumeric" rule would break.
  CHECK(StripCommentsAndLiterals("u8'v' L't' 'x'").find('v') ==
        std::string::npos);
  // And a comment is still removed whole, separator or not.
  CHECK(StripCommentsAndLiterals("// vt::Scale 16'000\nint a = 1;")
            .find("vt::") == std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. THE FRONT END against `ref_front`.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the front end agrees with an INDEPENDENT double reference") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  REQUIRE(cfg.present);
  CHECK(cfg.sampling_rate == 16000);
  CHECK(cfg.chunk_seconds == spec.a_chunk_seconds);
  CHECK(cfg.n_mels == spec.a_mels);
  CHECK(cfg.token_stride() == 1280);
  CHECK(cfg.chunk_samples() == spec.a_chunk_samples());
  CHECK(cfg.chunk_mel_frames() == spec.a_chunk_mel_frames());
  cfg.audio_start_token_id = dots3_tiny::kAudStartId;
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  cfg.audio_end_token_id = dots3_tiny::kAudEndId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  const std::vector<float> wav = dots3_tiny::FixtureAudioF32(0);
  const vllm::multimodal::AudioKwargs got = proc.ProcessWaveform(
      wav.data(), static_cast<int64_t>(wav.size()), 16000);

  CHECK(got.n_mels == spec.a_mels);
  CHECK(got.n_frames == spec.a_chunk_mel_frames());
  CHECK(got.num_samples == dots3_tiny::kAudioSamples);
  // `ceil(8000 / 1280)` = 7 (`common/processor.py:771`).
  CHECK(got.num_tokens == dots3_tiny::kAudioTokens);

  const std::vector<std::vector<double>> want = ref_front::LogMel(
      wav, static_cast<int>(spec.a_chunk_samples()), cfg.n_fft, cfg.hop_length,
      static_cast<int>(spec.a_mels), 16000);
  REQUIRE(want.size() == static_cast<std::size_t>(spec.a_mels));
  REQUIRE(want[0].size() == static_cast<std::size_t>(spec.a_chunk_mel_frames()));

  double worst = 0.0;
  for (int64_t m = 0; m < spec.a_mels; ++m) {
    for (int64_t f = 0; f < spec.a_chunk_mel_frames(); ++f) {
      const double a =
          got.input_features[static_cast<std::size_t>(m * got.n_frames + f)];
      const double b = want[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)];
      worst = std::max(worst, std::fabs(a - b));
    }
  }
  // Both sides compute in double and differ only in float summation ORDER
  // inside the DFT and the mel projection; the implementation stores f32.
  MESSAGE("front end vs reference: worst |delta| " << worst);
  CHECK(worst < 1e-5);

  // THE PADDED TAIL IS NOT ZERO, which is the whole reason the tower masks. The
  // clip is half the chunk, so frames 50..99 are silence — and their value is
  // the `-8` global-max floor pushed through `(x + 4) / 4`, a NONZERO constant.
  const double tail =
      got.input_features[static_cast<std::size_t>(0 * got.n_frames + 99)];
  MESSAGE("the padded tail sits at " << tail << ", not at 0");
  CHECK(std::fabs(tail) > 0.1);
  // ...and it is the SAME constant everywhere in the tail, which is what makes
  // it leak as a bias rather than as noise.
  for (int64_t m = 0; m < spec.a_mels; ++m) {
    const double v =
        got.input_features[static_cast<std::size_t>(m * got.n_frames + 99)];
    CHECK(std::fabs(v - tail) < 1e-6);
  }
}

TEST_CASE("dots3-note W7a: the front end REFUSES a wrong rate and an over-long clip, BY NAME") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);
  const std::vector<float> wav = dots3_tiny::FixtureAudioF32(0);

  SUBCASE("a rate that is not `audio_config.sampling_rate` names W7c") {
    std::string msg;
    try {
      proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()), 22050);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("22050") != std::string::npos);
    CHECK(msg.find("W7c") != std::string::npos);
    CHECK(msg.find("RESAMPLING IS NOT PORTED") != std::string::npos);
  }
  SUBCASE("a clip over `chunk_seconds` names W7b, and says why it is not a convenience") {
    std::vector<float> too_long(static_cast<std::size_t>(spec.a_chunk_samples() + 1),
                                0.1f);
    std::string msg;
    try {
      proc.ProcessWaveform(too_long.data(),
                           static_cast<int64_t>(too_long.size()), 16000);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("W7b") != std::string::npos);
    CHECK(msg.find("SEGMENTATION IS NOT PORTED") != std::string::npos);
    CHECK(msg.find("splices audio features") != std::string::npos);
  }
  SUBCASE("...and exactly `chunk_samples` is ACCEPTED, so the bound is not off by one") {
    std::vector<float> exact(static_cast<std::size_t>(spec.a_chunk_samples()), 0.1f);
    const vllm::multimodal::AudioKwargs kw = proc.ProcessWaveform(
        exact.data(), static_cast<int64_t>(exact.size()), 16000);
    // `ceil`, not floor: 16000 / 1280 is 12.5, so a full chunk is THIRTEEN
    // tokens. That is also exactly the stem's output length (100 mel frames ->
    // 50 -> 25 -> 13), which is why a full chunk is the largest span the tower
    // can produce and why `chunk_seconds` is the right place to refuse.
    CHECK(kw.num_tokens ==
          (spec.a_chunk_samples() + 1280 - 1) / 1280);
    CHECK(kw.num_tokens == dots3_tiny::kAudioStemFrames);
  }
}

TEST_CASE("dots3-note W7a: `num_tokens` and the mask length are TWO numbers") {
  // §4.14: halving `samples // 160` three times is NOT `ceil(samples / 1280)`.
  // At 1281 samples the mask says 1 and the span says 2, so the span covers a
  // stem row the mask zeroed. A port that derived one from the other would be
  // wrong here and right on the fixture clip, which is why this case exists.
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  cfg.audio_token_id = dots3_tiny::kAudPadId;
  const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);

  const auto mask_len = [](int64_t n) {
    int64_t v = n / 160;
    for (int i = 0; i < 3; ++i) v = (v + 1) / 2;
    return v;
  };
  CHECK(proc.NumAudioTokens(1280) == 1);
  CHECK(mask_len(1280) == 1);
  CHECK(proc.NumAudioTokens(1281) == 2);
  CHECK(mask_len(1281) == 1);
  MESSAGE("at 1281 samples: span " << proc.NumAudioTokens(1281) << ", mask "
                                   << mask_len(1281));
  // The fixture clip is a case where they AGREE, which is why it cannot see the
  // difference on its own.
  CHECK(proc.NumAudioTokens(dots3_tiny::kAudioSamples) == dots3_tiny::kAudioTokens);
  CHECK(mask_len(dots3_tiny::kAudioSamples) == dots3_tiny::kAudioTokens);
}

TEST_CASE("dots3-note W7a: the three marker ids come from the TOKENIZER, and refuse BY NAME") {
  const TinySpec spec = AudioSpec();
  const TinyCheckpoint ckpt(FixtureDir(), spec);
  vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                          "tiny");
  REQUIRE(cfg.present);
  CHECK(cfg.audio_comp_start == "<|audio_comp_start|>");
  CHECK(cfg.audio_comp_span == "<|audio_comp_pad|>");
  CHECK(cfg.audio_comp_end == "<|audio_comp_end|>");
  // NOT resolved by the config loader: only a tokenizer can answer.
  CHECK(cfg.audio_token_id == -1);

  SUBCASE("a tokenizer that carries all three resolves them BY STRING") {
    vllm::multimodal::Dots3NoteAudioProcessorConfig c = cfg;
    vllm::multimodal::Dots3NoteResolveAudioTokenIds(
        &c, [](const std::string& m) -> int32_t {
          if (m == "<|audio_comp_start|>") return dots3_tiny::kAudStartId;
          if (m == "<|audio_comp_end|>") return dots3_tiny::kAudEndId;
          if (m == "<|audio_comp_pad|>") return dots3_tiny::kAudPadId;
          return -1;
        });
    CHECK(c.audio_start_token_id == dots3_tiny::kAudStartId);
    CHECK(c.audio_end_token_id == dots3_tiny::kAudEndId);
    CHECK(c.audio_token_id == dots3_tiny::kAudPadId);
    // THE ORDER IS start, END, pad — the released checkpoint's own
    // (151718 / 151719 / 151720). A port that guessed "start, start+1,
    // start+2" would put the PAD id where the END id belongs, and this is the
    // assertion that catches it.
    CHECK(c.audio_end_token_id == c.audio_start_token_id + 1);
    CHECK(c.audio_token_id == c.audio_start_token_id + 2);
    CHECK(c.audio_token_id != c.audio_start_token_id + 1);
  }
  SUBCASE("a tokenizer missing ONE of them refuses, naming the marker") {
    vllm::multimodal::Dots3NoteAudioProcessorConfig c = cfg;
    std::string msg;
    try {
      vllm::multimodal::Dots3NoteResolveAudioTokenIds(
          &c, [](const std::string& m) -> int32_t {
            if (m == "<|audio_comp_pad|>") return -1;  // the one that is absent
            return 5;
          });
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("<|audio_comp_pad|>") != std::string::npos);
    CHECK(msg.find("audio_comp_span") != std::string::npos);
    CHECK(msg.find("REFUSING rather than defaulting") != std::string::npos);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. THE TOWER against `ref_tower`.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// The fixture checkpoint's bf16-rounded values, reshaped for the reference.
// Reading them from `TinyCheckpoint::value_of` rather than from the loaded
// `OwnedTensor` is deliberate: the reference is driven from the SAME BYTES the
// loader read, so the comparison measures the FORWARD and not a second copy of
// the weights.
struct TowerRefWeights {
  std::vector<ref_tower::Mat> conv1, conv2, conv3;  // [out][in][9]
  std::vector<double> conv1_b, conv2_b, conv3_b;
  ref_tower::Mat conv_out;  // [D][C*F]
  struct Layer {
    std::vector<double> attn_norm, final_norm;
    ref_tower::Mat q, k, v, o;
    std::vector<double> qb, vb, ob;
    ref_tower::Mat fc1, fc2;
    std::vector<double> fc1_b, fc2_b;
  };
  std::vector<Layer> layers;
  std::vector<double> final_norm;
  std::vector<double> ln_w, ln_b;
  ref_tower::Mat a1, a2;
  std::vector<double> a1_b, a2_b;
};

ref_tower::Mat Reshape(const std::vector<double>& v, std::size_t rows,
                       std::size_t cols) {
  REQUIRE(v.size() == rows * cols);
  ref_tower::Mat out(rows, std::vector<double>(cols));
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) out[r][c] = v[r * cols + c];
  return out;
}

std::vector<ref_tower::Mat> Reshape3(const std::vector<double>& v,
                                     std::size_t out_ch, std::size_t in_ch) {
  REQUIRE(v.size() == out_ch * in_ch * 9);
  std::vector<ref_tower::Mat> out(out_ch,
                                  ref_tower::Mat(in_ch, std::vector<double>(9)));
  std::size_t i = 0;
  for (std::size_t co = 0; co < out_ch; ++co)
    for (std::size_t ci = 0; ci < in_ch; ++ci)
      for (std::size_t k = 0; k < 9; ++k) out[co][ci][k] = v[i++];
  return out;
}

TowerRefWeights ReadTowerWeights(const TinyCheckpoint& ckpt, const TinySpec& s) {
  const std::string se = "audio_encoder.dots_encoder.speech_encoder.";
  const std::size_t D = static_cast<std::size_t>(s.a_d_model);
  const std::size_t F = static_cast<std::size_t>(s.a_ffn);
  const std::size_t dhs = static_cast<std::size_t>(s.a_dhs);
  TowerRefWeights w;
  w.conv1 = Reshape3(ckpt.value_of(se + "conv2d1.weight"), dhs, 1);
  w.conv1_b = ckpt.value_of(se + "conv2d1.bias");
  w.conv2 = Reshape3(ckpt.value_of(se + "conv2d2.weight"), dhs, dhs);
  w.conv2_b = ckpt.value_of(se + "conv2d2.bias");
  w.conv3 = Reshape3(ckpt.value_of(se + "conv2d3.weight"), dhs, dhs);
  w.conv3_b = ckpt.value_of(se + "conv2d3.bias");
  w.conv_out = Reshape(ckpt.value_of(se + "conv_out.weight"), D,
                       dhs * static_cast<std::size_t>(s.a_freq_after()));
  for (int64_t l = 0; l < s.a_layers; ++l) {
    const std::string p = se + "layers." + std::to_string(l) + ".";
    TowerRefWeights::Layer lw;
    lw.attn_norm = ckpt.value_of(p + "self_attn_layer_norm.weight");
    lw.final_norm = ckpt.value_of(p + "final_layer_norm.weight");
    lw.q = Reshape(ckpt.value_of(p + "self_attn.q_proj.weight"), D, D);
    lw.qb = ckpt.value_of(p + "self_attn.q_proj.bias");
    lw.k = Reshape(ckpt.value_of(p + "self_attn.k_proj.weight"), D, D);
    lw.v = Reshape(ckpt.value_of(p + "self_attn.v_proj.weight"), D, D);
    lw.vb = ckpt.value_of(p + "self_attn.v_proj.bias");
    lw.o = Reshape(ckpt.value_of(p + "self_attn.out_proj.weight"), D, D);
    lw.ob = ckpt.value_of(p + "self_attn.out_proj.bias");
    lw.fc1 = Reshape(ckpt.value_of(p + "fc1.weight"),
                     static_cast<std::size_t>(s.a_fc1_out()), D);
    lw.fc1_b = ckpt.value_of(p + "fc1.bias");
    lw.fc2 = Reshape(ckpt.value_of(p + "fc2.weight"), D, F);
    lw.fc2_b = ckpt.value_of(p + "fc2.bias");
    w.layers.push_back(std::move(lw));
  }
  w.final_norm = ckpt.value_of(se + "layer_norm.weight");
  const std::string ad = "audio_encoder.audio_adapter.proj.";
  const std::size_t AO = static_cast<std::size_t>(s.a_adapter_out());
  w.ln_w = ckpt.value_of(ad + "0.weight");
  w.ln_b = ckpt.value_of(ad + "0.bias");
  w.a1 = Reshape(ckpt.value_of(ad + "1.weight"), AO, D);
  w.a1_b = ckpt.value_of(ad + "1.bias");
  w.a2 = Reshape(ckpt.value_of(ad + "3.weight"), AO, AO);
  w.a2_b = ckpt.value_of(ad + "3.bias");
  return w;
}

// THE WHOLE TOWER, in double precision, from `nvidia/audio_encoder.py:611-736`
// and `nvidia/audio.py:193-282` @ `9035151d6`. `apply_mask` and `rotate_all`
// are MUTATION HANDLES, not options: a case flips one and asserts the answer
// moves, which is how §4.14.8's mutations C and (the rope half of) the
// partial-RoPE risk are measured rather than argued.
ref_tower::Mat RefTower(const TowerRefWeights& w, const TinySpec& s,
                        const std::vector<std::vector<double>>& mel,
                        int64_t num_samples, int64_t num_tokens,
                        bool apply_mask = true, bool rotate_all = false,
                        bool k_has_bias = false) {
  using ref_tower::Mat;
  const std::size_t dhs = static_cast<std::size_t>(s.a_dhs);
  const std::size_t D = static_cast<std::size_t>(s.a_d_model);
  const std::size_t nh = static_cast<std::size_t>(s.a_heads);
  const std::size_t hd = static_cast<std::size_t>(s.a_head_dim());

  // ── the stem, `_conv2d_stem_one_chunk` (:535-562) ───────────────────────
  std::vector<Mat> x(1, mel);
  std::vector<std::int64_t> valid(4);
  valid[0] = num_samples / 160;
  valid[1] = (valid[0] + 1) / 2;
  valid[2] = (valid[1] + 1) / 2;
  valid[3] = (valid[2] + 1) / 2;
  if (apply_mask) ref_tower::MaskTime(&x, valid[0]);
  x = ref_tower::Conv2dGelu(x, w.conv1, w.conv1_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[1]);
  x = ref_tower::Conv2dGelu(x, w.conv2, w.conv2_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[2]);
  x = ref_tower::Conv2dGelu(x, w.conv3, w.conv3_b, dhs);
  if (apply_mask) ref_tower::MaskTime(&x, valid[3]);

  // `x.permute(0, 3, 1, 2).reshape(B, T, C * F)` (:579-580) — CHANNEL-major.
  const std::size_t Fo = x[0].size(), To = x[0][0].size();
  Mat rows(To, std::vector<double>(dhs * Fo));
  for (std::size_t t = 0; t < To; ++t)
    for (std::size_t c = 0; c < dhs; ++c)
      for (std::size_t f = 0; f < Fo; ++f)
        rows[t][c * Fo + f] = x[c][f][t];

  // `conv_out` (:581), then the varlen pack keeps the first `num_tokens`
  // (:679-681 with B == 1).
  Mat hidden = ref_tower::Linear(rows, w.conv_out, nullptr);
  hidden.resize(static_cast<std::size_t>(num_tokens));

  // ── the rope cache (:69-79, :126-131) ──────────────────────────────────
  const std::size_t rd =
      rotate_all ? hd : static_cast<std::size_t>(s.a_rotary_dim());
  const std::size_t nf = rd / 2;
  Mat cos_t(hidden.size(), std::vector<double>(nf));
  Mat sin_t(hidden.size(), std::vector<double>(nf));
  for (std::size_t t = 0; t < hidden.size(); ++t) {
    for (std::size_t i = 0; i < nf; ++i) {
      const double inv =
          1.0 / std::pow(s.a_rope_theta,
                         static_cast<double>(2 * i) / static_cast<double>(rd));
      cos_t[t][i] = std::cos(static_cast<double>(t) * inv);
      sin_t[t][i] = std::sin(static_cast<double>(t) * inv);
    }
  }

  for (std::size_t l = 0; l < w.layers.size(); ++l) {
    const TowerRefWeights::Layer& lw = w.layers[l];
    const Mat n1 = ref_tower::RmsNorm(hidden, lw.attn_norm, s.rms_eps);
    Mat q = ref_tower::Linear(n1, lw.q, &lw.qb);
    // NO BIAS ON K (`:221`), unless a mutation case asks for one.
    Mat k = ref_tower::Linear(n1, lw.k, k_has_bias ? &lw.qb : nullptr);
    const Mat v = ref_tower::Linear(n1, lw.v, &lw.vb);

    // `apply_rotary_pos_emb` (:146-181): rotate the LEADING `rd` dims of each
    // head, NeoX half-split, and leave the tail alone.
    const auto rope = [&](Mat& m) {
      for (std::size_t t = 0; t < m.size(); ++t) {
        for (std::size_t h = 0; h < nh; ++h) {
          const std::size_t base = h * hd;
          std::vector<double> rot(rd);
          for (std::size_t i = 0; i < rd; ++i) rot[i] = m[t][base + i];
          for (std::size_t i = 0; i < rd; ++i) {
            const double c = cos_t[t][i % nf];
            const double sn = sin_t[t][i % nf];
            const double other = i < nf ? -rot[i + nf] : rot[i - nf];
            m[t][base + i] = rot[i] * c + other * sn;
          }
        }
      }
    };
    rope(q);
    rope(k);

    // Full non-causal MHA per head, softmax in double.
    Mat attn(hidden.size(), std::vector<double>(D, 0.0));
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    for (std::size_t h = 0; h < nh; ++h) {
      const std::size_t base = h * hd;
      for (std::size_t i = 0; i < q.size(); ++i) {
        std::vector<double> sc(k.size());
        double m = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < k.size(); ++j) {
          double dot = 0.0;
          for (std::size_t c = 0; c < hd; ++c)
            dot += q[i][base + c] * k[j][base + c];
          sc[j] = dot * scale;
          m = std::max(m, sc[j]);
        }
        double sum = 0.0;
        for (double& e : sc) {
          e = std::exp(e - m);
          sum += e;
        }
        for (std::size_t c = 0; c < hd; ++c) {
          double acc = 0.0;
          for (std::size_t j = 0; j < k.size(); ++j)
            acc += sc[j] / sum * v[j][base + c];
          attn[i][base + c] = acc;
        }
      }
    }
    const Mat proj = ref_tower::Linear(attn, lw.o, &lw.ob);
    for (std::size_t r = 0; r < hidden.size(); ++r)
      for (std::size_t c = 0; c < D; ++c) hidden[r][c] += proj[r][c];

    const Mat n2 = ref_tower::RmsNorm(hidden, lw.final_norm, s.rms_eps);
    const Mat f1 = ref_tower::Linear(n2, lw.fc1, &lw.fc1_b);
    const Mat act = ref_tower::SwiGlu(f1);
    const Mat f2 = ref_tower::Linear(act, lw.fc2, &lw.fc2_b);
    for (std::size_t r = 0; r < hidden.size(); ++r)
      for (std::size_t c = 0; c < D; ++c) hidden[r][c] += f2[r][c];
  }

  hidden = ref_tower::RmsNorm(hidden, w.final_norm, s.rms_eps);

  // `AudioAdapter` (`audio.py:240-248`): LayerNorm -> Linear -> GELU-erf ->
  // Linear.
  Mat a = ref_tower::LayerNorm(hidden, w.ln_w, w.ln_b, 1e-5);
  a = ref_tower::Linear(a, w.a1, &w.a1_b);
  for (auto& row : a)
    for (double& e : row) e = ref_tower::GeluErf(e);
  return ref_tower::Linear(a, w.a2, &w.a2_b);
}

// The relative L2 the row's other tower gates report.
double RelL2(const std::vector<float>& got, const ref_tower::Mat& want) {
  double num = 0.0, den = 0.0;
  std::size_t i = 0;
  for (const std::vector<double>& row : want) {
    for (double w : row) {
      const double d = static_cast<double>(got[i++]) - w;
      num += d * d;
      den += w * w;
    }
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Everything a tower case needs: the fixture checkpoint, the loaded weights and
// the mel the front end produced.
struct LoadedTower {
  TinySpec spec;
  TinyCheckpoint ckpt;
  vllm::HfConfig config;
  vllm::Dots3NoteAudioParams params;
  vllm::Dots3NoteAudioWeights weights;
  vllm::multimodal::AudioKwargs mel;
  std::vector<std::vector<double>> mel_ref;

  explicit LoadedTower(TinySpec s = AudioSpec(), int variant = 0)
      : spec(s),
        ckpt(FixtureDir(), s),
        config(vllm::LoadHfConfig(ckpt.config_path())) {
    params = vllm::ParseDots3NoteAudioParams(config);
    REQUIRE(params.present);
    REQUIRE(vllm::Dots3NoteAudioRefusal(params, "", {}).empty());
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.weights_path()));
    weights = vllm::MaterializeDots3NoteAudio(shards, params);
    REQUIRE(weights.present);

    vllm::multimodal::Dots3NoteAudioProcessorConfig cfg =
        vllm::multimodal::LoadDots3NoteAudioProcessorConfig(ckpt.config_path(),
                                                            "tiny");
    cfg.audio_token_id = dots3_tiny::kAudPadId;
    const vllm::multimodal::Dots3NoteAudioProcessor proc(cfg);
    const std::vector<float> wav = dots3_tiny::FixtureAudioF32(variant);
    mel = proc.ProcessWaveform(wav.data(), static_cast<int64_t>(wav.size()),
                               16000);
    mel_ref.assign(static_cast<std::size_t>(mel.n_mels),
                   std::vector<double>(static_cast<std::size_t>(mel.n_frames)));
    for (int64_t m = 0; m < mel.n_mels; ++m)
      for (int64_t f = 0; f < mel.n_frames; ++f)
        mel_ref[static_cast<std::size_t>(m)][static_cast<std::size_t>(f)] =
            mel.input_features[static_cast<std::size_t>(m * mel.n_frames + f)];
  }

  std::vector<float> Run(vllm::Dots3NoteAudioCapture* cap = nullptr) const {
    return vllm::Dots3NoteAudioForward(mel.input_features, mel.num_samples,
                                       mel.num_tokens, /*hop_length=*/160,
                                       weights, params,
                                       vt::GetBackend(vt::DeviceType::kCPU), cap);
  }
};

}  // namespace

TEST_CASE("dots3-note W7a: the AUDIO tower agrees with an INDEPENDENT double reference") {
  const LoadedTower t;
  vllm::Dots3NoteAudioCapture cap;
  const std::vector<float> got = t.Run(&cap);

  REQUIRE(got.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens *
                                                 t.spec.a_adapter_out()));
  // The four mask lengths, in stage order — captured because a mask that halved
  // with the wrong rounding would still zero SOMETHING.
  REQUIRE(cap.valid_lens.size() == 4u);
  CHECK(cap.valid_lens[0] == dots3_tiny::kAudioSamples / 160);
  CHECK(cap.valid_lens[1] == 25);
  CHECK(cap.valid_lens[2] == 13);
  CHECK(cap.valid_lens[3] == dots3_tiny::kAudioTokens);
  MESSAGE("mask stages: " << cap.valid_lens[0] << " -> " << cap.valid_lens[1]
                          << " -> " << cap.valid_lens[2] << " -> "
                          << cap.valid_lens[3]
                          << ", against a padded mel of " << t.mel.n_frames
                          << " frames and a stem output of "
                          << dots3_tiny::kAudioStemFrames);

  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat want =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens);
  REQUIRE(want.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens));
  REQUIRE(want[0].size() == static_cast<std::size_t>(t.spec.a_adapter_out()));

  const double rel = RelL2(got, want);
  MESSAGE("tower vs reference rel-L2: " << rel);
  // The bf16 envelope of a 2-block tower plus a 3-layer conv stem, with the
  // deliberate `vt::RmsNorm` rounding difference `dots3_note_audio.h` records.
  CHECK(rel < 5e-2);

  // AND THE ANSWER IS NOT A CONSTANT. A tower replaced by a correctly-shaped
  // constant passes every SHAPE assertion above; it cannot pass this one.
  double lo = got[0], hi = got[0];
  for (float v : got) {
    lo = std::min(lo, static_cast<double>(v));
    hi = std::max(hi, static_cast<double>(v));
  }
  MESSAGE("tower output spans [" << lo << ", " << hi << "]");
  CHECK(hi - lo > 1e-3);
}

TEST_CASE("dots3-note W7a: the TEMPORAL MASK changes the answer, at every stage") {
  // §4.14.3. The mel of a zero-padded tail is the `-8` floor through
  // `(x + 4) / 4`, a NONZERO constant, so an unmasked stem leaks it through the
  // 3x3 receptive fields into the LAST VALID tokens. This case measures the
  // leak rather than arguing it: the same reference is run with the four mask
  // stages deleted, and the two answers are asserted to DIFFER.
  const LoadedTower t;
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat masked =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true);
  const ref_tower::Mat unmasked =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/false);

  double worst = 0.0;
  std::size_t worst_row = 0;
  for (std::size_t r = 0; r < masked.size(); ++r) {
    for (std::size_t c = 0; c < masked[r].size(); ++c) {
      const double d = std::fabs(masked[r][c] - unmasked[r][c]);
      if (d > worst) {
        worst = d;
        worst_row = r;
      }
    }
  }
  MESSAGE("deleting the mask moves the answer by up to " << worst
          << ", worst at token " << worst_row << " of " << masked.size());
  CHECK(worst > 1e-3);

  // AND IT REACHES THE LAST KEPT TOKEN, which is the claim that matters: a leak
  // that only touched tokens the span discards would be harmless.
  double last_row = 0.0;
  const std::size_t last = masked.size() - 1;
  for (std::size_t c = 0; c < masked[last].size(); ++c)
    last_row = std::max(last_row, std::fabs(masked[last][c] - unmasked[last][c]));
  MESSAGE("the leak reaches the LAST kept token by " << last_row);
  CHECK(last_row > 1e-3);

  // The IMPLEMENTATION is the masked one.
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, masked) < 5e-2);
  CHECK(RelL2(got, unmasked) > RelL2(got, masked));
}

TEST_CASE("dots3-note W7a: PARTIAL rope rotates HALF the head, and the tail is untouched") {
  const LoadedTower t;
  CHECK(t.params.head_dim() == t.spec.a_head_dim());
  CHECK(t.params.rotary_dim() == t.spec.a_rotary_dim());
  CHECK(t.params.rotary_dim() * 2 == t.params.head_dim());

  // The cache is [T, rotary_dim] = [cos(rd/2) | sin(rd/2)], and at t == 0 every
  // angle is zero — so the first row is all ones then all zeros. A cache built
  // over `head_dim` frequencies instead would be twice as wide.
  const std::vector<float> cache =
      vllm::Dots3NoteAudioRopeCache(dots3_tiny::kAudioTokens, t.params);
  REQUIRE(cache.size() == static_cast<std::size_t>(dots3_tiny::kAudioTokens *
                                                   t.params.rotary_dim()));
  const int64_t nf = t.params.rotary_dim() / 2;
  for (int64_t i = 0; i < nf; ++i) {
    CHECK(cache[static_cast<std::size_t>(i)] == doctest::Approx(1.0));
    CHECK(cache[static_cast<std::size_t>(nf + i)] == doctest::Approx(0.0));
  }
  // The DENOMINATOR is `rotary_dim`, not `head_dim`. At rotary_dim 4 the two
  // frequencies are theta^0 = 1 and theta^-0.5; over head_dim 8 the second
  // would be theta^-0.25, which is a different number.
  const double want_inv = 1.0 / std::pow(t.spec.a_rope_theta,
                                         2.0 / static_cast<double>(t.params.rotary_dim()));
  CHECK(cache[static_cast<std::size_t>(1 * t.params.rotary_dim() + 1)] ==
        doctest::Approx(std::cos(want_inv)).epsilon(1e-5));

  // ROTATING THE WHOLE HEAD moves the answer, which is what makes the partial
  // rotation an assertion rather than a description.
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat partial =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false);
  const ref_tower::Mat full =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/true);
  double worst = 0.0;
  for (std::size_t r = 0; r < partial.size(); ++r)
    for (std::size_t c = 0; c < partial[r].size(); ++c)
      worst = std::max(worst, std::fabs(partial[r][c] - full[r][c]));
  MESSAGE("rotating the whole head instead of half moves the answer by "
          << worst);
  CHECK(worst > 1e-3);
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, partial) < RelL2(got, full));
}

TEST_CASE("dots3-note W7a: `k_proj` has NO bias, and giving it one moves the answer") {
  // `nn.Linear(embed_dim, embed_dim, bias=False)` for k against `bias=bias`
  // (default True) for q, v and out (`audio_encoder.py:221-224`) — Whisper's
  // own convention and the obvious thing to get wrong. The checkpoint agrees:
  // 32 each of q/v/out bias and none for k.
  const LoadedTower t;
  const TowerRefWeights rw = ReadTowerWeights(t.ckpt, t.spec);
  const ref_tower::Mat no_k_bias =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false, /*k_has_bias=*/false);
  const ref_tower::Mat with_k_bias =
      RefTower(rw, t.spec, t.mel_ref, t.mel.num_samples, t.mel.num_tokens,
               /*apply_mask=*/true, /*rotate_all=*/false, /*k_has_bias=*/true);
  double worst = 0.0;
  for (std::size_t r = 0; r < no_k_bias.size(); ++r)
    for (std::size_t c = 0; c < no_k_bias[r].size(); ++c)
      worst = std::max(worst, std::fabs(no_k_bias[r][c] - with_k_bias[r][c]));
  MESSAGE("adding a k bias moves the answer by " << worst);
  CHECK(worst > 1e-3);
  const std::vector<float> got = t.Run();
  CHECK(RelL2(got, no_k_bias) < RelL2(got, with_k_bias));

  // AND THE ENUMERATION AGREES: no `k_proj.bias` is claimed, so a checkpoint
  // shipping one would be UNACCOUNTED rather than silently loaded.
  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteAudioTensors(t.params);
  std::size_t k_bias = 0, q_bias = 0;
  for (const vllm::Dots3NoteTensor& c : claimed) {
    if (c.name.find("k_proj.bias") != std::string::npos) ++k_bias;
    if (c.name.find("q_proj.bias") != std::string::npos) ++q_bias;
  }
  CHECK(k_bias == 0u);
  CHECK(q_bias == static_cast<std::size_t>(t.spec.a_layers));
}

TEST_CASE("dots3-note W7a: two DIFFERENT waveforms give two different tower outputs") {
  const LoadedTower a(AudioSpec(), /*variant=*/0);
  const LoadedTower b(AudioSpec(), /*variant=*/1);
  const std::vector<float> ga = a.Run();
  const std::vector<float> gb = b.Run();
  REQUIRE(ga.size() == gb.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < ga.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(ga[i] - gb[i])));
  MESSAGE("two waveforms differ in the tower output by up to " << worst);
  CHECK(worst > 1e-3);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. THE ENUMERATION, and the RELEASED checkpoint's own 430.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: the tower claims ALL 430 of the RELEASED tower's tensors") {
  // The released `audio_config`, read from the committed fixture. 32 layers x
  // 13 tensors = 416, plus 7 stem, 1 final norm and 6 adapter = 430 — the
  // number `bucket_totals` in `index_full.json` records and W2 asserted.
  const vllm::HfConfig cfg =
      vllm::LoadHfConfig(FixtureDir() + "/config.json");
  const vllm::Dots3NoteAudioParams a = vllm::ParseDots3NoteAudioParams(cfg);
  REQUIRE(a.present);
  CHECK(a.d_model == 1280);
  CHECK(a.num_heads == 20);
  CHECK(a.num_layers == 32);
  CHECK(a.ffn_dim == 5120);
  CHECK(a.num_mel_bins == 128);
  CHECK(a.max_source_positions == 6000);
  CHECK(a.downsample_hidden_size == 480);
  CHECK(a.adapter_in_dim == 1280);
  CHECK(a.adapter_out_dim == 5120);
  CHECK(a.head_dim() == 64);
  CHECK(a.rotary_dim() == 32);
  CHECK(a.freq_after() == 16);
  CHECK(a.conv_out_in_dim() == 7680);
  CHECK(a.fc1_out() == 10240);
  // The three MEASURED-DEAD knobs are READ and then never used.
  CHECK(a.conv_chunksize == 500);
  CHECK(a.conv_bucket_step == 10);
  CHECK(a.conv_bucket_max_elements == 20000);

  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteAudioTensors(a);
  MESSAGE("the released audio tower claims " << claimed.size() << " tensors");
  CHECK(claimed.size() == 430u);
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK(t.name.rfind("audio_encoder.", 0) == 0);
    CHECK_FALSE(t.consumer.empty());
  }
  // NO `conv_out.bias` and NO `k_proj.bias`: both are `bias=False` upstream and
  // both are absent from the released checkpoint. Claiming either would refuse
  // every real load.
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK(t.name.find("conv_out.bias") == std::string::npos);
    CHECK(t.name.find("k_proj.bias") == std::string::npos);
  }
  // And the released tower is ACCEPTED — the whole point of the brick.
  CHECK(vllm::Dots3NoteAudioRefusal(a, "", {}).empty());
}

TEST_CASE("dots3-note W7a: every unported audio arm refuses BY NAME, with its brick") {
  const vllm::HfConfig base =
      vllm::LoadHfConfig(FixtureDir() + "/config.json");
  const auto refuse = [&](const std::function<void(nlohmann::json&)>& edit) {
    nlohmann::json raw = base.raw;
    edit(raw["audio_config"]);
    vllm::HfConfig c = base;
    c.raw = raw;
    return vllm::Dots3NoteAudioRefusal(vllm::ParseDots3NoteAudioParams(c), "", {});
  };

  SUBCASE("the RELEASED config is accepted — the premise of every case below") {
    CHECK(refuse([](nlohmann::json&) {}).empty());
  }
  SUBCASE("a BLOCKWISE-QUANTIZED checkpoint is W9, and outranks everything") {
    const std::string why = vllm::Dots3NoteAudioRefusal(
        vllm::ParseDots3NoteAudioParams(base), "fp8", {128, 128});
    CHECK(why.find("W9") != std::string::npos);
    CHECK(why.find("BLOCKWISE") != std::string::npos);
    // ...and it says the released tower is NOT one, so a reader does not
    // conclude the checkpoint they have is refused.
    CHECK(why.find("430") != std::string::npos);
  }
  SUBCASE("`encoder_type` other than 'dots' — upstream refuses it too") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["encoder_type"] = "whisper"; });
    CHECK(why.find("encoder_type") != std::string::npos);
    CHECK(why.find("audio.py:255-256") != std::string::npos);
  }
  SUBCASE("a non-swiglu activation is a DIFFERENT state dict, not a swap") {
    const std::string why = refuse([](nlohmann::json& a) {
      a["whisper_config"]["activation_function"] = "gelu";
    });
    CHECK(why.find("swiglu") != std::string::npos);
    CHECK(why.find("state dict") != std::string::npos);
  }
  SUBCASE("`use_conv2d_stem` false selects the conv1d stem") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_conv2d_stem"] = false; });
    CHECK(why.find("use_conv2d_stem") != std::string::npos);
    CHECK(why.find("1280") != std::string::npos);  // the stride it changes
  }
  SUBCASE("`use_latent_input` selects the latent stem, and OUTRANKS the conv1d arm") {
    // Both flags at once, because `use_latent_input` alone is REFUSED AT PARSE:
    // `use_conv2d_stem and use_latent_input` is mutually exclusive upstream
    // (`audio_encoder.py:450-453`) and this port mirrors that VT_CHECK. The
    // refusal order is upstream's constructor order — latent first, then the
    // stem selection — so a latent config is told about the latent stem rather
    // than about the conv1d one it also implies.
    const std::string why = refuse(
        [](nlohmann::json& a) { a["whisper_config"]["use_latent_input"] = true;
                                a["use_conv2d_stem"] = false; });
    CHECK(why.find("use_latent_input") != std::string::npos);
    CHECK(why.find("LATENT stem") != std::string::npos);
  }
  SUBCASE("...and `use_latent_input` WITH the conv2d stem refuses at PARSE") {
    nlohmann::json raw = base.raw;
    raw["audio_config"]["whisper_config"]["use_latent_input"] = true;
    vllm::HfConfig c = base;
    c.raw = raw;
    std::string msg;
    try {
      vllm::ParseDots3NoteAudioParams(c);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("mutually exclusive") != std::string::npos);
  }
  SUBCASE("`use_causal` changes THREE things at once") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_causal"] = true; });
    CHECK(why.find("use_causal") != std::string::npos);
    CHECK(why.find("THREE") != std::string::npos);
  }
  SUBCASE("`use_rms_norm` false is a different state dict") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_rms_norm"] = false; });
    CHECK(why.find("use_rms_norm") != std::string::npos);
    // A LayerNorm ships a BIAS none of the 65 norms in the released checkpoint
    // carries, which is what makes this a state-dict change and not a formula
    // one.
    CHECK(why.find("BIAS") != std::string::npos);
    CHECK(why.find("65 norms") != std::string::npos);
  }
  SUBCASE("`use_rope` false needs a positional table nothing ships") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["use_rope"] = false; });
    CHECK(why.find("use_rope") != std::string::npos);
    CHECK(why.find("embed_positions") != std::string::npos);
  }
  SUBCASE("`merge_factor != 1` changes the adapter's INPUT width") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["merge_factor"] = 2; });
    CHECK(why.find("merge_factor") != std::string::npos);
    CHECK(why.find("adapter") != std::string::npos);
  }
  SUBCASE("an adapter that does not land in the TEXT hidden space refuses") {
    const std::string why =
        refuse([](nlohmann::json& a) { a["whisper_adapter_out_dim"] = 4096; });
    CHECK(why.find("whisper_adapter_out_dim") != std::string::npos);
    CHECK(why.find("EncodeMmDots3NoteForCausalLM") != std::string::npos);
  }
  SUBCASE("a checkpoint with NO audio_config refuses, naming the absence") {
    nlohmann::json raw = base.raw;
    raw.erase("audio_config");
    vllm::HfConfig c = base;
    c.raw = raw;
    const vllm::Dots3NoteAudioParams a = vllm::ParseDots3NoteAudioParams(c);
    CHECK_FALSE(a.present);
    const std::string why = vllm::Dots3NoteAudioRefusal(a, "", {});
    CHECK(why.find("no `audio_config`") != std::string::npos);
    CHECK(why.find("multimodal.py:119-126") != std::string::npos);
    // And the enumeration claims NOTHING, so the 430 stay in the deferral
    // bucket rather than becoming missing tensors.
    CHECK(vllm::EnumerateDots3NoteAudioTensors(a).empty());
  }
}
