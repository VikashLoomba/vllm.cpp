// Qwen4-Exp (Qwen3.8-Flash-Next) W6-CUDA DEVICE-ARM GATE — the CUDA arms of
// `vt::Qwen4ExpPleConv`, `vt::Qwen4ExpPleGate` and
// `vt::Qwen4ExpGatedResidualWriteBack`, the first CUDA kernels this
// architecture has ever had.
// Row MODEL-MM-QWEN4-EXP, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. Two things, and the
// distinction is the point of the file:
//
//   1. THE ORACLE. The conv and the gate are compared DIRECTLY against
//      `qwen4_exp_ple_goldens.inc`, dumped by `scripts/gen-qwen4-exp-ple-goldens.py`
//      which lifts `Qwen4ExpTextPLELayer` and `LinearAttentionLayer.update_conv_state`
//      VERBATIM by line range out of transformers v5.16.0 and EXECUTES them under
//      torch. The CPU arms are held to the same file, so the two arms answer to
//      ONE oracle instead of to each other — the arrangement
//      `test_qwen4_exp_ple_device.cpp` already states for the host reference.
//   2. THE CPU ARM. Every op is additionally compared against its own CPU arm on
//      identical inputs. This is what catches a defect the goldens' 16 channels
//      and 12 tokens cannot see, and it is the only gate the write-back has an
//      exact form of — see the transitivity note below.
//
// A DEVICE ARM GATED ONLY AGAINST ITSELF WOULD BE WORTHLESS, which is why
// neither comparison stands alone here.
//
// ─── THE TOLERANCES, AND WHY EACH ONE IS THE VALUE IT IS ─────────────────────
// These are not "close enough" numbers. Each is derived from what the two arms
// can differ by, and two of the three are ZERO.
//
// `vt::Qwen4ExpGatedResidualWriteBack` is BYTE-IDENTICAL and is gated by
// `std::memcmp`, not by a tolerance. The host provider is pinned to
// `-ffp-contract=off` (CMakeLists.txt:41-56) and the CUDA arm spells the same
// two roundings with `__fmul_rn`/`__fadd_rn`, so every bit of every output must
// agree. A tolerance would be the wrong instrument: a transposed `injection`
// axis or a dropped `hc` stride lands well inside any epsilon anyone would
// write. This is the position `tests/vt/test_ops_conv1d_general.cpp` takes for
// the same reason.
//
// **AND THAT MAKES THE WRITE-BACK'S ORACLE GATE TRANSITIVE, EXACTLY.** The CPU
// arm is held to the transformers goldens by `test_qwen4_exp_hc_device.cpp`;
// the CUDA arm is held to the CPU arm by bitwise equality; equality composes, so
// the CUDA arm meets the oracle by the same margin the CPU arm does. That
// argument is only available BECAUSE the relation is equality — it would be
// invalid for a tolerance, and it is stated rather than assumed.
//
// The conv and the gate are held to `kUlpTol`, ONE f32 ulp relative
// (2^-23 ~ 1.19e-7) plus a 1e-30 absolute floor for the denormal neighbourhood.
// The single divergence source between the two arms is `exp()`: both evaluate it
// in DOUBLE, and CUDA's libdevice `exp` is not required to return the same double
// as glibc's. Every other operation on both paths is IEEE-exact or spelled with
// an `_rn` intrinsic. A sub-ulp double difference is below the f32 store's own
// rounding for all but an exact tie, so these arms are EXPECTED to be
// byte-identical and are deliberately NOT asserted to be: the cases below MEASURE
// the difference and report whether it is exactly zero, so a future toolkit that
// moves `exp` is visible as a number rather than as a red gate nobody can read.
//
// NEVER WIDENED. `kUlpTol` is one ulp because one ulp is what the analysis above
// permits, and the defects it has to separate are orders of magnitude away: the
// conv's three oracle dilations are 0.44 to 0.72 apart (`kConvDilationsSeparate`
// in the CPU suite), and the gate's clamp separation is 1.56e-3
// (`kGateClampSeparation`). If a case here needs a looser bound to pass, the
// kernel is wrong and the bound is not the thing to change.
//
// ─── SCOPE, HONESTLY ─────────────────────────────────────────────────────────
// THREE OF THE SIX `qwen4_exp` OPS HAVE CUDA ARMS. `vt::Qwen4ExpGatedResidual`,
// `vt::RmsNormGroup`, `vt::Qwen4ExpQsaCompress` and
// `vt::Qwen4ExpQsaGatherAttention` do not, and neither does the block-decoding
// n-gram gather `vt::Embedding` needs (`EmbeddingKernelCuda` refuses a
// block-quantized table by name). So `ModelRegistry::Forward` still cannot run
// this architecture on a CUDA queue, NOTHING IN PRODUCTION REACHES THESE THREE
// KERNELS, and their reachability from a production entry point is VACUOUS
// rather than proven. The spec's `## Owed` names each missing arm and the row
// that owns the wiring. No token claim and no speed claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpGatedResidualArgs;
using vt::Qwen4ExpPleConvArgs;
using vt::Qwen4ExpPleGateArgs;
using vt::Queue;
using vt::Tensor;

namespace {

#include "qwen4_exp_ple_goldens.inc"  // NOLINT — golden literals

// ONE f32 ulp relative, with an absolute floor for the denormal neighbourhood.
// Justified in the header; never widen it.
constexpr double kUlpTol = 1.20e-7;
constexpr double kAbsFloor = 1e-30;

constexpr int64_t kChannels = 16;  // kConvChannels
constexpr int64_t kKernel = 4;     // kConvKernel
constexpr int64_t kT = 6;          // kGateT
constexpr int64_t kHc = 2;         // kGateHc
constexpr int64_t kH = 8;          // kGateH

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// LOUD, because a silent skip on a CPU box is how a device arm goes un-gated for
// a release — the position `tests/vt/test_ops_conv1d_general.cpp` states.
bool SkipNoCuda(const char* what) {
  if (HasCuda()) return false;
  std::printf("[SKIP] no CUDA backend: %s NOT exercised\n", what);
  return true;
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// max|a-b| with the NaN-blindness of `std::max`/`>` removed, and the count of
// elements that are not BITWISE equal. Both are reported, because "within one
// ulp" and "identical" are different findings and this file claims the second
// wherever it can get it.
struct Agreement {
  double worst = 0.0;
  size_t not_bitwise_equal = 0;
};

Agreement Compare(const std::vector<float>& got, const std::vector<float>& want) {
  REQUIRE(got.size() == want.size());
  Agreement a;
  // MaxAbsDiff returns +infinity on ANY non-finite operand and raises its own
  // doctest failure, so an all-NaN device output cannot read as a perfect match
  // here. That defect is issue #449 and this file must not re-introduce it by
  // hand-rolling a reduction.
  a.worst = MaxAbsDiff(got, want);
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) ++a.not_bitwise_equal;
  }
  return a;
}

// BOTH REPORTERS PRINT ON SUCCESS, not only on failure. doctest's INFO/CAPTURE
// are emitted only when an assertion fails, so a passing run of a numeric gate
// says nothing about HOW closely it passed — and "max|diff| = 0" versus
// "max|diff| = one ulp" is the difference between this file's byte-identity
// claim and its within-an-ulp one. The numbers below are the wave's evidence and
// have to survive a green run.
void CheckWithinUlp(const std::vector<float>& got, const std::vector<float>& want,
                    const char* what) {
  const Agreement a = Compare(got, want);
  double scale = 0.0;
  for (float v : want) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  const double bound = kAbsFloor + kUlpTol * scale;
  std::printf("[MEASURED] %-42s max|diff| = %.9g  bound = %.9g  not-bitwise-equal = %zu/%zu\n",
              what, a.worst, bound, a.not_bitwise_equal, want.size());
  INFO(what << ": max|diff| = " << a.worst << " vs bound " << bound << "; "
            << a.not_bitwise_equal << " of " << want.size()
            << " elements not bitwise equal (0 means byte-identical)");
  CHECK(a.worst <= bound);
}

void CheckBitwise(const std::vector<float>& got, const std::vector<float>& want,
                  const char* what) {
  const Agreement a = Compare(got, want);
  std::printf("[MEASURED] %-42s BYTE gate: %zu/%zu differ, max|diff| = %.9g\n", what,
              a.not_bitwise_equal, want.size(), a.worst);
  INFO(what << ": " << a.not_bitwise_equal << " of " << want.size()
            << " elements differ; max|diff| = " << a.worst);
  CHECK(a.not_bitwise_equal == 0);
}

// ── the conv, on either device ───────────────────────────────────────────────
struct ConvResult {
  std::vector<float> out;
  std::vector<float> state;
};

ConvResult RunConv(DeviceType dt, int64_t dilation, const std::vector<float>& x_in,
                   int64_t tokens, const std::vector<float>& w_in,
                   const std::vector<float>& state_in) {
  const int64_t state_len = (kKernel - 1) * dilation;
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(tokens)};
  Qwen4ExpPleConvArgs args;
  args.dilation = dilation;
  ConvResult r;
  r.out.assign(static_cast<size_t>(tokens * kChannels), 0.0f);
  r.state = state_in;
  if (dt == DeviceType::kCPU) {
    Queue q = CpuQ();
    std::vector<float> x = x_in;
    std::vector<float> w = w_in;
    Tensor t_x = MakeTensor(x.data(), DType::kF32, Cpu(), {tokens, kChannels});
    Tensor t_w = MakeTensor(w.data(), DType::kF32, Cpu(), {kChannels, kKernel});
    Tensor t_s = MakeTensor(r.state.data(), DType::kF32, Cpu(), {1, kChannels, state_len});
    Tensor t_o = MakeTensor(r.out.data(), DType::kF32, Cpu(), {tokens, kChannels});
    Tensor t_q = MakeTensor(qsl.data(), DType::kI32, Cpu(), {2});
    vt::Qwen4ExpPleConv(q, t_o, t_x, t_w, t_s, t_q, nullptr, args);
    return r;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_x(b, qg.q, DType::kF32, {tokens, kChannels}, x_in.data());
  DeviceTensor d_w(b, qg.q, DType::kF32, {kChannels, kKernel}, w_in.data());
  DeviceTensor d_s(b, qg.q, DType::kF32, {1, kChannels, state_len}, state_in.data());
  DeviceTensor d_o(b, qg.q, DType::kF32, {tokens, kChannels});
  DeviceTensor d_q(b, qg.q, DType::kI32, {2}, qsl.data());
  vt::Qwen4ExpPleConv(qg.q, d_o.tensor(), d_x.tensor(), d_w.tensor(), d_s.tensor(),
                      d_q.tensor(), nullptr, args);
  b.Synchronize(qg.q);
  d_o.Download(qg.q, r.out.data());
  d_s.Download(qg.q, r.state.data());
  return r;
}

// ── the gate, on either device ───────────────────────────────────────────────
std::vector<float> RunGate(DeviceType dt, const std::vector<float>& score,
                           const std::vector<float>& value, int64_t tokens, int64_t hc,
                           int64_t hidden, float divisor, float clamp_min = 1e-6f) {
  Qwen4ExpPleGateArgs args;
  args.gate_divisor = divisor;
  args.clamp_min = clamp_min;
  std::vector<float> out(static_cast<size_t>(tokens * hc * hidden), 0.0f);
  if (dt == DeviceType::kCPU) {
    Queue q = CpuQ();
    std::vector<float> s = score;
    std::vector<float> v = value;
    Tensor t_s = MakeTensor(s.data(), DType::kF32, Cpu(), {tokens, hc});
    Tensor t_v = MakeTensor(v.data(), DType::kF32, Cpu(), {tokens, hidden});
    Tensor t_o = MakeTensor(out.data(), DType::kF32, Cpu(), {tokens, hc * hidden});
    vt::Qwen4ExpPleGate(q, t_o, t_s, t_v, args);
    return out;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_s(b, qg.q, DType::kF32, {tokens, hc}, score.data());
  DeviceTensor d_v(b, qg.q, DType::kF32, {tokens, hidden}, value.data());
  DeviceTensor d_o(b, qg.q, DType::kF32, {tokens, hc * hidden});
  vt::Qwen4ExpPleGate(qg.q, d_o.tensor(), d_s.tensor(), d_v.tensor(), args);
  b.Synchronize(qg.q);
  d_o.Download(qg.q, out.data());
  return out;
}

// ── the write-back, on either device ─────────────────────────────────────────
std::vector<float> RunWriteBack(DeviceType dt, const std::vector<float>& hyper_in,
                                const std::vector<float>& block_out,
                                const std::vector<float>& injection, int64_t T, int64_t hc,
                                int64_t hidden) {
  Qwen4ExpGatedResidualArgs args;
  args.hc_count = hc;
  args.hidden_size = hidden;
  // lowrank and eps are INERT at this op — the CPU arm reads neither and neither
  // does the CUDA arm. They are set to values that would be visibly wrong if
  // either kernel ever started reading them.
  args.lowrank = 7;
  args.eps = 0.5f;
  std::vector<float> hyper = hyper_in;
  const int64_t flat = hc * hidden;
  if (dt == DeviceType::kCPU) {
    Queue q = CpuQ();
    std::vector<float> bo = block_out;
    std::vector<float> inj = injection;
    Tensor t_h = MakeTensor(hyper.data(), DType::kF32, Cpu(), {T, flat});
    Tensor t_b = MakeTensor(bo.data(), DType::kF32, Cpu(), {T, hidden});
    Tensor t_i = MakeTensor(inj.data(), DType::kF32, Cpu(), {T, hc});
    vt::Qwen4ExpGatedResidualWriteBack(q, t_h, t_b, t_i, args);
    return hyper;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor d_h(b, qg.q, DType::kF32, {T, flat}, hyper_in.data());
  DeviceTensor d_b(b, qg.q, DType::kF32, {T, hidden}, block_out.data());
  DeviceTensor d_i(b, qg.q, DType::kF32, {T, hc}, injection.data());
  vt::Qwen4ExpGatedResidualWriteBack(qg.q, d_h.tensor(), d_b.tensor(), d_i.tensor(), args);
  b.Synchronize(qg.q);
  d_h.Download(qg.q, hyper.data());
  return hyper;
}

const float* ConvExpectedFor(int64_t dilation) {
  switch (dilation) {
    case 1: return &kConvExpectedD1[0];
    case 2: return &kConvExpectedD2[0];
    default: return &kConvExpectedD3[0];
  }
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpPleConv
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("CUDA vt::Qwen4ExpPleConv reproduces the pinned transformers oracle") {
  if (SkipNoCuda("vt::Qwen4ExpPleConv CUDA arm vs the transformers oracle")) return;
  const std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  // ALL THREE DILATIONS. The dilation is the one field of this op's args, and a
  // kernel that ignored it computes an ordinary K=4 causal conv over the same
  // state width and returns a plausible tensor. The oracle supplies both sides
  // of the variable: upstream's own `_short_conv` was run at 1, 2 and 3 over the
  // SAME input and the SAME weight.
  for (int i = 0; i < 3; ++i) {
    const int64_t dilation = kConvDilations[i];
    CAPTURE(dilation);
    const int64_t state_len = (kKernel - 1) * dilation;
    const std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
    const std::vector<float> zero(static_cast<size_t>(kChannels * state_len), 0.0f);
    const ConvResult got = RunConv(DeviceType::kCUDA, dilation, x, kConvSeqLen, w, zero);
    const std::vector<float> want(ConvExpectedFor(dilation),
                                  ConvExpectedFor(dilation) + kConvSeqLen * kChannels);
    CheckWithinUlp(got.out, want, "conv vs oracle");
  }
}

TEST_CASE("CUDA vt::Qwen4ExpPleConv separates the three oracle dilations") {
  if (SkipNoCuda("vt::Qwen4ExpPleConv CUDA dilation separation")) return;
  // A gate that only checked dilation 3 could not tell a dilation-reading kernel
  // from one that hard-codes a lag set. This case asserts that the CUDA arm's
  // OWN three answers differ from each other by far more than the tolerance, so
  // the case above is measuring the field and not a coincidence.
  const std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  const std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
  std::vector<std::vector<float>> answers;
  for (int i = 0; i < 3; ++i) {
    const int64_t dilation = kConvDilations[i];
    const std::vector<float> zero(
        static_cast<size_t>(kChannels * (kKernel - 1) * dilation), 0.0f);
    answers.push_back(RunConv(DeviceType::kCUDA, dilation, x, kConvSeqLen, w, zero).out);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a + 1; b < 3; ++b) {
      const double sep = MaxAbsDiff(answers[static_cast<size_t>(a)],
                                    answers[static_cast<size_t>(b)]);
      INFO("dilation " << kConvDilations[a] << " vs " << kConvDilations[b]
                       << " separation " << sep);
      CHECK(sep > 1.0e-2);
    }
  }
}

TEST_CASE("CUDA vt::Qwen4ExpPleConv agrees with the CPU arm, ring and all") {
  if (SkipNoCuda("vt::Qwen4ExpPleConv CPU-vs-CUDA agreement")) return;
  constexpr int64_t kDil = 3;
  constexpr int64_t kStateLen = (kKernel - 1) * kDil;
  // A NON-ZERO incoming ring, which the oracle cases above never exercise: they
  // all start from a zeroed cache, so the `[old state | this chunk]` join and the
  // ring write-back are only half-gated by them. `tokens < state_len` is included
  // deliberately — that is the case where the write-back keeps the tail of the
  // OLD state ahead of the new chunk, and where a read-after-write hazard in the
  // device kernel's in-place ring update would show.
  for (int64_t tokens : {int64_t{1}, int64_t{4}, int64_t{9}, int64_t{12}}) {
    CAPTURE(tokens);
    const std::vector<float> x = RandomF32(static_cast<size_t>(tokens * kChannels), 11u);
    const std::vector<float> w = RandomF32(static_cast<size_t>(kChannels * kKernel), 22u);
    const std::vector<float> st =
        RandomF32(static_cast<size_t>(kChannels * kStateLen), 33u);
    const ConvResult cpu = RunConv(DeviceType::kCPU, kDil, x, tokens, w, st);
    const ConvResult gpu = RunConv(DeviceType::kCUDA, kDil, x, tokens, w, st);
    CheckWithinUlp(gpu.out, cpu.out, "conv output");
    // THE RING IS GATED SEPARATELY FROM THE OUTPUT. It is persistent state: a
    // kernel that computed every output correctly and left the cache unshifted
    // would pass a value-only comparison and then be wrong on the NEXT step.
    CheckBitwise(gpu.state, cpu.state, "conv ring write-back");
  }
}

TEST_CASE("CUDA vt::Qwen4ExpPleConv keeps the double tap accumulator") {
  if (SkipNoCuda("vt::Qwen4ExpPleConv accumulator width")) return;
  // WELL-SCALED DATA CANNOT SEE THIS, and neither can magnitude-separated data
  // laid out carelessly. The first draft of this case put `big + c` in one input
  // element: at big = 2^40 an f32 ulp is 2^17, so `c` was lost ON THE STORE and
  // both arms agreed trivially. Adjacent `+big, -big` does not work either,
  // because a SEQUENTIAL f32 accumulator cancels them before the small term
  // arrives and keeps it exactly.
  //
  // What separates the two widths is the small term arriving FIRST, ahead of a
  // cancelling pair. Tap k reads `hist[t + k*dilation]`, so at dilation 3 the
  // four taps of output 0 are hist[0], hist[3], hist[6], hist[9]. Set those to
  // 1.0, 2^40, -2^40, 0 and the two accumulators disagree completely:
  //
  //   double: ((0 + 1) + 2^40) - 2^40 + 0  ==  1.0        -> silu(1) = 0.7310586
  //   float:  1 + 2^40 rounds to 2^40, so  ==  0.0        -> silu(0) = 0.0
  //
  // The case is therefore held to the DOUBLE answer directly rather than only to
  // the CPU arm, so it states what it is measuring and cannot be satisfied by
  // two arms being wrong together.
  constexpr int64_t kDil = 3;
  constexpr int64_t kStateLen = (kKernel - 1) * kDil;  // 9
  constexpr int64_t kTokens = 4;
  const float big = 1099511627776.0f;  // 2^40, exactly representable in f32
  std::vector<float> x(static_cast<size_t>(kTokens * kChannels), 0.0f);
  std::vector<float> w(static_cast<size_t>(kChannels * kKernel), 1.0f);
  std::vector<float> st(static_cast<size_t>(kChannels * kStateLen), 0.0f);
  for (int64_t c = 0; c < kChannels; ++c) {
    st[static_cast<size_t>(c * kStateLen + 0)] = 1.0f;
    st[static_cast<size_t>(c * kStateLen + 3)] = big;
    st[static_cast<size_t>(c * kStateLen + 6)] = -big;
  }
  const ConvResult cpu = RunConv(DeviceType::kCPU, kDil, x, kTokens, w, st);
  const ConvResult gpu = RunConv(DeviceType::kCUDA, kDil, x, kTokens, w, st);
  // silu(1.0) in double, which is what a double accumulator must produce.
  const double want = 1.0 * (1.0 / (1.0 + std::exp(-1.0)));
  INFO("a float accumulator gives silu(0) = 0.0 here; a double one gives " << want);
  for (int64_t c = 0; c < kChannels; ++c) {
    const double got_gpu = static_cast<double>(gpu.out[static_cast<size_t>(c)]);
    const double got_cpu = static_cast<double>(cpu.out[static_cast<size_t>(c)]);
    CAPTURE(c);
    CHECK(std::fabs(got_cpu - want) < 1.0e-6);
    CHECK(std::fabs(got_gpu - want) < 1.0e-6);
  }
  // And the two arms still agree with each other everywhere else in the block.
  CheckWithinUlp(gpu.out, cpu.out, "conv under catastrophic cancellation");
  CheckBitwise(gpu.state, cpu.state, "conv ring under catastrophic cancellation");
}

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpPleGate
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("CUDA vt::Qwen4ExpPleGate reproduces the pinned transformers oracle") {
  if (SkipNoCuda("vt::Qwen4ExpPleGate CUDA arm vs the transformers oracle")) return;
  REQUIRE(kGateT == kT);
  REQUIRE(kGateHc == kHc);
  REQUIRE(kGateH == kH);
  // The scores are upstream's OWN :1180 output, so `gate_divisor` is the identity
  // here and the op is held to :1181-1182 alone.
  const std::vector<float> score(kGateScaledDot, kGateScaledDot + kT * kHc);
  const std::vector<float> value(kGateValueIn, kGateValueIn + kT * kH);
  const std::vector<float> got = RunGate(DeviceType::kCUDA, score, value, kT, kHc, kH, 1.0f);
  const std::vector<float> want(kGateExpectedOut, kGateExpectedOut + kT * kHc * kH);
  CheckWithinUlp(got, want, "gate vs oracle");
}

TEST_CASE("CUDA vt::Qwen4ExpPleGate reproduces the oracle's signed-sqrt table") {
  if (SkipNoCuda("vt::Qwen4ExpPleGate CUDA signed-sqrt arms")) return;
  // `kGateInput`/`kGateExpected` are upstream's own
  // `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` over eleven points chosen
  // to walk every arm of it: the ORIGIN (sign 0, where the 1e-3 floor is
  // cancelled), both sides of the clamp, and both signs well above it. A kernel
  // that reordered the clamp past the sqrt, or that returned -1e-3 at zero,
  // fails here and passes every shape check.
  constexpr int64_t kN = 11;
  const std::vector<float> score(kGateInput, kGateInput + kN);
  const std::vector<float> value(static_cast<size_t>(kN), 1.0f);
  // score [kN, 1] against value [kN, 1]: the gate weight is sigmoid(signed_sqrt)
  // and value 1.0, so the output IS the sigmoid of the table. Invert it to
  // recover the pre-sigmoid value the golden states.
  const std::vector<float> got = RunGate(DeviceType::kCUDA, score, value, kN, 1, 1, 1.0f);
  std::vector<float> recovered(static_cast<size_t>(kN));
  for (int64_t i = 0; i < kN; ++i) {
    const double s = static_cast<double>(got[static_cast<size_t>(i)]);
    REQUIRE(s > 0.0);
    REQUIRE(s < 1.0);
    recovered[static_cast<size_t>(i)] = static_cast<float>(std::log(s / (1.0 - s)));
  }
  const std::vector<float> want(kGateExpected, kGateExpected + kN);
  // The logit inversion is not exact in f32, so this case is held to a bound
  // derived from that inversion rather than to kUlpTol. It is still three orders
  // of magnitude below the 1.56e-3 clamp separation it has to see.
  INFO("signed-sqrt table recovered through the sigmoid inverse");
  CHECK(MaxAbsDiff(recovered, want) < 1.0e-5);
}

TEST_CASE("CUDA vt::Qwen4ExpPleGate propagates NaN as upstream does") {
  if (SkipNoCuda("vt::Qwen4ExpPleGate CUDA NaN arm")) return;
  // THE INHERITED OBLIGATION. `include/vt/ops.h` states that a CUDA arm of this
  // op "inherits the NaN obligation above and owes its own case for it". Upstream
  // propagates a NaN gate to a NaN output: `torch.sign(NaN) == 0` but
  // `NaN * 0.0 == NaN`. Without the explicit NaN guard the fall-through returns
  // 0.0, which sigmoids to a perfectly plausible `0.5 * value` — a poison value
  // rendered as a number, and one no tolerance can catch. This case asserts the
  // NaN ARRIVES, which is why it cannot go through MaxAbsDiff.
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> score = {nan_v, 1.0f};
  const std::vector<float> value = {2.0f, 3.0f};
  const std::vector<float> got = RunGate(DeviceType::kCUDA, score, value, 2, 1, 1, 1.0f);
  INFO("got[0] = " << got[0] << " (must be NaN, NOT 0.5 * value = 1.0)");
  CHECK(std::isnan(got[0]));
  CHECK_FALSE(std::isnan(got[1]));
  // The 0.5*value the missing guard would produce, named so a reader can see
  // what this case separates.
  CHECK(got[0] != doctest::Approx(1.0));
}

TEST_CASE("CUDA vt::Qwen4ExpPleGate agrees with the CPU arm at width") {
  if (SkipNoCuda("vt::Qwen4ExpPleGate CPU-vs-CUDA agreement")) return;
  // Wider than the goldens' 6x2x8, and with a real divisor, so the divide and
  // the [T, hc] x [T, H] broadcast are both exercised outside the fixture.
  constexpr int64_t kTt = 17;
  constexpr int64_t kHcc = 4;
  constexpr int64_t kHh = 129;  // deliberately not a multiple of the block size
  const std::vector<float> score =
      RandomF32(static_cast<size_t>(kTt * kHcc), 101u, -6.0f, 6.0f);
  const std::vector<float> value = RandomF32(static_cast<size_t>(kTt * kHh), 202u);
  const std::vector<float> cpu =
      RunGate(DeviceType::kCPU, score, value, kTt, kHcc, kHh, kGateDivisor);
  const std::vector<float> gpu =
      RunGate(DeviceType::kCUDA, score, value, kTt, kHcc, kHh, kGateDivisor);
  CheckWithinUlp(gpu, cpu, "gate at width");
}

// ═════════════════════════════════════════════════════════════════════════════
// vt::Qwen4ExpGatedResidualWriteBack
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("CUDA vt::Qwen4ExpGatedResidualWriteBack is BYTE-IDENTICAL to the CPU arm") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidualWriteBack CPU-vs-CUDA byte equality")) return;
  // Not a tolerance. See the file header: both arms perform the SAME two
  // roundings (`-ffp-contract=off` on the host, `__fmul_rn`/`__fadd_rn` on the
  // device), so equality is the correct relation, and it is what makes the CPU
  // arm's oracle gate transitive to this one.
  struct Shape {
    int64_t T, hc, hidden;
  };
  // The last is the released model's real hyper-connection geometry.
  const Shape shapes[] = {{1, 2, 1}, {3, 4, 8}, {7, 3, 129}, {2, 4, 2560}};
  for (const Shape& s : shapes) {
    CAPTURE(s.T);
    CAPTURE(s.hc);
    CAPTURE(s.hidden);
    const std::vector<float> hyper =
        RandomF32(static_cast<size_t>(s.T * s.hc * s.hidden), 7u);
    const std::vector<float> block = RandomF32(static_cast<size_t>(s.T * s.hidden), 8u);
    const std::vector<float> inj = RandomF32(static_cast<size_t>(s.T * s.hc), 9u);
    const std::vector<float> cpu =
        RunWriteBack(DeviceType::kCPU, hyper, block, inj, s.T, s.hc, s.hidden);
    const std::vector<float> gpu =
        RunWriteBack(DeviceType::kCUDA, hyper, block, inj, s.T, s.hc, s.hidden);
    CheckBitwise(gpu, cpu, "write-back");
  }
}

TEST_CASE("CUDA vt::Qwen4ExpGatedResidualWriteBack puts each hc stream in its own slot") {
  if (SkipNoCuda("vt::Qwen4ExpGatedResidualWriteBack CUDA stride")) return;
  // A value comparison against the CPU arm cannot see a TRANSPOSED hc/hidden
  // decomposition when the two extents are equal, and a random fixture with
  // hc == hidden is exactly the shape a careless index would survive. This case
  // makes the addressing structural: every injection weight is distinct and
  // every block output is 1.0, so hyper[t, j*H + h] must land on injection[t, j]
  // and on nothing else.
  constexpr int64_t T = 3, hc = 4, H = 4;  // hc == H on purpose
  const std::vector<float> hyper(static_cast<size_t>(T * hc * H), 0.0f);
  const std::vector<float> block(static_cast<size_t>(T * H), 1.0f);
  std::vector<float> inj(static_cast<size_t>(T * hc));
  for (size_t i = 0; i < inj.size(); ++i) inj[i] = static_cast<float>(i + 1);
  const std::vector<float> gpu = RunWriteBack(DeviceType::kCUDA, hyper, block, inj, T, hc, H);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      for (int64_t h = 0; h < H; ++h) {
        const float want = inj[static_cast<size_t>(t * hc + j)];
        const float got = gpu[static_cast<size_t>(t * hc * H + j * H + h)];
        INFO("t=" << t << " j=" << j << " h=" << h);
        CHECK(got == doctest::Approx(want));
      }
    }
  }
}

TEST_CASE("the qwen4_exp CUDA arms are registered for kCUDA and refuse BY NAME elsewhere") {
  // Runs WITHOUT A GPU, and that is the point: registration happens in the .cu
  // translation units' static initialisers, which exist in any CUDA-ENABLED
  // BUILD whether or not a device is present. `GetOp` is the dispatcher path
  // every production caller reaches these kernels through, so this case is what
  // says the two `RegisterOp` lines are live rather than vestigial.
  //
  // The guard is on the BUILD, not on the device. Without -DVLLM_CPP_CUDA the
  // .cu files are not compiled at all and there is nothing to assert; asserting
  // anyway would red every CPU-only gate in the tree.
  //
  // GetOp THROWS on an unregistered (op, device) rather than returning null, and
  // the portable CPU reference tier CANNOT rescue a missing CUDA arm: it is
  // gated on `Backend::DeviceMemoryIsHostAddressable()`, which CudaBackend
  // leaves at the base `false` because CUDA on GB10 allocates with `cudaMalloc`
  // and the host may not dereference it (#844, #1435). So the refusals below are
  // real refusals and not a fallback in disguise.
#ifndef VLLM_CPP_CUDA
  std::printf("[SKIP] built without VLLM_CPP_CUDA: qwen4_exp CUDA registrations NOT asserted\n");
#else
  CHECK(vt::GetOp(vt::OpId::kQwen4ExpPleConv, DeviceType::kCUDA) != nullptr);
  CHECK(vt::GetOp(vt::OpId::kQwen4ExpPleGate, DeviceType::kCUDA) != nullptr);
  CHECK(vt::GetOp(vt::OpId::kQwen4ExpGatedResidualWriteBack, DeviceType::kCUDA) != nullptr);
  // The three ops this wave did NOT give a CUDA arm. They must still refuse, by
  // name, rather than silently falling back to a CPU kernel that would then
  // dereference device pointers. If a later wave registers one, this case fails
  // and the reader is sent to the spec's `## Owed` to strike the entry.
  CHECK_THROWS(vt::GetOp(vt::OpId::kQwen4ExpGatedResidual, DeviceType::kCUDA));
  CHECK_THROWS(vt::GetOp(vt::OpId::kQwen4ExpQsaCompress, DeviceType::kCUDA));
  CHECK_THROWS(vt::GetOp(vt::OpId::kQwen4ExpQsaGatherAttention, DeviceType::kCUDA));
  CHECK_THROWS(vt::GetOp(vt::OpId::kRmsNormGroup, DeviceType::kCUDA));
#endif
}
