// The gamma's dtype is INDEPENDENT of the activation's, and the pairing that
// PRODUCTION reaches is f32 activation against a bf16 gamma (#2477).
//
// vllm.cpp original (vt runtime).
//
// WHICH PAIRING IS REAL, and an earlier revision of this file got it wrong.
// The released `unsloth/Qwen3.8-Flash-Next-GGUF` stores its norm gammas as F32
// ON DISK, but the loader does not hand that to the kernel:
// `qwen4_exp_weights.cpp`'s `LoadNormBf16` runs `DequantAll` into an f32 vector
// and then `Bf16From`, which is `MakeTensor(kBF16, ...)`
// (`glm_moe_dsa_loader.cpp:145-151`). All four QSA gammas take that path, so at
// RUNTIME every gamma is bf16. On-disk dtype is not runtime dtype.
//
// The activation is what diverges. `qwen4_exp_qsa_block.cpp:694` allocates
// `q_f32` at `DType::kF32` and `vt::AttnGateSplit` fills it, where vLLM chunks
// `q` straight off the bf16 QKV GEMM and hands it to `q_norm` unwidened
// (`vllm/model_executor/models/qwen3_next.py:384-440`). So the refusing site is
// `qwen4_exp_qsa_block.cpp:705` -- f32 activation, bf16 gamma, `Dh = 256` -- and
// the sites at `:632` and `:736` pair bf16 with bf16 and always passed.
//
// RED BEFORE GREEN, stated exactly as it was taken: the pre-fix dispatcher
// refused ANY `w.dtype != x.dtype`, so case 1 below could not have run at all.
// That refusal was observed as a server 500 from `#2477`, NOT captured by
// running this file against a pre-fix binary. The transition is therefore
// evidence that the pairing is now accepted, and not evidence about which
// pairing was at fault; case 1 exists to pin the one production actually uses.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

uint16_t F32ToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}
float Bf16ToF32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

constexpr int64_t kRows = 5;    // the CPU control's prompt_tokens
constexpr int64_t kProdH = 256; // qwen4_exp head_dim: attn_q_norm.weight is [256]
constexpr int64_t kIdxH = 128;  // indexer head_dim, for the bf16/f32 blast-radius case

// Widen both operands to f32, multiply, round once at the store. This is
// GemmaRMSNorm's own order and is recomputed here rather than taken from the op,
// because a gate that compares the op against itself proves consistency and not
// correctness.
std::vector<float> Reference(const std::vector<float>& x, const std::vector<float>& w,
                             int64_t rows, int64_t h, float eps) {
  std::vector<float> out(static_cast<size_t>(rows * h));
  for (int64_t r = 0; r < rows; ++r) {
    double sumsq = 0.0;
    for (int64_t j = 0; j < h; ++j) {
      const float v = x[static_cast<size_t>(r * h + j)];
      sumsq += static_cast<double>(v) * static_cast<double>(v);
    }
    const float inv =
        1.0f / std::sqrt(static_cast<float>(sumsq / static_cast<double>(h)) + eps);
    for (int64_t j = 0; j < h; ++j)
      out[static_cast<size_t>(r * h + j)] =
          x[static_cast<size_t>(r * h + j)] * inv * (1.0f + w[static_cast<size_t>(j)]);
  }
  return out;
}

std::vector<float> SpreadX(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.37f) * (1.0f + 0.01f * static_cast<float>(i % 17));
  return v;
}
std::vector<float> SpreadW(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j)
    v[static_cast<size_t>(j)] = 0.05f * std::cos(static_cast<float>(j) * 0.11f);
  return v;
}

// ─── CASE 1 SHAPE: f32 activation, bf16 gamma, bf16 out. THE PRODUCTION ONE.
// Instantiates RmsNormRowKernel<float, __nv_bfloat16, __nv_bfloat16, float>.
std::vector<uint16_t> RunF32ActBf16W(Device dev, const std::vector<float>& x,
                                     const std::vector<uint16_t>& w, int64_t h) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * h), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<float*>(x.data()), DType::kF32, dev, {kRows, h});
    Tensor tw = Tensor::Contiguous(const_cast<uint16_t*>(w.data()), DType::kBF16, dev, {h});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, h});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(float));
  void* wd = b.Alloc(w.size() * sizeof(uint16_t));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(float));
  b.Copy(q, wd, w.data(), w.size() * sizeof(uint16_t));
  Tensor tx = Tensor::Contiguous(xd, DType::kF32, dev, {kRows, h});
  Tensor tw = Tensor::Contiguous(wd, DType::kBF16, dev, {h});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, h});
  vt::RmsNorm(q, to, tx, tw, args);
  b.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
  return out;
}

// ─── CASE 2 SHAPE: bf16 activation, f32 gamma. NOT reached by qwen4_exp, but
// newly ADMITTED by this change and reachable from the GGUF arms of glm4,
// gemma/gemma2/gemma3 and deepseek_v2 whose gammas stay F32 (#2503).
std::vector<uint16_t> RunBf16ActF32W(Device dev, const std::vector<uint16_t>& x,
                                     const std::vector<float>& w, int64_t h) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * h), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, dev, {kRows, h});
    Tensor tw = Tensor::Contiguous(const_cast<float*>(w.data()), DType::kF32, dev, {h});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, h});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(uint16_t));
  void* wd = b.Alloc(w.size() * sizeof(float));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(uint16_t));
  b.Copy(q, wd, w.data(), w.size() * sizeof(float));
  Tensor tx = Tensor::Contiguous(xd, DType::kBF16, dev, {kRows, h});
  Tensor tw = Tensor::Contiguous(wd, DType::kF32, dev, {h});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, h});
  vt::RmsNorm(q, to, tx, tw, args);
  b.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
  return out;
}

size_t BadCount(const std::vector<uint16_t>& got, const std::vector<float>& want) {
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - want[i]) > 0.01f * (std::fabs(want[i]) + 1e-3f)) ++bad;
  return bad;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// CASE 1. The pairing qwen4_exp_qsa_block.cpp:705 actually issues.
TEST_CASE("rmsnorm: f32 activation against a bf16 gamma is the production pairing") {
  const std::vector<float> x = SpreadX(kRows * kProdH);
  const std::vector<float> wf = SpreadW(kProdH);
  std::vector<uint16_t> wb(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wb[static_cast<size_t>(j)] = F32ToBf16(wf[static_cast<size_t>(j)]);
  // The reference must use the ROUNDED gamma, since that is what the op reads.
  std::vector<float> wr(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wr[static_cast<size_t>(j)] = Bf16ToF32(wb[static_cast<size_t>(j)]);

  const std::vector<uint16_t> got = RunF32ActBf16W(Cpu(), x, wb, kProdH);
  CHECK(BadCount(got, Reference(x, wr, kRows, kProdH, 1e-6f)) == 0);

  // The gamma must be LOAD-BEARING or the tolerance above is vacuous.
  std::vector<uint16_t> bumped = wb;
  bumped[7] = F32ToBf16(wr[7] + 4.0f);
  CHECK(got[7] != RunF32ActBf16W(Cpu(), x, bumped, kProdH)[7]);
}

TEST_CASE("rmsnorm: CUDA runs the production f32-activation/bf16-gamma pairing") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: production pairing device arm NOT exercised\n");
    return;
  }
  const std::vector<float> x = SpreadX(kRows * kProdH);
  const std::vector<float> wf = SpreadW(kProdH);
  std::vector<uint16_t> wb(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wb[static_cast<size_t>(j)] = F32ToBf16(wf[static_cast<size_t>(j)]);
  const std::vector<uint16_t> want = RunF32ActBf16W(Cpu(), x, wb, kProdH);
  const std::vector<uint16_t> got = RunF32ActBf16W(Device{DeviceType::kCUDA, 0}, x, wb, kProdH);
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - Bf16ToF32(want[i])) >
        0.01f * (std::fabs(Bf16ToF32(want[i])) + 1e-3f)) ++bad;
  CHECK(bad == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 2. The pairing this change newly ADMITS. Not qwen4_exp's, but reachable.
TEST_CASE("rmsnorm: bf16 activation against an f32 gamma, the newly admitted pairing") {
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  std::vector<float> xr(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xr[i] = Bf16ToF32(xb[i]);
  const std::vector<float> w = SpreadW(kIdxH);

  const std::vector<uint16_t> got = RunBf16ActF32W(Cpu(), xb, w, kIdxH);
  CHECK(BadCount(got, Reference(xr, w, kRows, kIdxH, 1e-6f)) == 0);
}

TEST_CASE("rmsnorm: CUDA runs the bf16-activation/f32-gamma pairing") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: admitted-pairing device arm NOT exercised\n");
    return;
  }
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  const std::vector<float> w = SpreadW(kIdxH);
  const std::vector<uint16_t> want = RunBf16ActF32W(Cpu(), xb, w, kIdxH);
  const std::vector<uint16_t> got = RunBf16ActF32W(Device{DeviceType::kCUDA, 0}, xb, w, kIdxH);
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - Bf16ToF32(want[i])) >
        0.01f * (std::fabs(Bf16ToF32(want[i])) + 1e-3f)) ++bad;
  CHECK(bad == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 3. The KNOWN DIVERGENCE this dispatcher carries (#2503).
//
// `vt::RmsNorm` admits `IsFloat(weight.dtype)` (`ops.cpp:1030`, `:24`), which
// includes kF16, and the CPU kernel widens a kF16 gamma like any other
// (`cpu_ops.cpp:554-557`). The CUDA dispatcher refuses it. That refusal is NOT
// new -- before #2477 the equality check refused it too, because `x` is only ever
// dispatched as f32 or bf16 -- but it is a device arm refusing a dtype its CPU
// sibling accepts, which `cuda_qwen4_exp.cu:60-62` says must be recorded.
//
// It is pinned here so the `default:` arm cannot be quietly turned into a
// fall-through: doing that would read a kF16 gamma through `const float*`, and
// without this case every test would stay green.
TEST_CASE("rmsnorm: a kF16 gamma is admitted by the seam, taken by CPU, refused by CUDA") {
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  std::vector<uint16_t> w16(static_cast<size_t>(kIdxH), 0x3400);  // 0.25 in binary16
  std::vector<uint16_t> out(static_cast<size_t>(kRows * kIdxH), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};

  Queue qc{Cpu(), nullptr};
  Tensor cx = Tensor::Contiguous(xb.data(), DType::kBF16, Cpu(), {kRows, kIdxH});
  Tensor cw = Tensor::Contiguous(w16.data(), DType::kF16, Cpu(), {kIdxH});
  Tensor co = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {kRows, kIdxH});
  CHECK_NOTHROW(vt::RmsNorm(qc, co, cx, cw, args));

  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: kF16-gamma refusal NOT exercised on device\n");
    return;
  }
  const Device gpu{DeviceType::kCUDA, 0};
  vt::Backend& b = vt::GetBackend(gpu.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(xb.size() * sizeof(uint16_t));
  void* wd = b.Alloc(w16.size() * sizeof(uint16_t));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, xb.data(), xb.size() * sizeof(uint16_t));
  b.Copy(q, wd, w16.data(), w16.size() * sizeof(uint16_t));
  Tensor gx = Tensor::Contiguous(xd, DType::kBF16, gpu, {kRows, kIdxH});
  Tensor gw = Tensor::Contiguous(wd, DType::kF16, gpu, {kIdxH});
  Tensor go = Tensor::Contiguous(od, DType::kBF16, gpu, {kRows, kIdxH});
  // CHECK_THROWS would accept ANY exception, including one from an unrelated
  // allocation failure -- and worse for this case's purpose: if a fall-through
  // mutation's 256-byte over-read happened to fault, `Check(cudaGetLastError(), ...)`
  // would throw and a bare CHECK_THROWS would PASS on the mutated build, reading as
  // "mutation detected" when it was not. Assert the dispatcher's own message, so the
  // case is true by construction rather than by allocator luck.
  CHECK_THROWS_WITH(vt::RmsNorm(q, go, gx, gw, args),
                    doctest::Contains("unsupported weight dtype"));
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
}
