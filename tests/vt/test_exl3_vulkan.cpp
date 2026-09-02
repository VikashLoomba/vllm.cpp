// The Vulkan arm of the two ops an EXL3 checkpoint ran on the CPU reference tier
// — row BACKEND-VULKAN-EXL3, issue #2530, spec
// .agents/specs/backend-vulkan-exl3.md.
//
// WHAT THIS SUITE GATES, AND WHY THE BOUND IS ZERO.
//
// `src/vt/vulkan/shaders/vt_exl3_had.comp` and `vt_exl3_gemm.comp` are
// transcribed from the PORTABLE CPU reference (`cpu_exl3_kernels.cpp`,
// `cpu_exl3_dequant.cpp`) rather than ported from `cuda_exl3.cu`, whose 90 KiB
// shared-memory budget already exceeds Vulkan's 16 KiB GUARANTEE before one
// reaches `mma.sync`, `ldmatrix`, `cp.async` or a grid-wide barrier Vulkan has
// at no version. Because they are a transcription of that reference and use no
// cooperative matrices and no split-K reduction, all three steps run the same
// IEEE f32 operations on the same values in the same ORDER — so the gate here is
// BYTE equality with the CPU arm, and not the 1.0e-3 RMS bound
// `test_exl3_gemm`'s CUDA cases carry. A tolerance here would be slack this arm
// has not earned and does not need.
//
// fp16 IS A SOFTWARE CODEC ON BOTH SIDES, which is why a byte claim survives the
// absence of any fp16 arithmetic extension in these shaders. `vt_common.glsl`
// requires only 16-bit STORAGE, and its `vt_f16_to_f32` / `vt_f32_to_f16` are
// integer transcriptions of `src/vt/dtype.cpp` — the same functions the host
// `RoundHalf` calls. The device therefore rounds where the host rounds rather
// than where hardware fp16 would.
//
// The comparison is INTEGER: the fp16 (or f32) output words are compared as bit
// patterns. That is what makes it a decode gate as well as a GEMM gate. A single
// mis-decoded codeword changes an f32 accumulation and the stored bits with it,
// and the codebook and bits arguments are proven to REACH the device decode by
// the two discrimination cases below, which require the arms to DISAGREE with
// each other on one trellis.
//
// Every device case SKIPS when no Vulkan backend is registered, says so, and
// STILL ASSERTS the precondition it skipped on. `assertions: 0` printed under
// `SUCCESS!` is a skip wearing a pass, and this family has already paid for that
// once.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;

bool HasVulkanBackend() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kVULKAN);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool HasVulkanCast() {
  return HasVulkanBackend() && vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN);
}

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

// A `const char*` handed to doctest's MESSAGE stringifies as a BOOL and prints
// "1", which turns an honest skip notice into noise. std::string prints.
const std::string kNoDevice =
    "SKIPPED, no Vulkan device: BACKEND-VULKAN-EXL3 parity is PENDING. This "
    "suite needs NO GPU and NO lease — llvmpipe (mesa-vulkan-drivers, "
    "/usr/share/vulkan/icd.d/lvp_icd.json) is a conformant Vulkan 1.1 device and "
    "is what the gate ran on. Reproduce with: cmake -S . -B build -G Ninja "
    "-DVLLM_CPP_VULKAN=ON && ctest --test-dir build -R '^test_exl3_vulkan$' "
    "--output-on-failure";

}  // namespace

// ─── V1: kCastF16 ────────────────────────────────────────────────────────────

TEST_CASE("exl3 vulkan: CastF16 is registered on Vulkan, like its two siblings") {
  // The op-table half of #2530's second reference-tier hit. `kCastBf16` and
  // `kCastF32` were registered on this backend from the W0 skeleton; `kCastF16`
  // was not, while all three are ONE kernel differing by a specialization
  // constant. This case is what reds if the registrar entry is deleted while the
  // kernel still compiles.
  if (!HasVulkanBackend()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
  CHECK(vt::OpRegistered(vt::OpId::kCastBf16, vt::DeviceType::kVULKAN));
  CHECK(vt::OpRegistered(vt::OpId::kCastF32, vt::DeviceType::kVULKAN));
}

TEST_CASE("exl3 vulkan: CastF16 is BYTE-identical to the CPU arm, from f32 and from bf16") {
  if (!HasVulkanCast()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kVULKAN));
    return;
  }
  vt::Backend& vb = vt::GetBackend(vt::DeviceType::kVULKAN);
  vt::Queue dq = vb.CreateQueue();
  vt::Queue hq = CpuQueue();
  const int64_t rows = 5, cols = 71;  // deliberately not a multiple of the block
  const int64_t nelem = rows * cols;

  Rng rng;
  rng.s = 0x9E3779B9u;
  std::vector<float> src_f32(static_cast<size_t>(nelem));
  for (auto& v : src_f32) v = rng.next(6.0f);
  // Values a plain conversion agrees on are not the interesting ones. These are:
  // an exact fp16 tie of each parity (rounds to EVEN), a value below the
  // subnormal floor, a subnormal, an overflow, and a signed zero. They are what
  // separates vt_common.glsl's transcribed codec from the driver's own
  // packHalf2x16, whose subnormal handling and rounding are NOT contractually
  // vt::F32ToF16's — which is exactly why that codec was transcribed.
  src_f32[0] = 1.0f + 1.0f / 2048.0f;
  src_f32[1] = 3.0f + 3.0f / 2048.0f;
  src_f32[2] = 5.0e-8f;
  src_f32[3] = 1.0e-6f;
  src_f32[4] = 1.0e6f;
  src_f32[5] = -0.0f;

  std::vector<uint16_t> src_bf16(static_cast<size_t>(nelem));
  for (int64_t i = 0; i < nelem; ++i) src_bf16[i] = vt::F32ToBF16(src_f32[i]);

  for (int arm = 0; arm < 2; ++arm) {
    const bool from_f32 = arm == 0;
    CAPTURE(from_f32);
    const size_t src_bytes =
        static_cast<size_t>(nelem) * (from_f32 ? sizeof(float) : sizeof(uint16_t));
    const void* src = from_f32 ? static_cast<const void*>(src_f32.data())
                               : static_cast<const void*>(src_bf16.data());
    const vt::DType sdt = from_f32 ? vt::DType::kF32 : vt::DType::kBF16;

    std::vector<uint16_t> host_out(static_cast<size_t>(nelem), 0);
    vt::Tensor hi = vt::Tensor::Contiguous(const_cast<void*>(src), sdt, hq.device, {rows, cols});
    vt::Tensor ho =
        vt::Tensor::Contiguous(host_out.data(), vt::DType::kF16, hq.device, {rows, cols});
    vt::CastF16(hq, ho, hi);

    void* d_in = vb.Alloc(src_bytes);
    void* d_out = vb.Alloc(static_cast<size_t>(nelem) * sizeof(uint16_t));
    vb.Copy(dq, d_in, src, src_bytes);
    std::memset(d_out, 0xFF, static_cast<size_t>(nelem) * sizeof(uint16_t));  // poison
    vt::Tensor di = vt::Tensor::Contiguous(d_in, sdt, dq.device, {rows, cols});
    vt::Tensor dof = vt::Tensor::Contiguous(d_out, vt::DType::kF16, dq.device, {rows, cols});
    vt::CastF16(dq, dof, di);
    vb.Synchronize(dq);
    std::vector<uint16_t> dev_out(static_cast<size_t>(nelem), 0);
    vb.Copy(dq, dev_out.data(), d_out, dev_out.size() * sizeof(uint16_t));
    vb.Synchronize(dq);
    vb.Free(d_in);
    vb.Free(d_out);

    int mismatches = 0;
    for (size_t i = 0; i < host_out.size(); ++i)
      if (host_out[i] != dev_out[i]) ++mismatches;
    CHECK(mismatches == 0);
    // Not vacuous: the cast must have produced something other than the poison
    // the device buffer was filled with.
    bool any_nonpoison = false;
    for (uint16_t v : dev_out)
      if (v != 0xFFFF) any_nonpoison = true;
    CHECK(any_nonpoison);
  }
  vb.DestroyQueue(dq);
}
