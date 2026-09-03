// The EXL3 m<=8 GEMV arm and its selection envelope — MODEL-DSV4-EXL3 W2c.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//   exllamav3_ext/quant/exl3_gemv.cu:29-42    the two env knobs
//   exllamav3_ext/quant/exl3_gemv.cu:46-72    exl3_gemv_cfg, the shape envelope
//   exllamav3_ext/quant/exl3_gemv.cu:108-114  the hard eligibility checks
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh:31  EXL3_GEMV_MAX_M
//
// WHAT IS GATED HERE, AND WHAT IS NOT.
//
// The ENVELOPE is pure integer arithmetic over (cc, m, k, n, K, cb, mode,
// narrow_coresident) and is gated on any machine. That matters more here than it
// did for the GEMM shape table, because the sentence this row inherited about
// this envelope — "Ada/Blackwell are memory-bound here and keep the regular
// kernel" — describes a guard that is COMMENTED OUT at `exl3_gemv.cu:53`. A
// quoted comment is not a gate; these cases are.
//
// The KERNEL is not gated here on a machine with no GPU, and its bound is not
// tier 3. It accumulates in fp16 and folds to f32 only every FOLD iterations
// (`exl3_gemv_kernel.cuh:37-52,317-330`), which is a different NUMERIC arm from
// the f32-accumulating regular kernel. `.agents/specs/model-dsv4-exl3.md`
// `## W2cd design` W2c-3 states its own bound, tier 3c, and the device case
// below is where it is measured. That case SKIPS loudly and still asserts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;

bool HasCudaExl3() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

// The two shapes this checkpoint's TP1-coalesced experts have (spec
// `## The format`): w1/w3 are k=4096 n=2048, w2 is k=2048 n=4096.
constexpr int kW13K = 4096, kW13N = 2048;
constexpr int kW2K = 2048, kW2N = 4096;

}  // namespace

// ─── W2c-1: the compute-capability guard is DISABLED upstream ────────────────

TEST_CASE("exl3 gemv: no compute-capability test is live in the envelope") {
  // `exl3_gemv.cu:53` is `//if (cc != CC_AMPERE) return -1;`. If that line were
  // live, every non-Ampere bucket would return -1 for every shape. It is not, so
  // w2's shape is eligible on EVERY bucket that reaches the shape tests, and the
  // buckets differ from each other only where upstream's LIVE branches say they
  // do (`:65`, the Ada K==3 row).
  const int kMode = 1;  // the heuristic, upstream's default
  for (vt::Exl3Cc cc : {vt::Exl3Cc::kOld, vt::Exl3Cc::kAmpere, vt::Exl3Cc::kAda,
                        vt::Exl3Cc::kHopper, vt::Exl3Cc::kBlackwell}) {
    // w2, K = 3, cb = 1: `:67` `size_k <= 2048 && size_n <= 8192` fires for
    // every bucket, so config 0 (narrow), with NO occupancy input.
    CHECK(vt::Exl3GemvSelectConfig(cc, 1, kW2K, kW2N, 3, 1, kMode,
                                   /*narrow_coresident=*/0) == 0);
  }
}

// ─── W2c-2: what the envelope resolves to for THIS checkpoint ────────────────

TEST_CASE("exl3 gemv: w2 is eligible on GB10 and w1/w3 rest on an occupancy query") {
  const vt::Exl3Cc bw = vt::Exl3Cc::kBlackwell;
  const int kMode = 1;

  // w2 (k=2048, n=4096): `:67` fires. Independent of occupancy, so BOTH ends of
  // the occupancy range agree.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW2K, kW2N, 3, 1, kMode, 1 << 20) == 0);

  // w1/w3 (k=4096, n=2048): `:67` does NOT fire, so `:68` `K == 3` returns -1
  // UNLESS `:66` fires first, which needs `2048 / 32 = 64 <= narrow_coresident`.
  // The threshold is EXACTLY 64 and both sides of it are pinned, because the
  // whole point of the entry is that the verdict is a device query this row has
  // not made.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 63) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 64) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, kW13K, kW13N, 3, 1, kMode, 65) == 0);
}

TEST_CASE("exl3 gemv: the other branches of the envelope are upstream's too") {
  const int kMode = 1;
  // `:64` K == 2 splits on n at 8192, on EVERY bucket.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 2, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 2, 1, kMode, 0) == 1);
  // `:65` K == 3 on ADA takes the same split; on Blackwell it does not (that is
  // the one branch where the buckets genuinely differ).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8192, 3, 1, kMode, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 8320, 3, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8320, 3, 1, kMode, 0) == -1);
  // `:69` K == 4, big n, small k -> the wide config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 4096, 8192, 4, 1, kMode, 0) == 1);
  // `:70` is Ampere-only even though nothing else is.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAmpere, 1, 5120, 10240, 4, 1, kMode, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, 5120, 10240, 4, 1, kMode, 0) == -1);
  // `:48` mode 0 turns the whole path off; `:54` mode 2 takes it wherever the
  // hard constraints allow; `:55`/`:56` force one config.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW2K, kW2N, 3, 1, 0, 0) == -1);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 2, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 3, 0) == 0);
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, 1, kW13K, kW13N, 3, 1, 4, 0) == 1);
}

TEST_CASE("exl3 gemv: the hard constraints refuse before any heuristic runs") {
  // `exl3_gemv.cu:110-114`, in upstream's own order.
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 1, /*has_su_sv=*/false));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 1, 1, true));   // K < 2
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 5, 1, true));   // K > 4
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 3, 0, true));   // K != 4 && cb == 0
  CHECK(vt::Exl3GemvHardEligible(1, kW2K, kW2N, 4, 0, true));         // K == 4 admits cb 0
  CHECK(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 2048 + 16, kW2N, 3, 1, true));  // size_k % 128
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, kW2K, 4096 + 16, 3, 1, true));  // size_n % 128
  // The constant itself, so a change to it is a red rather than a silent
  // widening (`exl3_gemv_kernel.cuh:31`).
  CHECK(vt::kExl3GemvMaxM == 8);
  // The envelope also enforces the hard bound on m, independently, because
  // upstream repeats it inside `exl3_gemv_cfg` (`:51`).
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kBlackwell, vt::kExl3GemvMaxM + 1, kW2K, kW2N, 3, 1,
                                 1, 0) == -1);
}

// ─── QUANT-EXL3-PERF A3: the envelope AT THE CHECKPOINT'S OWN SHAPES ────────
//
// The spec's admission table, executable. Read from the local safetensors
// headers of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` on 2026-09-02: 409 trellis
// modules, all `mul1` (cb 2), widths {3: 137, 4: 270, 5: 1, 6: 1}, every one
// 128-aligned on both k and n.
//
// WHY THIS CASE EXISTS. Instantiating an arm is necessary and it is NOT
// sufficient: `Exl3GemvSelectConfig` returns -1 to DECLINE, and on Blackwell
// every bits-3 branch above the `if (K == 3) return -1;` line depends on
// `narrow_coresident`, which is an OCCUPANCY QUERY and therefore the one term a
// unit test cannot supply. So this case pins the THRESHOLD instead: it asserts
// the exact `narrow_coresident` at which each shape flips, which makes the
// spec's table a gate and turns a device measurement into a lookup rather than
// a guess. A zero end-to-end effect with the arm declined and a zero with the
// arm taken are DIFFERENT results, and this is what tells them apart.
TEST_CASE("exl3 gemv: the envelope's verdict at the #2495 checkpoint's real shapes") {
  constexpr int kMode = 1;  // the DEFAULT: production, not a forced testing mode
  const auto bw = vt::Exl3Cc::kBlackwell;

  // bits 3, cb 2 — 137 modules, two shapes. On Blackwell the only branch that
  // can admit them is `size_n / 32 <= narrow_coresident`; `size_k <= 2048` is
  // false at both (k is 5120 and 17408) and `if (K == 3) return -1;` follows.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, kMode, 543) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, kMode, 544) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, kMode, 159) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, kMode, 160) == 0);

  // The envelope is per-K and NOT per-cb, so (3, 2) inherits (3, 1)'s exactly.
  // If that ever stops holding, this row's whole "no envelope change" claim is
  // wrong, and it fails here rather than in a benchmark.
  for (int n : {17408, 5120}) {
    const int k = n == 17408 ? 5120 : 17408;
    for (int nc : {0, 159, 160, 543, 544, 1 << 20}) {
      CHECK(vt::Exl3GemvSelectConfig(bw, 1, k, n, 3, 1, kMode, nc) ==
            vt::Exl3GemvSelectConfig(bw, 1, k, n, 3, 2, kMode, nc));
    }
  }

  // bits 4, cb 2 — 270 modules. NOT instantiated here (the arm is owed), but
  // the envelope is upstream's and would admit them, so the thresholds are
  // recorded now: they are what the (4, 2) port would be measured against.
  // n = 1024 (34 modules) is admitted by ANY plausible occupancy.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 1024, 4, 2, kMode, 32) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 1024, 4, 2, kMode, 31) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 6144, 5120, 4, 2, kMode, 160) == 0);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 10240, 4, 2, kMode, 320) == 0);
  // ... and that K == 4 does NOT hit the bits-3 early decline: with the
  // narrow branch missed it falls to the wide-config band, which needs
  // size_k <= 4096 and every k here is larger.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 10240, 4, 2, kMode, 319) == -1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 4096, 10240, 4, 2, kMode, 319) == 1);

  // bits 5 and 6 have NO GEMV upstream either (`exl3_gemv.cu:110-111`), so the
  // one 5-bit tensor and the 6-bit lm_head falling to the regular shape table
  // is upstream's arrangement and not a gap. Asserted so it is not re-filed.
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 6144, 5120, 5, 2, true));
  CHECK_FALSE(vt::Exl3GemvHardEligible(1, 5120, 248320, 6, 2, true));

  // Mode 2 is upstream's "wherever the hard constraints allow" testing mode
  // (`exl3_gemv.cu:22`). It is what the diagnostic leg of the A/B uses, and it
  // must admit every bits-3 shape here regardless of occupancy, or that leg
  // measures the same path twice.
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 5120, 17408, 3, 2, 2, 0) == 1);
  CHECK(vt::Exl3GemvSelectConfig(bw, 1, 17408, 5120, 3, 2, 2, 0) == 0);
}

TEST_CASE("exl3 gemv: the env knobs parse exactly as upstream's do") {
  // `exl3_gemv.cu:29-34`: unset is 1, everything else is atoi.
  CHECK(vt::Exl3GemvParseMode(nullptr) == 1);
  CHECK(vt::Exl3GemvParseMode("0") == 0);
  CHECK(vt::Exl3GemvParseMode("1") == 1);
  CHECK(vt::Exl3GemvParseMode("2") == 2);
  CHECK(vt::Exl3GemvParseMode("4") == 4);
  // `exl3_gemv.cu:37-42`: unset is -1.
  CHECK(vt::Exl3GemvParseSmemMode(nullptr) == -1);
  CHECK(vt::Exl3GemvParseSmemMode("0") == 0);
  CHECK(vt::Exl3GemvParseSmemMode("1") == 1);
}

// ─── W2c-3: the device arm, and the bound that is NOT tier 3 ─────────────────

TEST_CASE("exl3 device: every instantiated GEMV arm meets tier 3c") {
  if (!HasCudaExl3()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2c's tier-3c bound is PENDING, and so is "
        "`narrow_coresident`, the occupancy query that alone decides whether the w1/w3 shape "
        "is GEMV-eligible at all. dgx.casa is flapping. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemv -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  // FORCED, through `Exl3GemmArgs::force_gemv`, which mirrors upstream's own
  // direct entry point (`exl3_gemv.cu:171-241`, "errors if the call is not
  // hard-eligible"). Forcing is what makes this a gate rather than a coin flip
  // on a heuristic: without it a device whose occupancy declines the shape would
  // measure the REGULAR kernel and report tier 3c green.
  vt::Backend& cb_be = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  const int64_t m = 1, k = kW2K, n = kW2N;
  Exl3Fixture f = MakeFixture(k, n, 3, 0x5EEDu);
  std::vector<uint16_t> a(static_cast<size_t>(m * k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  // Both codebooks' device outputs are kept, because they MUST differ: (3, 1)
  // and (3, 2) are the confusable pair. Same width, same `dq8` route, same tile
  // shapes, and a `cb` threaded wrongly between them neither fails to compile
  // nor changes a shape — it decodes with the other codebook's tail and yields a
  // weight with the right DISTRIBUTION and no correlation to the true one. A
  // per-arm tolerance alone cannot see that; the cross-arm check below can.
  std::vector<std::vector<uint16_t>> per_arm;

  // (3, 1) is the SparkInfer DeepSeek-V4 artifact's arm. (3, 2) is 137 of the
  // 409 trellis modules of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — every MLP
  // projection quantized at the low end of its 3.5 bpw average, and the single
  // largest population in that checkpoint that this arm can reach
  // (QUANT-EXL3-PERF, #2570).
  for (int codebook : {1, 2}) {
    CAPTURE(codebook);
    vt::Queue dq = cb_be.CreateQueue();

    // The reference is the CPU arm, which `test_exl3_gemm` already gates against
    // the f64 chain at tier 3. Comparing against it rather than re-deriving f64
    // here keeps ONE reference for both device arms.
    std::vector<uint16_t> ref(static_cast<size_t>(m * n), 0), got(static_cast<size_t>(m * n), 0);
    std::vector<uint16_t> a_had_h(static_cast<size_t>(m * k), 0);
    vt::Exl3GemmArgs ha;
    ha.bits = 3;
    ha.codebook = codebook;
    {
      vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tc = vt::Tensor::Contiguous(ref.data(), vt::DType::kF16, hq.device, {m, n});
      vt::Tensor tb = vt::Tensor::Contiguous(f.trellis.data(), vt::DType::kI8, hq.device,
                                             {k / 16, n / 16, 32 * 3});
      vt::Tensor tsuh = vt::Tensor::Contiguous(f.suh.data(), vt::DType::kF16, hq.device, {k});
      vt::Tensor tsvh = vt::Tensor::Contiguous(f.svh.data(), vt::DType::kF16, hq.device, {n});
      vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, ha);
    }

    void* d_a = cb_be.Alloc(a.size() * 2);
    void* d_ah = cb_be.Alloc(a.size() * 2);
    void* d_c = cb_be.Alloc(got.size() * 2);
    void* d_b = cb_be.Alloc(f.trellis.size() * 2);
    void* d_suh = cb_be.Alloc(f.suh.size() * 2);
    void* d_svh = cb_be.Alloc(f.svh.size() * 2);
    cb_be.Copy(dq, d_a, a.data(), a.size() * 2);
    cb_be.Copy(dq, d_b, f.trellis.data(), f.trellis.size() * 2);
    cb_be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
    cb_be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);
    vt::Tensor da = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor dah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor dc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
    vt::Tensor db =
        vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * 3});
    vt::Tensor dsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
    vt::Tensor dsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
    vt::Exl3GemmArgs da_args = ha;
    da_args.force_gemv = 1;  // upstream's `force`: bypasses the heuristic, not the hard checks
    vt::Exl3Gemm(dq, dc, da, db, dsuh, dsvh, dah, da_args);
    cb_be.Synchronize(dq);
    cb_be.Copy(dq, got.data(), d_c, got.size() * 2);
    cb_be.Synchronize(dq);

    double sq = 0.0, rq = 0.0, worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
      worst = std::max(worst, std::fabs(g - r));
    }
    const double rms_ref = std::sqrt(rq / static_cast<double>(got.size()));
    const double rel = std::sqrt(sq / static_cast<double>(got.size())) / rms_ref;
    MESSAGE("cb ", codebook, " tier 3c: relative RMS ", rel, ", worst elementwise ", worst);
    // `## W2cd design` W2c-3. NOT tier 3's 1.0e-3: this arm accumulates in fp16.
    // A NEW arm INHERITS this bound; it is never widened to admit one.
    CHECK(rel <= 6.0e-3);
    CHECK(worst <= 64.0 * UlpF16(rms_ref));
    // A GEMV that silently declined would leave `got` at its allocated content
    // and could still pass a tolerance against a reference that is also near
    // zero. It cannot pass this.
    CHECK(rms_ref > 0.0);

    per_arm.push_back(got);

    cb_be.Free(d_a);
    cb_be.Free(d_ah);
    cb_be.Free(d_c);
    cb_be.Free(d_b);
    cb_be.Free(d_suh);
    cb_be.Free(d_svh);
    cb_be.DestroyQueue(dq);
  }

  // THE DISCRIMINATION CHECK. The two codebooks are different decodes of the
  // same bits, so on the same trellis they must produce different numbers. If
  // they agree the case is measuring one arm twice and its tolerances mean
  // nothing — which is the spec's stop condition, not a tolerance to widen.
  REQUIRE(per_arm.size() == 2);
  size_t differing = 0;
  for (size_t i = 0; i < per_arm[0].size(); ++i)
    if (per_arm[0][i] != per_arm[1][i]) ++differing;
  MESSAGE("(3,1) vs (3,2) differ in ", differing, " of ", per_arm[0].size(), " outputs");
  CHECK(differing > per_arm[0].size() / 2);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}
