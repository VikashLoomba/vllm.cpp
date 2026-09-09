// vllm.cpp original. Typed shared BF16 MoE contracts (#3094).
#include <doctest/doctest.h>
#include <array>
#include <type_traits>
#include "rocm_moe_test_helpers.h"

// Adding native modes must not change any existing caller's function signature.
static_assert(std::is_same_v<vt::MoeGroupedGemmBf16Fn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, const vt::Tensor&)>);
static_assert(std::is_same_v<vt::MoeGroupedGemmBf16GateUpSiluFn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, const vt::Tensor&, const vt::Tensor&)>);
static_assert(std::is_same_v<vt::MoeCombineFn,
    void (*)(vt::Queue&, vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
             const vt::Tensor*, float)>);

TEST_CASE("shared native BF16 MoE validates routes and physical tensor formats") {
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::array<uint16_t, 16> values{};
  std::array<float, 4> weights{};
  std::array<int32_t, 4> ids{};
  std::array<int64_t, 4> ptrs{};
  auto a = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto out = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {2, 4});
  auto expert = rocm_moe_test::View(ids.data(), vt::DType::kI32, cpu, {2});
  auto pointers = rocm_moe_test::View(ptrs.data(), vt::DType::kI64, cpu, {4});
  auto route = rocm_moe_test::View(weights.data(), vt::DType::kF32, cpu, {1});
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr, pointers, route),
      doctest::Contains("route_weights must be contiguous f32 [P]"));
  route.shape[0] = 2;
  route.dtype = vt::DType::kBF16;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16Weighted(q, out, a, expert, nullptr, pointers, route),
      doctest::Contains("route_weights must be contiguous f32 [P]"));
  a.dtype = vt::DType::kF32;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, nullptr,
                                                        pointers, pointers),
                    doctest::Contains("act must be bf16"));
  a.dtype = vt::DType::kBF16;
  out.dtype = vt::DType::kF32;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, nullptr,
                                                        pointers, pointers),
                    doctest::Contains("out must be bf16"));
  auto rows = expert;
  rows.stride[0] = 2;
  out.dtype = vt::DType::kBF16;
  CHECK_THROWS_WITH(vt::MoeGroupedGemmBf16GateUpSiluNative(q, out, a, expert, &rows,
                                                        pointers, pointers),
                    doctest::Contains("row_map must be contiguous i32 [P]"));
  auto combined = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {1, 4});
  auto weighted = rocm_moe_test::View(values.data(), vt::DType::kF32, cpu, {1, 2, 4});
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted),
                    doctest::Contains("expert_out must be bf16"));
  weighted.dtype = vt::DType::kBF16;
  weighted.shape[1] = 4;
  weighted.shape[2] = 2;
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted),
                    doctest::Contains("must match out"));
  weighted = rocm_moe_test::View(values.data(), vt::DType::kBF16, cpu, {1, 2, 4});
  auto shared = combined;
  shared.dtype = vt::DType::kF16;
  CHECK_THROWS_WITH(vt::MoeCombinePreweighted(q, combined, weighted, &shared),
                    doctest::Contains("shared must be f32/bf16"));
  CHECK_FALSE(vt::MoeGroupedBf16NativeAvailable(vt::DeviceType::kCPU));
}
