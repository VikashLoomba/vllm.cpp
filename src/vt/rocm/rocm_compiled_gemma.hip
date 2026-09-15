// Gemma compiled fusion boundaries, vLLM e126687a9a828d513c01a07cd69f025f27d63280.
// Generated cpjgc4x (embedding), cycnctz (sandwich), and cduy4zd (final norm).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "vt/compiled_gemma.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {
namespace {
constexpr int kThreads = 256;
__device__ float Load(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
// The executing Inductor reductions use R0_BLOCK=1024, four adjacent
// elements per lane, eight wave32 warps, then an eight-warp reduction.
__device__ float Sum(float value, float* scratch) {
  for (int stride = 8; stride; stride /= 2) value += __shfl_xor(value, stride, 32);
  value += __shfl_xor(value, 16, 32);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  for (int stride = 4; stride; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}
__device__ float Inverse(const __hip_bfloat16* x, int64_t width, float eps, float* scratch,
                         float scale = 1.f) {
  float sums[4] = {};
  for (int64_t tile = 0; tile < width; tile += kThreads * 4) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int64_t j = tile + threadIdx.x * 4 + element;
      if (j < width) {
        const float v = Load(x, j) * scale;
        sums[element] = fmaf(v, v, sums[element]);
      }
    }
  }
  const float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  return __builtin_amdgcn_rsqf(Sum(sum, scratch) / static_cast<float>(width) + eps);
}
__global__ void ScaledKernel(__hip_bfloat16* out, const __hip_bfloat16* input,
                             const __hip_bfloat16* weight, __hip_bfloat16* residual, int64_t width,
                             float scale, float eps) {
  __shared__ float scratch[kThreads];  // FP32 row reduction only, never a model buffer.
  const auto offset = static_cast<int64_t>(blockIdx.x) * width;
  const float inv = Inverse(input + offset, width, eps, scratch, scale);
  for (int64_t j = threadIdx.x; j < width; j += kThreads) {
    const auto scaled = __float2bfloat16(Load(input + offset, j) * scale);
    residual[offset + j] = scaled;
    out[offset + j] = __float2bfloat16((__bfloat162float(scaled) * inv) * (1.f + Load(weight, j)));
  }
}
__device__ float Expression(const __hip_bfloat16* a, const __hip_bfloat16* base,
                            const __hip_bfloat16* delta, const __hip_bfloat16* aw,
                            const __hip_bfloat16* dw, int64_t j, float ai, float di) {
  const float av = Load(a, j) * ai;
  const float sum = fmaf(av, 1.f + Load(aw, j), Load(base, j));
  if (!delta) return sum;
  const float dv = Load(delta, j) * di;
  return fmaf(dv, 1.f + Load(dw, j), sum);
}
__global__ void SandwichKernel(__hip_bfloat16* out, const __hip_bfloat16* a,
                               const __hip_bfloat16* base, const __hip_bfloat16* delta,
                               const __hip_bfloat16* aw, const __hip_bfloat16* dw,
                               const __hip_bfloat16* weight, __hip_bfloat16* residual,
                               int64_t width, float eps) {
  __shared__ float scratch[kThreads];  // FP32 variance scratch, BF16 activation operands.
  const auto offset = static_cast<int64_t>(blockIdx.x) * width;
  a += offset;
  base += offset;
  if (delta) delta += offset;
  const float ai = Inverse(a, width, eps, scratch);
  const float di = delta ? Inverse(delta, width, eps, scratch) : 0.f;
  float sums[4] = {};
  for (int64_t tile = 0; tile < width; tile += kThreads * 4) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int64_t j = tile + threadIdx.x * 4 + element;
      if (j < width) {
        const float value = Expression(a, base, delta, aw, dw, j, ai, di);
        sums[element] = fmaf(value, value, sums[element]);
      }
    }
  }
  const float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  const float inv = __builtin_amdgcn_rsqf(Sum(sum, scratch) / static_cast<float>(width) + eps);
  for (int64_t j = threadIdx.x; j < width; j += kThreads) {
    const float value = Expression(a, base, delta, aw, dw, j, ai, di);
    out[offset + j] = __float2bfloat16((value * inv) * (1.f + Load(weight, j)));
    if (residual) residual[offset + j] = __float2bfloat16(value);
  }
}
__global__ void GeluKernel(__hip_bfloat16* out, const __hip_bfloat16* packed, int64_t total,
                           int64_t width) {
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < total;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const auto row = at / width, j = at % width;
    const float gate = Load(packed, row * 2 * width + j),
                up = Load(packed, row * 2 * width + width + j);
    const float activated = (.5f * gate) * (1.f + erff(gate * .7071067811865476f));
    out[at] = __float2bfloat16(activated * up);
  }
}
void Check() {
  const auto status = hipGetLastError();
  VT_CHECK(status == hipSuccess, hipGetErrorString(status));
}
void Scaled(Queue& q, Tensor& out, const Tensor& input, const Tensor& weight,
            const ScaledRmsNormArgs& args, Tensor& residual) {
  EnsureQueueDevice(q);
  ScaledKernel<<<static_cast<unsigned>(input.shape[0]), kThreads, 0,
                 static_cast<hipStream_t>(q.handle)>>>(
      out.Ptr<__hip_bfloat16>(), input.Ptr<__hip_bfloat16>(), weight.Ptr<__hip_bfloat16>(),
      residual.Ptr<__hip_bfloat16>(), input.shape[1], args.scale, args.eps);
  Check();
}
void Sandwich(Queue& q, Tensor& out, const SandwichNormInputs& in, float eps, Tensor* residual) {
  EnsureQueueDevice(q);
  SandwichKernel<<<static_cast<unsigned>(in.a.shape[0]), kThreads, 0,
                   static_cast<hipStream_t>(q.handle)>>>(
      out.Ptr<__hip_bfloat16>(), in.a.Ptr<__hip_bfloat16>(), in.base.Ptr<__hip_bfloat16>(),
      in.delta ? in.delta->Ptr<__hip_bfloat16>() : nullptr, in.a_weight.Ptr<__hip_bfloat16>(),
      in.delta_weight ? in.delta_weight->Ptr<__hip_bfloat16>() : nullptr,
      in.weight.Ptr<__hip_bfloat16>(), residual ? residual->Ptr<__hip_bfloat16>() : nullptr,
      in.a.shape[1], eps);
  Check();
}
void Gelu(Queue& q, Tensor& out, const Tensor& packed) {
  EnsureQueueDevice(q);
  const auto total = out.Numel();
  GeluKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
               static_cast<hipStream_t>(q.handle)>>>(
      out.Ptr<__hip_bfloat16>(), packed.Ptr<__hip_bfloat16>(), total, out.shape[1]);
  Check();
}
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kScaledRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<ScaledRmsNormFn>(&Scaled)));
    RegisterOp(OpId::kSandwichRmsNorm, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<SandwichRmsNormFn>(&Sandwich)));
    RegisterOp(OpId::kCompiledGeluErfMul, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&Gelu)));
  }
} registrar;
}  // namespace
}  // namespace vt::rocm
