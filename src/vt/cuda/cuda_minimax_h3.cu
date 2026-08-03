// CUDA half of the MiniMax-H3 DiT device-forward glue table (vt::OpId::kMiniMaxH3).
//
// Three small ops the shared vt:: surface does not cover; see
// include/vllm/model_executor/models/minimax_h3_device.h for why the table is this
// short. Each is a 1:1 CUDA port of the host reference it stands in for
// (minimax_h3.cpp: ModulateScaleShift / ModulateGate / Silu), and is gated against
// that reference through the SAME upstream goldens the CPU forward uses.
//
// All three are elementwise over independent (row, column) pairs — the indexed
// AdaLN row lookup is a READ of a shared row, never a write — so a flat
// grid-stride loop is both correct and the natural shape. There is no reduction
// here, hence no reduction-order question: these kernels are bit-identical to the
// CPU sibling. (The device forward as a whole is NOT bit-identical to the CPU
// reference, but that comes from the tuned SHARED ops it reuses -- notably
// vt::RmsNorm's f32 reduction vs the reference's double -- not from this file.)
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/minimax_h3_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vllm::minimax_h3 {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

constexpr int kBlock = 256;

cudaStream_t AsStream(Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  return static_cast<unsigned>(capped < 1 ? 1 : capped);
}

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda minimax_h3: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

// _modulate_scale_shift (minimax_h3_transformer.py:183-192):
//   x[r, i] = x[r, i] * (1 + scale[idx[r], i]) + shift[idx[r], i]
__global__ void ModulateScaleShiftKernel(float* x, const float* shift, const float* scale,
                                         const int32_t* idx, int64_t rows, int64_t width,
                                         int64_t src_stride) {
  const int64_t n = rows * width;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; t < n; t += step) {
    const int64_t r = t / width, i = t % width;
    const int64_t row = idx[r];
    x[t] = x[t] * (1.0f + scale[row * src_stride + i]) + shift[row * src_stride + i];
  }
}

// _modulate_gate (minimax_h3_transformer.py:195-204):
//   residual[r, i] += gate[idx[r], i] * other[r, i]
__global__ void ModulateGateKernel(float* residual, const float* gate, const float* other,
                                   const int32_t* idx, int64_t rows, int64_t width,
                                   int64_t src_stride) {
  const int64_t n = rows * width;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; t < n; t += step) {
    const int64_t r = t / width, i = t % width;
    const int64_t row = idx[r];
    residual[t] += gate[row * src_stride + i] * other[t];
  }
}

// x / (1 + exp(-x)) -- the SAME form as the host Silu, not the algebraically
// equivalent x * sigmoid(x), which would not reproduce its bits.
__global__ void SiluKernel(float* x, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; t < n; t += step) {
    // expf, NOT the __expf fast intrinsic: __expf trades accuracy for speed and
    // would silently break the bit-identity this file claims.
    x[t] = x[t] / (1.0f + expf(-x[t]));
  }
}

void ModulateScaleShiftCuda(Queue& q, float* x, const float* shift, const float* scale,
                            const int32_t* idx, int64_t rows, int64_t width,
                            int64_t src_stride) {
  const int64_t n = rows * width;
  if (n == 0) return;
  ModulateScaleShiftKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, shift, scale, idx, rows,
                                                                   width, src_stride);
  Check(cudaGetLastError(), "modulate_scale_shift launch");
}

void ModulateGateCuda(Queue& q, float* residual, const float* gate, const float* other,
                      const int32_t* idx, int64_t rows, int64_t width, int64_t src_stride) {
  const int64_t n = rows * width;
  if (n == 0) return;
  ModulateGateKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(residual, gate, other, idx, rows,
                                                             width, src_stride);
  Check(cudaGetLastError(), "modulate_gate launch");
}

void SiluCuda(Queue& q, float* x, int64_t n) {
  if (n == 0) return;
  SiluKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(x, n);
  Check(cudaGetLastError(), "silu launch");
}

const MiniMaxH3DeviceKernels kKernels{
    &ModulateScaleShiftCuda,
    &ModulateGateCuda,
    &SiluCuda,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMiniMaxH3, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vllm::minimax_h3
