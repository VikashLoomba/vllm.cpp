// CPU half of the MiniMax-H3 DiT device-forward glue table (vt::OpId::kMiniMaxH3).
//
// Three small ops the shared vt:: surface does not cover; see
// include/vllm/model_executor/models/minimax_h3_device.h for why the table is this
// short (everything else in the DiT forward reuses tuned shared ops).
//
// Each kernel is a 1:1 transcription of the host reference it stands in for
// (minimax_h3.cpp: ModulateScaleShift / ModulateGate / Silu), in the same
// arithmetic order — so on CPU the device forward is BYTE-IDENTICAL to
// MiniMaxH3DitForward's own helpers, not merely close. The CUDA sibling lives in
// src/vt/cuda/cuda_minimax_h3.cu.
//
// Registering this on kCPU (Laguna's equivalent table is CUDA-only) is what lets
// the whole device-forward code path be covered by CPU CI, so a GPU is needed to
// gate the KERNELS, not the port's structure.
#include <cmath>
#include <cstdint>

#include "vllm/model_executor/models/minimax_h3_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vt::cpu {
namespace {

// _modulate_scale_shift (minimax_h3_transformer.py:183-192).
void MiniMaxH3ModulateScaleShift(Queue&, float* x, const float* shift, const float* scale,
                                 const int32_t* idx, int64_t rows, int64_t width,
                                 int64_t src_stride) {
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row = idx[r];
    float* dst = x + r * width;
    const float* s = scale + row * src_stride;
    const float* h = shift + row * src_stride;
    for (int64_t i = 0; i < width; ++i) dst[i] = dst[i] * (1.0f + s[i]) + h[i];
  }
}

// _modulate_gate (minimax_h3_transformer.py:195-204).
void MiniMaxH3ModulateGate(Queue&, float* residual, const float* gate, const float* other,
                           const int32_t* idx, int64_t rows, int64_t width,
                           int64_t src_stride) {
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row = idx[r];
    float* dst = residual + r * width;
    const float* g = gate + row * src_stride;
    const float* o = other + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] += g[i] * o[i];
  }
}

// Matches minimax_h3.cpp's Silu EXACTLY (x / (1 + exp(-x))); the algebraically
// equivalent x * sigmoid(x) is NOT bit-identical, and this path is gated against
// the host reference.
void MiniMaxH3Silu(Queue&, float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

const vllm::minimax_h3::MiniMaxH3DeviceKernels kKernels{
    &MiniMaxH3ModulateScaleShift,
    &MiniMaxH3ModulateGate,
    &MiniMaxH3Silu,
};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMiniMaxH3, DeviceType::kCPU,
               const_cast<void*>(static_cast<const void*>(&kKernels)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
