// MiniMax-H3 NVFP4 arm — the quantized safetensors checkpoint that is also the
// SPEED path on sm_121 (native FP4 tensor cores).
//
// The real 1051-tensor manifest of `lilcheaty/MiniMax-H3-NVFP4` is gated in
// tests/vllm/models/minimax_h3_nvfp4_manifest.inc and shows the checkpoint is the
// textbook compressed-tensors triple this project ALREADY consumes:
//
//   <name>.weight          U8       FP4 packed 2-per-byte, [out, in/2]
//   <name>.weight_scale    F8_E4M3  one per group of 16 along K, [out, in/16]
//   <name>.weight_scale_2  F32      one global scalar
//
// 258 quantized projections carry all three; the fp32/bf16 ISLANDS (both patch
// projections, the time embedder, both output heads, the norms, rope.inv_freq)
// are left unquantized so the DiT's dtype policy survives. The names are
// IDENTICAL to the contract derived from upstream source, so — as with the GGUF
// arm — there is no rename table, only a dequantization decision per tensor.
//
// This loader materializes to f32 for the reference forward. The DEVICE path
// (brick W2b/W10b) will instead keep the packed FP4 resident and route the
// projections through the existing cutlass FP4 GEMM, which is where the speed
// actually comes from; nothing here claims a speed result.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// vt bf16 bit pattern -> f32.
float Bf16ToF32(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

float F16ToF32(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 0x1u;
  const uint32_t exponent = (bits >> 10) & 0x1Fu;
  const uint32_t mantissa = bits & 0x3FFu;
  uint32_t out_bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      out_bits = sign << 31;
    } else {  // subnormal: renormalize
      uint32_t e = exponent, m = mantissa;
      int shift = 0;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3FFu;
      e = 127 - 15 - shift + 1;
      out_bits = (sign << 31) | (e << 23) | (m << 13);
    }
  } else if (exponent == 0x1F) {
    out_bits = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
  } else {
    out_bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

// Read an UNQUANTIZED island tensor into f32, whatever its storage dtype.
std::vector<float> ReadPlain(const StTensor& t) {
  int64_t numel = 1;
  for (int64_t d : t.shape) numel *= d;
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.dtype == "F32") {
    VT_CHECK(t.nbytes == static_cast<size_t>(numel) * 4,
             "minimax_h3 nvfp4: F32 tensor span does not match its shape");
    std::memcpy(out.data(), t.data, t.nbytes);
  } else if (t.dtype == "BF16") {
    VT_CHECK(t.nbytes == static_cast<size_t>(numel) * 2,
             "minimax_h3 nvfp4: BF16 tensor span does not match its shape");
    const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = Bf16ToF32(src[i]);
  } else if (t.dtype == "F16") {
    VT_CHECK(t.nbytes == static_cast<size_t>(numel) * 2,
             "minimax_h3 nvfp4: F16 tensor span does not match its shape");
    const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = F16ToF32(src[i]);
  } else {
    VT_CHECK(false, "minimax_h3 nvfp4: unsupported island dtype (expected F32/BF16/F16)");
  }
  return out;
}

}  // namespace

MiniMaxH3GgufDit LoadMiniMaxH3DitFromNvfp4(const SafetensorsFile& file) {
  MiniMaxH3GgufDit out;

  // --- pass 1: materialize every tensor to f32 ---
  for (const std::string& name : file.Names()) {
    // Quantization sidecars are consumed with their parent, never on their own.
    if (name.size() > 12 &&
        (name.compare(name.size() - 12, 12, "weight_scale") == 0 ||
         (name.size() > 14 && name.compare(name.size() - 14, 14, "weight_scale_2") == 0))) {
      continue;
    }
    const StTensor& t = file.Get(name);
    if (t.dtype != "U8") {
      out.storage[name] = ReadPlain(t);
      out.shapes[name] = t.shape;
      continue;
    }

    // NVFP4: [out, in/2] packed + [out, in/16] E4M3 group scales + f32 global.
    VT_CHECK(t.shape.size() == 2, "minimax_h3 nvfp4: a packed weight must be rank 2");
    const int64_t out_dim = t.shape[0];
    const int64_t in_dim = t.shape[1] * 2;
    const StTensor& scale = file.Get(name + "_scale");
    const StTensor& global = file.Get(name + "_scale_2");
    VT_CHECK(scale.dtype == "F8_E4M3", "minimax_h3 nvfp4: weight_scale must be F8_E4M3");
    VT_CHECK(global.dtype == "F32", "minimax_h3 nvfp4: weight_scale_2 must be F32");
    VT_CHECK(scale.shape.size() == 2 && scale.shape[0] == out_dim &&
                 scale.shape[1] * 16 == in_dim,
             "minimax_h3 nvfp4: weight_scale must be [out, in/16] (NVFP4 group size 16)");
    float global_scale = 0.0f;
    VT_CHECK(global.nbytes >= sizeof(float), "minimax_h3 nvfp4: weight_scale_2 is too small");
    std::memcpy(&global_scale, global.data, sizeof(float));

    std::vector<uint16_t> bf16(static_cast<size_t>(out_dim * in_dim));
    DequantNvfp4ToBf16(t.data, scale.data, global_scale, out_dim, in_dim, bf16.data());
    std::vector<float> values(bf16.size());
    for (size_t i = 0; i < bf16.size(); ++i) values[i] = Bf16ToF32(bf16[i]);
    out.storage[name] = std::move(values);
    out.shapes[name] = {out_dim, in_dim};
  }

  // --- pass 2: geometry from the materialized shapes, then bind the views ---
  std::vector<MiniMaxH3TensorSpec> manifest;
  manifest.reserve(out.shapes.size());
  for (const auto& entry : out.shapes) {
    MiniMaxH3TensorSpec spec;
    spec.name = entry.first;
    spec.shape = entry.second;
    manifest.push_back(std::move(spec));
  }
  out.params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  BindMiniMaxH3DitViews(&out);
  return out;
}

}  // namespace vllm
