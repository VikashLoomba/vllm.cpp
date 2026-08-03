// MiniMax-H3 VAE checkpoint loaders — materializing the shipped safetensors files
// into the weight structs the (already gated) VAE forwards consume.
//
// The forwards were gated against the checkpoint's OWN remote code at reduced
// dimensions, with weights rebuilt from the shared PRNG. That proved the MATH.
// What it could not prove is that the SHIPPED file's tensors bind onto those
// structs — and for the audio VAE they do not, not directly. Both mismatches were
// found by reading the real 1087-tensor header (an HTTP range request over the
// file's first 2 MiB, no payload downloaded) and are gated against it:
//
//   1. WEIGHT-NORM SPELLING. The checkpoint ships torch's LEGACY weight_norm pair
//      `weight_g` / `weight_v`. The decoder reads the PARAMETRIZATION spelling
//      `parametrizations.weight.original0` / `original1`, because the generator
//      that produced its goldens ran the checkpoint's remote code under a modern
//      torch, where weight_norm is a parametrization. Same tensors, different era.
//   2. PREFIX. Every BigVGAN tensor lives under `decoder.`, but `dec_in_proj.*` —
//      the Conv1d that runs BEFORE BigVGAN — sits at the top level.
//
// Either mismatch alone yields a loader that throws by name (best case) or, if a
// future refactor made lookups lenient, a decoder reading zeros. The mapping is
// therefore asserted against the real manifest in the test, not just exercised.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

float Bf16ToF32(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

float F16ToF32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exp = (bits >> 10) & 0x1Fu;
  const uint32_t mant = bits & 0x3FFu;
  uint32_t out_bits;
  if (exp == 0) {
    if (mant == 0) {
      out_bits = sign;  // +/- zero
    } else {
      // Subnormal: renormalize into the f32 exponent range.
      uint32_t e = 0;
      uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FFu;
      out_bits = sign | ((127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    out_bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    out_bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

}  // namespace

std::vector<float> MiniMaxH3ReadSafetensorF32(const StTensor& tensor) {
  int64_t numel = 1;
  for (int64_t d : tensor.shape) numel *= d;
  std::vector<float> out(static_cast<size_t>(numel));
  if (tensor.dtype == "F32") {
    VT_CHECK(tensor.nbytes == static_cast<size_t>(numel) * 4,
             "minimax_h3: F32 tensor span does not match its shape");
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
  } else if (tensor.dtype == "BF16") {
    VT_CHECK(tensor.nbytes == static_cast<size_t>(numel) * 2,
             "minimax_h3: BF16 tensor span does not match its shape");
    const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.data);
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = Bf16ToF32(src[i]);
  } else if (tensor.dtype == "F16") {
    VT_CHECK(tensor.nbytes == static_cast<size_t>(numel) * 2,
             "minimax_h3: F16 tensor span does not match its shape");
    const uint16_t* src = reinterpret_cast<const uint16_t*>(tensor.data);
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = F16ToF32(src[i]);
  } else {
    VT_CHECK(false, "minimax_h3: unsupported tensor dtype (expected F32/BF16/F16)");
  }
  return out;
}

MiniMaxH3AudioVaeWeights LoadMiniMaxH3AudioVaeWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    // The audio ENCODER shares this file. Generation only decodes, so its tensors
    // are ignored rather than loaded — silently carrying ~half the file would cost
    // memory for weights nothing reads.
    if (name.rfind("encoder.", 0) == 0) continue;

    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(std::strlen("decoder."));

    // The kaiser-sinc anti-aliasing filters are COMPUTED at load
    // (MiniMaxH3KaiserSincFilter1d), never read. Loading them would be harmless
    // but misleading: it would suggest the decoder consumes them.
    if (key.size() >= 7 && key.compare(key.size() - 7, 7, ".filter") == 0) continue;

    // LEGACY weight_norm -> the parametrization spelling the decoder reads.
    const std::string g = ".weight_g";
    const std::string v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      key = key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    } else if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      key = key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    }

    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 audio vae: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(file.Get(name));
  }
  VT_CHECK(!out.tensors.empty(), "minimax_h3 audio vae: checkpoint contained no decoder tensors");
  return out;
}

}  // namespace vllm
