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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
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
    // The audio ENCODER shares this file, as do the VAE's mean/logvar heads and
    // pre_block in repackaged copies. Generation only decodes, so none are loaded.
    if (name.rfind("encoder.", 0) == 0) continue;
    if (name.rfind("mean_proj.", 0) == 0 || name.rfind("logs_proj.", 0) == 0) continue;
    if (name.rfind("pre_block.", 0) == 0) continue;

    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(std::strlen("decoder."));

    // The kaiser-sinc anti-aliasing filters are COMPUTED at load
    // (MiniMaxH3KaiserSincFilter1d), never read.
    if (key.size() >= 7 && key.compare(key.size() - 7, 7, ".filter") == 0) continue;

    const StTensor& tensor = file.Get(name);

    // --- the three weight-norm spellings this decoder must accept ---
    // (1) LEGACY  `weight_g` / `weight_v`  — the OFFICIAL MiniMax-H3 checkpoint.
    // (2) MODERN  `parametrizations.weight.original0/1` — what the decoder reads,
    //     and what the generator produced (it ran the remote code under a torch
    //     where weight_norm is a parametrization).
    // (3) MATERIALIZED plain `weight` — repackaged community bundles, which folded
    //     the norm at conversion time.
    const std::string g = ".weight_g", v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      key = key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    } else if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      key = key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    } else if (key.size() > 7 && key.compare(key.size() - 7, 7, ".weight") == 0 &&
               key != "dec_in_proj.weight" && tensor.shape.size() == 3) {
      // (3) Reconstruct an EXACT pair rather than an approximate one: with v = w and
      // g = per-dim-0-slice ||w||, the decoder's own g * v / ||v|| returns w.
      // dec_in_proj is a PLAIN Conv1d (not weight-normalized), so it is excluded.
      const std::string base = key.substr(0, key.size() - 7);
      std::vector<float> w = MiniMaxH3ReadSafetensorF32(tensor);
      const int64_t rows = tensor.shape[0];
      VT_CHECK(rows > 0 && static_cast<int64_t>(w.size()) % rows == 0,
               "minimax_h3 audio vae: materialized conv weight has an implausible shape");
      const int64_t per_row = static_cast<int64_t>(w.size()) / rows;
      std::vector<float> mag(static_cast<size_t>(rows));
      for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < per_row; ++i) {
          const double e = w[static_cast<size_t>(r * per_row + i)];
          sum += e * e;
        }
        mag[static_cast<size_t>(r)] = static_cast<float>(std::sqrt(sum));
      }
      VT_CHECK(out.tensors.count(base + ".parametrizations.weight.original1") == 0,
               "minimax_h3 audio vae: two checkpoint tensors map to the same name");
      out.tensors[base + ".parametrizations.weight.original1"] = std::move(w);
      out.tensors[base + ".parametrizations.weight.original0"] = std::move(mag);
      continue;
    }

    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 audio vae: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(tensor);
  }
  VT_CHECK(!out.tensors.empty(), "minimax_h3 audio vae: checkpoint contained no decoder tensors");
  return out;
}

MiniMaxH3AudioVaeWeights LoadMiniMaxH3VideoVaeDecoderWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    // The 3D-CNN ENCODER shares this file (conditioning only), and `quant_conv`
    // is its output stage. Generation decodes, so neither is loaded.
    if (name.rfind("encoder.", 0) == 0) continue;
    if (name.rfind("quant_conv.", 0) == 0) continue;

    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(std::strlen("decoder."));
    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 video vae: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(file.Get(name));
  }
  VT_CHECK(!out.tensors.empty(), "minimax_h3 video vae: checkpoint contained no decoder tensors");
  return out;
}

std::vector<float> MiniMaxH3VideoVaePostQuantConv(const MiniMaxH3AudioVaeWeights& weights,
                                                  const std::vector<float>& latent,
                                                  int64_t channels, int64_t elems_per_channel) {
  const std::vector<float>& w = weights.Get("post_quant_conv.weight");
  const std::vector<float>& b = weights.Get("post_quant_conv.bias");
  VT_CHECK(static_cast<int64_t>(w.size()) == channels * channels,
           "minimax_h3 post_quant_conv: weight must be [C, C, 1, 1, 1]");
  VT_CHECK(static_cast<int64_t>(b.size()) == channels,
           "minimax_h3 post_quant_conv: bias must have one value per channel");
  VT_CHECK(static_cast<int64_t>(latent.size()) == channels * elems_per_channel,
           "minimax_h3 post_quant_conv: latent size does not match [C, ...]");

  // A 1x1x1 Conv3d over a CHANNEL-MAJOR latent: out[o, p] = sum_i w[o, i] * in[i, p]
  // + b[o]. The accumulation is f32 in input-channel order, matching torch's
  // contraction over a length-C reduction.
  std::vector<float> out(latent.size());
  for (int64_t o = 0; o < channels; ++o) {
    const float bias = b[static_cast<size_t>(o)];
    float* dst = out.data() + o * elems_per_channel;
    for (int64_t p = 0; p < elems_per_channel; ++p) dst[p] = bias;
    for (int64_t i = 0; i < channels; ++i) {
      const float coeff = w[static_cast<size_t>(o * channels + i)];
      const float* src = latent.data() + i * elems_per_channel;
      for (int64_t p = 0; p < elems_per_channel; ++p) dst[p] += coeff * src[p];
    }
  }
  return out;
}


// --- H3-Encoder (FL2VA/text_encoder) ---------------------------------------
// The one loader that TRANSFORMS rather than renames: HF ships q/k/v and gate/up
// SEPARATE, the port (like vLLM) consumes them FUSED. Row-concatenation order is
// load-bearing — the forward slices qkv_proj at [0, q_width), [q_width,
// q_width+kv_width), [q_width+kv_width, ...), so any other order silently feeds
// keys into the query path.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3EncoderWeights(const std::vector<SafetensorsFile>& shards,
                                                     int64_t max_layers) {
  // One index over every shard, so a tensor is found wherever it lives.
  std::map<std::string, std::pair<size_t, const StTensor*>> index;
  for (size_t s = 0; s < shards.size(); ++s) {
    for (const std::string& name : shards[s].Names()) {
      VT_CHECK(index.count(name) == 0, "minimax_h3 encoder: tensor appears in two shards");
      index.emplace(name, std::make_pair(s, &shards[s].Get(name)));
    }
  }
  VT_CHECK(!index.empty(), "minimax_h3 encoder: no shards contained any tensor");

  MiniMaxH3AudioVaeWeights out;
  auto read = [&](const std::string& name) -> std::vector<float> {
    const auto it = index.find(name);
    VT_CHECK(it != index.end(), "minimax_h3 encoder: checkpoint is missing a required tensor");
    return MiniMaxH3ReadSafetensorF32(*it->second.second);
  };
  auto has = [&](const std::string& name) { return index.count(name) != 0; };

  const std::string lm = "model.language_model.";
  const std::string vis = "model.visual.";

  // --- text tower: strip the prefix, FUSE q/k/v and gate/up ---
  for (int64_t layer = 0;; ++layer) {
    const std::string src = lm + "layers." + std::to_string(layer) + ".";
    if (!has(src + "input_layernorm.weight")) break;
    if (max_layers > 0 && layer >= max_layers) break;
    const std::string dst = "layers." + std::to_string(layer) + ".";

    out.tensors[dst + "input_layernorm.weight"] = read(src + "input_layernorm.weight");
    out.tensors[dst + "post_attention_layernorm.weight"] =
        read(src + "post_attention_layernorm.weight");
    out.tensors[dst + "self_attn.q_norm.weight"] = read(src + "self_attn.q_norm.weight");
    out.tensors[dst + "self_attn.k_norm.weight"] = read(src + "self_attn.k_norm.weight");
    out.tensors[dst + "self_attn.o_proj.weight"] = read(src + "self_attn.o_proj.weight");
    out.tensors[dst + "mlp.down_proj.weight"] = read(src + "mlp.down_proj.weight");

    // [q_all | k_all | v_all], the order the forward slices.
    std::vector<float> q = read(src + "self_attn.q_proj.weight");
    const std::vector<float> k = read(src + "self_attn.k_proj.weight");
    const std::vector<float> v = read(src + "self_attn.v_proj.weight");
    q.reserve(q.size() + k.size() + v.size());
    q.insert(q.end(), k.begin(), k.end());
    q.insert(q.end(), v.begin(), v.end());
    out.tensors[dst + "self_attn.qkv_proj.weight"] = std::move(q);

    // [gate | up], matching MergedColumnParallelLinear.
    std::vector<float> gate = read(src + "mlp.gate_proj.weight");
    const std::vector<float> up = read(src + "mlp.up_proj.weight");
    gate.reserve(gate.size() + up.size());
    gate.insert(gate.end(), up.begin(), up.end());
    out.tensors[dst + "mlp.gate_up_proj.weight"] = std::move(gate);
  }
  VT_CHECK(out.tensors.count("layers.0.self_attn.qkv_proj.weight") != 0,
           "minimax_h3 encoder: no text-tower layers were loaded");

  if (has(lm + "embed_tokens.weight")) {
    // Kept: the text forward takes inputs_embeds, so a caller needs this to embed.
    out.tensors["embed_tokens.weight"] = read(lm + "embed_tokens.weight");
  }
  // `model.language_model.norm.weight` is deliberately NOT loaded — H3 reads the
  // UNNORMALIZED truncated output, and carrying the tensor would imply otherwise.

  // --- vision tower: prefix strip only; HF already ships qkv fused ---
  for (const auto& kv : index) {
    const std::string& name = kv.first;
    if (name.rfind(vis, 0) != 0) continue;
    const std::string dst = name.substr(vis.size());
    VT_CHECK(out.tensors.count(dst) == 0, "minimax_h3 encoder: vision name collides");
    out.tensors[dst] = MiniMaxH3ReadSafetensorF32(*kv.second.second);
  }
  return out;
}

}  // namespace vllm
