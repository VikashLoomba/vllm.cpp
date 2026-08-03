// MiniMax-H3 ENCODER text tower — a Qwen3-VL with three H3-specific deltas.
//
// The H3-Encoder produces the `[seq, 5120]` `prompt_embeds` the DiT consumes. It
// is a Qwen3-VL, which this project already ports (qwen3_vl_{text,vision}.cpp), so
// the ARCHITECTURE is reuse; what is H3-specific — and what this file pins down and
// gates — are the three deltas (upstream encoder.py:1-30):
//
//   1. LAYER TRUNCATION. Only the first `MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER`
//      (50) decoder layers are kept: `num_layers = min(config.num_hidden_layers,
//      selected_layer)`.
//   2. UNNORMALIZED OUTPUT. The checkpoint consumes the hidden state straight out
//      of layer 49 — there is NO final RMSNorm, unlike a stock Qwen3-VL text model.
//      Applying one silently shifts every conditioning vector.
//   3. DEEPSTACK. Visual features are ADDED at the visual token positions after
//      each of the first `len(deepstack_visual_embeds)` layers.
//
// The layer itself is the familiar pre-norm block: RMSNorm -> fused-QKV attention
// with per-head q/k RMSNorm and interleaved M-RoPE -> causal GQA SDPA -> o_proj ->
// residual; RMSNorm -> gated-SiLU MLP -> residual.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

void RmsNormRowsEnc(const float* in, const float* weight, float* out, int64_t rows, int64_t width,
                    double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(src[i]) * src[i];
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(width) + eps);
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) dst[i] = static_cast<float>(src[i] * inv * weight[i]);
  }
}

float SiluEnc(float x) { return x / (1.0f + std::exp(-x)); }

// y = x @ W^T (no bias; Qwen3-VL text projections are bias-free), W [out, in],
// optionally reading a ROW SLICE of a larger fused weight.
std::vector<float> LinearSlice(const std::vector<float>& x, int64_t rows, int64_t in_features,
                               const std::vector<float>& weight, int64_t row_offset,
                               int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = 0.0;
      const float* w = weight.data() + (row_offset + o) * in_features;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(w[i]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

// Interleaved M-RoPE (encoder.py:292-300 + 319-329). `positions` is [3, seq]
// (temporal, height, width). Produces cos/sin of [seq, head_dim].
void MiniMaxH3EncoderMrope(const int64_t* positions, int64_t seq, int64_t head_dim,
                           double rope_theta, const std::vector<int64_t>& mrope_section,
                           std::vector<float>* cos_out, std::vector<float>* sin_out) {
  VT_CHECK(head_dim % 2 == 0, "minimax_h3 encoder: head_dim must be even");
  VT_CHECK(mrope_section.size() == 3, "minimax_h3 encoder: mrope_section must have 3 entries");
  const int64_t half = head_dim / 2;

  cos_out->assign(static_cast<size_t>(seq * head_dim), 0.0f);
  sin_out->assign(static_cast<size_t>(seq * head_dim), 0.0f);
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t i = 0; i < half; ++i) {
      // The frequency slot i takes its position from the TEMPORAL axis by
      // default; slots at (1, 4, 7, ...) below 3*section[1] come from HEIGHT and
      // slots at (2, 5, 8, ...) below 3*section[2] come from WIDTH. That is the
      // "chunked [TTT HHH WWW] -> interleaved [THW THW ...]" reshuffle.
      int64_t axis = 0;
      if (i % 3 == 1 && i < 3 * mrope_section[1]) {
        axis = 1;
      } else if (i % 3 == 2 && i < 3 * mrope_section[2]) {
        axis = 2;
      }
      const double inv_freq =
          1.0 / std::pow(rope_theta, static_cast<double>(2 * i) / static_cast<double>(head_dim));
      const double angle = static_cast<double>(positions[axis * seq + s]) * inv_freq;
      const float c = static_cast<float>(std::cos(angle));
      const float sn = static_cast<float>(std::sin(angle));
      // emb = cat(freqs, freqs) => the second half repeats the first.
      (*cos_out)[static_cast<size_t>(s * head_dim + i)] = c;
      (*cos_out)[static_cast<size_t>(s * head_dim + half + i)] = c;
      (*sin_out)[static_cast<size_t>(s * head_dim + i)] = sn;
      (*sin_out)[static_cast<size_t>(s * head_dim + half + i)] = sn;
    }
  }
}

// The H3 layer budget: min(config.num_hidden_layers, selected_layer).
int64_t MiniMaxH3EncoderNumLayers(int64_t config_num_hidden_layers, int64_t selected_layer) {
  VT_CHECK(config_num_hidden_layers > 0 && selected_layer > 0,
           "minimax_h3 encoder: layer counts must be positive");
  return std::min(config_num_hidden_layers, selected_layer);
}

std::vector<float> MiniMaxH3EncoderTextForward(const MiniMaxH3EncoderConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::vector<float>& inputs_embeds,
                                               const int64_t* positions, int64_t seq,
                                               const uint8_t* visual_pos_mask,
                                               const std::vector<std::vector<float>>& deepstack) {
  const int64_t hidden = config.hidden_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t head_dim = config.head_dim;
  const int64_t q_width = heads * head_dim;
  const int64_t kv_width = kv_heads * head_dim;
  VT_CHECK(static_cast<int64_t>(inputs_embeds.size()) == seq * hidden,
           "minimax_h3 encoder: inputs_embeds size does not match [seq, hidden]");
  VT_CHECK(heads % kv_heads == 0, "minimax_h3 encoder: heads must be a multiple of kv_heads");
  const int64_t groups = heads / kv_heads;

  std::vector<float> cos, sin;
  MiniMaxH3EncoderMrope(positions, seq, head_dim, config.rope_theta, config.mrope_section, &cos,
                        &sin);

  const int64_t num_layers =
      MiniMaxH3EncoderNumLayers(config.num_hidden_layers, config.selected_layer);
  std::vector<float> h = inputs_embeds;
  std::vector<float> normed(h.size());

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    RmsNormRowsEnc(h.data(), weights.Get(p + "input_layernorm.weight").data(), normed.data(), seq,
                   hidden, config.rms_norm_eps);

    // Fused qkv weight rows are [q_all | k_all | v_all].
    const std::vector<float>& qkv_w = weights.Get(p + "self_attn.qkv_proj.weight");
    std::vector<float> q = LinearSlice(normed, seq, hidden, qkv_w, 0, q_width);
    std::vector<float> k = LinearSlice(normed, seq, hidden, qkv_w, q_width, kv_width);
    const std::vector<float> v = LinearSlice(normed, seq, hidden, qkv_w, q_width + kv_width, kv_width);

    // Per-head q/k RMSNorm, THEN RoPE.
    std::vector<float> qn(q.size()), kn(k.size());
    RmsNormRowsEnc(q.data(), weights.Get(p + "self_attn.q_norm.weight").data(), qn.data(),
                   seq * heads, head_dim, config.rms_norm_eps);
    RmsNormRowsEnc(k.data(), weights.Get(p + "self_attn.k_norm.weight").data(), kn.data(),
                   seq * kv_heads, head_dim, config.rms_norm_eps);
    const int64_t rot_half = head_dim / 2;
    auto apply_rope = [&](std::vector<float>& x, int64_t n_heads) {
      for (int64_t s = 0; s < seq; ++s) {
        const float* c = cos.data() + s * head_dim;
        const float* sn = sin.data() + s * head_dim;
        for (int64_t head = 0; head < n_heads; ++head) {
          float* row = x.data() + (s * n_heads + head) * head_dim;
          for (int64_t i = 0; i < rot_half; ++i) {
            const float lo = row[i], hi = row[i + rot_half];
            row[i] = lo * c[i] - hi * sn[i];
            row[i + rot_half] = hi * c[i + rot_half] + lo * sn[i + rot_half];
          }
        }
      }
    };
    apply_rope(qn, heads);
    apply_rope(kn, kv_heads);

    // CAUSAL GQA attention (is_causal=True upstream).
    std::vector<float> attn(static_cast<size_t>(seq * q_width), 0.0f);
    std::vector<double> probs(static_cast<size_t>(seq));
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int64_t head = 0; head < heads; ++head) {
      const int64_t kv_head = head / groups;
      for (int64_t i = 0; i < seq; ++i) {
        double max_score = -1e30;
        for (int64_t j = 0; j <= i; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(qn[static_cast<size_t>((i * heads + head) * head_dim + d)]) *
                   static_cast<double>(kn[static_cast<size_t>((j * kv_heads + kv_head) * head_dim + d)]);
          }
          probs[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j <= i; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(v[static_cast<size_t>((j * kv_heads + kv_head) * head_dim + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }
    const std::vector<float> projected =
        LinearSlice(attn, seq, q_width, weights.Get(p + "self_attn.o_proj.weight"), 0, hidden);
    for (size_t i = 0; i < h.size(); ++i) h[i] += projected[i];

    // gated-SiLU MLP; gate_up_proj rows are [gate | up].
    RmsNormRowsEnc(h.data(), weights.Get(p + "post_attention_layernorm.weight").data(),
                   normed.data(), seq, hidden, config.rms_norm_eps);
    const std::vector<float>& gate_up = weights.Get(p + "mlp.gate_up_proj.weight");
    const std::vector<float> gate = LinearSlice(normed, seq, hidden, gate_up, 0, config.intermediate_size);
    const std::vector<float> up =
        LinearSlice(normed, seq, hidden, gate_up, config.intermediate_size, config.intermediate_size);
    std::vector<float> act(gate.size());
    for (size_t i = 0; i < act.size(); ++i) act[i] = SiluEnc(gate[i]) * up[i];
    const std::vector<float> down =
        LinearSlice(act, seq, config.intermediate_size, weights.Get(p + "mlp.down_proj.weight"), 0,
                    hidden);
    for (size_t i = 0; i < h.size(); ++i) h[i] += down[i];

    // DeepStack: add the visual features into the visual token rows, for the
    // FIRST len(deepstack) layers only (encoder.py:770-779, 792-798).
    if (layer < static_cast<int64_t>(deepstack.size())) {
      VT_CHECK(visual_pos_mask != nullptr,
               "minimax_h3 encoder: deepstack embeds require a visual position mask");
      const std::vector<float>& embeds = deepstack[static_cast<size_t>(layer)];
      int64_t visual_index = 0;
      for (int64_t s = 0; s < seq; ++s) {
        if (!visual_pos_mask[s]) continue;
        for (int64_t i = 0; i < hidden; ++i) {
          h[static_cast<size_t>(s * hidden + i)] +=
              embeds[static_cast<size_t>(visual_index * hidden + i)];
        }
        ++visual_index;
      }
    }
  }
  // NO final norm: the checkpoint consumes the UNNORMALIZED layer-49 state.
  return h;
}

// ---------------------------------------------------------------------------
// Vision tower block (encoder.py:417-481) — the repeated unit of the ViT.
//
// Differs from the TEXT tower in several ways that all matter numerically:
//   * LayerNorm (with bias), not RMSNorm, at eps 1e-6;
//   * the qkv output is reshaped [seq, 3, heads, head_dim] and PERMUTED, so the
//     layout is [q_all | k_all | v_all] per token — not the per-head interleave
//     the video VAE's ViT uses;
//   * rotary is applied in FP32 with cos/sin supplied per token;
//   * attention is NON-CAUSAL and segmented by `cu_seqlens` (one segment per
//     image/frame), so it never crosses a packed-image boundary;
//   * the MLP uses the TANH-approximate GELU (`gelu_pytorch_tanh`), not exact erf.
// ---------------------------------------------------------------------------

namespace {

// nn.LayerNorm over the last dim.
void LayerNormRows(const float* in, const float* weight, const float* bias, float* out,
                   int64_t rows, int64_t width, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double mean = 0.0;
    for (int64_t i = 0; i < width; ++i) mean += src[i];
    mean /= static_cast<double>(width);
    double var = 0.0;
    for (int64_t i = 0; i < width; ++i) var += (src[i] - mean) * (src[i] - mean);
    var /= static_cast<double>(width);
    const double inv = 1.0 / std::sqrt(var + eps);
    float* dst = out + r * width;
    for (int64_t i = 0; i < width; ++i) {
      double value = (src[i] - mean) * inv * weight[i];
      if (bias != nullptr) value += bias[i];
      dst[i] = static_cast<float>(value);
    }
  }
}

// nn.GELU(approximate="tanh").
float GeluTanh(float x) {
  const double xd = x;
  const double inner = 0.7978845608028654 * (xd + 0.044715 * xd * xd * xd);
  return static_cast<float>(0.5 * xd * (1.0 + std::tanh(inner)));
}

// y = x @ W^T + b.
std::vector<float> LinearBias(const std::vector<float>& x, int64_t rows, int64_t in_features,
                              const std::vector<float>& weight, const std::vector<float>* bias,
                              int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(o)] : 0.0;
      const float* w = weight.data() + o * in_features;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(w[i]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

std::vector<float> MiniMaxH3VisionBlockForward(const MiniMaxH3VisionBlockConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::string& prefix,
                                               const std::vector<float>& hidden, int64_t seq,
                                               const float* cos, const float* sin,
                                               const int32_t* cu_seqlens, int64_t num_segments) {
  const int64_t dim = config.hidden_size;
  const int64_t heads = config.num_heads;
  const int64_t head_dim = dim / heads;
  VT_CHECK(dim > 0 && heads > 0 && dim % heads == 0,
           "minimax_h3 vision: hidden_size must divide by num_heads");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == seq * dim,
           "minimax_h3 vision: hidden size does not match [seq, dim]");

  std::vector<float> h = hidden;
  std::vector<float> normed(h.size());

  // --- attention ---
  LayerNormRows(h.data(), weights.Get(prefix + ".norm1.weight").data(),
                weights.Get(prefix + ".norm1.bias").data(), normed.data(), seq, dim, config.eps);
  const std::vector<float> qkv =
      LinearBias(normed, seq, dim, weights.Get(prefix + ".attn.qkv.weight"),
                 &weights.Get(prefix + ".attn.qkv.bias"), 3 * dim);

  // reshape(seq, 3, heads, head_dim).permute(1,0,2,3) => q/k/v each [seq, heads, head_dim].
  std::vector<float> q(static_cast<size_t>(seq * dim));
  std::vector<float> k(static_cast<size_t>(seq * dim));
  std::vector<float> v(static_cast<size_t>(seq * dim));
  for (int64_t s = 0; s < seq; ++s) {
    const float* row = qkv.data() + s * 3 * dim;
    std::copy(row, row + dim, q.begin() + s * dim);
    std::copy(row + dim, row + 2 * dim, k.begin() + s * dim);
    std::copy(row + 2 * dim, row + 3 * dim, v.begin() + s * dim);
  }

  // Rotary in FP32, cos/sin shared across heads (encoder.py:404-415).
  const int64_t half = head_dim / 2;
  for (int64_t s = 0; s < seq; ++s) {
    const float* c = cos + s * head_dim;
    const float* sn = sin + s * head_dim;
    for (int64_t head = 0; head < heads; ++head) {
      for (std::vector<float>* target : {&q, &k}) {
        float* row = target->data() + (s * heads + head) * head_dim;
        for (int64_t i = 0; i < half; ++i) {
          const double lo = row[i], hi = row[i + half];
          row[i] = static_cast<float>(lo * c[i] - hi * sn[i]);
          row[i + half] = static_cast<float>(hi * c[i + half] + lo * sn[i + half]);
        }
      }
    }
  }

  // Non-causal attention, segmented by cu_seqlens.
  std::vector<float> attn(static_cast<size_t>(seq * dim), 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<double> probs;
  for (int64_t seg = 0; seg < num_segments; ++seg) {
    const int64_t begin = cu_seqlens[seg];
    const int64_t end = cu_seqlens[seg + 1];
    const int64_t len = end - begin;
    if (len <= 0) continue;
    probs.resize(static_cast<size_t>(len));
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t i = begin; i < end; ++i) {
        double max_score = -1e30;
        for (int64_t j = begin; j < end; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q[static_cast<size_t>((i * heads + head) * head_dim + d)]) *
                   static_cast<double>(k[static_cast<size_t>((j * heads + head) * head_dim + d)]);
          }
          probs[static_cast<size_t>(j - begin)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j - begin)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j < len; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < len; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(
                       v[static_cast<size_t>(((begin + j) * heads + head) * head_dim + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * head_dim + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }
  }
  const std::vector<float> projected =
      LinearBias(attn, seq, dim, weights.Get(prefix + ".attn.proj.weight"),
                 &weights.Get(prefix + ".attn.proj.bias"), dim);
  for (size_t i = 0; i < h.size(); ++i) h[i] += projected[i];

  // --- MLP with tanh-approximate GELU ---
  LayerNormRows(h.data(), weights.Get(prefix + ".norm2.weight").data(),
                weights.Get(prefix + ".norm2.bias").data(), normed.data(), seq, dim, config.eps);
  std::vector<float> mid =
      LinearBias(normed, seq, dim, weights.Get(prefix + ".mlp.linear_fc1.weight"),
                 &weights.Get(prefix + ".mlp.linear_fc1.bias"), config.intermediate_size);
  for (float& value : mid) value = GeluTanh(value);
  const std::vector<float> down =
      LinearBias(mid, seq, config.intermediate_size, weights.Get(prefix + ".mlp.linear_fc2.weight"),
                 &weights.Get(prefix + ".mlp.linear_fc2.bias"), dim);
  for (size_t i = 0; i < h.size(); ++i) h[i] += down[i];
  return h;
}

}  // namespace vllm
