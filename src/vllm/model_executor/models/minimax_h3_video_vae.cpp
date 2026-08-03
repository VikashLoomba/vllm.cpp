// MiniMax-H3 VIDEO VAE decoder block — the repeated unit of the ViT3D decoder.
//
// Like the audio VAE, the video VAE is checkpoint REMOTE CODE
// (`FL2VA/video_vae/*.py` under `trust_remote_code`), so it must be REIMPLEMENTED
// rather than adapted. The real 560-tensor manifest
// (tests/vllm/models/minimax_h3_video_vae_manifest.inc) shows the split:
//   * the ENCODER is the 3D CNN (116 tensors, rank-5 Conv3d down blocks);
//   * the DECODER — the only half generation needs — is a **36-block TRANSFORMER**
//     (440 tensors), which is what this file ports.
//
// Per block (base_module.py:200-281), all in fp32:
//   h = h + scale1 * Attention(RMSNorm(h))
//   h = h + scale2 * FeedForward(RMSNorm(h))
// with `scale1`/`scale2` LEARNED PER-CHANNEL VECTORS (not scalars), a gated SiLU
// feed-forward (`w1` produces [gate | up], `w2` projects back), and per-head RMS
// q/k normalization with NO affine weight.
//
// THE TRAP: this ViT's `to_qkv` output is PER-HEAD INTERLEAVED. Upstream does
// `qkv.view(B, S, -1, 3 * dim_head)` then `chunk(3, dim=-1)` (attention.py), so the
// layout is [head0_q | head0_k | head0_v | head1_q | ...] — NOT the
// [q_all | k_all | v_all] that the H3 DiT's fused qkv uses. Reading it the DiT way
// silently produces a plausible-but-wrong image.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

// nn.RMSNorm over the last dim, fp32 accumulation. `weight` may be empty, which
// is the qk-norm case (elementwise_affine=False).
void RmsNormLastDim(std::vector<float>& x, int64_t rows, int64_t width,
                    const std::vector<float>* weight, double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float* row = x.data() + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(row[i]) * row[i];
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(width) + eps);
    for (int64_t i = 0; i < width; ++i) {
      double value = row[i] * inv;
      if (weight != nullptr) value *= (*weight)[static_cast<size_t>(i)];
      row[i] = static_cast<float>(value);
    }
  }
}

float SiluF(float x) { return x / (1.0f + std::exp(-x)); }

// y = x @ W^T + b, with W [out, in].
std::vector<float> LinearRows(const std::vector<float>& x, int64_t rows, int64_t in_features,
                              const std::vector<float>& weight, const std::vector<float>* bias,
                              int64_t out_features) {
  std::vector<float> out(static_cast<size_t>(rows * out_features));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(o)] : 0.0;
      for (int64_t i = 0; i < in_features; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_features + i)]) *
               static_cast<double>(weight[static_cast<size_t>(o * in_features + i)]);
      }
      out[static_cast<size_t>(r * out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

// One decoder TransformerBlock (base_module.py:200-281). `hidden` is [seq, dim].
std::vector<float> MiniMaxH3VideoVaeBlockForward(const MiniMaxH3VideoVaeBlockConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& hidden, int64_t seq) {
  const int64_t dim = config.dim;
  const int64_t heads = config.heads;
  const int64_t dim_head = config.dim_head;
  const int64_t inner = heads * dim_head;
  VT_CHECK(dim > 0 && heads > 0 && dim_head > 0, "minimax_h3 video vae: bad block geometry");
  VT_CHECK(static_cast<int64_t>(hidden.size()) == seq * dim,
           "minimax_h3 video vae: hidden size does not match [seq, dim]");

  std::vector<float> h = hidden;

  // --- attention branch ---
  {
    std::vector<float> normed = h;
    RmsNormLastDim(normed, seq, dim, &weights.Get(prefix + ".norm1.weight"), config.eps);

    const std::vector<float> qkv =
        LinearRows(normed, seq, dim, weights.Get(prefix + ".attn.to_qkv.weight"),
                   &weights.Get(prefix + ".attn.to_qkv.bias"), 3 * inner);

    // PER-HEAD INTERLEAVED: [head][q|k|v] within each row.
    std::vector<float> q(static_cast<size_t>(seq * inner));
    std::vector<float> k(static_cast<size_t>(seq * inner));
    std::vector<float> v(static_cast<size_t>(seq * inner));
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t head = 0; head < heads; ++head) {
        const int64_t src = s * 3 * inner + head * 3 * dim_head;
        const int64_t dst = s * inner + head * dim_head;
        for (int64_t d = 0; d < dim_head; ++d) {
          q[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + d)];
          k[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + dim_head + d)];
          v[static_cast<size_t>(dst + d)] = qkv[static_cast<size_t>(src + 2 * dim_head + d)];
        }
      }
    }
    // qk RMSNorm has NO affine weight in this checkpoint.
    RmsNormLastDim(q, seq * heads, dim_head, nullptr, config.eps);
    RmsNormLastDim(k, seq * heads, dim_head, nullptr, config.eps);

    // Full (non-causal) attention per head.
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim_head));
    std::vector<float> attn(static_cast<size_t>(seq * inner), 0.0f);
    std::vector<double> probs(static_cast<size_t>(seq));
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t i = 0; i < seq; ++i) {
        double max_score = -1e30;
        for (int64_t j = 0; j < seq; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < dim_head; ++d) {
            dot += static_cast<double>(q[static_cast<size_t>((i * heads + head) * dim_head + d)]) *
                   static_cast<double>(k[static_cast<size_t>((j * heads + head) * dim_head + d)]);
          }
          probs[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, probs[static_cast<size_t>(j)]);
        }
        double denom = 0.0;
        for (int64_t j = 0; j < seq; ++j) {
          probs[static_cast<size_t>(j)] = std::exp(probs[static_cast<size_t>(j)] - max_score);
          denom += probs[static_cast<size_t>(j)];
        }
        for (int64_t d = 0; d < dim_head; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < seq; ++j) {
            acc += probs[static_cast<size_t>(j)] *
                   static_cast<double>(v[static_cast<size_t>((j * heads + head) * dim_head + d)]);
          }
          attn[static_cast<size_t>((i * heads + head) * dim_head + d)] =
              static_cast<float>(acc / denom);
        }
      }
    }

    const std::vector<float> projected =
        LinearRows(attn, seq, inner, weights.Get(prefix + ".attn.to_out.weight"),
                   &weights.Get(prefix + ".attn.to_out.bias"), dim);
    // scale1 is a learned PER-CHANNEL vector.
    const std::vector<float>& scale1 = weights.Get(prefix + ".scale1");
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        h[static_cast<size_t>(s * dim + d)] +=
            projected[static_cast<size_t>(s * dim + d)] * scale1[static_cast<size_t>(d)];
      }
    }
  }

  // --- feed-forward branch (gated SiLU) ---
  {
    std::vector<float> normed = h;
    RmsNormLastDim(normed, seq, dim, &weights.Get(prefix + ".norm2.weight"), config.eps);

    const int64_t ff_inner = config.ff_inner;
    const std::vector<float> fused =
        LinearRows(normed, seq, dim, weights.Get(prefix + ".ff.w1.weight"),
                   &weights.Get(prefix + ".ff.w1.bias"), 2 * ff_inner);
    std::vector<float> act(static_cast<size_t>(seq * ff_inner));
    for (int64_t s = 0; s < seq; ++s) {
      const float* row = fused.data() + s * 2 * ff_inner;
      for (int64_t i = 0; i < ff_inner; ++i) {
        // chunk(2) gives [gate, up]; the gate is the FIRST half.
        act[static_cast<size_t>(s * ff_inner + i)] = SiluF(row[i]) * row[ff_inner + i];
      }
    }
    const std::vector<float> projected =
        LinearRows(act, seq, ff_inner, weights.Get(prefix + ".ff.w2.weight"),
                   &weights.Get(prefix + ".ff.w2.bias"), dim);
    const std::vector<float>& scale2 = weights.Get(prefix + ".scale2");
    for (int64_t s = 0; s < seq; ++s) {
      for (int64_t d = 0; d < dim; ++d) {
        h[static_cast<size_t>(s * dim + d)] +=
            projected[static_cast<size_t>(s * dim + d)] * scale2[static_cast<size_t>(d)];
      }
    }
  }
  return h;
}

}  // namespace vllm
