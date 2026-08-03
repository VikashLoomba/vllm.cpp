// MiniMax-H3 AUDIO VAE decoder — a DAC-lineage BigVGAN vocoder, REIMPLEMENTED.
//
// ─── WHY REIMPLEMENTED AND NOT ADAPTED ───────────────────────────────────────
// H3's two VAEs are checkpoint REMOTE CODE: their implementations ship inside the
// HF repo (`FL2VA/audio_vae/*.py`) and are loaded through
// `get_class_from_dynamic_module` under `trust_remote_code`; vLLM-Omni's `vae.py`
// only ADAPTS them (vae.py:41-53). A no-Python engine cannot do that, so this is a
// from-scratch port of the checkpoint's own modules, gated against them by
// scripts/gen-minimax-h3-audio-vae-goldens.py (which imports the remote code and
// runs it at reduced dimensions as the oracle). The remote code is NOT vendored
// here — it ships under the MiniMax H3 Community License with the checkpoint.
//
// ─── ARCHITECTURE (checkpoint config.yaml + metadata.json) ───────────────────
//   decode(z[32, T]) = dec_in_proj (Conv1d 32 -> 2048, k=1) -> BigVGAN
//   BigVGAN: conv_pre(2048 -> 1024, k=7)
//            x7 [ ConvTranspose1d upsample (rates 5,5,2,2,2,2,2;
//                                           kernels 9,9,4,4,4,4,4)
//                 then 3 AMPBlock1 residual blocks (kernels 3,7,11,
//                 dilations 1,3,5) whose outputs are AVERAGED ]
//            anti-aliased SnakeBeta -> conv_post(-> 1 ch, k=7, no bias)
//            -> clamp[-1, 1]   (use_tanh_at_final is false for H3)
//   32 kHz, 2 channels; the DiT's audio rows are decoded per channel.
//
// Two details that are easy to get wrong and are gated explicitly:
//   * Every conv is WEIGHT-NORMALIZED. The checkpoint stores (g, v) as
//     `...parametrizations.weight.original0` / `original1`, and the effective
//     weight is `g * v / ||v||` with the norm taken over every dim except dim 0.
//   * The anti-aliased activation upsamples 2x, applies SnakeBeta, then
//     downsamples 2x, both through a KAISER-SINC filter built at load time (never
//     loaded from the checkpoint) with REPLICATE padding.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

const std::vector<float>& MiniMaxH3AudioVaeWeights::Get(const std::string& name) const {
  const auto it = tensors.find(name);
  VT_CHECK(it != tensors.end(), "minimax_h3 audio vae: missing checkpoint tensor");
  return it->second;
}

namespace {

// Zeroth-order modified Bessel function of the first kind, matching the series
// torch.kaiser_window uses.
double BesselI0(double x) {
  double sum = 1.0, term = 1.0;
  const double half_x_sq = (x / 2.0) * (x / 2.0);
  for (int k = 1; k < 64; ++k) {
    term *= half_x_sq / (static_cast<double>(k) * static_cast<double>(k));
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return sum;
}

double Sinc(double x) {
  if (x == 0.0) return 1.0;
  const double pix = M_PI * x;
  return std::sin(pix) / pix;
}

// torch.kaiser_window(n, periodic=false, beta).
std::vector<double> KaiserWindow(int64_t length, double beta) {
  std::vector<double> window(static_cast<size_t>(length));
  const double denom = BesselI0(beta);
  // periodic=false => the window spans [0, length-1] inclusive.
  const double n_minus_1 = static_cast<double>(length - 1);
  for (int64_t i = 0; i < length; ++i) {
    const double ratio = (2.0 * static_cast<double>(i) - n_minus_1) / n_minus_1;
    window[static_cast<size_t>(i)] = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
  }
  return window;
}

// One 1-D convolution over [C_in, T] with dilation/stride/groups.
// Weight is [C_out, C_in/groups, K]; input is assumed ALREADY padded.
std::vector<float> Conv1d(const std::vector<float>& in, int64_t in_channels, int64_t in_len,
                          const std::vector<float>& weight, const std::vector<float>* bias,
                          int64_t out_channels, int64_t kernel, int64_t stride, int64_t dilation,
                          int64_t groups, int64_t* out_len) {
  const int64_t effective = dilation * (kernel - 1) + 1;
  const int64_t length = (in_len - effective) / stride + 1;
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv1d output length is empty");
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<float> out(static_cast<size_t>(out_channels * length), 0.0f);
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    const int64_t g = oc / out_per_group;
    for (int64_t t = 0; t < length; ++t) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0;
      for (int64_t ic = 0; ic < in_per_group; ++ic) {
        const int64_t src_c = g * in_per_group + ic;
        for (int64_t k = 0; k < kernel; ++k) {
          const int64_t pos = t * stride + k * dilation;
          acc += static_cast<double>(in[static_cast<size_t>(src_c * in_len + pos)]) *
                 static_cast<double>(weight[static_cast<size_t>((oc * in_per_group + ic) * kernel + k)]);
        }
      }
      out[static_cast<size_t>(oc * length + t)] = static_cast<float>(acc);
    }
  }
  *out_len = length;
  return out;
}

// torch.nn.functional.conv_transpose1d over [C_in, T].
// Weight is [C_in, C_out/groups, K]; output length = (T-1)*stride - 2*padding + K.
std::vector<float> ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                   int64_t in_len, const std::vector<float>& weight,
                                   const std::vector<float>* bias, int64_t out_channels,
                                   int64_t kernel, int64_t stride, int64_t padding, int64_t groups,
                                   int64_t* out_len) {
  const int64_t full = (in_len - 1) * stride + kernel;
  const int64_t length = full - 2 * padding;
  VT_CHECK(length > 0, "minimax_h3 audio vae: conv_transpose1d output length is empty");
  const int64_t in_per_group = in_channels / groups;
  const int64_t out_per_group = out_channels / groups;
  std::vector<double> acc(static_cast<size_t>(out_channels * full), 0.0);
  for (int64_t ic = 0; ic < in_channels; ++ic) {
    const int64_t g = ic / in_per_group;
    for (int64_t t = 0; t < in_len; ++t) {
      const double value = in[static_cast<size_t>(ic * in_len + t)];
      if (value == 0.0) continue;
      for (int64_t oc = 0; oc < out_per_group; ++oc) {
        const int64_t dst_c = g * out_per_group + oc;
        for (int64_t k = 0; k < kernel; ++k) {
          acc[static_cast<size_t>(dst_c * full + t * stride + k)] +=
              value * static_cast<double>(weight[static_cast<size_t>((ic * out_per_group + oc) * kernel + k)]);
        }
      }
    }
  }
  std::vector<float> out(static_cast<size_t>(out_channels * length));
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      double value = acc[static_cast<size_t>(c * full + t + padding)];
      if (bias != nullptr) value += (*bias)[static_cast<size_t>(c)];
      out[static_cast<size_t>(c * length + t)] = static_cast<float>(value);
    }
  }
  *out_len = length;
  return out;
}

// F.pad(..., mode="replicate") along the time axis.
std::vector<float> PadReplicate(const std::vector<float>& in, int64_t channels, int64_t in_len,
                                int64_t left, int64_t right, int64_t* out_len) {
  const int64_t length = in_len + left + right;
  std::vector<float> out(static_cast<size_t>(channels * length));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < length; ++t) {
      int64_t src = t - left;
      src = std::max<int64_t>(0, std::min<int64_t>(in_len - 1, src));
      out[static_cast<size_t>(c * length + t)] = in[static_cast<size_t>(c * in_len + src)];
    }
  }
  *out_len = length;
  return out;
}

// SnakeBeta: x + (beta + 1e-9)^-1 * sin^2(alpha * x), with alpha/beta exponentiated
// when the checkpoint stores them in log scale (H3 does).
void SnakeBeta(std::vector<float>& x, int64_t channels, int64_t length,
               const std::vector<float>& alpha, const std::vector<float>& beta, bool logscale) {
  for (int64_t c = 0; c < channels; ++c) {
    double a = alpha[static_cast<size_t>(c)];
    double b = beta[static_cast<size_t>(c)];
    if (logscale) {
      a = std::exp(a);
      b = std::exp(b);
    }
    const double inv_beta = 1.0 / (b + 1e-9);
    for (int64_t t = 0; t < length; ++t) {
      const double v = x[static_cast<size_t>(c * length + t)];
      const double s = std::sin(a * v);
      x[static_cast<size_t>(c * length + t)] = static_cast<float>(v + inv_beta * s * s);
    }
  }
}

}  // namespace

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60). Returns [kernel_size].
std::vector<float> MiniMaxH3KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size) {
  VT_CHECK(kernel_size > 0, "minimax_h3 audio vae: kernel_size must be positive");
  VT_CHECK(cutoff >= 0.0 && cutoff <= 0.5, "minimax_h3 audio vae: cutoff must be in [0, 0.5]");
  const bool even = (kernel_size % 2) == 0;
  const int64_t half_size = kernel_size / 2;

  const double delta_f = 4.0 * half_width;
  const double a = 2.285 * (static_cast<double>(half_size) - 1.0) * M_PI * delta_f + 7.95;
  double beta = 0.0;
  if (a > 50.0) {
    beta = 0.1102 * (a - 8.7);
  } else if (a >= 21.0) {
    beta = 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  }
  const std::vector<double> window = KaiserWindow(kernel_size, beta);

  std::vector<double> time(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    time[static_cast<size_t>(i)] = even ? (static_cast<double>(-half_size + i) + 0.5)
                                        : static_cast<double>(i - half_size);
  }

  std::vector<double> filter(static_cast<size_t>(kernel_size), 0.0);
  if (cutoff == 0.0) {
    return std::vector<float>(static_cast<size_t>(kernel_size), 0.0f);
  }
  double sum = 0.0;
  for (int64_t i = 0; i < kernel_size; ++i) {
    filter[static_cast<size_t>(i)] =
        2.0 * cutoff * window[static_cast<size_t>(i)] * Sinc(2.0 * cutoff * time[static_cast<size_t>(i)]);
    sum += filter[static_cast<size_t>(i)];
  }
  // Normalized to sum 1 so a constant input does not leak.
  std::vector<float> out(static_cast<size_t>(kernel_size));
  for (int64_t i = 0; i < kernel_size; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(filter[static_cast<size_t>(i)] / sum);
  }
  return out;
}

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0.
std::vector<float> MiniMaxH3MaterializeWeightNorm(const std::vector<float>& g,
                                                  const std::vector<float>& v,
                                                  int64_t out_channels) {
  VT_CHECK(out_channels > 0 && v.size() % static_cast<size_t>(out_channels) == 0,
           "minimax_h3 audio vae: weight-norm direction does not divide by out_channels");
  const int64_t per_out = static_cast<int64_t>(v.size()) / out_channels;
  VT_CHECK(static_cast<int64_t>(g.size()) == out_channels,
           "minimax_h3 audio vae: weight-norm magnitude must have one value per output channel");
  std::vector<float> out(v.size());
  for (int64_t c = 0; c < out_channels; ++c) {
    double norm = 0.0;
    for (int64_t i = 0; i < per_out; ++i) {
      const double value = v[static_cast<size_t>(c * per_out + i)];
      norm += value * value;
    }
    norm = std::sqrt(norm);
    const double scale = norm > 0.0 ? static_cast<double>(g[static_cast<size_t>(c)]) / norm : 0.0;
    for (int64_t i = 0; i < per_out; ++i) {
      out[static_cast<size_t>(c * per_out + i)] =
          static_cast<float>(v[static_cast<size_t>(c * per_out + i)] * scale);
    }
  }
  return out;
}

namespace {

// The anti-aliased activation: upsample 2x -> SnakeBeta -> downsample 2x
// (dac_alias_free_act.py + dac_alias_free_resample.py).
struct AliasFreeActivation {
  int64_t ratio = 2;
  int64_t kernel_size = 12;
  std::vector<float> up_filter;
  std::vector<float> down_filter;

  void Build() {
    up_filter = MiniMaxH3KaiserSincFilter1d(0.5 / static_cast<double>(ratio),
                                            0.6 / static_cast<double>(ratio), kernel_size);
    down_filter = up_filter;  // same cutoff/half_width/kernel for ratio 2
  }

  std::vector<float> Apply(const std::vector<float>& in, int64_t channels, int64_t in_len,
                           const std::vector<float>& alpha, const std::vector<float>& beta,
                           bool logscale, int64_t* out_len) const {
    // --- UpSample1d ---
    const int64_t pad = kernel_size / ratio - 1;
    const int64_t pad_left = pad * ratio + (kernel_size - ratio) / 2;
    const int64_t pad_right = pad * ratio + (kernel_size - ratio + 1) / 2;
    int64_t padded_len = 0;
    const std::vector<float> padded = PadReplicate(in, channels, in_len, pad, pad, &padded_len);
    // Depthwise transposed conv: filter.expand(C, -1, -1) => weight [C, 1, K].
    std::vector<float> up_weight(static_cast<size_t>(channels * kernel_size));
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t k = 0; k < kernel_size; ++k) {
        up_weight[static_cast<size_t>(c * kernel_size + k)] = up_filter[static_cast<size_t>(k)];
      }
    }
    int64_t up_len = 0;
    std::vector<float> up = ConvTranspose1d(padded, channels, padded_len, up_weight, nullptr,
                                            channels, kernel_size, ratio, /*padding=*/0,
                                            /*groups=*/channels, &up_len);
    for (float& value : up) value *= static_cast<float>(ratio);
    // x[..., pad_left : -pad_right]
    const int64_t trimmed_len = up_len - pad_left - pad_right;
    VT_CHECK(trimmed_len > 0, "minimax_h3 audio vae: upsample trim emptied the signal");
    std::vector<float> trimmed(static_cast<size_t>(channels * trimmed_len));
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < trimmed_len; ++t) {
        trimmed[static_cast<size_t>(c * trimmed_len + t)] =
            up[static_cast<size_t>(c * up_len + pad_left + t)];
      }
    }

    // --- SnakeBeta ---
    SnakeBeta(trimmed, channels, trimmed_len, alpha, beta, logscale);

    // --- DownSample1d (LowPassFilter1d, stride = ratio, replicate padding) ---
    const bool even = (kernel_size % 2) == 0;
    const int64_t lp_left = kernel_size / 2 - (even ? 1 : 0);
    const int64_t lp_right = kernel_size / 2;
    int64_t lp_padded_len = 0;
    const std::vector<float> lp_padded =
        PadReplicate(trimmed, channels, trimmed_len, lp_left, lp_right, &lp_padded_len);
    std::vector<float> down_weight(static_cast<size_t>(channels * kernel_size));
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t k = 0; k < kernel_size; ++k) {
        down_weight[static_cast<size_t>(c * kernel_size + k)] = down_filter[static_cast<size_t>(k)];
      }
    }
    return Conv1d(lp_padded, channels, lp_padded_len, down_weight, nullptr, channels, kernel_size,
                  /*stride=*/ratio, /*dilation=*/1, /*groups=*/channels, out_len);
  }
};

int64_t GetPadding(int64_t kernel_size, int64_t dilation) {
  return (kernel_size * dilation - dilation) / 2;
}

}  // namespace

// BigVGAN.forward (dac_bigvgan.py:170-195) preceded by DacAudioVAE.dec_in_proj.
std::vector<float> MiniMaxH3AudioVaeDecode(const MiniMaxH3AudioVaeConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t frames,
                                           int64_t* out_samples) {
  const int64_t num_upsamples = static_cast<int64_t>(config.upsample_rates.size());
  const int64_t num_kernels = static_cast<int64_t>(config.resblock_kernel_sizes.size());
  VT_CHECK(num_upsamples > 0 && num_kernels > 0, "minimax_h3 audio vae: empty decoder config");
  VT_CHECK(static_cast<int64_t>(config.upsample_kernel_sizes.size()) == num_upsamples,
           "minimax_h3 audio vae: upsample rates/kernels length mismatch");
  VT_CHECK(static_cast<int64_t>(config.resblock_dilation_sizes.size()) == num_kernels,
           "minimax_h3 audio vae: resblock kernels/dilations length mismatch");
  VT_CHECK(static_cast<int64_t>(latent.size()) == config.num_mels * frames,
           "minimax_h3 audio vae: latent size does not match [num_mels, frames]");

  AliasFreeActivation act;
  act.Build();

  auto conv_weight = [&](const std::string& prefix, int64_t out_channels) {
    return MiniMaxH3MaterializeWeightNorm(weights.Get(prefix + ".parametrizations.weight.original0"),
                                          weights.Get(prefix + ".parametrizations.weight.original1"),
                                          out_channels);
  };

  // --- conv_pre: Conv1d(num_mels -> upsample_initial_channel, k=7, padding=3) ---
  int64_t channels = config.upsample_initial_channel;
  int64_t length = 0;
  std::vector<float> x;
  {
    int64_t padded_len = 0;
    std::vector<float> padded(static_cast<size_t>(config.num_mels * (frames + 6)), 0.0f);
    for (int64_t c = 0; c < config.num_mels; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        padded[static_cast<size_t>(c * (frames + 6) + 3 + t)] =
            latent[static_cast<size_t>(c * frames + t)];
      }
    }
    padded_len = frames + 6;  // zero padding 3 on each side
    const std::vector<float> w = conv_weight("conv_pre", channels);
    const std::vector<float>& b = weights.Get("conv_pre.bias");
    x = Conv1d(padded, config.num_mels, padded_len, w, &b, channels, 7, 1, 1, 1, &length);
  }

  // --- upsample stages ---
  for (int64_t i = 0; i < num_upsamples; ++i) {
    const int64_t rate = config.upsample_rates[static_cast<size_t>(i)];
    const int64_t kernel = config.upsample_kernel_sizes[static_cast<size_t>(i)];
    const int64_t out_channels = config.upsample_initial_channel / (int64_t{1} << (i + 1));
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    // ConvTranspose1d weight is [in, out/groups, k] -> weight-norm dim 0 is IN.
    const std::vector<float> w = conv_weight(prefix, channels);
    const std::vector<float>& b = weights.Get(prefix + ".bias");
    int64_t up_len = 0;
    x = ConvTranspose1d(x, channels, length, w, &b, out_channels, kernel, rate,
                        /*padding=*/(kernel - rate) / 2, /*groups=*/1, &up_len);
    channels = out_channels;
    length = up_len;

    // --- the num_kernels AMPBlock1s, AVERAGED ---
    std::vector<float> sum(static_cast<size_t>(channels * length), 0.0f);
    for (int64_t j = 0; j < num_kernels; ++j) {
      const int64_t kernel_size = config.resblock_kernel_sizes[static_cast<size_t>(j)];
      const std::vector<int64_t>& dilations = config.resblock_dilation_sizes[static_cast<size_t>(j)];
      const std::string block = "resblocks." + std::to_string(i * num_kernels + j);
      std::vector<float> h = x;
      for (size_t d = 0; d < dilations.size(); ++d) {
        const int64_t dilation = dilations[d];
        const std::string c1 = block + ".convs1." + std::to_string(d);
        const std::string c2 = block + ".convs2." + std::to_string(d);
        const std::string a1 = block + ".activations." + std::to_string(2 * d);
        const std::string a2 = block + ".activations." + std::to_string(2 * d + 1);

        int64_t act_len = 0;
        std::vector<float> xt = act.Apply(h, channels, length, weights.Get(a1 + ".act.alpha"),
                                          weights.Get(a1 + ".act.beta"), config.snake_logscale,
                                          &act_len);
        VT_CHECK(act_len == length, "minimax_h3 audio vae: anti-aliased activation changed length");
        int64_t padded_len = 0;
        const int64_t pad1 = GetPadding(kernel_size, dilation);
        std::vector<float> padded(static_cast<size_t>(channels * (length + 2 * pad1)), 0.0f);
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t t = 0; t < length; ++t) {
            padded[static_cast<size_t>(c * (length + 2 * pad1) + pad1 + t)] =
                xt[static_cast<size_t>(c * length + t)];
          }
        }
        padded_len = length + 2 * pad1;
        int64_t conv_len = 0;
        xt = Conv1d(padded, channels, padded_len, conv_weight(c1, channels),
                    &weights.Get(c1 + ".bias"), channels, kernel_size, 1, dilation, 1, &conv_len);

        xt = act.Apply(xt, channels, conv_len, weights.Get(a2 + ".act.alpha"),
                       weights.Get(a2 + ".act.beta"), config.snake_logscale, &act_len);
        const int64_t pad2 = GetPadding(kernel_size, 1);
        std::vector<float> padded2(static_cast<size_t>(channels * (act_len + 2 * pad2)), 0.0f);
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t t = 0; t < act_len; ++t) {
            padded2[static_cast<size_t>(c * (act_len + 2 * pad2) + pad2 + t)] =
                xt[static_cast<size_t>(c * act_len + t)];
          }
        }
        int64_t conv2_len = 0;
        xt = Conv1d(padded2, channels, act_len + 2 * pad2, conv_weight(c2, channels),
                    &weights.Get(c2 + ".bias"), channels, kernel_size, 1, 1, 1, &conv2_len);
        VT_CHECK(conv2_len == length, "minimax_h3 audio vae: resblock changed the sequence length");
        for (size_t n = 0; n < h.size(); ++n) h[n] += xt[n];
      }
      for (size_t n = 0; n < sum.size(); ++n) sum[n] += h[n];
    }
    for (float& value : sum) value /= static_cast<float>(num_kernels);
    x.swap(sum);
  }

  // --- post activation + conv_post + bound ---
  {
    int64_t act_len = 0;
    x = act.Apply(x, channels, length, weights.Get("activation_post.act.alpha"),
                  weights.Get("activation_post.act.beta"), config.snake_logscale, &act_len);
    length = act_len;
  }
  {
    const int64_t pad = 3;
    std::vector<float> padded(static_cast<size_t>(channels * (length + 2 * pad)), 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < length; ++t) {
        padded[static_cast<size_t>(c * (length + 2 * pad) + pad + t)] =
            x[static_cast<size_t>(c * length + t)];
      }
    }
    const std::vector<float> w = conv_weight("conv_post", 1);
    const std::vector<float>* b = nullptr;
    std::vector<float> bias_storage;
    if (config.use_bias_at_final) {
      bias_storage = weights.Get("conv_post.bias");
      b = &bias_storage;
    }
    int64_t final_len = 0;
    x = Conv1d(padded, channels, length + 2 * pad, w, b, 1, 7, 1, 1, 1, &final_len);
    length = final_len;
  }
  // H3 sets use_tanh_at_final=false, so the output is CLAMPED, not squashed.
  for (float& value : x) {
    value = config.use_tanh_at_final ? std::tanh(value) : std::min(1.0f, std::max(-1.0f, value));
  }
  *out_samples = length;
  return x;
}

}  // namespace vllm
