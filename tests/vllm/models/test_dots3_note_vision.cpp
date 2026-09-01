// dots3-note W6a — the DENSE vision tower, against an INDEPENDENT
// double-precision reference (#2512, `.agents/specs/dots3-note.md` §4.11).
//
// WHAT THIS GATE IS, IN ITS OWN WORDS. §6.4 of the spec records option B,
// decided 2026-08-15: the checkpoint is 298.67 GB fp8 / 576.89 GB bf16 against
// 119-122 GiB hosts, so vLLM cannot be run on it on any hardware this project
// owns and NO DENOMINATOR EXISTS. This is therefore a CONSISTENCY gate. It
// establishes that two implementations of the same formula agree. It does NOT
// establish that either matches vLLM, and no performance number is claimable on
// any axis while B holds. Nothing below claims otherwise.
//
// The reference is written from `vllm/models/dots3_note/nvidia/vision.py` and
// `nvidia/vision_attention.py` read in `~/_git/vllm` at **`9035151d6`** — the
// merge of vllm#51255 — and shares NO helper with the implementation. It is a
// scalar `double` loop with its own GEMM, its own softmax, its own rope and its
// own norms; the implementation is `vt::MatmulBT` / `vt::RmsNorm` /
// `vt::RopeFromCache` / `vt::AttentionDenseFlash` over bf16 device buffers
// through the shared seams. Every anchor names that SHA because upstream has
// already moved under this row: `vision_attention.py` is 477 lines at
// `9035151d6` and 494 at vLLM `main` `7a100bb61`.
//
// THE ONE FORMULA DIFFERENCE, and why the reference does not copy it. Upstream's
// `RMSNorm.forward` (`vision.py:112-114`) casts the normalized value back to the
// ACTIVATION dtype before multiplying by the weight; `vt::RmsNorm` keeps f32
// through that multiply. At infinite precision the two are the same function,
// so the reference — which is double throughout — is the algebra BOTH implement
// and the tolerance below covers our bf16 storage AND upstream's intermediate
// cast together. Copying the cast into the reference would make the reference
// agree with a rounding choice instead of with the maths.
#include "vllm/model_executor/models/dots3_note_vision.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <system_error>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dots3_note_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using dots3_tiny::TinyCheckpoint;
using dots3_tiny::TinySpec;
using vllm::Dots3NoteVisionForward;
using vllm::Dots3NoteVisionParams;
using vllm::Dots3NoteVisionPosIds;
using vllm::Dots3NoteVisionRefusal;
using vllm::Dots3NoteVisionWeights;
using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ParseDots3NoteVisionParams;

std::string FixtureDir() { return DOTS3_NOTE_CKPT_FIXTURE_DIR; }

nlohmann::json ReleasedConfigDoc() {
  std::ifstream in(FixtureDir() + "/config.json");
  REQUIRE_MESSAGE(in.good(), "cannot open " << FixtureDir() << "/config.json");
  nlohmann::json j;
  in >> j;
  return j;
}

// A throwaway `config.json` holding an arbitrary document, so a case can drive
// the REAL `LoadHfConfig` -> `ParseDots3NoteVisionParams` path rather than
// building an `HfConfig` by hand.
class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    static int counter = 0;
    static const unsigned salt = std::random_device{}();
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_vision_cfg_" + std::to_string(salt) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json", std::ios::binary) << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

Dots3NoteVisionParams ParseDoc(const nlohmann::json& doc) {
  const TempConfig cfg(doc);
  return ParseDots3NoteVisionParams(LoadHfConfig(cfg.path()));
}

// ═══════════════════════════════════════════════════════════════════════════
// THE INDEPENDENT REFERENCE. Every line below is written from the upstream
// source at `9035151d6`; it calls nothing the implementation calls.
// ═══════════════════════════════════════════════════════════════════════════
namespace ref {

// out[M,N] = x[M,K] @ w[N,K]^T (+ bias). Plain triple loop, double accumulator.
std::vector<double> Linear(const std::vector<double>& x,
                           const std::vector<double>& w,
                           const std::vector<double>* bias, int64_t M,
                           int64_t K, int64_t N) {
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(n)] : 0.0;
      for (int64_t k = 0; k < K; ++k) {
        acc += x[static_cast<size_t>(m * K + k)] *
               w[static_cast<size_t>(n * K + k)];
      }
      out[static_cast<size_t>(m * N + n)] = acc;
    }
  }
  return out;
}

// `RMSNorm.forward` (vision.py:107-124) and `_RMSNorm` (vision_attention.py:97-110),
// which are the same function: `x * rsqrt(mean(x^2) + eps) * weight`. The
// intermediate `.type_as(x)` is a no-op in double (see the file header).
std::vector<double> Rms(const std::vector<double>& x,
                        const std::vector<double>& w, int64_t rows, int64_t dim,
                        double eps) {
  std::vector<double> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double v = x[static_cast<size_t>(r * dim + c)];
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (int64_t c = 0; c < dim; ++c) {
      out[static_cast<size_t>(r * dim + c)] =
          x[static_cast<size_t>(r * dim + c)] * inv * w[static_cast<size_t>(c)];
    }
  }
  return out;
}

// `nn.LayerNorm` with weight and bias, the adapter's `ln_q` (vision.py:466).
std::vector<double> LayerNorm(const std::vector<double>& x,
                              const std::vector<double>& w,
                              const std::vector<double>& b, int64_t rows,
                              int64_t dim, double eps) {
  std::vector<double> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < dim; ++c) mean += x[static_cast<size_t>(r * dim + c)];
    mean /= static_cast<double>(dim);
    double var = 0.0;
    for (int64_t c = 0; c < dim; ++c) {
      const double d = x[static_cast<size_t>(r * dim + c)] - mean;
      var += d * d;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int64_t c = 0; c < dim; ++c) {
      out[static_cast<size_t>(r * dim + c)] =
          (x[static_cast<size_t>(r * dim + c)] - mean) * inv *
              w[static_cast<size_t>(c)] +
          b[static_cast<size_t>(c)];
    }
  }
  return out;
}

// `get_pos_ids_by_grid` (vision.py:566-603), written straight from the reshape /
// permute / flatten upstream spells, rather than from the loop the
// implementation collapsed it into.
std::vector<std::array<int64_t, 2>> PosIds(int64_t t, int64_t h, int64_t w,
                                           int64_t rope_merge) {
  // hpos[i][j] = i ; wpos[i][j] = j, both [h, w]
  std::vector<int64_t> hp(static_cast<size_t>(h * w));
  std::vector<int64_t> wp(static_cast<size_t>(h * w));
  for (int64_t i = 0; i < h; ++i) {
    for (int64_t j = 0; j < w; ++j) {
      hp[static_cast<size_t>(i * w + j)] = i;
      wp[static_cast<size_t>(i * w + j)] = j;
    }
  }
  // reshape(h/m, m, w/m, m) then permute(0, 2, 1, 3) then flatten.
  const int64_t m = rope_merge;
  const auto regroup = [&](const std::vector<int64_t>& src) {
    std::vector<int64_t> out;
    out.reserve(src.size());
    for (int64_t a = 0; a < h / m; ++a) {
      for (int64_t c = 0; c < w / m; ++c) {
        for (int64_t b = 0; b < m; ++b) {
          for (int64_t dd = 0; dd < m; ++dd) {
            // index into the [h/m, m, w/m, m] view: (a, b, c, dd)
            const int64_t row = a * m + b;
            const int64_t col = c * m + dd;
            out.push_back(src[static_cast<size_t>(row * w + col)]);
          }
        }
      }
    }
    return out;
  };
  const std::vector<int64_t> hf = regroup(hp);
  const std::vector<int64_t> wf = regroup(wp);
  std::vector<std::array<int64_t, 2>> out;
  for (int64_t f = 0; f < t; ++f) {
    for (size_t i = 0; i < hf.size(); ++i) out.push_back({hf[i], wf[i]});
  }
  return out;
}

// `apply_rotary_pos_emb_vision` (vision_attention.py:33-49) applied to ONE
// [L, nh, hd] tensor, with the frequency table built as
// `VisionRotaryEmbedding(hd // 2)` (vision_attention.py:60-89).
void ApplyRope(std::vector<double>* x,
               const std::vector<std::array<int64_t, 2>>& pos, int64_t L,
               int64_t nh, int64_t hd) {
  const int64_t dim = hd / 2;        // the table's own `dim`
  const int64_t nf = dim / 2;        // frequencies per spatial axis
  for (int64_t l = 0; l < L; ++l) {
    // freqs[l] is [2, nf] flattened to [dim]; cos/sin are then REPEATED to hd
    // as [f | f] (`.repeat(1, 1, 2)` at vision_attention.py:46-47).
    std::vector<double> c(static_cast<size_t>(hd)), s(static_cast<size_t>(hd));
    for (int64_t axis = 0; axis < 2; ++axis) {
      for (int64_t i = 0; i < nf; ++i) {
        const double invf =
            1.0 / std::pow(10000.0, static_cast<double>(2 * i) /
                                        static_cast<double>(dim));
        const double ang =
            static_cast<double>(pos[static_cast<size_t>(l)][
                static_cast<size_t>(axis)]) * invf;
        const int64_t k = axis * nf + i;
        c[static_cast<size_t>(k)] = std::cos(ang);
        c[static_cast<size_t>(dim + k)] = std::cos(ang);
        s[static_cast<size_t>(k)] = std::sin(ang);
        s[static_cast<size_t>(dim + k)] = std::sin(ang);
      }
    }
    for (int64_t h = 0; h < nh; ++h) {
      const size_t base = static_cast<size_t>((l * nh + h) * hd);
      std::vector<double> in(x->begin() + static_cast<ptrdiff_t>(base),
                             x->begin() + static_cast<ptrdiff_t>(base + hd));
      for (int64_t d = 0; d < hd; ++d) {
        // rotate_half: (-x2, x1) over the two halves (vision_attention.py:33-36)
        const double rot = d < dim ? -in[static_cast<size_t>(d + dim)]
                                   : in[static_cast<size_t>(d - dim)];
        (*x)[base + static_cast<size_t>(d)] =
            in[static_cast<size_t>(d)] * c[static_cast<size_t>(d)] +
            rot * s[static_cast<size_t>(d)];
      }
    }
  }
}

// `VisionAttentionV2.forward` (vision_attention.py:210-239) for ONE window:
// bidirectional, scaled by 1/sqrt(head_dim), softmax in f32 (here, double).
std::vector<double> Attention(const std::vector<double>& q,
                              const std::vector<double>& k,
                              const std::vector<double>& v, int64_t L,
                              int64_t nh, int64_t hd) {
  std::vector<double> out(static_cast<size_t>(L * nh * hd), 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  for (int64_t h = 0; h < nh; ++h) {
    for (int64_t i = 0; i < L; ++i) {
      std::vector<double> sc(static_cast<size_t>(L));
      double mx = -1e300;
      for (int64_t j = 0; j < L; ++j) {
        double acc = 0.0;
        for (int64_t d = 0; d < hd; ++d) {
          acc += q[static_cast<size_t>((i * nh + h) * hd + d)] *
                 k[static_cast<size_t>((j * nh + h) * hd + d)];
        }
        sc[static_cast<size_t>(j)] = acc * scale;
        mx = std::max(mx, sc[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int64_t j = 0; j < L; ++j) {
        sc[static_cast<size_t>(j)] = std::exp(sc[static_cast<size_t>(j)] - mx);
        sum += sc[static_cast<size_t>(j)];
      }
      for (int64_t j = 0; j < L; ++j) {
        const double p = sc[static_cast<size_t>(j)] / sum;
        for (int64_t d = 0; d < hd; ++d) {
          out[static_cast<size_t>((i * nh + h) * hd + d)] +=
              p * v[static_cast<size_t>((j * nh + h) * hd + d)];
        }
      }
    }
  }
  return out;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }
// `nn.GELU()` with no `approximate=` is the EXACT erf gelu.
double GeluErf(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// `DotsMoEVitModel.forward` (vision.py:634-677), the all-DENSE path.
std::vector<double> Tower(const TinySpec& s, const TinyCheckpoint& ck,
                          const std::vector<double>& pixels, int64_t t,
                          int64_t gh, int64_t gw) {
  const int64_t E = s.v_embed, nh = s.v_heads, hd = s.v_head_dim();
  const int64_t L = t * gh * gw, P = s.v_patch_row(), VI = s.v_inter;
  const double eps = s.v_rms_eps;
  const std::string vp = "vision_encoder.";

  // patch_embed: the Conv2d over one non-overlapping patch IS a Linear over the
  // flattened patch row, then RMSNorm (vision.py:317-331).
  std::vector<double> hidden =
      Linear(pixels, ck.value_of(vp + "patch_embed.proj.weight"),
             &ck.value_of(vp + "patch_embed.proj.bias"), L, P, E);
  hidden = Rms(hidden, ck.value_of(vp + "patch_embed.norm.weight"), L, E, eps);

  const int64_t rope_merge =
      s.v_pre_pixel_shuffle ? (s.v_merge > 1 ? s.v_merge : 2) : 1;
  const std::vector<std::array<int64_t, 2>> pos = PosIds(t, gh, gw, rope_merge);

  for (int64_t b = 0; b < s.v_layers; ++b) {
    const std::string pre = vp + "blocks." + std::to_string(b) + ".";
    // `apply_vision_attention_residual`: hidden + attn(norm_1(hidden))
    const std::vector<double> n1 =
        Rms(hidden, ck.value_of(pre + "norm_1.weight"), L, E, eps);
    const std::vector<double> qkv =
        Linear(n1, ck.value_of(pre + "attn.qkv.weight"), nullptr, L, E, 3 * E);
    // `.reshape(L, 3, nh, -1).permute(1, 0, 2, 3).unbind(0)`
    std::vector<double> qh(static_cast<size_t>(L * E));
    std::vector<double> kh(static_cast<size_t>(L * E));
    std::vector<double> vh(static_cast<size_t>(L * E));
    for (int64_t l = 0; l < L; ++l) {
      for (int64_t c = 0; c < E; ++c) {
        qh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + c)];
        kh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + E + c)];
        vh[static_cast<size_t>(l * E + c)] =
            qkv[static_cast<size_t>(l * 3 * E + 2 * E + c)];
      }
    }
    // Q/K NORM FIRST, ROPE SECOND (vision_attention.py:161-165). The order is
    // silent when swapped: same shapes, same magnitudes, different numbers.
    qh = Rms(qh, ck.value_of(pre + "attn.q_norm.weight"), L * nh, hd, eps);
    kh = Rms(kh, ck.value_of(pre + "attn.k_norm.weight"), L * nh, hd, eps);
    ApplyRope(&qh, pos, L, nh, hd);
    ApplyRope(&kh, pos, L, nh, hd);
    const std::vector<double> ao = Attention(qh, kh, vh, L, nh, hd);
    const std::vector<double> proj =
        Linear(ao, ck.value_of(pre + "attn.proj.weight"), nullptr, L, E, E);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += proj[i];

    // hidden + mlp(norm_2(hidden)), `fc2(silu(fc1(x)) * fc3(x))`
    const std::vector<double> n2 =
        Rms(hidden, ck.value_of(pre + "norm_2.weight"), L, E, eps);
    const std::vector<double> g =
        Linear(n2, ck.value_of(pre + "mlp.fc1.weight"), nullptr, L, E, VI);
    const std::vector<double> u =
        Linear(n2, ck.value_of(pre + "mlp.fc3.weight"), nullptr, L, E, VI);
    std::vector<double> act(g.size());
    for (size_t i = 0; i < g.size(); ++i) act[i] = Silu(g[i]) * u[i];
    const std::vector<double> down =
        Linear(act, ck.value_of(pre + "mlp.fc2.weight"), nullptr, L, VI, E);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += down[i];
  }

  if (s.v_post_norm) {
    hidden = Rms(hidden, ck.value_of(vp + "post_trunk_norm.weight"), L, E, eps);
  }

  // `PatchMergerAdapter.forward` (vision.py:474-484): ln_q over the per-token
  // dim at a HARD-CODED eps of 1e-6, then `reshape(-1, merged_dim)`, then the
  // two-layer MLP with an exact-erf GELU between.
  const std::vector<double> lnq =
      LayerNorm(hidden, ck.value_of(vp + "adapter.ln_q.weight"),
                ck.value_of(vp + "adapter.ln_q.bias"), L, E, 1e-6);
  const int64_t M = s.v_merged_dim(), O = s.v_adapter_out();
  const int64_t Nm = L * E / M;
  std::vector<double> f1 =
      Linear(lnq, ck.value_of(vp + "adapter.mlp.0.weight"),
             &ck.value_of(vp + "adapter.mlp.0.bias"), Nm, M, M);
  for (double& x : f1) x = GeluErf(x);
  return Linear(f1, ck.value_of(vp + "adapter.mlp.2.weight"),
                &ck.value_of(vp + "adapter.mlp.2.bias"), Nm, M, O);
}

}  // namespace ref

// The whole loaded model, built once per case through the REAL registry over the
// REAL loader — never a hand-built weights struct.
struct Bench {
  TinySpec spec;
  TinyCheckpoint ckpt;
  HfConfig config;
  std::unique_ptr<vllm::LoadedModel> model;

  explicit Bench(TinySpec s = TinySpec{})
      : spec(s),
        ckpt(FixtureDir(), s),
        config(LoadHfConfig(ckpt.config_path())) {
    const std::vector<std::string> arch{"Dots3NoteForCausalLM"};
    const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.weights_path()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    model = reg.factory->load_weights(reg, config, source);
  }
};

// The bf16 patch rows the tower consumes, widened to double, so the reference
// and the implementation start from the SAME bytes and the comparison measures
// the forward rather than the processor.
std::vector<double> WidenBf16(const std::vector<uint16_t>& bits) {
  std::vector<double> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i)
    out[i] = static_cast<double>(vt::BF16ToF32(bits[i]));
  return out;
}

double MaxAbs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE RELEASED `vision_config` resolves to the geometry the checkpoint's own
//    shard index carries. Every number here was read from the COMMITTED
//    fixture, not from the issue text.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the RELEASED vision_config resolves to the measured geometry") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  REQUIRE(v.present);
  CHECK(v.embed_dim == 1536);
  CHECK(v.num_attention_heads == 24);
  CHECK(v.head_dim() == 64);
  CHECK(v.num_hidden_layers == 42);
  CHECK(v.intermediate_size == 4224);
  CHECK(v.moe_intermediate_size == 2112);
  CHECK(v.patch_size == 14);
  CHECK(v.temporal_patch_size == 1);
  CHECK(v.spatial_merge_size == 2);
  CHECK(v.rms_norm_eps == doctest::Approx(1e-5));
  CHECK_FALSE(v.use_bias);
  CHECK(v.use_qk_norm);
  CHECK_FALSE(v.is_causal);
  CHECK(v.post_norm);

  // THE TWO FLAGS #2512's PROSE CONFLATES, asserted apart. `adapter_type` is
  // `patch_merger`, which is upstream's name for the arm that SKIPS the
  // pixel-shuffle permutation (vision.py:441-449 @ 9035151d6); the 2x2
  // regrouping did not disappear, `pre_pixel_shuffle` moved it into the
  // PREPROCESSOR and into the RoPE. The issue's own tensor inventory —
  // `adapter.{ln_q, mlp.0, mlp.2}` — is `PatchMergerAdapter`'s state dict and
  // agrees with this; `PixelShuffleAdapter` spells its parameters
  // `proj.0`/`proj.1`/`proj.3` (vision.py:397-406). See spec §4.11.1.
  CHECK(v.adapter_type == "patch_merger");
  CHECK(v.pre_pixel_shuffle);
  CHECK(v.adapter_in_dim == 1536);
  CHECK(v.adapter_out_dim == 5120);
  CHECK(v.adapter_merge_size == 2);
  CHECK(v.merged_dim() == 6144);  // 4 x 1536

  // 25 dense + 17 MoE, counted from `pyramid_num_routed` rather than assumed:
  // `is_moe` is `> 0` (vision.py:346-350), so the leading -1s are DENSE.
  REQUIRE(v.pyramid_num_routed.size() == 42u);
  CHECK(v.num_dense_blocks() == 25);
  CHECK(v.num_moe_blocks() == 17);
  CHECK(v.is_moe_block(24) == false);
  CHECK(v.is_moe_block(25) == true);
  CHECK(v.pyramid_num_routed[25] == 4);
  CHECK(v.pyramid_num_routed[41] == 64);
  int64_t experts = 0;
  for (int64_t i = 25; i < 42; ++i) experts += v.pyramid_num_routed[static_cast<size_t>(i)];
  // 1960 of the 2195 vision tensors: 17 x 8 block tensors + 608 x 3 experts.
  CHECK(experts == 608);
  CHECK(17 * 8 + experts * 3 == 1960);
}

// ---------------------------------------------------------------------------
// 2. THE RELEASED CHECKPOINT STILL REFUSES, by name, and names W6b.
//    This is the row's established pattern — W3 refused the LANGUAGE tower's
//    MoE for four bricks before W5 lifted it — not a new exception.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the RELEASED vision tower REFUSES BY NAME, and names W6b") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  const std::string why = Dots3NoteVisionRefusal(v, "", {});
  INFO("refusal: ", why);
  REQUIRE_FALSE(why.empty());
  CHECK(why.find("W6b") != std::string::npos);
  CHECK(why.find("25") != std::string::npos);   // the FIRST routed block
  CHECK(why.find("17") != std::string::npos);   // how many there are
  CHECK(why.find("MoE") != std::string::npos);
  // The message names what to build, not that something is missing.
  CHECK(why.find("gate_weight") != std::string::npos);
  CHECK(why.find("router_bias") != std::string::npos);
}

TEST_CASE("dots3-note W6a: every unported vision shape refuses BY NAME with its brick") {
  const nlohmann::json released = ReleasedConfigDoc();

  SUBCASE("an all-DENSE tower is ACCEPTED — the premise of every case below") {
    nlohmann::json d = released;
    for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
    CHECK(Dots3NoteVisionRefusal(ParseDoc(d), "", {}).empty());
  }
  SUBCASE("the BLOCKWISE-FP8 arm is W9, and it outranks the MoE refusal") {
    nlohmann::json d = released;
    for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
    const std::string why =
        Dots3NoteVisionRefusal(ParseDoc(d), "fp8", {128, 128});
    INFO(why);
    CHECK(why.find("W9") != std::string::npos);
    CHECK(why.find("weight_block_size") != std::string::npos);
  }
  SUBCASE("`pixel_shuffle_mlp` is a DIFFERENT token order and is refused") {
    nlohmann::json d = released;
    for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
    d["vision_config"]["adapter_type"] = "pixel_shuffle_mlp";
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK(why.find("pixel_shuffle_mlp") != std::string::npos);
    CHECK(why.find("patch_merger") != std::string::npos);
  }
  SUBCASE("an UNKNOWN adapter refuses at PARSE, not at load") {
    nlohmann::json d = released;
    d["vision_config"]["adapter_type"] = "something_else";
    CHECK_THROWS_AS((void)ParseDoc(d), std::runtime_error);
  }
  SUBCASE("`temporal_patch_size != 1` is the VIDEO arm and names W7") {
    nlohmann::json d = released;
    for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
    d["vision_config"]["temporal_patch_size"] = 2;
    const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
    INFO(why);
    CHECK(why.find("W7") != std::string::npos);
  }
  SUBCASE("`use_bias`, `use_qk_norm`, `is_causal` and `post_norm` each refuse") {
    for (const char* key : {"use_bias", "use_qk_norm", "is_causal", "post_norm"}) {
      nlohmann::json d = released;
      for (auto& e : d["vision_config"]["pyramid_num_routed"]) e = -1;
      d["vision_config"][key] = !d["vision_config"][key].get<bool>();
      const std::string why = Dots3NoteVisionRefusal(ParseDoc(d), "", {});
      INFO("key ", key, " -> ", why);
      CHECK_MESSAGE(!why.empty(), "flipping " << key << " was accepted");
      CHECK(why.find(key) != std::string::npos);
    }
  }
  SUBCASE("a checkpoint with NO vision_config refuses, naming the absence") {
    nlohmann::json d = released;
    d.erase("vision_config");
    const Dots3NoteVisionParams v = ParseDoc(d);
    CHECK_FALSE(v.present);
    const std::string why = Dots3NoteVisionRefusal(v, "", {});
    CHECK(why.find("vision_config") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// 3. THE 235 DENSE TENSORS, counted against the COMMITTED shard index.
//    "Nothing was left over" is also true of a map that claimed the MoE blocks
//    as dense, so the counts are asserted BY NUMBER and cross-checked against
//    the released index's own names.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the DENSE arm claims 235 of the released tower's 2195 tensors") {
  const Dots3NoteVisionParams v = ParseDoc(ReleasedConfigDoc());
  const std::vector<vllm::Dots3NoteTensor> claimed =
      vllm::EnumerateDots3NoteVisionTensors(v);
  CHECK(claimed.size() == 235u);

  // 25 blocks x 9, plus 3 patch_embed, 1 post_trunk_norm, 6 adapter.
  CHECK(25 * 9 + 3 + 1 + 6 == 235);

  std::set<std::string> names;
  for (const vllm::Dots3NoteTensor& t : claimed) {
    CHECK_MESSAGE(!t.consumer.empty(), t.name << " has no named consumer");
    CHECK_MESSAGE(names.insert(t.name).second, t.name << " is claimed twice");
    CHECK(t.name.rfind("vision_encoder.", 0) == 0);
  }

  // NOT ONE MoE block's tensor is claimed. A map that walked all 42 blocks
  // would claim 42 x 9 = 378 names, of which 153 do not exist on disk — and the
  // load would then refuse for the wrong reason.
  for (int64_t b = 25; b < 42; ++b) {
    const std::string pre = "vision_encoder.blocks." + std::to_string(b) + ".";
    for (const std::string& n : names) {
      CHECK_MESSAGE(n.rfind(pre, 0) != 0,
                    "the DENSE arm claims " << n << ", which is a W6b block");
    }
  }
  // ...and every DENSE block IS claimed, so a leading-run bug that stopped
  // early would show.
  for (int64_t b = 0; b < 25; ++b) {
    const std::string pre = "vision_encoder.blocks." + std::to_string(b) + ".";
    CHECK_MESSAGE(names.count(pre + "attn.qkv.weight") == 1,
                  "block " << b << " is dense and unclaimed");
    CHECK_MESSAGE(names.count(pre + "mlp.fc3.weight") == 1,
                  "block " << b << " is dense and its fc3 is unclaimed");
  }
  CHECK(names.count("vision_encoder.adapter.ln_q.bias") == 1);
  CHECK(names.count("vision_encoder.adapter.mlp.2.weight") == 1);
  CHECK(names.count("vision_encoder.post_trunk_norm.weight") == 1);
  CHECK(names.count("vision_encoder.patch_embed.proj.bias") == 1);
}

// ---------------------------------------------------------------------------
// 4. THE POSITION GRID. `pre_pixel_shuffle` selects between two DIFFERENT token
//    orders, and the flag is read by the processor AND by the tower — so a case
//    that only checked one order could not see the two disagreeing.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: pre_pixel_shuffle regroups the RoPE positions, and NOT setting it does not") {
  TinySpec s;
  const std::array<int64_t, 3> grid{1, 4, 4};

  s.v_pre_pixel_shuffle = true;
  Dots3NoteVisionParams grouped = ParseDoc(dots3_tiny::TinyConfigDoc(FixtureDir(), s));
  s.v_pre_pixel_shuffle = false;
  Dots3NoteVisionParams flat = ParseDoc(dots3_tiny::TinyConfigDoc(FixtureDir(), s));

  const auto ours_grouped = Dots3NoteVisionPosIds(grid, grouped);
  const auto ours_flat = Dots3NoteVisionPosIds(grid, flat);
  const auto ref_grouped = ref::PosIds(1, 4, 4, /*rope_merge=*/2);
  const auto ref_flat = ref::PosIds(1, 4, 4, /*rope_merge=*/1);

  REQUIRE(ours_grouped.size() == 16u);
  CHECK(ours_grouped == ref_grouped);
  CHECK(ours_flat == ref_flat);
  // THE PREMISE, asserted rather than assumed: the two orders really differ, so
  // agreeing with the wrong one is a detectable defect rather than a tie.
  CHECK(ours_grouped != ours_flat);
  // Spot the grouped layout by hand: token 1 is the 2x2 block's top-RIGHT, so
  // it is row 0 column 1, where the flat order would put row 0 column 1 too but
  // token 2 apart — flat token 2 is (0, 2), grouped token 2 is (1, 0).
  CHECK(ours_grouped[2] == std::array<int64_t, 2>{1, 0});
  CHECK(ours_flat[2] == std::array<int64_t, 2>{0, 2});
}

// ---------------------------------------------------------------------------
// 5. THE TOWER, against the independent double-precision reference.
//    THE CONSISTENCY GATE. It says two implementations agree; it does not say
//    either matches vLLM, because vLLM cannot be run on this model here
//    (spec §6.4 option B).
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the DENSE tower agrees with an INDEPENDENT double reference") {
  Bench bench;
  // The processor is the production one, so the patch rows the tower sees are
  // the rows a served request would produce.
  vllm::multimodal::Dots3NoteProcessorConfig pcfg =
      vllm::multimodal::LoadDots3NoteProcessorConfig(
          bench.ckpt.dir() + "/preprocessor_config.json",
          bench.ckpt.config_path(), "tiny-dots3");
  const vllm::multimodal::Dots3NoteImageProcessor proc(pcfg);
  const std::vector<uint8_t> rgb = dots3_tiny::FixtureImage(0);
  const vllm::multimodal::ImageKwargs kw =
      proc.ProcessImage(rgb.data(), dots3_tiny::kImageSide,
                        dots3_tiny::kImageSide);
  REQUIRE(kw.image_grid_thw[0] == 1);
  REQUIRE(kw.image_grid_thw[1] == 4);
  REQUIRE(kw.image_grid_thw[2] == 4);
  REQUIRE(kw.num_patches == 16);

  const Dots3NoteVisionParams v = ParseDots3NoteVisionParams(bench.config);
  REQUIRE(Dots3NoteVisionRefusal(v, "", {}).empty());
  Dots3NoteVisionWeights vw = vllm::MaterializeDots3NoteVision(
      [&] {
        std::vector<vllm::SafetensorsFile> shards;
        shards.push_back(
            vllm::SafetensorsFile::Open(bench.ckpt.weights_path()));
        return shards;
      }(),
      v);
  REQUIRE(vw.present);

  // THE MEMORY FORMAT (porting.md). A widened store passes every shape check
  // and every token gate while moving twice the bytes, and this row's W2 F1
  // fixture row already proves a re-typed tensor fires.
  CHECK(vw.patch_proj_w.dtype == vt::DType::kBF16);
  CHECK(vw.adapter_mlp2_w.dtype == vt::DType::kBF16);
  for (const auto& blk : vw.blocks) {
    CHECK(blk.qkv.dtype == vt::DType::kBF16);
    CHECK(blk.gate_up.dtype == vt::DType::kBF16);
  }
  // The merge is real: [2I, E], gate then up.
  REQUIRE(vw.blocks.size() == static_cast<size_t>(bench.spec.v_layers));
  CHECK(vw.blocks[0].gate_up.shape[0] == 2 * bench.spec.v_inter);
  CHECK(vw.blocks[0].gate_up.shape[1] == bench.spec.v_embed);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  const std::vector<float> ours = Dots3NoteVisionForward(
      kw.pixel_values_bf16, kw.image_grid_thw, vw, v, backend);

  const std::vector<double> want =
      ref::Tower(bench.spec, bench.ckpt, WidenBf16(kw.pixel_values_bf16), 1, 4, 4);
  REQUIRE(ours.size() == want.size());
  // FOUR merger rows (16 patches / 2x2), each in the TEXT hidden space.
  CHECK(ours.size() ==
        static_cast<size_t>(dots3_tiny::kExpectedImageTokens * bench.spec.hidden));

  double max_abs = 0.0;
  for (size_t i = 0; i < ours.size(); ++i)
    max_abs = std::max(max_abs, std::abs(static_cast<double>(ours[i]) - want[i]));
  const double scale = MaxAbs(want);
  const double rel = scale > 0.0 ? max_abs / scale : max_abs;
  MESSAGE("tower vs double reference: max |diff| ", max_abs, " over a scale of ",
          scale, " => relative ", rel);
  // THE BOUND, and where it comes from. The implementation stores every
  // activation and every weight in bf16 (8 mantissa bits, ~3.9e-3 relative) and
  // runs two blocks plus a 64-wide adapter GEMM over it; the reference is
  // double throughout and also absorbs upstream's intermediate `.type_as(x)`
  // cast, which is a no-op at infinite precision. The bound is set at a
  // MEASURED multiple of the observed deviation rather than at a round number,
  // and the mutation evidence in the spec is what proves it is still tight
  // enough to see a defect — a bound above the real error is a mute switch.
  // MEASURED 2026-09-01 on this fixture: max |diff| 0.0533 over a scale of
  // 6.314, i.e. 8.44e-3 relative. The bound is 0.02 — a 2.4x margin over the
  // observation, wide enough that a different libm or a different GEMM
  // reduction order does not red it, and tight enough that the mutations
  // recorded in spec §4.11 all exceed it.
  CHECK(rel < 0.02);
  // ...and the two are not trivially equal, which would mean one of them is
  // reading the other's answer.
  CHECK(scale > 1e-3);
}

// ---------------------------------------------------------------------------
// 6. THE PROCESSOR. `resized_size` is NOT `smart_resize`, the normalization is
//    PER CHANNEL, and the placeholder count follows from the grid.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the image processor mirrors upstream's own resize and normalization") {
  using vllm::multimodal::Dots3NoteResizedSize;
  // `factor = patch * merge` = 28 on the released geometry.
  SUBCASE("a conformant size is the identity") {
    const auto rs = Dots3NoteResizedSize(56, 84, 28, 16, 1 << 22);
    CHECK(rs[0] == 56);
    CHECK(rs[1] == 84);
  }
  SUBCASE("each side rounds INDEPENDENTLY to a multiple of factor") {
    const auto rs = Dots3NoteResizedSize(57, 83, 28, 16, 1 << 22);
    CHECK(rs[0] == 56);
    CHECK(rs[1] == 84);
  }
  SUBCASE("a side under factor/4 refuses, naming the bound") {
    CHECK_THROWS_AS((void)Dots3NoteResizedSize(6, 84, 28, 16, 1 << 22),
                    std::runtime_error);
  }
  SUBCASE("an aspect ratio over 200 refuses") {
    CHECK_THROWS_AS((void)Dots3NoteResizedSize(28, 28 * 300, 28, 16, 1 << 30),
                    std::runtime_error);
  }
  SUBCASE("the max-pixel budget shrinks BOTH sides by the same beta") {
    const auto rs = Dots3NoteResizedSize(280, 280, 28, 16, 28 * 28 * 4);
    CHECK(rs[0] % 28 == 0);
    CHECK(rs[1] % 28 == 0);
    CHECK(rs[0] * rs[1] <= 28 * 28 * 4);
  }

  // PER-CHANNEL normalization, which is the second thing that separates this
  // processor from Qwen3-VL's. The fixture's mean/std differ per channel, so a
  // processor that read only `image_mean[0]` would put red's statistics on
  // green and blue.
  TinySpec s;
  TinyCheckpoint ck(FixtureDir(), s);
  const vllm::multimodal::Dots3NoteProcessorConfig cfg =
      vllm::multimodal::LoadDots3NoteProcessorConfig(
          ck.dir() + "/preprocessor_config.json", ck.config_path(), "tiny");
  CHECK(cfg.image_mean[0] == doctest::Approx(0.5));
  CHECK(cfg.image_mean[1] == doctest::Approx(0.45));
  CHECK(cfg.image_mean[2] == doctest::Approx(0.4));
  CHECK(cfg.image_std[0] == doctest::Approx(0.25));
  CHECK(cfg.image_std[2] == doctest::Approx(0.35));
  // The three marker ids resolved from `added_tokens.json`.
  CHECK(cfg.image_token_id == dots3_tiny::kImgPadId);
  CHECK(cfg.image_start_token_id == dots3_tiny::kImgStartId);
  CHECK(cfg.image_end_token_id == dots3_tiny::kImgEndId);

  const vllm::multimodal::Dots3NoteImageProcessor proc(cfg);
  const std::vector<uint8_t> rgb = dots3_tiny::FixtureImage(0);
  const vllm::multimodal::ImageKwargs kw = proc.ProcessImage(
      rgb.data(), dots3_tiny::kImageSide, dots3_tiny::kImageSide);
  CHECK(kw.num_patches == 16);
  CHECK(kw.patch_feature_dim == s.v_patch_row());
  // The placeholder run: `prod(grid) // merge**2` (multimodal.py:151-155).
  CHECK(kw.num_patches / (s.v_merge * s.v_merge) ==
        dots3_tiny::kExpectedImageTokens);

  // One patch row, computed by hand against the pre_pixel_shuffle layout:
  // row 0 is the 2x2 block (0,0)'s sub-patch (0,0), i.e. source pixels
  // [0..1] x [0..1], and its channel-c element (ph, pw) is
  // (raw - mean_c/rescale) / (std_c/rescale).
  const int64_t patch = s.v_patch;
  for (int64_t c = 0; c < 3; ++c) {
    for (int64_t ph = 0; ph < patch; ++ph) {
      for (int64_t pw = 0; pw < patch; ++pw) {
        const uint8_t raw =
            rgb[static_cast<size_t>(ph * dots3_tiny::kImageSide * 3 + pw * 3 + c)];
        const double shift = cfg.image_mean[static_cast<size_t>(c)] /
                             cfg.rescale_factor;
        const double sc =
            cfg.image_std[static_cast<size_t>(c)] / cfg.rescale_factor;
        const double want = (static_cast<double>(raw) - shift) / sc;
        const int64_t k = (c * patch + ph) * patch + pw;
        CHECK(kw.pixel_values_f32[static_cast<size_t>(k)] ==
              doctest::Approx(want).epsilon(1e-6));
      }
    }
  }

  // A non-conformant image is REFUSED by name rather than patchified at the
  // wrong grid: the bicubic resize is a named residual, not a silent path.
  std::vector<uint8_t> odd(static_cast<size_t>(9 * 9 * 3), 128);
  CHECK_THROWS_AS((void)proc.ProcessImage(odd.data(), 9, 9), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 7. THE LOADER refuses a MoE tower over a checkpoint that really ships one,
//    and leaves the language tower loadable beside it. A checkpoint whose
//    vision arm is owed must still load its text half — that is what makes the
//    refusal a deferral rather than a load failure.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: a PYRAMID vision block defers the tower and leaves the text tower loaded") {
  TinySpec s;
  s.v_pyramid = {-1, 4};  // block 1 is routed
  Bench bench(s);
  REQUIRE(bench.model != nullptr);

  const Dots3NoteVisionParams v = ParseDots3NoteVisionParams(bench.config);
  const std::string why = Dots3NoteVisionRefusal(v, "", {});
  INFO(why);
  CHECK_FALSE(why.empty());
  CHECK(why.find("W6b") != std::string::npos);

  // The LANGUAGE tower still materialized: the load did not fail, and the
  // registration still resolves. A refusal that took the whole checkpoint down
  // would make the released model unloadable, which it is not.
  const std::vector<std::string> arch{"Dots3NoteForCausalLM"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
  CHECK(reg.factory->encode_mm != nullptr);
  CHECK(reg.factory->embed_mm != nullptr);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(*bench.model));
  // ...and NOT an M-RoPE model.
  CHECK_FALSE(vllm::ModelRegistry::UsesMrope(*bench.model));
}
