// A TINY, COMPLETE dots3-note checkpoint on disk — language tower + DENSE
// vision tower — shared by the W6a tower gate and the W6a served-request gate
// (#2512).
//
// WHY A SHARED HEADER AND NOT A COPY. The two gates must agree on the geometry
// byte-for-byte: the tower gate measures the arithmetic at a grid, and the
// server gate asserts the PLACEHOLDER COUNT that same grid implies. Two
// hand-typed copies of "8x8 image, patch 2, merge 2 -> 4 placeholder tokens" are
// one edit away from disagreeing, and the disagreement would show up as a
// passing tower gate beside a server gate measuring a different model.
//
// WHY IT IS NEW RATHER THAN EXTRACTED FROM `test_dots3_note_attn.cpp`. That
// file's checkpoint builder is 5867 lines deep in one anonymous namespace and
// carries W3's double-precision attention reference with it. Lifting it would be
// a refactor of four bricks' evidence in a brick that is adding a tower. This
// header builds only what a LOAD needs, which is much less.
//
// The language geometry mirrors `test_dots3_note_attn.cpp`'s device bench and
// for the same measured reason (its review finding F1): `q_lora` 3 and
// `kv_lora` 2 over `hidden` 16 give the two §4-trap-5 rescales sqrt(16/3) and
// sqrt(16/2), which are DIFFERENT from each other and both far from 1, so a
// dropped or swapped rescale cannot hide.
#ifndef VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_
#define VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/dtype.h"

namespace dots3_tiny {

// ── the geometry ────────────────────────────────────────────────────────────
//
// Every dimension is the smallest that still exercises the branch it stands
// for:
//   * vision `head_dim` 8, so the 2-D rope has TWO frequencies per spatial axis
//     (`head_dim/2 = 4` split into a height half and a width half,
//     vision.py:503-504 @ 9035151d6). At `head_dim` 4 there would be one
//     frequency per axis and a swapped height/width axis could not show.
//   * TWO dense vision blocks, so a block that read the previous block's
//     weights would move the answer.
//   * a 4x4 patch grid over a 2x2 merge, so the adapter folds FOUR rows into
//     one and produces FOUR merger rows. One row would make a mis-sized scatter
//     invisible.
//   * `intermediate_size` 6 against `embed_dim` 16, so the SwiGLU is not square
//     and a transposed gate/up merge refuses on shape instead of computing.
struct TinySpec {
  // text tower
  int64_t hidden = 16;
  int64_t heads = 2;
  int64_t qk_nope = 4;
  int64_t qk_rope = 4;
  int64_t v_head = 8;
  int64_t q_lora = 3;
  int64_t kv_lora = 2;
  int64_t layers = 2;
  int64_t vocab = 17;
  int64_t inter = 10;
  int64_t max_pos = 64;
  int64_t index_topk = 32;
  int64_t index_n_heads = 2;
  int64_t index_head_dim = 6;
  double rope_theta = 137.0;
  double rms_eps = 1e-3;

  // vision tower
  bool with_vision = true;
  int64_t v_embed = 16;
  int64_t v_heads = 2;
  int64_t v_layers = 2;
  int64_t v_inter = 6;
  int64_t v_patch = 2;
  int64_t v_merge = 2;
  int64_t v_channels = 3;
  int64_t v_temporal = 1;
  double v_rms_eps = 1e-3;
  // `pyramid_num_routed` written into the config. EMPTY means "no MoE at all";
  // a case that wants the W6b refusal sets one entry positive.
  std::vector<int64_t> v_pyramid{-1, -1};
  std::string v_adapter_type = "patch_merger";
  bool v_pre_pixel_shuffle = true;
  bool v_post_norm = true;

  int64_t v_head_dim() const { return v_embed / v_heads; }
  int64_t v_patch_row() const {
    return v_channels * v_temporal * v_patch * v_patch;
  }
  int64_t v_merged_dim() const { return v_embed * v_merge * v_merge; }
  // The adapter lands in the TEXT hidden space; anything else cannot be
  // scattered into the prompt.
  int64_t v_adapter_out() const { return hidden; }
  int64_t qk_head_dim() const { return qk_nope + qk_rope; }
};

// The three vision marker ids. They are the tokenizer fixture's ADDED tokens in
// the server gate, and `config.json`'s `image_token_id` / `image_start_token_id`
// / `image_end_token_id` here, so the marker string the chat seam injects
// tokenizes to exactly one `<|imgpad|>` id the expansion can expand.
inline constexpr int32_t kImgStartId = 14;
inline constexpr int32_t kImgPadId = 15;
inline constexpr int32_t kImgEndId = 16;

// An 8x8 RGB image over a 2-pixel patch and a 2x2 merge: `factor` is 4, the
// image needs no resize, the grid is (1, 4, 4) = 16 patches, and
// 16 / (2*2) = FOUR placeholder tokens.
inline constexpr int64_t kImageSide = 8;
inline constexpr int64_t kExpectedImageTokens = 4;

// ── deterministic values ────────────────────────────────────────────────────
inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Values already ROUNDED to the bf16 the checkpoint stores, so a comparison
// against a double reference measures the FORWARD rather than the weights'
// storage width.
inline std::vector<double> Values(int64_t n, uint64_t seed, double amp,
                                  double bias = 0.0) {
  std::vector<double> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double u =
        static_cast<double>(Mix(seed + static_cast<uint64_t>(i)) >> 40) /
        static_cast<double>(1 << 24);
    const float f = static_cast<float>((u * 2.0 - 1.0) * amp + bias);
    v[static_cast<size_t>(i)] =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(f)));
  }
  return v;
}

// ── safetensors ─────────────────────────────────────────────────────────────
struct StOut {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<double> values;  // already rounded to `dtype`
  std::string dtype = "BF16";
};

inline void WriteSafetensors(const std::vector<StOut>& entries,
                             const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StOut& e : entries) {
    size_t n = 1;
    for (int64_t s : e.shape) n *= static_cast<size_t>(s);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StOut& e : entries) {
    for (double v : e.values) {
      if (e.dtype == "F32") {
        const float f = static_cast<float>(v);
        out.write(reinterpret_cast<const char*>(&f), 4);
      } else {
        const uint16_t b = vt::F32ToBF16(static_cast<float>(v));
        out.write(reinterpret_cast<const char*>(&b), 2);
      }
    }
  }
}

// ── the config document ─────────────────────────────────────────────────────
//
// Built on the COMMITTED released `config.json`, with the geometry overridden.
// That is deliberate: all 36 keys `ParseDots3NoteParams` requires, and the whole
// W1 validation, still apply to this fixture, so a tiny config cannot pass
// through a hole the released one would have hit.
inline nlohmann::json TinyConfigDoc(const std::string& fixture_dir,
                                    const TinySpec& s) {
  nlohmann::json d;
  {
    std::ifstream in(fixture_dir + "/config.json");
    in >> d;
  }
  d["hidden_size"] = s.hidden;
  d["num_hidden_layers"] = s.layers;
  nlohmann::json lt = nlohmann::json::array();
  for (int64_t i = 0; i < s.layers; ++i) lt.push_back("full_attention");
  d["layer_types"] = lt;
  d["num_attention_heads"] = s.heads;
  d["num_key_value_heads"] = s.heads;
  d["qk_nope_head_dim"] = s.qk_nope;
  d["qk_rope_head_dim"] = s.qk_rope;
  d["v_head_dim"] = s.v_head;
  d["q_lora_rank"] = s.q_lora;
  d["kv_lora_rank"] = s.kv_lora;
  d["rope_theta"] = s.rope_theta;
  d["rms_norm_eps"] = s.rms_eps;
  d["max_position_embeddings"] = s.max_pos;
  d["index_n_heads"] = s.index_n_heads;
  d["index_head_dim"] = s.index_head_dim;
  d["index_topk"] = s.index_topk;
  d["indexer_rope_interleave"] = true;
  // The SWA geometry is required by the parse even with zero sliding layers;
  // `swa_kv_lora_rank == kv_lora_rank` keeps the PHYSICAL latent row equal to
  // the logical one.
  d["swa_num_attention_heads"] = 1;
  d["swa_num_key_value_heads"] = 1;
  d["swa_q_lora_rank"] = s.q_lora;
  d["swa_kv_lora_rank"] = s.kv_lora;
  d["swa_qk_nope_head_dim"] = s.qk_nope;
  d["swa_qk_rope_head_dim"] = s.qk_rope;
  d["swa_v_head_dim"] = s.v_head;
  d["vocab_size"] = s.vocab;
  d["intermediate_size"] = s.inter;
  d["moe_intermediate_size"] = 6;
  d["n_routed_experts"] = 4;
  d["num_experts_per_tok"] = 2;
  d["first_k_dense_replace"] = s.layers;   // every layer DENSE
  d["num_nextn_predict_layers"] = 0;       // no MTP tail (W10 owns it)
  d["tie_word_embeddings"] = false;
  // The three image marker ids, so the processor can resolve them from
  // `config.json` the way a converted checkpoint carries them.
  d["image_token_id"] = kImgPadId;
  d["image_start_token_id"] = kImgStartId;
  d["image_end_token_id"] = kImgEndId;

  if (s.with_vision) {
    nlohmann::json v = nlohmann::json::object();
    v["embed_dim"] = s.v_embed;
    v["hidden_size"] = s.hidden;
    v["intermediate_size"] = s.v_inter;
    v["moe_intermediate_size"] = 4;
    v["num_hidden_layers"] = s.v_layers;
    v["num_attention_heads"] = s.v_heads;
    v["num_channels"] = s.v_channels;
    v["patch_size"] = s.v_patch;
    v["spatial_merge_size"] = s.v_merge;
    v["temporal_patch_size"] = s.v_temporal;
    v["rms_norm_eps"] = s.v_rms_eps;
    v["use_bias"] = false;
    v["use_qk_norm"] = true;
    v["is_causal"] = false;
    v["post_norm"] = s.v_post_norm;
    v["pre_pixel_shuffle"] = s.v_pre_pixel_shuffle;
    v["pyramid_num_routed"] = s.v_pyramid;
    v["capacity_factor"] = 2;
    v["router_scoring_func"] = "sigmoid";
    v["router_scale"] = 1.0;
    v["adapter_type"] = s.v_adapter_type;
    v["adapter_in_dim"] = s.v_embed;
    v["adapter_out_dim"] = s.v_adapter_out();
    v["adapter_merge_size"] = s.v_merge;
    d["vision_config"] = v;
  } else {
    d.erase("vision_config");
  }
  // The AUDIO tower is W7's and this fixture ships none, so the key that would
  // make the loader expect one is removed.
  d.erase("audio_config");
  return d;
}

// The `preprocessor_config.json` the chat seam's factory reads. Its geometry
// reproduces the `vision_config`'s, and `min_pixels`/`max_pixels` are chosen so
// `Dots3NoteResizedSize` is the IDENTITY on the fixture image — the bicubic
// resize path is a named residual, so a fixture that needed it would refuse.
inline nlohmann::json TinyPreprocessorDoc(const TinySpec& s) {
  return nlohmann::json{
      {"patch_size", s.v_patch},
      {"temporal_patch_size", s.v_temporal},
      {"merge_size", s.v_merge},
      {"pre_pixel_shuffle", s.v_pre_pixel_shuffle},
      {"image_mean", {0.5, 0.45, 0.4}},
      {"image_std", {0.25, 0.3, 0.35}},
      {"min_pixels", 16},
      {"max_pixels", 1 << 20}};
}

inline nlohmann::json TinyAddedTokensDoc() {
  return nlohmann::json{{"<|img|>", kImgStartId},
                        {"<|imgpad|>", kImgPadId},
                        {"<|endofimg|>", kImgEndId}};
}

// ── the tensors ─────────────────────────────────────────────────────────────
//
// The LANGUAGE half is exactly what `EnumerateDots3NoteTensors` claims for a
// config whose every layer is full-attention with a dense MLP, and the VISION
// half is exactly what `EnumerateDots3NoteVisionTensors` claims for an all-dense
// tower. A name this list gets wrong refuses the load by name rather than being
// skipped, which is what makes the fixture self-checking.
inline std::vector<StOut> TinyEntries(const TinySpec& s, uint64_t seed = 7) {
  const int64_t H = s.hidden, N = s.heads, QK = s.qk_head_dim();
  std::vector<StOut> e;
  uint64_t k = seed;
  const auto next = [&k]() { return (k += 0x9E37ULL) * 1000003ULL; };
  // A norm weight sits around 1.0; a projection sits around 0.
  const auto norm = [&](int64_t n) { return Values(n, next(), 0.25, 1.0); };
  const auto proj = [&](int64_t n) { return Values(n, next(), 0.5); };

  e.push_back({"model.embed_tokens.weight", {s.vocab, H}, proj(s.vocab * H)});
  e.push_back({"model.norm.weight", {H}, norm(H)});
  e.push_back({"lm_head.weight", {s.vocab, H}, proj(s.vocab * H)});
  for (int64_t l = 0; l < s.layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const std::string sa = p + "self_attn.";
    e.push_back({p + "input_layernorm.weight", {H}, norm(H)});
    e.push_back({p + "post_attention_layernorm.weight", {H}, norm(H)});
    e.push_back({sa + "q_a_proj.weight", {s.q_lora, H}, proj(s.q_lora * H)});
    e.push_back({sa + "q_a_layernorm.weight", {s.q_lora}, norm(s.q_lora)});
    e.push_back({sa + "q_b_proj.weight", {N * QK, s.q_lora},
                 proj(N * QK * s.q_lora)});
    e.push_back({sa + "kv_a_proj_with_mqa.weight",
                 {s.kv_lora + s.qk_rope, H}, proj((s.kv_lora + s.qk_rope) * H)});
    e.push_back({sa + "kv_a_layernorm.weight", {s.kv_lora}, norm(s.kv_lora)});
    e.push_back({sa + "kv_b_proj.weight",
                 {N * (s.qk_nope + s.v_head), s.kv_lora},
                 proj(N * (s.qk_nope + s.v_head) * s.kv_lora)});
    e.push_back({sa + "o_proj.weight", {H, N * s.v_head}, proj(H * N * s.v_head)});
    e.push_back({sa + "g_proj.weight", {N, H}, proj(N * H)});
    e.push_back({sa + "k_rope_only_layernorm.weight", {s.qk_rope},
                 norm(s.qk_rope)});
    e.push_back({sa + "indexer.wq_b.weight",
                 {s.index_n_heads * s.index_head_dim, s.q_lora},
                 proj(s.index_n_heads * s.index_head_dim * s.q_lora)});
    e.push_back({sa + "indexer.wk.weight", {s.index_head_dim, H},
                 proj(s.index_head_dim * H)});
    e.push_back({sa + "indexer.k_norm.weight", {s.index_head_dim},
                 norm(s.index_head_dim)});
    e.push_back({sa + "indexer.k_norm.bias", {s.index_head_dim},
                 Values(s.index_head_dim, next(), 0.1)});
    e.push_back({sa + "indexer.weights_proj.weight", {s.index_n_heads, H},
                 proj(s.index_n_heads * H)});
    e.push_back({p + "mlp.gate_proj.weight", {s.inter, H}, proj(s.inter * H)});
    e.push_back({p + "mlp.up_proj.weight", {s.inter, H}, proj(s.inter * H)});
    e.push_back({p + "mlp.down_proj.weight", {H, s.inter}, proj(H * s.inter)});
  }

  if (!s.with_vision) return e;

  const int64_t E = s.v_embed, VI = s.v_inter, D = s.v_head_dim();
  const std::string vp = "vision_encoder.";
  e.push_back({vp + "patch_embed.proj.weight",
               {E, s.v_channels, s.v_patch, s.v_patch},
               proj(E * s.v_patch_row())});
  e.push_back({vp + "patch_embed.proj.bias", {E}, Values(E, next(), 0.2)});
  e.push_back({vp + "patch_embed.norm.weight", {E}, norm(E)});
  for (int64_t b = 0; b < s.v_layers; ++b) {
    const std::string pre = vp + "blocks." + std::to_string(b) + ".";
    const bool moe = b < static_cast<int64_t>(s.v_pyramid.size()) &&
                     s.v_pyramid[static_cast<size_t>(b)] > 0;
    e.push_back({pre + "norm_1.weight", {E}, norm(E)});
    e.push_back({pre + "norm_2.weight", {E}, norm(E)});
    e.push_back({pre + "attn.qkv.weight", {3 * E, E}, proj(3 * E * E)});
    e.push_back({pre + "attn.proj.weight", {E, E}, proj(E * E)});
    e.push_back({pre + "attn.q_norm.weight", {D}, norm(D)});
    e.push_back({pre + "attn.k_norm.weight", {D}, norm(D)});
    if (moe) {
      // Present so a REFUSAL case can load a checkpoint that really does carry
      // a pyramid block, rather than one that merely says so in its config.
      const int64_t ne = s.v_pyramid[static_cast<size_t>(b)];
      e.push_back({pre + "mlp.gate_weight", {ne, E}, proj(ne * E)});
      e.push_back({pre + "mlp.router_bias", {ne}, Values(ne, next(), 0.1),
                   "F32"});
      for (int64_t x = 0; x < ne; ++x) {
        const std::string ep = pre + "mlp.experts." + std::to_string(x) + ".";
        e.push_back({ep + "fc1.weight", {4, E}, proj(4 * E)});
        e.push_back({ep + "fc2.weight", {E, 4}, proj(E * 4)});
        e.push_back({ep + "fc3.weight", {4, E}, proj(4 * E)});
      }
    } else {
      e.push_back({pre + "mlp.fc1.weight", {VI, E}, proj(VI * E)});
      e.push_back({pre + "mlp.fc2.weight", {E, VI}, proj(E * VI)});
      e.push_back({pre + "mlp.fc3.weight", {VI, E}, proj(VI * E)});
    }
  }
  if (s.v_post_norm) {
    e.push_back({vp + "post_trunk_norm.weight", {E}, norm(E)});
  }
  const int64_t M = s.v_merged_dim(), O = s.v_adapter_out();
  e.push_back({vp + "adapter.ln_q.weight", {E}, norm(E)});
  e.push_back({vp + "adapter.ln_q.bias", {E}, Values(E, next(), 0.1)});
  e.push_back({vp + "adapter.mlp.0.weight", {M, M}, proj(M * M)});
  e.push_back({vp + "adapter.mlp.0.bias", {M}, Values(M, next(), 0.1)});
  e.push_back({vp + "adapter.mlp.2.weight", {O, M}, proj(O * M)});
  e.push_back({vp + "adapter.mlp.2.bias", {O}, Values(O, next(), 0.1)});
  return e;
}

// ── the checkpoint DIRECTORY ────────────────────────────────────────────────
//
// A complete model directory: `config.json`, `model.safetensors`,
// `preprocessor_config.json` and `added_tokens.json`. The last two are what the
// PRODUCTION chat factory reads, so the server gate loads its processor off
// disk on the production path rather than being handed one pre-built.
class TinyCheckpoint {
 public:
  TinyCheckpoint(const std::string& fixture_dir, const TinySpec& spec,
                 uint64_t seed = 7)
      : entries_(TinyEntries(spec, seed)) {
    static int counter = 0;
    static const unsigned salt = std::random_device{}();
    dir_ = std::filesystem::temp_directory_path() /
           ("dots3_note_tiny_" + std::to_string(salt) + "_" +
            std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json", std::ios::binary)
        << TinyConfigDoc(fixture_dir, spec).dump();
    std::ofstream(dir_ / "preprocessor_config.json", std::ios::binary)
        << TinyPreprocessorDoc(spec).dump();
    std::ofstream(dir_ / "added_tokens.json", std::ios::binary)
        << TinyAddedTokensDoc().dump();
    WriteSafetensors(entries_, (dir_ / "model.safetensors").string());
  }
  ~TinyCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  TinyCheckpoint(const TinyCheckpoint&) = delete;
  TinyCheckpoint& operator=(const TinyCheckpoint&) = delete;

  std::string dir() const { return dir_.string(); }
  std::string config_path() const { return (dir_ / "config.json").string(); }
  std::string weights_path() const {
    return (dir_ / "model.safetensors").string();
  }
  const std::vector<StOut>& entries() const { return entries_; }
  // The bf16-rounded values of one tensor, by name. Used by the tower gate to
  // drive its DOUBLE reference from the SAME bytes the loader read, so the
  // comparison measures the forward and not a second copy of the weights.
  const std::vector<double>& value_of(const std::string& name) const {
    for (const StOut& e : entries_)
      if (e.name == name) return e.values;
    static const std::vector<double> kEmpty;
    return kEmpty;
  }

 private:
  std::filesystem::path dir_;
  std::vector<StOut> entries_;
};

// The fixture image: HWC uint8, `kImageSide` square. `variant` picks a
// genuinely DIFFERENT image rather than a shifted one — a high-frequency
// sawtooth against a smooth vertical ramp — so the two disagree in every patch.
inline std::vector<uint8_t> FixtureImage(int variant) {
  std::vector<uint8_t> rgb(static_cast<size_t>(kImageSide * kImageSide * 3));
  for (size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = variant == 0
                 ? static_cast<uint8_t>((i * 37 + 11) & 0xFF)
                 : static_cast<uint8_t>(((i / (kImageSide * 3)) * 29) & 0xFF);
  }
  return rgb;
}

}  // namespace dots3_tiny

#endif  // VLLM_TESTS_DOTS3_NOTE_TINY_FIXTURE_H_
