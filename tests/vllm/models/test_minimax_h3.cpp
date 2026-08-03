// MiniMax-H3 parity gate. Every assertion here compares our port against the
// UPSTREAM vLLM-Omni implementation (vllm-project/vllm-omni,
// vllm_omni/diffusion/models/minimax_h3/), executed at reduced dimensions by
// scripts/gen-minimax-h3-goldens.py and frozen into minimax_h3_goldens.inc.
//
// WHY A REDUCED-DIMENSION GATE. The shipped H3 checkpoint is ~354 GB and its
// validated serving config is 4x NVIDIA B300 (~133 GB peak per rank); it does not
// fit this project's hardware, so nothing here claims an end-to-end video result.
// What it does claim is exact: the packed layout (including the FP64 position grid
// that feeds RoPE), the flow-matching scheduler, the latent<->token packing, and
// the full DiT forward all reproduce upstream's numbers. See
// .agents/specs/minimax-h3.md sections 0 and 4.
#include "vllm/model_executor/models/minimax_h3.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "minimax_h3_goldens.inc"

#include "vt/device.h"
#include "vt/tensor.h"

using vllm::BuildMiniMaxH3PackedSequence;
using vllm::BuildMiniMaxH3PackedSequenceRef2va;
using vllm::EnumerateMiniMaxH3DitTensors;
using vllm::MiniMaxH3DitForward;
using vllm::MiniMaxH3DitInputs;
using vllm::MiniMaxH3DitOutputs;
using vllm::MiniMaxH3DitParams;
using vllm::MiniMaxH3DitWeights;
using vllm::MiniMaxH3EulerEta0Step;
using vllm::MiniMaxH3PackAudioLatent;
using vllm::MiniMaxH3PackedSequence;
using vllm::MiniMaxH3PatchifyVideoLatent;
using vllm::MiniMaxH3RefBlock;
using vllm::MiniMaxH3ReorderGroupedQkv;
using vllm::MiniMaxH3RfVToX0;
using vllm::MiniMaxH3UnpackAudioTokens;
using vllm::MiniMaxH3UnpatchifyVideoTokens;
using vllm::ParseMiniMaxH3DitParams;

namespace {

// ---------------------------------------------------------------------------
// H3Rand — the exact mirror of the generator's deterministic stream
// (scripts/gen-minimax-h3-goldens.py :: h3_rand). A per-tensor FNV-1a seed plus a
// splitmix64 counter, so both sides build identical tensors without shipping a
// single weight byte.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<double> H3Rand(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<double> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = static_cast<double>(u >> 11) * 0x1p-53;
    out[static_cast<size_t>(i)] = unit * 2.0 - 1.0;
  }
  return out;
}

// make_param: h3_rand * scale + offset, rounded to f32 (the generator's astype).
std::vector<float> MakeParam(const std::string& name, int64_t count, double scale,
                             double offset = 0.0) {
  const std::vector<double> raw = H3Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)] * scale + offset);
  }
  return out;
}

vt::Device Cpu() { return vt::Device{}; }

vt::Tensor View1D(std::vector<float>& buffer) {
  return vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(),
                                {static_cast<int64_t>(buffer.size())});
}

vt::Tensor View2D(std::vector<float>& buffer, int64_t rows, int64_t cols) {
  REQUIRE(static_cast<int64_t>(buffer.size()) == rows * cols);
  return vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(), {rows, cols});
}

// Max absolute difference against a golden array.
double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

template <typename T>
void CheckI64(const std::vector<T>& got, const int64_t* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(static_cast<int64_t>(got[i]) == want[i]);
}

// The reduced-dimension arch the generator used (section 5 scalars).
MiniMaxH3DitParams GoldenParams() {
  MiniMaxH3DitParams p;
  p.num_layers = vllm_test::kH3Dit_num_layers;
  p.token_refiner_num_layers = vllm_test::kH3Dit_token_refiner_num_layers;
  p.hidden_size = vllm_test::kH3Dit_hidden_size;
  p.num_attention_heads = vllm_test::kH3Dit_num_attention_heads;
  p.attention_head_dim = vllm_test::kH3Dit_attention_head_dim;
  p.ffn_hidden_size = vllm_test::kH3Dit_ffn_hidden_size;
  p.latents_dim = vllm_test::kH3Dit_latents_dim;
  p.audio_latents_dim = vllm_test::kH3Dit_audio_latents_dim;
  p.patch_size_t = 1;
  p.patch_size_h = 2;
  p.patch_size_w = 2;
  p.text_dim = vllm_test::kH3Dit_text_dim;
  p.timestep_input_dim = vllm_test::kH3Dit_timestep_input_dim;
  p.time_embed_hidden_size = vllm_test::kH3Dit_time_embed_hidden_size;
  p.time_embed_dim = vllm_test::kH3Dit_time_embed_dim;
  p.adaln_out_features = 18 * p.hidden_size;
  p.final_adaln_out_features = 2 * p.hidden_size;
  p.rope_inv_freq_len = vllm_test::kH3Dit_rope_inv_freq_len;
  return p;
}

// Owns every reduced-dimension parameter so the vt::Tensor views stay valid.
struct GoldenWeights {
  MiniMaxH3DitParams params;
  std::vector<std::vector<float>> storage;
  MiniMaxH3DitWeights views;

  std::vector<float>& Add(const std::string& name, int64_t count, double scale,
                          double offset = 0.0) {
    storage.push_back(MakeParam(name, count, scale, offset));
    return storage.back();
  }
};

// Mirrors RefDiT.__init__ in the generator: same names, same scales, same shapes.
// storage is reserved up front because the vt::Tensor views are non-owning.
void BuildBlock(GoldenWeights& w, const std::string& prefix, bool with_adaln) {
  const MiniMaxH3DitParams& p = w.params;
  const int64_t h = p.hidden_size;
  const int64_t inner = p.num_attention_heads * p.attention_head_dim;
  vllm::MiniMaxH3DitBlockWeights block;
  block.norm1 = View1D(w.Add(prefix + ".norm1.weight", h, 0.1, 1.0));
  block.norm2 = View1D(w.Add(prefix + ".norm2.weight", h, 0.1, 1.0));
  block.qkv_proj =
      View2D(w.Add(prefix + ".attn.qkv_proj.weight", 3 * inner * h, 0.05), 3 * inner, h);
  block.q_norm = View1D(w.Add(prefix + ".attn.q_norm.weight", p.attention_head_dim, 0.1, 1.0));
  block.k_norm = View1D(w.Add(prefix + ".attn.k_norm.weight", p.attention_head_dim, 0.1, 1.0));
  block.out_proj = View2D(w.Add(prefix + ".attn.out_proj.weight", h * inner, 0.05), h, inner);
  block.fc1 = View2D(w.Add(prefix + ".mlp.fc1.weight", 2 * p.ffn_hidden_size * h, 0.05),
                     2 * p.ffn_hidden_size, h);
  block.fc2 = View2D(w.Add(prefix + ".mlp.fc2.weight", h * p.ffn_hidden_size, 0.05), h,
                     p.ffn_hidden_size);
  if (with_adaln) {
    block.adaln_w = View2D(
        w.Add(prefix + ".adaln_proj.linear.weight", p.adaln_out_features * p.time_embed_dim, 0.05),
        p.adaln_out_features, p.time_embed_dim);
    block.adaln_b = View1D(w.Add(prefix + ".adaln_proj.linear.bias", p.adaln_out_features, 0.02));
    w.views.blocks.push_back(block);
  } else {
    w.views.refiner.push_back(block);
  }
}

std::unique_ptr<GoldenWeights> BuildGoldenWeights() {
  auto w = std::make_unique<GoldenWeights>();
  w->params = GoldenParams();
  const MiniMaxH3DitParams& p = w->params;
  const int64_t h = p.hidden_size;
  const int64_t video_width = p.video_row_width();
  // Reserve so no reallocation invalidates a previously taken view.
  w->storage.reserve(512);

  w->views.video_patch_proj_w =
      View2D(w->Add("video_patch_proj.weight", h * video_width, 0.05), h, video_width);
  w->views.video_patch_proj_b = View1D(w->Add("video_patch_proj.bias", h, 0.02));
  w->views.audio_patch_proj_w =
      View2D(w->Add("audio_patch_proj.weight", h * p.audio_latents_dim, 0.05), h, p.audio_latents_dim);
  w->views.audio_patch_proj_b = View1D(w->Add("audio_patch_proj.bias", h, 0.02));
  w->views.condition_proj_w =
      View2D(w->Add("condition_proj.weight", h * p.text_dim, 0.05), h, p.text_dim);
  w->views.condition_proj_b = View1D(w->Add("condition_proj.bias", h, 0.02));
  w->views.time_proj_in_w =
      View2D(w->Add("time_embedder.proj_in.weight", p.time_embed_hidden_size * p.timestep_input_dim,
                    0.05),
             p.time_embed_hidden_size, p.timestep_input_dim);
  w->views.time_proj_in_b =
      View1D(w->Add("time_embedder.proj_in.bias", p.time_embed_hidden_size, 0.02));
  w->views.time_proj_out_w = View2D(
      w->Add("time_embedder.proj_out.weight", p.time_embed_dim * p.time_embed_hidden_size, 0.05),
      p.time_embed_dim, p.time_embed_hidden_size);
  w->views.time_proj_out_b = View1D(w->Add("time_embedder.proj_out.bias", p.time_embed_dim, 0.02));

  // inv_freq = 10000^(-2i / 2L), computed (not random) on both sides.
  {
    std::vector<float> inv(static_cast<size_t>(p.rope_inv_freq_len));
    for (int64_t i = 0; i < p.rope_inv_freq_len; ++i) {
      inv[static_cast<size_t>(i)] = static_cast<float>(
          std::pow(10000.0, -(2.0 * static_cast<double>(i)) / (2.0 * static_cast<double>(p.rope_inv_freq_len))));
    }
    w->storage.push_back(std::move(inv));
    w->views.rope_inv_freq = View1D(w->storage.back());
  }

  for (int64_t i = 0; i < p.token_refiner_num_layers; ++i) {
    BuildBlock(*w, "token_refiner.blocks." + std::to_string(i), /*with_adaln=*/false);
  }
  w->views.refiner_final_norm = View1D(w->Add("token_refiner.final_norm.weight", h, 0.1, 1.0));
  for (int64_t i = 0; i < p.num_layers; ++i) {
    BuildBlock(*w, "blocks." + std::to_string(i), /*with_adaln=*/true);
  }
  w->views.final_norm = View1D(w->Add("final_layer.norm.weight", h, 0.1, 1.0));
  w->views.final_adaln_w =
      View2D(w->Add("final_layer.adaln_proj.linear.weight", p.final_adaln_out_features * p.time_embed_dim,
                    0.05),
             p.final_adaln_out_features, p.time_embed_dim);
  w->views.final_adaln_b =
      View1D(w->Add("final_layer.adaln_proj.linear.bias", p.final_adaln_out_features, 0.02));
  w->views.video_out_w =
      View2D(w->Add("final_layer.video_out.weight", video_width * h, 0.05), video_width, h);
  w->views.video_out_b = View1D(w->Add("final_layer.video_out.bias", video_width, 0.02));
  w->views.audio_out_w =
      View2D(w->Add("final_layer.audio_out.weight", p.audio_latents_dim * h, 0.05),
             p.audio_latents_dim, h);
  w->views.audio_out_b = View1D(w->Add("final_layer.audio_out.bias", p.audio_latents_dim, 0.02));
  return w;
}

}  // namespace

TEST_CASE("minimax_h3: the deterministic weight stream matches the generator") {
  // If this fails, nothing else in this file is meaningful — the two sides would
  // be comparing different tensors, not different implementations.
  // Bit-exact on purpose: the two streams must agree exactly, not approximately.
  const std::vector<double> probe = H3Rand("h3.probe", 8);
  REQUIRE(probe.size() == std::size(vllm_test::kH3RandProbe));
  for (size_t i = 0; i < probe.size(); ++i) {
    CHECK(probe[i] == vllm_test::kH3RandProbe[i]);
  }
}

TEST_CASE("minimax_h3: fl2va packed sequence matches upstream") {
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  CHECK(packed.seq_len == vllm_test::kH3Fl2va_seq_len);
  CheckI64(packed.input_ids, vllm_test::kH3Fl2vaInputIds,
           std::size(vllm_test::kH3Fl2vaInputIds));
  CheckI64(packed.token_tags, vllm_test::kH3Fl2vaTokenTags,
           std::size(vllm_test::kH3Fl2vaTokenTags));
  CheckI64(packed.img_pos, vllm_test::kH3Fl2vaImgPos, std::size(vllm_test::kH3Fl2vaImgPos));
  CheckI64(packed.audio_pos, vllm_test::kH3Fl2vaAudioPos, std::size(vllm_test::kH3Fl2vaAudioPos));
  CheckI64(packed.text_pos, vllm_test::kH3Fl2vaTextPos, std::size(vllm_test::kH3Fl2vaTextPos));
  CheckI64(packed.update_mask, vllm_test::kH3Fl2vaUpdateMask,
           std::size(vllm_test::kH3Fl2vaUpdateMask));
  CheckI64(packed.cu_seqlens, vllm_test::kH3Fl2vaCuSeqlens,
           std::size(vllm_test::kH3Fl2vaCuSeqlens));
  CheckI64(packed.document_id, vllm_test::kH3Fl2vaDocumentId,
           std::size(vllm_test::kH3Fl2vaDocumentId));

  // The FP64 position grid feeds RoPE directly and is gated BIT-EXACT: a last-ulp
  // drift here would silently rotate every video token.
  REQUIRE(packed.img_position_ids.size() == std::size(vllm_test::kH3Fl2vaImgPositionIds));
  for (size_t i = 0; i < packed.img_position_ids.size(); ++i) {
    CHECK(packed.img_position_ids[i] == vllm_test::kH3Fl2vaImgPositionIds[i]);
  }
}

TEST_CASE("minimax_h3: ref2va block packed sequence matches upstream") {
  std::vector<MiniMaxH3RefBlock> blocks(2);
  blocks[0].kind = MiniMaxH3RefBlock::Kind::kImage;
  blocks[0].latent_h = 4;
  blocks[0].latent_w = 4;
  blocks[1].kind = MiniMaxH3RefBlock::Kind::kVideoAudio;
  blocks[1].ref_audio_t = 2;
  blocks[1].latent_t = 2;
  blocks[1].latent_h = 4;
  blocks[1].latent_w = 4;

  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequenceRef2va(
      vllm_test::kH3Ref2va_text_len, vllm_test::kH3Ref2va_latent_t, vllm_test::kH3Ref2va_latent_h,
      vllm_test::kH3Ref2va_latent_w, vllm_test::kH3Ref2va_audio_t, blocks,
      vllm_test::kH3Ref2va_audio_channel);

  CHECK(packed.seq_len == vllm_test::kH3Ref2va_seq_len);
  CheckI64(packed.input_ids, vllm_test::kH3Ref2vaInputIds,
           std::size(vllm_test::kH3Ref2vaInputIds));
  CheckI64(packed.token_tags, vllm_test::kH3Ref2vaTokenTags,
           std::size(vllm_test::kH3Ref2vaTokenTags));
  CheckI64(packed.img_pos, vllm_test::kH3Ref2vaImgPos, std::size(vllm_test::kH3Ref2vaImgPos));
  CheckI64(packed.audio_pos, vllm_test::kH3Ref2vaAudioPos, std::size(vllm_test::kH3Ref2vaAudioPos));
  CheckI64(packed.update_mask, vllm_test::kH3Ref2vaUpdateMask,
           std::size(vllm_test::kH3Ref2vaUpdateMask));
  CheckI64(packed.audio_update_mask, vllm_test::kH3Ref2vaAudioUpdateMask,
           std::size(vllm_test::kH3Ref2vaAudioUpdateMask));
  CheckI64(packed.cu_seqlens, vllm_test::kH3Ref2vaCuSeqlens,
           std::size(vllm_test::kH3Ref2vaCuSeqlens));
  REQUIRE(packed.img_position_ids.size() == std::size(vllm_test::kH3Ref2vaImgPositionIds));
  for (size_t i = 0; i < packed.img_position_ids.size(); ++i) {
    CHECK(packed.img_position_ids[i] == vllm_test::kH3Ref2vaImgPositionIds[i]);
  }
}

TEST_CASE("minimax_h3: video patchify and audio pack match upstream, and round-trip") {
  const int64_t c = vllm_test::kH3PatchifyC, t = vllm_test::kH3PatchifyT;
  const int64_t h = vllm_test::kH3PatchifyH, w = vllm_test::kH3PatchifyW;
  const std::vector<float> latent = MakeParam("packing.video_latent", c * t * h * w, 1.0);
  const std::vector<float> rows =
      MiniMaxH3PatchifyVideoLatent(latent, 1, c, t, h, w, 1, 2, 2);
  CHECK(static_cast<int64_t>(rows.size()) ==
        vllm_test::kH3PatchifyRows * vllm_test::kH3PatchifyRowWidth);
  CHECK(MaxAbsDiff(rows, vllm_test::kH3PatchifyRowsGolden, rows.size()) == 0.0);

  const std::vector<float> back =
      MiniMaxH3UnpatchifyVideoTokens(rows, t, h / 2, w / 2, c, 1, 2, 2);
  REQUIRE(back.size() == latent.size());
  for (size_t i = 0; i < back.size(); ++i) CHECK(back[i] == latent[i]);

  const int64_t ac = vllm_test::kH3AudioPackChannels, ad = vllm_test::kH3AudioPackDim;
  const int64_t at = vllm_test::kH3AudioPackT;
  const std::vector<float> audio = MakeParam("packing.audio_latent", ac * ad * at, 1.0);
  const std::vector<float> audio_rows = MiniMaxH3PackAudioLatent(audio, ac, ad, at);
  CHECK(MaxAbsDiff(audio_rows, vllm_test::kH3AudioPackRowsGolden, audio_rows.size()) == 0.0);
  const std::vector<float> audio_back = MiniMaxH3UnpackAudioTokens(audio_rows, ac * at, ac, ad);
  REQUIRE(audio_back.size() == audio.size());
  for (size_t i = 0; i < audio_back.size(); ++i) CHECK(audio_back[i] == audio[i]);
}

TEST_CASE("minimax_h3: flow-matching scheduler matches upstream") {
  const int64_t n = vllm_test::kH3SchedN;
  const std::vector<float> xt = MakeParam("sched.xt", n, 1.0);
  const std::vector<float> v = MakeParam("sched.v", n, 1.0);
  for (int64_t s = 0; s < vllm_test::kH3SchedSteps; ++s) {
    const double t = vllm_test::kH3SchedTimesteps[s];
    const std::vector<float> x0 = MiniMaxH3RfVToX0(xt, v, t);
    CHECK(MaxAbsDiff(x0, vllm_test::kH3SchedX0Golden + s * n, static_cast<size_t>(n)) == 0.0);
    const double sigma_curr = 1.0 - t;
    const std::vector<float> stepped =
        MiniMaxH3EulerEta0Step(xt, x0, sigma_curr, sigma_curr * 0.5);
    CHECK(MaxAbsDiff(stepped, vllm_test::kH3SchedStepGolden + s * n, static_cast<size_t>(n)) <=
          1e-6);
  }
  // The terminal step is a documented identity (scheduling:89-92).
  const std::vector<float> terminal = MiniMaxH3EulerEta0Step(xt, v, 0.0, 0.0);
  for (size_t i = 0; i < terminal.size(); ++i) CHECK(terminal[i] == xt[i]);
}

TEST_CASE("minimax_h3: DiT forward matches upstream at reduced dimensions") {
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;

  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  const int64_t seq_len = packed.seq_len;
  const int64_t video_width = p.video_row_width();
  REQUIRE(video_width == vllm_test::kH3Dit_video_row_width);
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());

  // Scatter the same rows the generator scattered.
  std::vector<float> x(static_cast<size_t>(seq_len * video_width), 0.0f);
  const std::vector<float> video_rows =
      MakeParam("dit.video_rows", num_img * video_width, 1.0);
  for (int64_t r = 0; r < num_img; ++r) {
    std::memcpy(x.data() + packed.img_pos[static_cast<size_t>(r)] * video_width,
                video_rows.data() + r * video_width,
                static_cast<size_t>(video_width) * sizeof(float));
  }
  std::vector<float> audio_x(static_cast<size_t>(seq_len * p.audio_latents_dim), 0.0f);
  const std::vector<float> audio_rows =
      MakeParam("dit.audio_rows", num_audio * p.audio_latents_dim, 1.0);
  for (int64_t r = 0; r < num_audio; ++r) {
    std::memcpy(audio_x.data() + packed.audio_pos[static_cast<size_t>(r)] * p.audio_latents_dim,
                audio_rows.data() + r * p.audio_latents_dim,
                static_cast<size_t>(p.audio_latents_dim) * sizeof(float));
  }
  const std::vector<float> prompt_embeds =
      MakeParam("dit.prompt_embeds", num_text * p.text_dim, 1.0);

  const std::vector<float> unique_timesteps(
      vllm_test::kH3DitUniqueTimesteps,
      vllm_test::kH3DitUniqueTimesteps + vllm_test::kH3Dit_unique_timesteps);
  const std::vector<int64_t> inverse(vllm_test::kH3DitInverseIndices,
                                     vllm_test::kH3DitInverseIndices + seq_len);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};

  MiniMaxH3DitInputs in;
  in.seq_len = seq_len;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_timesteps.data();
  in.num_unique_timesteps = static_cast<int64_t>(unique_timesteps.size());
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt_embeds.data();
  in.img_pos = packed.img_pos.data();
  in.num_img_pos = num_img;
  in.audio_pos = packed.audio_pos.data();
  in.num_audio_pos = num_audio;
  in.text_pos = packed.text_pos.data();
  in.num_text_pos = num_text;
  in.infer_out_pos = packed.img_pos.data();
  in.num_infer_out_pos = num_img;
  in.update_mask = packed.update_mask.data();
  in.cu_seqlens = packed.cu_seqlens.data();
  in.num_cu_seqlens = static_cast<int64_t>(packed.cu_seqlens.size());
  in.refiner_cu_seqlens = refiner_cu.data();
  in.num_refiner_cu_seqlens = static_cast<int64_t>(refiner_cu.size());

  const MiniMaxH3DitOutputs got =
      MiniMaxH3DitForward(Cpu(), p, weights->views, in, vt::DType::kF32);

  // f32 throughout on both sides, so the only slack is summation order inside the
  // GEMM and the softmax; 2e-5 absolute is far below any structural error.
  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsGolden, got.audio_logits.size());
  INFO("video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 2e-5);
  CHECK(audio_err <= 2e-5);

  // The pinned keyframe rows must be zeroed by the update mask.
  for (int64_t r = 0; r < num_img; ++r) {
    if (packed.update_mask[static_cast<size_t>(r)]) continue;
    for (int64_t i = 0; i < video_width; ++i) {
      CHECK(got.video_logits[static_cast<size_t>(r * video_width + i)] == 0.0f);
    }
  }
}

TEST_CASE("minimax_h3: the denoise loop advances targets and pins condition rows") {
  // The loop itself has no upstream golden (upstream's own loop test needs the
  // checkpoint), so this gates its INVARIANTS, which are what the CFG-distilled
  // schedule actually guarantees (denoise_loop.py:191-238): pinned keyframe rows
  // are reset to their anchor after EVERY step, target rows move, audio advances
  // on its own sigma schedule, and nothing goes non-finite.
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;

  vllm::MiniMaxH3DenoiseBranch branch;
  branch.packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);
  branch.token_tags = branch.packed.token_tags;

  const int64_t num_img = static_cast<int64_t>(branch.packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(branch.packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(branch.packed.text_pos.size());
  const int64_t video_width = p.video_row_width();
  branch.text_embeddings = MakeParam("loop.prompt_embeds", num_text * p.text_dim, 1.0);

  const std::vector<float> initial_video = MakeParam("loop.video", num_img * video_width, 1.0);
  const std::vector<float> initial_audio =
      MakeParam("loop.audio", num_audio * p.audio_latents_dim, 1.0);
  int64_t cond_rows = 0;
  for (uint8_t flag : branch.packed.update_mask) cond_rows += flag ? 0 : 1;
  REQUIRE(cond_rows > 0);
  const std::vector<float> keyframe = MakeParam("loop.keyframe", cond_rows * video_width, 1.0);

  // A 2-step schedule: enough to exercise the chaining and the per-step reset.
  const std::vector<double> sigmas_video = {0.6, 0.3, 0.0};
  const std::vector<double> sigmas_audio = {0.5, 0.25, 0.0};

  const vllm::MiniMaxH3DenoiseResult result = vllm::MiniMaxH3DenoiseLoop(
      Cpu(), p, weights->views, branch, initial_video, initial_audio, keyframe, {}, sigmas_video,
      sigmas_audio, vt::DType::kF32);

  REQUIRE(result.video_rows.size() == initial_video.size());
  REQUIRE(result.audio_rows.size() == initial_audio.size());
  for (float value : result.video_rows) REQUIRE(std::isfinite(value));
  for (float value : result.audio_rows) REQUIRE(std::isfinite(value));

  // Pinned rows must equal their anchors exactly; target rows must have moved.
  int64_t cond_index = 0, moved = 0;
  for (int64_t r = 0; r < num_img; ++r) {
    if (branch.packed.update_mask[static_cast<size_t>(r)]) {
      for (int64_t i = 0; i < video_width; ++i) {
        if (result.video_rows[static_cast<size_t>(r * video_width + i)] !=
            initial_video[static_cast<size_t>(r * video_width + i)]) {
          ++moved;
          break;
        }
      }
      continue;
    }
    for (int64_t i = 0; i < video_width; ++i) {
      CHECK(result.video_rows[static_cast<size_t>(r * video_width + i)] ==
            keyframe[static_cast<size_t>(cond_index * video_width + i)]);
    }
    ++cond_index;
  }
  CHECK(cond_index == cond_rows);
  CHECK(moved == num_img - cond_rows);  // every target row advanced

  // The terminal sigma is 0, so the last step is the identity on the state and the
  // schedule must land somewhere finite and different from where it started.
  bool audio_moved = false;
  for (size_t i = 0; i < result.audio_rows.size(); ++i) {
    if (result.audio_rows[i] != initial_audio[i]) audio_moved = true;
  }
  CHECK(audio_moved);
}

TEST_CASE("minimax_h3: config parse enforces the upstream invariants") {
  nlohmann::json config = {
      {"num_layers", 50},        {"token_refiner_num_layers", 2}, {"hidden_size", 5376},
      {"num_attention_heads", 56}, {"attention_head_dim", 128},   {"ffn_hidden_size", 14336},
      {"latents_dim", 24},       {"audio_latents_dim", 32},       {"patch_size", {1, 2, 2}},
      {"text_dim", 5120},        {"timestep_input_dim", 256},     {"time_embed_hidden_size", 5376},
      {"time_embed_dim", 2688},  {"adaln_out_features", 18 * 5376},
      {"final_adaln_out_features", 2 * 5376},                     {"rope_inv_freq_len", 16},
  };
  const MiniMaxH3DitParams p = ParseMiniMaxH3DitParams(config);
  // The SHIPPED MiniMax-H3 geometry.
  CHECK(p.num_layers == 50);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.video_row_width() == 96);   // 24 * 1 * 2 * 2
  CHECK(p.rope_rot_dim() == 96);      // 6 * 16, rotating 96 of 128 head dims
  CHECK(p.rope_rot_dim() <= p.attention_head_dim);

  // adaln_out_features must stay 6 vectors x 3 modalities x hidden_size.
  nlohmann::json bad = config;
  bad["adaln_out_features"] = 17 * 5376;
  CHECK_THROWS(ParseMiniMaxH3DitParams(bad));
  // patch_size must carry exactly three values.
  nlohmann::json bad_patch = config;
  bad_patch["patch_size"] = {1, 2};
  CHECK_THROWS(ParseMiniMaxH3DitParams(bad_patch));
}

TEST_CASE("minimax_h3: the DiT weight contract covers the shipped checkpoint") {
  MiniMaxH3DitParams p;  // shipped defaults
  const std::vector<vllm::MiniMaxH3TensorSpec> specs = EnumerateMiniMaxH3DitTensors(p);

  // 11 stem entries + refiner (2 x 8 + 1) + blocks (50 x 10) + 7 final entries.
  CHECK(specs.size() == 11 + (2 * 8 + 1) + (50 * 10) + 7);

  auto find = [&](const std::string& name) -> const vllm::MiniMaxH3TensorSpec* {
    for (const vllm::MiniMaxH3TensorSpec& spec : specs) {
      if (spec.name == name) return &spec;
    }
    return nullptr;
  };
  // The 12 fp32 parameters plus the fp32 rope buffer
  // (minimax_h3_transformer.py:85-101) must be marked fp32 and nothing else.
  const char* fp32_names[] = {
      "video_patch_proj.weight",     "video_patch_proj.bias",
      "audio_patch_proj.weight",     "audio_patch_proj.bias",
      "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
      "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
      "final_layer.video_out.weight", "final_layer.video_out.bias",
      "final_layer.audio_out.weight", "final_layer.audio_out.bias",
      "rope.inv_freq",
  };
  size_t fp32_count = 0;
  for (const vllm::MiniMaxH3TensorSpec& spec : specs) fp32_count += spec.fp32 ? 1 : 0;
  CHECK(fp32_count == std::size(fp32_names));
  for (const char* name : fp32_names) {
    const vllm::MiniMaxH3TensorSpec* spec = find(name);
    REQUIRE(spec != nullptr);
    CHECK(spec->fp32);
  }

  // MHA: qkv is 3 * heads * head_dim rows wide, and the text condition projection
  // consumes the encoder's 5120-wide hidden state.
  const vllm::MiniMaxH3TensorSpec* qkv = find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(qkv != nullptr);
  CHECK(qkv->shape[0] == 3 * 56 * 128);
  CHECK(qkv->shape[1] == 5376);
  const vllm::MiniMaxH3TensorSpec* cond = find("condition_proj.weight");
  REQUIRE(cond != nullptr);
  CHECK(cond->shape[1] == 5120);
}

TEST_CASE("minimax_h3: grouped-qkv checkpoint reorder is a pure permutation") {
  // The checkpoint stores [q, k, v] PER HEAD; the fused projection wants
  // [q_all, k_all, v_all] (minimax_h3_transformer.py:139-168). H3 is MHA, so
  // heads_per_group == 1.
  const int64_t groups = 3, head_dim = 2, in_features = 2;
  const int64_t rows = groups * (1 + 2) * head_dim;
  std::vector<float> weight(static_cast<size_t>(rows * in_features));
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = static_cast<float>(i);

  const std::vector<float> out =
      MiniMaxH3ReorderGroupedQkv(weight, groups, 1, head_dim, in_features);
  REQUIRE(out.size() == weight.size());

  const int64_t q_rows = groups * head_dim;
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t r = 0; r < head_dim; ++r) {
      for (int64_t c = 0; c < in_features; ++c) {
        const int64_t src_q = (g * 3 * head_dim + r) * in_features + c;
        const int64_t src_k = (g * 3 * head_dim + head_dim + r) * in_features + c;
        const int64_t src_v = (g * 3 * head_dim + 2 * head_dim + r) * in_features + c;
        CHECK(out[static_cast<size_t>((g * head_dim + r) * in_features + c)] == weight[src_q]);
        CHECK(out[static_cast<size_t>((q_rows + g * head_dim + r) * in_features + c)] ==
              weight[src_k]);
        CHECK(out[static_cast<size_t>((2 * q_rows + g * head_dim + r) * in_features + c)] ==
              weight[src_v]);
      }
    }
  }
}
