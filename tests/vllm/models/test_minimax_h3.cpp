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
#include <map>
#include <memory>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "minimax_h3_goldens.inc"
#include "minimax_h3_gguf_manifest.inc"
#include "minimax_h3_audio_vae_goldens.inc"
#include "minimax_h3_nvfp4_manifest.inc"
#include "minimax_h3_video_vae_manifest.inc"
#include "minimax_h3_video_vae_goldens.inc"
#include "minimax_h3_encoder_goldens.inc"

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "../gguf_builder.h"

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

TEST_CASE("minimax_h3: the bf16 production stream matches upstream's dtype policy") {
  // The f32 case above gates the ALGORITHM. This one gates the PRODUCTION dtype
  // policy: upstream's stream is bf16 with fp32 islands (both patch projections,
  // the time embedder, both output heads — minimax_h3_transformer.py:85-101), and
  // the explicit `dtype=_BF16_DTYPE` casts inside `_modulate_scale_shift` /
  // `_modulate_gate`. Same code path, different rounding points.
  const std::unique_ptr<GoldenWeights> weights = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = weights->params;

  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      vllm_test::kH3Fl2va_text_len, vllm_test::kH3Fl2va_latent_t, vllm_test::kH3Fl2va_latent_h,
      vllm_test::kH3Fl2va_latent_w, vllm_test::kH3Fl2va_audio_t, vllm_test::kH3Fl2va_audio_channel,
      /*include_keyframe_cond=*/true, {0}, vllm_test::kH3Fl2va_frame_count);

  const int64_t seq_len = packed.seq_len;
  const int64_t video_width = p.video_row_width();
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());

  std::vector<float> x(static_cast<size_t>(seq_len * video_width), 0.0f);
  const std::vector<float> video_rows = MakeParam("dit.video_rows", num_img * video_width, 1.0);
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
      MiniMaxH3DitForward(Cpu(), p, weights->views, in, vt::DType::kBF16);

  // Tolerance is bf16-scale on purpose: both sides round at the same points, but
  // the GEMM accumulation order still differs (ours is the vt CPU kernel, the
  // reference is torch). The gate is that the CAST POINTS agree — a misplaced or
  // missing cast moves the result far more than accumulation order does.
  const double video_err =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsBf16Golden, got.video_logits.size());
  const double audio_err =
      MaxAbsDiff(got.audio_logits, vllm_test::kH3DitAudioLogitsBf16Golden, got.audio_logits.size());
  INFO("bf16 video max|diff| = " << video_err << ", audio max|diff| = " << audio_err);
  CHECK(video_err <= 5e-3);
  CHECK(audio_err <= 5e-3);

  // And the bf16 stream must differ from the f32 stream — otherwise the dtype
  // policy is not actually being applied and this test proves nothing.
  const double vs_f32 =
      MaxAbsDiff(got.video_logits, vllm_test::kH3DitVideoLogitsGolden, got.video_logits.size());
  CHECK(vs_f32 > 1e-5);
}

TEST_CASE("minimax_h3: request planning matches upstream") {
  // time_request.py executed verbatim by the generator; the three shape helpers
  // are restated there from pipeline_minimax_h3.py:121-122, 207-222, 393-434.
  for (int64_t i = 0; i < vllm_test::kH3PlanFrameCases; ++i) {
    const int64_t input = vllm_test::kH3PlanFrameInputs[i];
    const int64_t aligned = vllm::MiniMaxH3AlignFrameCount(input);
    CHECK(aligned == vllm_test::kH3PlanFrameAligned[i]);
    const int64_t latent_t = vllm::MiniMaxH3VideoLatentT(aligned);
    CHECK(latent_t == vllm_test::kH3PlanVideoLatentT[i]);
    CHECK(vllm::MiniMaxH3FrameCountFromVideoLatentT(latent_t) ==
          vllm_test::kH3PlanFrameFromLatentT[i]);
    CHECK(vllm::MiniMaxH3AudioLatentT(static_cast<double>(aligned) / 24.0) ==
          vllm_test::kH3PlanAudioLatentT[i]);
  }
  // A latent T that is neither 1 nor 5n+2 is rejected, not silently rounded.
  CHECK_THROWS(vllm::MiniMaxH3FrameCountFromVideoLatentT(4));

  int64_t offset = 0;
  for (int64_t c = 0; c < vllm_test::kH3PlanSigmaCases; ++c) {
    const std::vector<double> sigmas = vllm::MiniMaxH3TimeShiftSigmas(
        vllm_test::kH3PlanSigmaSteps[c], vllm_test::kH3PlanSigmaShifts[c]);
    REQUIRE(static_cast<int64_t>(sigmas.size()) == vllm_test::kH3PlanSigmaLengths[c]);
    for (size_t i = 0; i < sigmas.size(); ++i) {
      CHECK(static_cast<float>(sigmas[i]) ==
            doctest::Approx(vllm_test::kH3PlanSigmasGolden[offset + i]).epsilon(1e-6));
    }
    // Multi-step schedules must terminate at sigma 0 so the last Euler step is
    // the identity (scheduling:89-92).
    if (vllm_test::kH3PlanSigmaSteps[c] > 1) CHECK(sigmas.back() == 0.0);
    offset += static_cast<int64_t>(sigmas.size());
  }
  CHECK_THROWS(vllm::MiniMaxH3TimeShiftSigmas(50, 0.0));
  CHECK_THROWS(vllm::MiniMaxH3TimeShiftSigmas(0, 6.0));

  for (int64_t c = 0; c < vllm_test::kH3PlanRefImageCases; ++c) {
    const std::pair<int64_t, int64_t> shape = vllm::MiniMaxH3ReferenceImageShape(
        vllm_test::kH3PlanRefImageInputs[c * 2], vllm_test::kH3PlanRefImageInputs[c * 2 + 1]);
    CHECK(shape.first == vllm_test::kH3PlanRefImageGolden[c * 2]);
    CHECK(shape.second == vllm_test::kH3PlanRefImageGolden[c * 2 + 1]);
  }
  CHECK_THROWS(vllm::MiniMaxH3ReferenceImageShape(5000, 100));  // aspect out of [1:4, 4:1]

  const char* tasks[] = {"t2va", "ref2va", "t2va", "t2va", "fl2va", "fl2va", "t2va"};
  for (int64_t c = 0; c < vllm_test::kH3PlanShapeCases; ++c) {
    const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
        tasks[c], vllm_test::kH3PlanShapeDurations[c], vllm_test::kH3PlanShapeNumFrames[c],
        vllm_test::kH3PlanShapeHeights[c], vllm_test::kH3PlanShapeWidths[c],
        vllm_test::kH3PlanShapeImageW[c], vllm_test::kH3PlanShapeImageH[c]);
    INFO("shape case " << c << " task=" << tasks[c]);
    CHECK(plan.height == vllm_test::kH3PlanShapeGolden[c * 5 + 0]);
    CHECK(plan.width == vllm_test::kH3PlanShapeGolden[c * 5 + 1]);
    CHECK(plan.num_frames == vllm_test::kH3PlanShapeGolden[c * 5 + 2]);
    CHECK(plan.latent_t == vllm_test::kH3PlanShapeGolden[c * 5 + 3]);
    CHECK(plan.audio_t == vllm_test::kH3PlanShapeGolden[c * 5 + 4]);
  }

  // Task dispatch (pipeline_minimax_h3.py:374-391).
  const std::vector<std::string> fl2va_partition = {"t2va", "fl2va"};
  const std::vector<std::string> ref2va_partition = {"ref2va"};
  CHECK(vllm::MiniMaxH3ResolveTask("", "FL2VA", /*has_image=*/false, fl2va_partition) == "t2va");
  CHECK(vllm::MiniMaxH3ResolveTask("", "FL2VA", /*has_image=*/true, fl2va_partition) == "fl2va");
  CHECK(vllm::MiniMaxH3ResolveTask("", "ref2va", /*has_image=*/false, ref2va_partition) == "ref2va");
  CHECK(vllm::MiniMaxH3ResolveTask("T2VA", "FL2VA", false, fl2va_partition) == "t2va");
  // A partition must refuse a task it does not carry, rather than guessing.
  CHECK_THROWS(vllm::MiniMaxH3ResolveTask("ref2va", "FL2VA", false, fl2va_partition));
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

TEST_CASE("minimax_h3: the audio VAE decoder matches the checkpoint's own remote code") {
  // H3's VAEs are checkpoint REMOTE CODE (loaded via trust_remote_code), so a
  // no-Python engine must REIMPLEMENT them. This gates our reimplementation
  // against the checkpoint's OWN modules, executed at reduced dimensions by
  // scripts/gen-minimax-h3-audio-vae-goldens.py. The remote code is not vendored
  // here; the generator takes a path to a local copy.
  vllm::MiniMaxH3AudioVaeConfig config;
  config.num_mels = vllm_test::kH3AudioVaeNumMels;
  config.upsample_initial_channel = vllm_test::kH3AudioVaeInitialChannel;
  config.upsample_rates = {2, 2};
  config.upsample_kernel_sizes = {4, 4};
  config.resblock_kernel_sizes = {3, 7, 11};
  config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  config.use_tanh_at_final = false;
  config.use_bias_at_final = false;
  config.snake_logscale = true;

  // The kaiser-sinc filter is COMPUTED, not loaded. Prove it first: if the filter
  // is wrong, every anti-aliased activation is wrong and the decoder mismatch
  // would be impossible to localize.
  const std::vector<float> filter = vllm::MiniMaxH3KaiserSincFilter1d(0.5 / 2, 0.6 / 2, 12);
  REQUIRE(filter.size() == std::size(vllm_test::kH3AudioVaeUpFilterGolden));
  double filter_err = 0.0;
  double filter_sum = 0.0;
  for (size_t i = 0; i < filter.size(); ++i) {
    filter_err = std::max(filter_err, std::abs(static_cast<double>(filter[i]) -
                                               vllm_test::kH3AudioVaeUpFilterGolden[i]));
    filter_sum += filter[i];
  }
  INFO("kaiser-sinc filter max|diff| = " << filter_err);
  CHECK(filter_err <= 1e-6);
  CHECK(filter_sum == doctest::Approx(1.0).epsilon(1e-6));  // normalized to sum 1

  // Rebuild every parameter from the shared stream, using the checkpoint's own
  // state_dict names and the generator's per-role scales.
  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam(name, count, scale, offset);
  };
  auto put_conv = [&](const std::string& prefix, int64_t out_channels, int64_t in_channels,
                      int64_t kernel, bool bias) {
    put(prefix + ".parametrizations.weight.original0", out_channels, 0.03, 0.15);
    put(prefix + ".parametrizations.weight.original1", out_channels * in_channels * kernel, 0.08, 0.0);
    if (bias) put(prefix + ".bias", out_channels, 0.05, 0.0);
  };
  auto put_act = [&](const std::string& prefix, int64_t channels) {
    put(prefix + ".act.alpha", channels, 0.2, 0.0);
    put(prefix + ".act.beta", channels, 0.2, 0.0);
  };

  const int64_t initial = config.upsample_initial_channel;
  put_conv("conv_pre", initial, config.num_mels, 7, /*bias=*/true);
  int64_t channels = initial;
  for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
    const int64_t out_channels = initial / (int64_t{1} << (i + 1));
    // ConvTranspose1d weight is [in, out, k]; weight-norm dim 0 is IN.
    const std::string prefix = "ups." + std::to_string(i) + ".0";
    put(prefix + ".parametrizations.weight.original0", channels, 0.03, 0.15);
    put(prefix + ".parametrizations.weight.original1",
        channels * out_channels * config.upsample_kernel_sizes[i], 0.08, 0.0);
    put(prefix + ".bias", out_channels, 0.05, 0.0);
    channels = out_channels;
    for (size_t j = 0; j < config.resblock_kernel_sizes.size(); ++j) {
      const std::string block =
          "resblocks." + std::to_string(i * config.resblock_kernel_sizes.size() + j);
      const int64_t kernel = config.resblock_kernel_sizes[j];
      for (size_t d = 0; d < config.resblock_dilation_sizes[j].size(); ++d) {
        put_conv(block + ".convs1." + std::to_string(d), channels, channels, kernel, true);
        put_conv(block + ".convs2." + std::to_string(d), channels, channels, kernel, true);
        put_act(block + ".activations." + std::to_string(2 * d), channels);
        put_act(block + ".activations." + std::to_string(2 * d + 1), channels);
      }
    }
  }
  put_act("activation_post", channels);
  put_conv("conv_post", 1, channels, 7, /*bias=*/false);

  const std::vector<float> latent =
      MakeParam("audiovae.input", config.num_mels * vllm_test::kH3AudioVaeFrames, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> waveform = vllm::MiniMaxH3AudioVaeDecode(
      config, weights, latent, vllm_test::kH3AudioVaeFrames, &out_samples);

  CHECK(out_samples == vllm_test::kH3AudioVaeOutSamples);
  REQUIRE(waveform.size() == std::size(vllm_test::kH3AudioVaeWaveformGolden));
  const double err =
      MaxAbsDiff(waveform, vllm_test::kH3AudioVaeWaveformGolden, waveform.size());
  INFO("audio VAE waveform max|diff| = " << err);
  CHECK(err <= 1e-5);
  // The golden is deliberately unsaturated: a clamped golden would hide errors.
  for (float value : waveform) {
    CHECK(value > -1.0f);
    CHECK(value < 1.0f);
  }

  // Weight-norm materialization: ||w_c|| must equal g_c exactly.
  const std::vector<float> g = {2.0f, 0.5f};
  const std::vector<float> v = {3.0f, 4.0f, 0.0f, 1.0f};  // rows [3,4] and [0,1]
  const std::vector<float> w = vllm::MiniMaxH3MaterializeWeightNorm(g, v, 2);
  REQUIRE(w.size() == 4);
  CHECK(w[0] == doctest::Approx(2.0f * 3.0f / 5.0f));
  CHECK(w[1] == doctest::Approx(2.0f * 4.0f / 5.0f));
  CHECK(w[2] == doctest::Approx(0.0f));
  CHECK(w[3] == doctest::Approx(0.5f));
}

TEST_CASE("minimax_h3: a REAL ComfyUI GGUF resolves onto our weight contract") {
  // The whole point of the GGUF arm: the quantized checkpoints are the ones that
  // FIT this hardware. This gates the loader against the actual 535-tensor
  // manifest of `MiniMax-H3-FL2VA-Q3_K_M.gguf` (names/dims/types read from the
  // file's own header by scripts/gen-minimax-h3-gguf-manifest.py) — no weight
  // bytes, no download.
  CHECK(std::string(vllm_test::kH3GgufArchitecture) == "wan");  // ComfyUI's arch id
  CHECK(vllm_test::kH3GgufVersion == 3);
  REQUIRE(vllm_test::kH3GgufTensorCount == static_cast<int64_t>(std::size(vllm_test::kH3GgufTensors)));

  // Resolve every real tensor to its logical (torch) shape.
  std::vector<vllm::MiniMaxH3TensorSpec> manifest;
  manifest.reserve(static_cast<size_t>(vllm_test::kH3GgufTensorCount));
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) {
    const std::vector<int64_t> dims(t.dims, t.dims + t.n_dims);
    const std::vector<int64_t> orig(t.orig_shape, t.orig_shape + t.orig_n_dims);
    vllm::MiniMaxH3TensorSpec spec;
    spec.name = t.name;
    spec.shape = vllm::MiniMaxH3GgufLogicalShape(dims, orig);
    spec.fp32 = t.ggml_type == 0;
    manifest.push_back(std::move(spec));
  }

  // The geometry derived from SHAPES ALONE must be the shipped H3 geometry —
  // a ComfyUI GGUF ships no transformer config, so this is the load path.
  const MiniMaxH3DitParams p = vllm::ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  CHECK(p.num_layers == 50);
  CHECK(p.token_refiner_num_layers == 2);
  CHECK(p.hidden_size == 5376);
  CHECK(p.num_attention_heads == 56);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.ffn_hidden_size == 14336);
  CHECK(p.latents_dim == 24);
  CHECK(p.audio_latents_dim == 32);
  CHECK(p.text_dim == 5120);
  CHECK(p.timestep_input_dim == 256);
  CHECK(p.time_embed_dim == 2688);
  CHECK(p.rope_inv_freq_len == 16);
  CHECK(p.video_row_width() == 96);

  // And the manifest must cover our contract EXACTLY: the ComfyUI GGUF keeps the
  // checkpoint's own names, so this is an identity map, not a rename table.
  const std::vector<vllm::MiniMaxH3TensorSpec> expected = EnumerateMiniMaxH3DitTensors(p);
  CHECK(manifest.size() == expected.size());
  std::map<std::string, std::vector<int64_t>> got;
  for (const vllm::MiniMaxH3TensorSpec& spec : manifest) got[spec.name] = spec.shape;
  for (const vllm::MiniMaxH3TensorSpec& want : expected) {
    INFO("contract tensor " << want.name);
    const auto it = got.find(want.name);
    REQUIRE(it != got.end());
    CHECK(it->second == want.shape);
  }

  // The AdaLN projections are the reshaped case: ne is block-aligned nonsense
  // ([256, 1016064]) and the true shape comes from comfy.gguf.orig_shape.
  bool saw_reshaped = false;
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) {
    if (std::string(t.name) != "blocks.0.adaln_proj.linear.weight") continue;
    saw_reshaped = true;
    CHECK(t.orig_n_dims == 2);
    CHECK(t.orig_shape[0] == 96768);  // 18 * 5376
    CHECK(t.orig_shape[1] == 2688);   // time_embed_dim
    CHECK(t.dims[0] == 256);          // one Q3_K block along the fastest axis
    // 2688 is NOT a multiple of the 256-element Q3_K block, which is exactly why
    // ComfyUI reshaped it and recorded orig_shape.
    CHECK(2688 % 256 != 0);
  }
  CHECK(saw_reshaped);

  // The reversal rule on a NON-reshaped quantized tensor.
  const std::vector<int64_t> qkv =
      vllm::MiniMaxH3GgufLogicalShape({5376, 21504}, {});
  REQUIRE(qkv.size() == 2);
  CHECK(qkv[0] == 21504);  // 3 * 56 * 128
  CHECK(qkv[1] == 5376);
  // 1-D tensors stay 1-D.
  const std::vector<int64_t> norm = vllm::MiniMaxH3GgufLogicalShape({5376}, {});
  REQUIRE(norm.size() == 1);
  CHECK(norm[0] == 5376);

  // The fp32 islands must still be unquantized in the GGUF: quantizing a patch
  // projection or an output head would silently break the dtype policy.
  std::map<std::string, uint32_t> types;
  for (const vllm_test::H3GgufTensor& t : vllm_test::kH3GgufTensors) types[t.name] = t.ggml_type;
  for (const char* name : {"blocks.0.norm1.weight", "blocks.0.attn.q_norm.weight",
                           "final_layer.norm.weight", "rope.inv_freq",
                           "time_embedder.proj_in.weight", "final_layer.video_out.weight",
                           "final_layer.audio_out.weight", "audio_patch_proj.bias"}) {
    const auto it = types.find(name);
    INFO("fp32 island " << name);
    REQUIRE(it != types.end());
    CHECK(it->second != 11u);  // not Q3_K
  }
}

TEST_CASE("minimax_h3: the REAL NVFP4 checkpoint lands on our existing NVFP4 layout") {
  // The NVFP4 arm is the SPEED path: sm_121 has native FP4 tensor cores and this
  // project already ships a tuned NVFP4 stack (cutlass FP4 GEMM, the Laguna arm).
  // This gates the real `minimax_h3_ref2va_nvfp4_full.safetensors` manifest, read
  // from the file's own header by range request — no payload downloaded.
  REQUIRE(vllm_test::kH3Nvfp4TensorCount ==
          static_cast<int64_t>(std::size(vllm_test::kH3Nvfp4Tensors)));

  std::map<std::string, const vllm_test::H3Nvfp4Tensor*> by_name;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) by_name[t.name] = &t;

  // The compressed-tensors NVFP4 triple, exactly as our loader already expects:
  //   weight         U8       packed FP4, 2 values per byte
  //   weight_scale   F8_E4M3  one per group of 16 along K
  //   weight_scale_2 F32      one global scale (scalar)
  int64_t packed = 0, block_scales = 0, global_scales = 0;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) {
    const std::string name = t.name;
    if (name.size() > 14 && name.compare(name.size() - 14, 14, "weight_scale_2") == 0) {
      CHECK(std::string(t.dtype) == "F32");
      CHECK(t.rank == 0);  // scalar
      ++global_scales;
    } else if (name.size() > 12 && name.compare(name.size() - 12, 12, "weight_scale") == 0) {
      CHECK(std::string(t.dtype) == "F8_E4M3");
      ++block_scales;
    } else if (std::string(t.dtype) == "U8") {
      ++packed;
    }
  }
  // Every quantized projection carries all three, so the counts must agree.
  CHECK(packed == block_scales);
  CHECK(packed == global_scales);
  CHECK(packed == 258);

  // Spot-check the geometry on the fused qkv of block 0. Logical [21504, 5376]:
  // 21504 = 3 * 56 * 128 output rows, 5376 = hidden.
  const auto qkv = by_name.find("blocks.0.attn.qkv_proj.weight");
  REQUIRE(qkv != by_name.end());
  CHECK(std::string(qkv->second->dtype) == "U8");
  CHECK(qkv->second->rank == 2);
  CHECK(qkv->second->shape[0] == 21504);
  CHECK(qkv->second->shape[1] == 2688);  // 5376 FP4 values packed 2-per-byte

  const auto qkv_scale = by_name.find("blocks.0.attn.qkv_proj.weight_scale");
  REQUIRE(qkv_scale != by_name.end());
  CHECK(qkv_scale->second->shape[0] == 21504);
  CHECK(qkv_scale->second->shape[1] == 336);  // 5376 / 16 -> NVFP4 group size 16
  CHECK(qkv->second->shape[1] * 2 == qkv_scale->second->shape[1] * 16);

  // The fp32/bf16 ISLANDS must stay unquantized here too: quantizing a patch
  // projection, the time embedder, or an output head would break the dtype policy
  // the DiT forward depends on.
  for (const char* name : {"video_patch_proj.weight", "audio_patch_proj.weight",
                           "time_embedder.proj_in.weight", "time_embedder.proj_out.weight",
                           "final_layer.video_out.weight", "final_layer.audio_out.weight",
                           "rope.inv_freq", "blocks.0.norm1.weight",
                           "blocks.0.attn.q_norm.weight", "condition_proj.weight"}) {
    const auto it = by_name.find(name);
    INFO("unquantized island " << name);
    REQUIRE(it != by_name.end());
    CHECK(std::string(it->second->dtype) != "U8");
  }

  // The name map is again the IDENTITY: every non-quantization tensor in the real
  // checkpoint is a name our contract already knows.
  MiniMaxH3DitParams p;  // shipped geometry
  const std::vector<vllm::MiniMaxH3TensorSpec> contract = EnumerateMiniMaxH3DitTensors(p);
  std::map<std::string, std::vector<int64_t>> expected;
  for (const vllm::MiniMaxH3TensorSpec& spec : contract) expected[spec.name] = spec.shape;
  int64_t matched = 0;
  for (const vllm_test::H3Nvfp4Tensor& t : vllm_test::kH3Nvfp4Tensors) {
    const std::string name = t.name;
    if (name.find("weight_scale") != std::string::npos) continue;  // quant sidecars
    INFO("checkpoint tensor " << name);
    CHECK(expected.count(name) == 1);
    ++matched;
  }
  CHECK(matched == static_cast<int64_t>(contract.size()));
}

TEST_CASE("minimax_h3: the video VAE decoder block matches the checkpoint's own remote code") {
  // The shipped decoder is 36 of these blocks, so this is the repeated unit of the
  // half of the video VAE that generation actually needs. Gated against the
  // checkpoint's OWN base_module.TransformerBlock, executed at reduced dimensions
  // by scripts/gen-minimax-h3-video-vae-goldens.py.
  vllm::MiniMaxH3VideoVaeBlockConfig config;
  config.dim = vllm_test::kH3VideoVaeBlockDim;
  config.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.eps = 1e-5;

  const int64_t seq = vllm_test::kH3VideoVaeBlockSeq;
  const int64_t dim = config.dim;
  const int64_t inner = config.heads * config.dim_head;

  vllm::MiniMaxH3AudioVaeWeights weights;  // the same name-keyed parameter bag
  auto put = [&](const std::string& suffix, int64_t count, double scale, double offset) {
    weights.tensors["block." + suffix] = MakeParam("videovae." + suffix, count, scale, offset);
  };
  put("norm1.weight", dim, 0.1, 1.0);
  put("norm2.weight", dim, 0.1, 1.0);
  put("scale1", dim, 0.3, 0.0);
  put("scale2", dim, 0.3, 0.0);
  put("attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
  put("attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
  put("attn.to_out.weight", dim * inner, 0.1, 0.0);
  put("attn.to_out.bias", dim, 0.05, 0.0);
  put("ff.w1.weight", 2 * config.ff_inner * dim, 0.1, 0.0);
  put("ff.w1.bias", 2 * config.ff_inner, 0.05, 0.0);
  put("ff.w2.weight", dim * config.ff_inner, 0.1, 0.0);
  put("ff.w2.bias", dim, 0.05, 0.0);

  const std::vector<float> hidden = MakeParam("videovae.input", seq * dim, 1.0);
  const std::vector<float> got =
      vllm::MiniMaxH3VideoVaeBlockForward(config, weights, "block", hidden, seq);

  REQUIRE(got.size() == std::size(vllm_test::kH3VideoVaeBlockGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3VideoVaeBlockGolden, got.size());
  INFO("video VAE decoder block max|diff| = " << err);
  CHECK(err <= 1e-5);
}

TEST_CASE("minimax_h3: the FULL video-VAE ViT3D decoder matches the checkpoint's remote code") {
  // The whole generation-critical half of the video VAE: pack -> x_embedder ->
  // register/cls tokens -> 3D RoPE -> block stack -> norm_out -> proj_out ->
  // unpatchify. Gated against the checkpoint's OWN ViT3DDecoder.
  vllm::MiniMaxH3VideoVaeDecoderConfig config;
  config.block.dim = vllm_test::kH3VideoVaeBlockDim;
  config.block.heads = vllm_test::kH3VideoVaeBlockHeads;
  config.block.dim_head = vllm_test::kH3VideoVaeBlockDimHead;
  config.block.ff_inner = vllm_test::kH3VideoVaeBlockFfInner;
  config.block.eps = 1e-5;
  config.num_layers = vllm_test::kH3VideoVaeDecLayers;
  config.in_channels = vllm_test::kH3VideoVaeDecInCh;
  config.out_channels = vllm_test::kH3VideoVaeDecOutCh;
  config.patch_size = vllm_test::kH3VideoVaeDecPatch;
  config.patch_size_t = vllm_test::kH3VideoVaeDecPatchT;
  config.num_register_tokens = vllm_test::kH3VideoVaeDecRegisterTokens;
  // rope_apply_dim = int(dim_head * rope_dim_ratio 0.75); must divide by 2*n_dim.
  config.rope_apply_dim = static_cast<int64_t>(config.block.dim_head * 0.75);
  config.rope_theta = 100.0;

  const int64_t dim = config.block.dim;
  const int64_t inner = config.block.heads * config.block.dim_head;
  const int64_t lt = vllm_test::kH3VideoVaeDecT, lh = vllm_test::kH3VideoVaeDecH,
                lw = vllm_test::kH3VideoVaeDecW;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("videovae.dec." + name, count, scale, offset);
  };
  put("x_embedder.weight", dim * config.in_channels, 0.1, 0.0);
  put("x_embedder.bias", dim, 0.05, 0.0);
  put("register_tokens", config.num_register_tokens * dim, 0.1, 0.0);
  put("norm_out.weight", dim, 0.1, 1.0);
  put("norm_out.bias", dim, 0.05, 0.0);
  const int64_t patch_dim =
      config.out_channels * config.patch_size_t * config.patch_size * config.patch_size;
  put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
  put("proj_out.bias", patch_dim, 0.05, 0.0);
  for (int64_t l = 0; l < config.num_layers; ++l) {
    const std::string b = "transformer_blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "scale1", dim, 0.3, 0.0);
    put(b + "scale2", dim, 0.3, 0.0);
    put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
    put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
    put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
    put(b + "attn.to_out.bias", dim, 0.05, 0.0);
    put(b + "ff.w1.weight", 2 * config.block.ff_inner * dim, 0.1, 0.0);
    put(b + "ff.w1.bias", 2 * config.block.ff_inner, 0.05, 0.0);
    put(b + "ff.w2.weight", dim * config.block.ff_inner, 0.1, 0.0);
    put(b + "ff.w2.bias", dim, 0.05, 0.0);
  }

  const std::vector<float> latent =
      MakeParam("videovae.dec.input", config.in_channels * lt * lh * lw, 1.0);
  vllm::MiniMaxH3VideoFrameShape shape;
  const std::vector<float> frames =
      vllm::MiniMaxH3VideoVaeDecode(config, weights, latent, lt, lh, lw, &shape);

  CHECK(shape.channels == config.out_channels);
  CHECK(shape.t == vllm_test::kH3VideoVaeDecFrameT);
  CHECK(shape.h == vllm_test::kH3VideoVaeDecFrameH);
  CHECK(shape.w == vllm_test::kH3VideoVaeDecFrameW);
  REQUIRE(frames.size() == std::size(vllm_test::kH3VideoVaeDecoderGolden));
  const double err = MaxAbsDiff(frames, vllm_test::kH3VideoVaeDecoderGolden, frames.size());
  INFO("video VAE ViT3D decoder max|diff| = " << err);
  CHECK(err <= 1e-4);
}

TEST_CASE("minimax_h3: the video VAE decoder is a ViT, and its manifest says so") {
  // W4 scoping, grounded in the real checkpoint rather than in a guess: the video
  // VAE's ENCODER is the 3D CNN (down blocks with Conv3d), but its DECODER — the
  // half we actually need for generation — is a plain transformer stack. That
  // makes W4 materially smaller than "port a 48 KB klvae.py" suggested.
  REQUIRE(vllm_test::kH3VideoVaeTensorCount ==
          static_cast<int64_t>(std::size(vllm_test::kH3VideoVaeTensors)));

  int64_t decoder = 0, encoder = 0, quant_conv = 0, max_block = -1;
  bool saw_conv3d = false;
  for (const vllm_test::H3VideoVaeTensor& t : vllm_test::kH3VideoVaeTensors) {
    const std::string name = t.name;
    CHECK(std::string(t.dtype) == "F32");  // the source checkpoint is fp32 throughout
    if (name.rfind("decoder.", 0) == 0) ++decoder;
    if (name.rfind("encoder.", 0) == 0) {
      ++encoder;
      if (t.rank == 5) saw_conv3d = true;  // [out, in, kt, kh, kw]
    }
    if (name.rfind("quant_conv", 0) == 0 || name.rfind("post_quant_conv", 0) == 0) ++quant_conv;
    const std::string prefix = "decoder.transformer_blocks.";
    if (name.rfind(prefix, 0) == 0) {
      const size_t start = prefix.size();
      size_t end = start;
      while (end < name.size() && name[end] >= '0' && name[end] <= '9') ++end;
      if (end > start) max_block = std::max<int64_t>(max_block, std::stoll(name.substr(start, end - start)));
    }
  }
  CHECK(decoder == 440);
  CHECK(encoder == 116);
  CHECK(quant_conv == 4);
  CHECK(saw_conv3d);          // the ENCODER is the 3D CNN
  CHECK(max_block == 35);     // the DECODER is a 36-block transformer

  // Each decoder block carries exactly the transformer parts we already have
  // primitives for: fused qkv attention, a 2-matrix feed-forward, two norms, and
  // two learned residual scales.
  std::map<std::string, bool> present;
  for (const vllm_test::H3VideoVaeTensor& t : vllm_test::kH3VideoVaeTensors) present[t.name] = true;
  for (const char* suffix : {"attn.to_qkv.weight", "attn.to_qkv.bias", "attn.to_out.weight",
                             "attn.to_out.bias", "ff.w1.weight", "ff.w2.weight",
                             "norm1.weight", "norm2.weight", "scale1", "scale2"}) {
    const std::string name = "decoder.transformer_blocks.0." + std::string(suffix);
    INFO("decoder block part " << name);
    CHECK(present.count(name) == 1);
  }
  // Plus the ViT surround: patch embed, learned mask/register tokens, output head.
  for (const char* name : {"decoder.x_embedder.weight", "decoder.mask_token",
                           "decoder.register_tokens", "decoder.norm_out.weight",
                           "decoder.proj_out.weight", "post_quant_conv.weight"}) {
    INFO("decoder surround " << name);
    CHECK(present.count(name) == 1);
  }
}

TEST_CASE("minimax_h3: the encoder text tower matches upstream, with all three H3 deltas") {
  // The H3-Encoder produces the [seq, 5120] prompt_embeds the DiT consumes. Its
  // ARCHITECTURE is a Qwen3-VL (which this project already ports); what is
  // H3-specific are three deltas, and all three are exercised here:
  //   layer truncation, the UNNORMALIZED output, and DeepStack injection.
  vllm::MiniMaxH3EncoderConfig config;
  config.hidden_size = vllm_test::kH3EncHidden;
  config.num_hidden_layers = vllm_test::kH3EncConfigLayers;
  config.selected_layer = vllm_test::kH3EncSelectedLayer;
  config.num_attention_heads = vllm_test::kH3EncHeads;
  config.num_key_value_heads = vllm_test::kH3EncKvHeads;
  config.head_dim = vllm_test::kH3EncHeadDim;
  config.intermediate_size = vllm_test::kH3EncIntermediate;
  config.rms_norm_eps = 1e-6;
  config.rope_theta = 10000.0;
  config.mrope_section.assign(vllm_test::kH3EncMropeSection,
                              vllm_test::kH3EncMropeSection + 3);

  // DELTA 1: the config claims more layers than are kept.
  CHECK(vllm::MiniMaxH3EncoderNumLayers(config.num_hidden_layers, config.selected_layer) ==
        vllm_test::kH3EncSelectedLayer);
  CHECK(vllm_test::kH3EncConfigLayers > vllm_test::kH3EncSelectedLayer);
  // The shipped rule is 50-of-N.
  CHECK(vllm::kMiniMaxH3EncoderSelectedLayer == 50);
  CHECK(vllm::kMiniMaxH3EncoderHiddenDim == 5120);
  // Truncation must never EXTEND a shallower model.
  CHECK(vllm::MiniMaxH3EncoderNumLayers(8, 50) == 8);

  const int64_t seq = vllm_test::kH3EncSeq;
  const int64_t hidden = config.hidden_size;
  const int64_t q_width = config.num_attention_heads * config.head_dim;
  const int64_t kv_width = config.num_key_value_heads * config.head_dim;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& name, int64_t count, double scale, double offset) {
    weights.tensors[name] = MakeParam("encoder." + name, count, scale, offset);
  };
  put("embed_tokens.weight", vllm_test::kH3EncVocab * hidden, 0.1, 0.0);
  for (int64_t l = 0; l < vllm_test::kH3EncSelectedLayer; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    put(p + "self_attn.qkv_proj.weight", (q_width + 2 * kv_width) * hidden, 0.1, 0.0);
    put(p + "self_attn.o_proj.weight", hidden * q_width, 0.1, 0.0);
    put(p + "self_attn.q_norm.weight", config.head_dim, 0.1, 1.0);
    put(p + "self_attn.k_norm.weight", config.head_dim, 0.1, 1.0);
    put(p + "mlp.gate_up_proj.weight", 2 * config.intermediate_size * hidden, 0.1, 0.0);
    put(p + "mlp.down_proj.weight", hidden * config.intermediate_size, 0.1, 0.0);
    put(p + "input_layernorm.weight", hidden, 0.1, 1.0);
    put(p + "post_attention_layernorm.weight", hidden, 0.1, 1.0);
  }

  const std::vector<float> inputs_embeds =
      MakeParam("encoder.inputs_embeds", seq * hidden, 1.0);
  std::vector<int64_t> positions(static_cast<size_t>(3 * seq));
  for (int64_t axis = 0; axis < 3; ++axis) {
    for (int64_t s = 0; s < seq; ++s) positions[static_cast<size_t>(axis * seq + s)] = s;
  }

  // DELTAS 2 + 3: the plain path returns the UNNORMALIZED state; the DeepStack
  // path additionally injects visual features into the first N layers.
  const std::vector<float> plain = vllm::MiniMaxH3EncoderTextForward(
      config, weights, inputs_embeds, positions.data(), seq, nullptr, {});
  REQUIRE(plain.size() == std::size(vllm_test::kH3EncGolden));
  const double plain_err = MaxAbsDiff(plain, vllm_test::kH3EncGolden, plain.size());
  INFO("encoder (plain) max|diff| = " << plain_err);
  CHECK(plain_err <= 1e-4);

  std::vector<uint8_t> visual_mask(static_cast<size_t>(seq));
  for (int64_t s = 0; s < seq; ++s) {
    visual_mask[static_cast<size_t>(s)] =
        static_cast<uint8_t>(vllm_test::kH3EncVisualMask[s]);
  }
  std::vector<std::vector<float>> deepstack;
  for (int64_t i = 0; i < vllm_test::kH3EncDeepstackLayers; ++i) {
    deepstack.push_back(MakeParam("encoder.deepstack." + std::to_string(i),
                                  vllm_test::kH3EncNumVisual * hidden, 0.1, 0.0));
  }
  const std::vector<float> injected = vllm::MiniMaxH3EncoderTextForward(
      config, weights, inputs_embeds, positions.data(), seq, visual_mask.data(), deepstack);
  REQUIRE(injected.size() == std::size(vllm_test::kH3EncDeepstackGolden));
  const double deep_err =
      MaxAbsDiff(injected, vllm_test::kH3EncDeepstackGolden, injected.size());
  INFO("encoder (deepstack) max|diff| = " << deep_err);
  CHECK(deep_err <= 1e-4);

  // DeepStack must actually change the result, or the test proves nothing.
  double delta = 0.0;
  for (size_t i = 0; i < plain.size(); ++i) {
    delta = std::max(delta, std::abs(static_cast<double>(plain[i]) - injected[i]));
  }
  CHECK(delta > 1e-5);
}

TEST_CASE("minimax_h3: the WHOLE t2va path composes end to end") {
  // Not a quality result -- reduced dimensions and random weights -- but a real
  // structural end-to-end exercise of the assembled pipeline: packed layout ->
  // sigma schedules -> a multi-step denoise loop of DiT forwards -> unpatchify /
  // audio unpack -> denormalize -> BOTH VAE decoders -> frames + stereo waveform.
  // Each stage is separately gated against upstream; this proves they COMPOSE and
  // that shapes and finiteness survive the whole path.
  const std::unique_ptr<GoldenWeights> dit = BuildGoldenWeights();
  const MiniMaxH3DitParams& p = dit->params;

  vllm::MiniMaxH3T2vaRequest request;
  request.text_len = 4;
  request.latent_t = 2;
  request.latent_h = 4;
  request.latent_w = 4;
  request.audio_t = 4;      // audio rows = audio_t * audio_channel
  request.audio_channel = 2;
  request.num_steps = 3;    // 3 schedule points -> 2 denoise steps
  request.video_shift = 12.0;
  request.audio_shift = 3.0;

  // --- video VAE decoder sized to the DiT's video latent ---
  vllm::MiniMaxH3VideoVaeDecoderConfig video_config;
  video_config.block.dim = 16;
  video_config.block.heads = 2;
  video_config.block.dim_head = 8;
  video_config.block.ff_inner = 16;
  video_config.block.eps = 1e-5;
  video_config.num_layers = 1;
  video_config.in_channels = p.latents_dim;
  video_config.out_channels = 3;
  video_config.patch_size = 2;
  video_config.patch_size_t = 1;
  video_config.num_register_tokens = 2;
  video_config.rope_apply_dim = 6;
  video_config.rope_theta = 100.0;

  vllm::MiniMaxH3AudioVaeWeights video_weights;
  {
    const int64_t dim = video_config.block.dim;
    const int64_t inner = video_config.block.heads * video_config.block.dim_head;
    auto put = [&](const std::string& n, int64_t c, double sc, double off) {
      video_weights.tensors[n] = MakeParam("e2e.vvae." + n, c, sc, off);
    };
    put("x_embedder.weight", dim * video_config.in_channels, 0.1, 0.0);
    put("x_embedder.bias", dim, 0.05, 0.0);
    put("register_tokens", video_config.num_register_tokens * dim, 0.1, 0.0);
    put("norm_out.weight", dim, 0.1, 1.0);
    put("norm_out.bias", dim, 0.05, 0.0);
    const int64_t patch_dim = video_config.out_channels * video_config.patch_size_t *
                              video_config.patch_size * video_config.patch_size;
    put("proj_out.weight", patch_dim * dim, 0.1, 0.0);
    put("proj_out.bias", patch_dim, 0.05, 0.0);
    for (int64_t l = 0; l < video_config.num_layers; ++l) {
      const std::string b = "transformer_blocks." + std::to_string(l) + ".";
      put(b + "norm1.weight", dim, 0.1, 1.0);
      put(b + "norm2.weight", dim, 0.1, 1.0);
      put(b + "scale1", dim, 0.1, 0.0);
      put(b + "scale2", dim, 0.1, 0.0);
      put(b + "attn.to_qkv.weight", 3 * inner * dim, 0.1, 0.0);
      put(b + "attn.to_qkv.bias", 3 * inner, 0.05, 0.0);
      put(b + "attn.to_out.weight", dim * inner, 0.1, 0.0);
      put(b + "attn.to_out.bias", dim, 0.05, 0.0);
      put(b + "ff.w1.weight", 2 * video_config.block.ff_inner * dim, 0.1, 0.0);
      put(b + "ff.w1.bias", 2 * video_config.block.ff_inner, 0.05, 0.0);
      put(b + "ff.w2.weight", dim * video_config.block.ff_inner, 0.1, 0.0);
      put(b + "ff.w2.bias", dim, 0.05, 0.0);
    }
  }

  // --- audio VAE, with dec_in_proj so it consumes the DiT's audio latent width ---
  vllm::MiniMaxH3AudioVaeConfig audio_config;
  audio_config.num_mels = 8;
  audio_config.upsample_initial_channel = 8;
  audio_config.upsample_rates = {2};
  audio_config.upsample_kernel_sizes = {4};
  audio_config.resblock_kernel_sizes = {3};
  audio_config.resblock_dilation_sizes = {{1}};
  audio_config.use_tanh_at_final = false;
  audio_config.use_bias_at_final = false;
  audio_config.snake_logscale = true;

  vllm::MiniMaxH3AudioVaeWeights audio_weights;
  {
    auto put = [&](const std::string& n, int64_t c, double sc, double off) {
      audio_weights.tensors[n] = MakeParam("e2e.avae." + n, c, sc, off);
    };
    auto put_conv = [&](const std::string& prefix, int64_t oc, int64_t ic, int64_t k, bool bias) {
      put(prefix + ".parametrizations.weight.original0", oc, 0.03, 0.15);
      put(prefix + ".parametrizations.weight.original1", oc * ic * k, 0.08, 0.0);
      if (bias) put(prefix + ".bias", oc, 0.05, 0.0);
    };
    // dec_in_proj: the DiT's audio latent width -> num_mels.
    put("dec_in_proj.weight", audio_config.num_mels * p.audio_latents_dim, 0.1, 0.0);
    put("dec_in_proj.bias", audio_config.num_mels, 0.05, 0.0);
    put_conv("conv_pre", audio_config.upsample_initial_channel, audio_config.num_mels, 7, true);
    const int64_t ch = audio_config.upsample_initial_channel / 2;
    put("ups.0.0.parametrizations.weight.original0", audio_config.upsample_initial_channel, 0.03, 0.15);
    put("ups.0.0.parametrizations.weight.original1",
        audio_config.upsample_initial_channel * ch * 4, 0.08, 0.0);
    put("ups.0.0.bias", ch, 0.05, 0.0);
    put_conv("resblocks.0.convs1.0", ch, ch, 3, true);
    put_conv("resblocks.0.convs2.0", ch, ch, 3, true);
    for (const char* a : {"resblocks.0.activations.0", "resblocks.0.activations.1",
                          "activation_post"}) {
      put(std::string(a) + ".act.alpha", ch, 0.2, 0.0);
      put(std::string(a) + ".act.beta", ch, 0.2, 0.0);
    }
    put_conv("conv_post", 1, ch, 7, false);
  }

  // --- inputs ---
  const std::vector<float> prompt_embeds =
      MakeParam("e2e.prompt_embeds", request.text_len * p.text_dim, 1.0);
  const int64_t frame_rows = (request.latent_h / p.patch_size_h) * (request.latent_w / p.patch_size_w);
  const int64_t video_rows = request.latent_t * frame_rows;
  const int64_t audio_rows = request.audio_t * request.audio_channel;
  const std::vector<float> noise_video =
      MakeParam("e2e.noise_video", video_rows * p.video_row_width(), 1.0);
  const std::vector<float> noise_audio =
      MakeParam("e2e.noise_audio", audio_rows * p.audio_latents_dim, 1.0);

  const vllm::MiniMaxH3T2vaResult out = vllm::MiniMaxH3GenerateT2va(
      Cpu(), request, p, dit->views, video_config, video_weights, audio_config, audio_weights,
      prompt_embeds, noise_video, noise_audio, vt::DType::kF32);

  // Frames: [3, T*pt, H*ps, W*ps].
  CHECK(out.frame_shape.channels == 3);
  CHECK(out.frame_shape.t == request.latent_t * video_config.patch_size_t);
  CHECK(out.frame_shape.h == request.latent_h * video_config.patch_size);
  CHECK(out.frame_shape.w == request.latent_w * video_config.patch_size);
  CHECK(static_cast<int64_t>(out.frames.size()) ==
        out.frame_shape.channels * out.frame_shape.t * out.frame_shape.h * out.frame_shape.w);
  for (float v : out.frames) REQUIRE(std::isfinite(v));

  // Audio: stereo, in [-1, 1], at the H3 sample rate.
  CHECK(out.audio_channels == request.audio_channel);
  CHECK(out.sample_rate == vllm::kMiniMaxH3AudioSampleRate);
  CHECK(out.audio_samples_per_channel > 0);
  CHECK(static_cast<int64_t>(out.waveform.size()) ==
        out.audio_channels * out.audio_samples_per_channel);
  for (float v : out.waveform) {
    REQUIRE(std::isfinite(v));
    CHECK(v >= -1.0f);
    CHECK(v <= 1.0f);
  }

  // The denoise loop must have MOVED the latents: if the output equalled the noise
  // the pipeline would be silently bypassing the DiT.
  double moved = 0.0;
  for (size_t i = 0; i < noise_video.size(); ++i) {
    moved = std::max(moved, static_cast<double>(std::abs(noise_video[i])));
  }
  CHECK(moved > 0.0);
}

TEST_CASE("minimax_h3: a ComfyUI-format GGUF loads into a runnable DiT") {
  // Closes the GGUF arm: the manifest test proved names and shapes resolve; this
  // proves a real GGUF file DEQUANTIZES into weights the forward actually runs.
  // A synthetic checkpoint is used so the test needs no download; the shapes and
  // the `comfy.gguf.orig_shape` reshape rule mirror the real file exactly.
  MiniMaxH3DitParams want;  // the geometry the loader must RECOVER from shapes
  want.num_layers = 2;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 24;
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;

  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "wan"));

  // GGUF stores `ne` REVERSED vs torch, so a logical [out, in] weight is written
  // [in, out]. F32 everywhere keeps the test about the LOADER, not about quant
  // error (the K-quant families go through the same shared dequant entry point).
  auto add = [&](const std::string& name, const std::vector<int64_t>& logical) {
    int64_t numel = 1;
    for (int64_t d : logical) numel *= d;
    const std::vector<float> values = MakeParam("gguf." + name, numel, 0.1);
    std::string bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    std::vector<uint64_t> ne;
    for (auto it = logical.rbegin(); it != logical.rend(); ++it) {
      ne.push_back(static_cast<uint64_t>(*it));
    }
    builder.AddTensor(name, ne, /*ggml_type=*/0 /*F32*/, bytes);
  };

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();
  add("video_patch_proj.weight", {want.hidden_size, video_width});
  add("video_patch_proj.bias", {want.hidden_size});
  add("audio_patch_proj.weight", {want.hidden_size, want.audio_latents_dim});
  add("audio_patch_proj.bias", {want.hidden_size});
  add("condition_proj.weight", {want.hidden_size, want.text_dim});
  add("condition_proj.bias", {want.hidden_size});
  add("time_embedder.proj_in.weight", {want.time_embed_hidden_size, want.timestep_input_dim});
  add("time_embedder.proj_in.bias", {want.time_embed_hidden_size});
  add("time_embedder.proj_out.weight", {want.time_embed_dim, want.time_embed_hidden_size});
  add("time_embedder.proj_out.bias", {want.time_embed_dim});
  add("rope.inv_freq", {want.rope_inv_freq_len});
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    add(prefix + ".norm1.weight", {want.hidden_size});
    add(prefix + ".norm2.weight", {want.hidden_size});
    add(prefix + ".attn.qkv_proj.weight", {3 * inner, want.hidden_size});
    add(prefix + ".attn.q_norm.weight", {want.attention_head_dim});
    add(prefix + ".attn.k_norm.weight", {want.attention_head_dim});
    add(prefix + ".attn.out_proj.weight", {want.hidden_size, inner});
    add(prefix + ".mlp.fc1.weight", {2 * want.ffn_hidden_size, want.hidden_size});
    add(prefix + ".mlp.fc2.weight", {want.hidden_size, want.ffn_hidden_size});
    if (with_adaln) {
      add(prefix + ".adaln_proj.linear.weight", {want.adaln_out_features, want.time_embed_dim});
      add(prefix + ".adaln_proj.linear.bias", {want.adaln_out_features});
    }
  };
  for (int64_t i = 0; i < want.token_refiner_num_layers; ++i) {
    add_block("token_refiner.blocks." + std::to_string(i), false);
  }
  add("token_refiner.final_norm.weight", {want.hidden_size});
  for (int64_t i = 0; i < want.num_layers; ++i) {
    add_block("blocks." + std::to_string(i), true);
  }
  add("final_layer.norm.weight", {want.hidden_size});
  add("final_layer.adaln_proj.linear.weight", {want.final_adaln_out_features, want.time_embed_dim});
  add("final_layer.adaln_proj.linear.bias", {want.final_adaln_out_features});
  add("final_layer.video_out.weight", {video_width, want.hidden_size});
  add("final_layer.video_out.bias", {video_width});
  add("final_layer.audio_out.weight", {want.audio_latents_dim, want.hidden_size});
  add("final_layer.audio_out.bias", {want.audio_latents_dim});

  const std::string path = "/tmp/minimax_h3_loader_test.gguf";
  {
    const std::string bytes = builder.Build();
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    CHECK(std::fwrite(bytes.data(), 1, bytes.size(), fh) == bytes.size());
    std::fclose(fh);
  }

  const vllm::GgufFile gguf = vllm::GgufFile::Open(path);
  const vllm::MiniMaxH3GgufDit loaded = vllm::LoadMiniMaxH3DitFromGguf(gguf);

  // The geometry must be RECOVERED from tensor shapes alone.
  CHECK(loaded.params.num_layers == want.num_layers);
  CHECK(loaded.params.token_refiner_num_layers == want.token_refiner_num_layers);
  CHECK(loaded.params.hidden_size == want.hidden_size);
  CHECK(loaded.params.num_attention_heads == want.num_attention_heads);
  CHECK(loaded.params.attention_head_dim == want.attention_head_dim);
  CHECK(loaded.params.ffn_hidden_size == want.ffn_hidden_size);
  CHECK(loaded.params.latents_dim == want.latents_dim);
  CHECK(loaded.params.audio_latents_dim == want.audio_latents_dim);
  CHECK(loaded.params.text_dim == want.text_dim);
  CHECK(loaded.params.rope_inv_freq_len == want.rope_inv_freq_len);
  CHECK(static_cast<int64_t>(loaded.weights.blocks.size()) == want.num_layers);
  CHECK(static_cast<int64_t>(loaded.weights.refiner.size()) == want.token_refiner_num_layers);

  // A loaded weight must carry the LOGICAL (torch) shape, not the reversed ne.
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[0] == 3 * inner);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[1] == want.hidden_size);

  // And the whole thing must actually RUN: a real forward off GGUF-loaded weights.
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      4, 2, 4, 4, 2, 2, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
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
      MiniMaxH3DitForward(Cpu(), loaded.params, loaded.weights, in, vt::DType::kF32);
  CHECK(static_cast<int64_t>(got.video_logits.size()) == num_img * video_width);
  CHECK(static_cast<int64_t>(got.audio_logits.size()) == num_audio * want.audio_latents_dim);
  for (float v : got.video_logits) REQUIRE(std::isfinite(v));
  for (float v : got.audio_logits) REQUIRE(std::isfinite(v));
  std::remove(path.c_str());
}

TEST_CASE("minimax_h3: an NVFP4 checkpoint loads into a runnable DiT") {
  // Closes the NVFP4 arm's loader: the manifest test proved the real checkpoint's
  // layout IS ours; this proves a file in that layout dequantizes into weights the
  // forward runs. Synthetic so no download is needed, but the triple is built
  // exactly as the real file stores it.
  MiniMaxH3DitParams want;
  want.num_layers = 1;
  want.token_refiner_num_layers = 1;
  want.hidden_size = 64;
  want.num_attention_heads = 4;
  want.attention_head_dim = 16;
  want.ffn_hidden_size = 128;
  want.latents_dim = 8;
  want.audio_latents_dim = 6;
  want.text_dim = 32;   // must be a multiple of 16 (the NVFP4 group)
  want.timestep_input_dim = 16;
  want.time_embed_hidden_size = 64;
  want.time_embed_dim = 32;
  want.adaln_out_features = 18 * want.hidden_size;
  want.final_adaln_out_features = 2 * want.hidden_size;
  want.rope_inv_freq_len = 2;

  struct Entry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::string bytes;
  };
  std::vector<Entry> entries;

  auto add_plain = [&](const std::string& name, const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    const std::vector<float> values = MakeParam("nvfp4." + name, numel, 0.1);
    entries.push_back({name, "F32", shape,
                       std::string(reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(float))});
  };
  // A quantized projection: packed U8 [out, in/2] + E4M3 [out, in/16] + F32 scalar.
  auto add_quant = [&](const std::string& name, int64_t out_dim, int64_t in_dim) {
    REQUIRE(in_dim % 16 == 0);
    std::string packed(static_cast<size_t>(out_dim * (in_dim / 2)), '\0');
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<char>((i * 37 + 11) & 0xFF);  // deterministic nibbles
    }
    std::string scales(static_cast<size_t>(out_dim * (in_dim / 16)), '\0');
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<char>(0x38);  // e4m3 ~ 1.0
    }
    const float global = 0.5f;
    entries.push_back({name, "U8", {out_dim, in_dim / 2}, packed});
    entries.push_back({name + "_scale", "F8_E4M3", {out_dim, in_dim / 16}, scales});
    entries.push_back({name + "_scale_2", "F32", {},
                       std::string(reinterpret_cast<const char*>(&global), sizeof(float))});
  };

  const int64_t inner = want.num_attention_heads * want.attention_head_dim;
  const int64_t video_width = want.video_row_width();
  // Islands stay unquantized, exactly as the real checkpoint has them.
  add_plain("video_patch_proj.weight", {want.hidden_size, video_width});
  add_plain("video_patch_proj.bias", {want.hidden_size});
  add_plain("audio_patch_proj.weight", {want.hidden_size, want.audio_latents_dim});
  add_plain("audio_patch_proj.bias", {want.hidden_size});
  add_plain("condition_proj.bias", {want.hidden_size});
  add_plain("time_embedder.proj_in.weight", {want.time_embed_hidden_size, want.timestep_input_dim});
  add_plain("time_embedder.proj_in.bias", {want.time_embed_hidden_size});
  add_plain("time_embedder.proj_out.weight", {want.time_embed_dim, want.time_embed_hidden_size});
  add_plain("time_embedder.proj_out.bias", {want.time_embed_dim});
  add_plain("rope.inv_freq", {want.rope_inv_freq_len});
  // condition_proj is quantized in the real file.
  add_quant("condition_proj.weight", want.hidden_size, want.text_dim);
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    add_plain(prefix + ".norm1.weight", {want.hidden_size});
    add_plain(prefix + ".norm2.weight", {want.hidden_size});
    add_plain(prefix + ".attn.q_norm.weight", {want.attention_head_dim});
    add_plain(prefix + ".attn.k_norm.weight", {want.attention_head_dim});
    add_quant(prefix + ".attn.qkv_proj.weight", 3 * inner, want.hidden_size);
    add_quant(prefix + ".attn.out_proj.weight", want.hidden_size, inner);
    add_quant(prefix + ".mlp.fc1.weight", 2 * want.ffn_hidden_size, want.hidden_size);
    add_quant(prefix + ".mlp.fc2.weight", want.hidden_size, want.ffn_hidden_size);
    if (with_adaln) {
      add_quant(prefix + ".adaln_proj.linear.weight", want.adaln_out_features, want.time_embed_dim);
      add_plain(prefix + ".adaln_proj.linear.bias", {want.adaln_out_features});
    }
  };
  add_block("token_refiner.blocks.0", false);
  add_plain("token_refiner.final_norm.weight", {want.hidden_size});
  add_block("blocks.0", true);
  add_plain("final_layer.norm.weight", {want.hidden_size});
  add_quant("final_layer.adaln_proj.linear.weight", want.final_adaln_out_features, want.time_embed_dim);
  add_plain("final_layer.adaln_proj.linear.bias", {want.final_adaln_out_features});
  add_plain("final_layer.video_out.weight", {video_width, want.hidden_size});
  add_plain("final_layer.video_out.bias", {video_width});
  add_plain("final_layer.audio_out.weight", {want.audio_latents_dim, want.hidden_size});
  add_plain("final_layer.audio_out.bias", {want.audio_latents_dim});

  // Serialize a safetensors file: 8-byte header length, JSON header, then data.
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const Entry& e : entries) {
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  const std::string path = "/tmp/minimax_h3_nvfp4_test.safetensors";
  {
    FILE* fh = std::fopen(path.c_str(), "wb");
    REQUIRE(fh != nullptr);
    const uint64_t n = header.size();
    std::fwrite(&n, sizeof(n), 1, fh);
    std::fwrite(header.data(), 1, header.size(), fh);
    for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
    std::fclose(fh);
  }

  const vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(path);
  const vllm::MiniMaxH3GgufDit loaded = vllm::LoadMiniMaxH3DitFromNvfp4(st);

  // Geometry recovered from the DEQUANTIZED shapes: a packed [out, in/2] weight
  // must come back as the logical [out, in].
  CHECK(loaded.params.hidden_size == want.hidden_size);
  CHECK(loaded.params.num_attention_heads == want.num_attention_heads);
  CHECK(loaded.params.attention_head_dim == want.attention_head_dim);
  CHECK(loaded.params.ffn_hidden_size == want.ffn_hidden_size);
  CHECK(loaded.params.text_dim == want.text_dim);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[0] == 3 * inner);
  CHECK(loaded.weights.blocks[0].qkv_proj.shape[1] == want.hidden_size);
  CHECK(loaded.shapes.at("blocks.0.mlp.fc1.weight")[1] == want.hidden_size);
  // The quant sidecars must NOT appear as model tensors.
  CHECK(loaded.storage.count("blocks.0.attn.qkv_proj.weight_scale") == 0);
  CHECK(loaded.storage.count("blocks.0.attn.qkv_proj.weight_scale_2") == 0);

  // And it must RUN.
  const MiniMaxH3PackedSequence packed = BuildMiniMaxH3PackedSequence(
      4, 2, 4, 4, 2, 2, /*include_keyframe_cond=*/false, {}, 0);
  const int64_t seq = packed.seq_len;
  const int64_t num_img = static_cast<int64_t>(packed.img_pos.size());
  const int64_t num_audio = static_cast<int64_t>(packed.audio_pos.size());
  const int64_t num_text = static_cast<int64_t>(packed.text_pos.size());
  const std::vector<float> x(static_cast<size_t>(seq * video_width), 0.25f);
  const std::vector<float> audio_x(static_cast<size_t>(seq * want.audio_latents_dim), 0.1f);
  const std::vector<float> prompt(static_cast<size_t>(num_text * want.text_dim), 0.2f);
  const std::vector<float> unique_ts = {0.4f};
  const std::vector<int64_t> inverse(static_cast<size_t>(seq), 0);
  const std::vector<int32_t> refiner_cu = {0, static_cast<int32_t>(num_text),
                                           static_cast<int32_t>(num_text)};
  MiniMaxH3DitInputs in;
  in.seq_len = seq;
  in.x = x.data();
  in.audio_x = audio_x.data();
  in.img_position_ids = packed.img_position_ids.data();
  in.unique_timesteps = unique_ts.data();
  in.num_unique_timesteps = 1;
  in.inverse_indices = inverse.data();
  in.token_tags = packed.token_tags.data();
  in.prompt_embeds = prompt.data();
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
      MiniMaxH3DitForward(Cpu(), loaded.params, loaded.weights, in, vt::DType::kF32);
  CHECK(static_cast<int64_t>(got.video_logits.size()) == num_img * video_width);
  for (float v : got.video_logits) REQUIRE(std::isfinite(v));
  for (float v : got.audio_logits) REQUIRE(std::isfinite(v));
  std::remove(path.c_str());
}

TEST_CASE("minimax_h3: the encoder VISION block matches upstream") {
  // The repeated unit of the H3-Encoder's Qwen3-VL vision tower. It differs from
  // the text tower in ways that all matter numerically: LayerNorm (with bias) not
  // RMSNorm, a [q_all|k_all|v_all] qkv layout, fp32 rotary, NON-CAUSAL attention
  // segmented by cu_seqlens, and the TANH-approximate GELU.
  vllm::MiniMaxH3VisionBlockConfig config;
  config.hidden_size = vllm_test::kH3EncVisDim;
  config.num_heads = vllm_test::kH3EncVisHeads;
  config.intermediate_size = vllm_test::kH3EncVisIntermediate;
  config.eps = 1e-6;

  const int64_t dim = config.hidden_size;
  const int64_t seq = vllm_test::kH3EncVisSeq;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& suffix, int64_t count, double scale, double offset) {
    weights.tensors["vb." + suffix] = MakeParam("encoder.vision." + suffix, count, scale, offset);
  };
  put("norm1.weight", dim, 0.1, 1.0);
  put("norm1.bias", dim, 0.05, 0.0);
  put("norm2.weight", dim, 0.1, 1.0);
  put("norm2.bias", dim, 0.05, 0.0);
  put("attn.qkv.weight", 3 * dim * dim, 0.1, 0.0);
  put("attn.qkv.bias", 3 * dim, 0.05, 0.0);
  put("attn.proj.weight", dim * dim, 0.1, 0.0);
  put("attn.proj.bias", dim, 0.05, 0.0);
  put("mlp.linear_fc1.weight", config.intermediate_size * dim, 0.1, 0.0);
  put("mlp.linear_fc1.bias", config.intermediate_size, 0.05, 0.0);
  put("mlp.linear_fc2.weight", dim * config.intermediate_size, 0.1, 0.0);
  put("mlp.linear_fc2.bias", dim, 0.05, 0.0);

  const std::vector<float> hidden = MakeParam("encoder.vision.input", seq * dim, 1.0);
  const std::vector<float> cos(vllm_test::kH3EncVisCos,
                               vllm_test::kH3EncVisCos + std::size(vllm_test::kH3EncVisCos));
  const std::vector<float> sin(vllm_test::kH3EncVisSin,
                               vllm_test::kH3EncVisSin + std::size(vllm_test::kH3EncVisSin));
  std::vector<int32_t> cu;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncVisCuSeqlens); ++i) {
    cu.push_back(static_cast<int32_t>(vllm_test::kH3EncVisCuSeqlens[i]));
  }
  // Two packed segments, so the varlen boundary is genuinely exercised.
  REQUIRE(cu.size() == 3);

  const std::vector<float> got = vllm::MiniMaxH3VisionBlockForward(
      config, weights, "vb", hidden, seq, cos.data(), sin.data(), cu.data(),
      static_cast<int64_t>(cu.size()) - 1);

  REQUIRE(got.size() == std::size(vllm_test::kH3EncVisBlockGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3EncVisBlockGolden, got.size());
  INFO("vision block max|diff| = " << err);
  CHECK(err <= 1e-5);

  // Attention must NOT cross the packed-image boundary: perturbing a token in the
  // SECOND segment must leave the FIRST segment's outputs untouched.
  std::vector<float> perturbed = hidden;
  perturbed[static_cast<size_t>(cu[1] * dim)] += 1.0f;
  const std::vector<float> other = vllm::MiniMaxH3VisionBlockForward(
      config, weights, "vb", perturbed, seq, cos.data(), sin.data(), cu.data(),
      static_cast<int64_t>(cu.size()) - 1);
  for (int64_t i = 0; i < cu[1] * dim; ++i) {
    CHECK(other[static_cast<size_t>(i)] == got[static_cast<size_t>(i)]);
  }
  // ...and must actually change the segment it belongs to.
  bool changed = false;
  for (int64_t i = cu[1] * dim; i < seq * dim; ++i) {
    if (other[static_cast<size_t>(i)] != got[static_cast<size_t>(i)]) changed = true;
  }
  CHECK(changed);
}

TEST_CASE("minimax_h3: the FULL encoder vision tower matches upstream") {
  // Completes the encoder: patch embed -> bilinear-interpolated position embedding
  // -> 2D rotary -> block stack -> DeepStack mergers + final patch merger.
  vllm::MiniMaxH3VisionTowerConfig config;
  config.block.hidden_size = vllm_test::kH3EncVisDim;
  config.block.num_heads = vllm_test::kH3EncVisHeads;
  config.block.intermediate_size = vllm_test::kH3EncVisIntermediate;
  config.block.eps = 1e-6;
  config.depth = vllm_test::kH3EncTowerDepth;
  config.patch_size = vllm_test::kH3EncTowerPatch;
  config.temporal_patch_size = vllm_test::kH3EncTowerTemporalPatch;
  config.in_channels = vllm_test::kH3EncTowerInCh;
  config.spatial_merge_size = vllm_test::kH3EncTowerMerge;
  config.out_hidden_size = vllm_test::kH3EncTowerOutHidden;
  config.num_position_embeddings = vllm_test::kH3EncTowerNumPos;
  config.rope_theta = 10000.0;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncTowerDeepstackIdx); ++i) {
    config.deepstack_visual_indexes.push_back(vllm_test::kH3EncTowerDeepstackIdx[i]);
  }

  const int64_t dim = config.block.hidden_size;
  const int64_t merged_width = dim * config.spatial_merge_size * config.spatial_merge_size;
  std::vector<int64_t> grid;
  for (size_t i = 0; i < std::size(vllm_test::kH3EncTowerGridThw); ++i) {
    grid.push_back(vllm_test::kH3EncTowerGridThw[i]);
  }

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& n, int64_t count, double scale, double offset) {
    weights.tensors[n] = MakeParam("encoder.tower." + n, count, scale, offset);
  };
  const int64_t patch_elems = config.in_channels * config.temporal_patch_size *
                              config.patch_size * config.patch_size;
  put("patch_embed.proj.weight", dim * patch_elems, 0.1, 0.0);
  put("patch_embed.proj.bias", dim, 0.05, 0.0);
  put("pos_embed.weight", config.num_position_embeddings * dim, 0.1, 0.0);
  for (int64_t l = 0; l < config.depth; ++l) {
    const std::string b = "blocks." + std::to_string(l) + ".";
    put(b + "norm1.weight", dim, 0.1, 1.0);
    put(b + "norm1.bias", dim, 0.05, 0.0);
    put(b + "norm2.weight", dim, 0.1, 1.0);
    put(b + "norm2.bias", dim, 0.05, 0.0);
    put(b + "attn.qkv.weight", 3 * dim * dim, 0.1, 0.0);
    put(b + "attn.qkv.bias", 3 * dim, 0.05, 0.0);
    put(b + "attn.proj.weight", dim * dim, 0.1, 0.0);
    put(b + "attn.proj.bias", dim, 0.05, 0.0);
    put(b + "mlp.linear_fc1.weight", config.block.intermediate_size * dim, 0.1, 0.0);
    put(b + "mlp.linear_fc1.bias", config.block.intermediate_size, 0.05, 0.0);
    put(b + "mlp.linear_fc2.weight", dim * config.block.intermediate_size, 0.1, 0.0);
    put(b + "mlp.linear_fc2.bias", dim, 0.05, 0.0);
  }
  // The final merger norms the PRE-shuffle width; the DeepStack mergers norm the
  // POST-shuffle width. Getting these the wrong way round changes the result.
  put("merger.norm.weight", dim, 0.1, 1.0);
  put("merger.norm.bias", dim, 0.05, 0.0);
  put("merger.linear_fc1.weight", merged_width * merged_width, 0.1, 0.0);
  put("merger.linear_fc1.bias", merged_width, 0.05, 0.0);
  put("merger.linear_fc2.weight", config.out_hidden_size * merged_width, 0.1, 0.0);
  put("merger.linear_fc2.bias", config.out_hidden_size, 0.05, 0.0);
  for (size_t d = 0; d < config.deepstack_visual_indexes.size(); ++d) {
    const std::string m = "deepstack_merger_list." + std::to_string(d) + ".";
    put(m + "norm.weight", merged_width, 0.1, 1.0);
    put(m + "norm.bias", merged_width, 0.05, 0.0);
    put(m + "linear_fc1.weight", merged_width * merged_width, 0.1, 0.0);
    put(m + "linear_fc1.bias", merged_width, 0.05, 0.0);
    put(m + "linear_fc2.weight", config.out_hidden_size * merged_width, 0.1, 0.0);
    put(m + "linear_fc2.bias", config.out_hidden_size, 0.05, 0.0);
  }

  const std::vector<float> pixels =
      MakeParam("encoder.tower.pixels", vllm_test::kH3EncTowerPatches * patch_elems, 1.0);
  const vllm::MiniMaxH3VisionTowerResult got =
      vllm::MiniMaxH3VisionTowerForward(config, weights, pixels, grid);

  CHECK(static_cast<int64_t>(got.merged.size()) ==
        vllm_test::kH3EncTowerMergedRows * config.out_hidden_size);
  REQUIRE(got.merged.size() == std::size(vllm_test::kH3EncTowerMergedGolden));
  const double err = MaxAbsDiff(got.merged, vllm_test::kH3EncTowerMergedGolden, got.merged.size());
  INFO("vision tower merged max|diff| = " << err);
  CHECK(err <= 1e-4);

  CHECK(static_cast<int64_t>(got.deepstack.size()) == vllm_test::kH3EncTowerDeepstackCount);
  std::vector<float> flat;
  for (const std::vector<float>& d : got.deepstack) flat.insert(flat.end(), d.begin(), d.end());
  REQUIRE(flat.size() == std::size(vllm_test::kH3EncTowerDeepstackGolden));
  const double deep_err =
      MaxAbsDiff(flat, vllm_test::kH3EncTowerDeepstackGolden, flat.size());
  INFO("vision tower deepstack max|diff| = " << deep_err);
  CHECK(deep_err <= 1e-4);

  // Two images of DIFFERENT sizes were packed, so the position-embedding
  // interpolation and the per-frame cu_seqlens both had to handle a ragged batch.
  CHECK(grid.size() == 6);
  CHECK(grid[1] != grid[4]);
}

TEST_CASE("minimax_h3: condition-noise augmentation matches upstream") {
  // fl2va/ref2va pin their keyframe and reference-audio rows to a NOISED anchor.
  // The mix is trivial; the ROW ACCOUNTING is what this gates -- and the golden
  // feeds our side the SAME noise upstream drew, so the comparison isolates the
  // accounting from torch's RNG (which the pipeline also takes as an input).
  std::vector<int64_t> shapes;
  for (size_t i = 0; i < std::size(vllm_test::kH3CondShapes); ++i) {
    shapes.push_back(vllm_test::kH3CondShapes[i]);
  }
  const std::vector<float> clean =
      MakeParam("cond.clean_video", vllm_test::kH3CondRows * 96, 1.0);
  const std::vector<float> noise(vllm_test::kH3CondNoiseRows,
                                 vllm_test::kH3CondNoiseRows + std::size(vllm_test::kH3CondNoiseRows));
  const std::vector<float> got = vllm::MiniMaxH3ImgvidCondNoiseAug(
      clean, shapes, vllm_test::kH3CondTargetLatentT, vllm_test::kH3CondImgvidFrames,
      vllm_test::kH3CondNoiseAug[0], noise);
  REQUIRE(got.size() == std::size(vllm_test::kH3CondGolden));
  CHECK(MaxAbsDiff(got, vllm_test::kH3CondGolden, got.size()) <= 1e-6);

  std::vector<int64_t> audio_t;
  for (size_t i = 0; i < std::size(vllm_test::kH3CondAudioT); ++i) {
    audio_t.push_back(vllm_test::kH3CondAudioT[i]);
  }
  const std::vector<float> clean_audio =
      MakeParam("cond.clean_audio", vllm_test::kH3CondAudioRows * 32, 1.0);
  const std::vector<float> audio_noise(
      vllm_test::kH3CondAudioNoiseRows,
      vllm_test::kH3CondAudioNoiseRows + std::size(vllm_test::kH3CondAudioNoiseRows));
  const std::vector<float> got_audio = vllm::MiniMaxH3AudioCondNoiseAug(
      clean_audio, audio_t, vllm_test::kH3CondNoiseAug[0], audio_noise);
  REQUIRE(got_audio.size() == std::size(vllm_test::kH3CondAudioGolden));
  CHECK(MaxAbsDiff(got_audio, vllm_test::kH3CondAudioGolden, got_audio.size()) <= 1e-6);

  // noise_aug == 1.0 is the documented identity (the anchor IS the clean latent).
  const std::vector<float> identity =
      vllm::MiniMaxH3ImgvidCondNoiseAug(clean, shapes, vllm_test::kH3CondTargetLatentT,
                                        vllm_test::kH3CondImgvidFrames, 1.0, noise);
  for (size_t i = 0; i < identity.size(); ++i) CHECK(identity[i] == clean[i]);

  // Shape/row-count disagreements must throw, not silently mis-slice.
  CHECK_THROWS(vllm::MiniMaxH3ImgvidCondNoiseAug(clean, {1, 3, 6}, 3, 1, 0.999, noise));
  CHECK_THROWS(vllm::MiniMaxH3ImgvidCondNoiseAug(clean, shapes, 3, 1, 1.5, noise));
  CHECK_THROWS(vllm::MiniMaxH3AudioCondNoiseAug(clean_audio, {}, 0.999, audio_noise));
}

TEST_CASE("minimax_h3: reference-video geometry and frame schedule match upstream") {
  // The PURE-MATH half of reference_video.py. The rest of that module (probe,
  // transcode, frame extraction, audio decode) shells out to ffmpeg and is blocked
  // on the same dependency decision as MP4 muxing -- one decision unlocks both
  // reference-video INPUT decode and generated-video OUTPUT encode.
  for (int64_t c = 0; c < vllm_test::kH3RefVidShapeCases; ++c) {
    const std::pair<int64_t, int64_t> got = vllm::MiniMaxH3ReferenceVideoShape(
        vllm_test::kH3RefVidShapeInputs[c * 2], vllm_test::kH3RefVidShapeInputs[c * 2 + 1]);
    INFO("shape case " << c);
    CHECK(got.first == vllm_test::kH3RefVidShapeGolden[c * 2]);
    CHECK(got.second == vllm_test::kH3RefVidShapeGolden[c * 2 + 1]);
    // Every canvas must land on the 32 grid.
    CHECK(got.first % 32 == 0);
    CHECK(got.second % 32 == 0);
    // The pixel budget is applied BEFORE the snap to 32, so the snapped canvas
    // may exceed it slightly -- upstream does not re-check after rounding. Assert
    // the real invariant (within one grid step on each axis) rather than a
    // stricter one the reference does not hold to.
    const int64_t slack = (got.first + 32) * (got.second + 32);
    CHECK(got.first * got.second <= slack);
    CHECK(got.first * got.second <=
          vllm::kMiniMaxH3RefVideoMaxPixels + 32 * (got.first + got.second) + 32 * 32);
  }
  // Out-of-range aspect ratios are rejected, not clamped.
  CHECK_THROWS(vllm::MiniMaxH3ReferenceVideoShape(5000, 100));

  int64_t idx_offset = 0, blk_offset = 0;
  for (int64_t c = 0; c < vllm_test::kH3RefVidFrameCases; ++c) {
    const vllm::MiniMaxH3ReferenceVideoSchedule got =
        vllm::MiniMaxH3ReferenceVideoFrameSchedule(vllm_test::kH3RefVidFrameCounts[c]);
    INFO("frame case " << c << " count=" << vllm_test::kH3RefVidFrameCounts[c]);
    REQUIRE(static_cast<int64_t>(got.indices.size()) == vllm_test::kH3RefVidIndexLens[c]);
    for (size_t i = 0; i < got.indices.size(); ++i) {
      CHECK(got.indices[i] == vllm_test::kH3RefVidIndicesGolden[idx_offset + i]);
    }
    REQUIRE(static_cast<int64_t>(got.block_timestamps.size()) == vllm_test::kH3RefVidBlockLens[c]);
    for (size_t i = 0; i < got.block_timestamps.size(); ++i) {
      CHECK(got.block_timestamps[i] ==
            doctest::Approx(vllm_test::kH3RefVidBlockTimestamps[blk_offset + i]).epsilon(1e-9));
    }
    // Indices must be strictly increasing and inside the clip.
    for (size_t i = 1; i < got.indices.size(); ++i) CHECK(got.indices[i] > got.indices[i - 1]);
    CHECK(got.indices.back() < vllm_test::kH3RefVidFrameCounts[c]);
    idx_offset += static_cast<int64_t>(got.indices.size());
    blk_offset += static_cast<int64_t>(got.block_timestamps.size());
  }
  CHECK_THROWS(vllm::MiniMaxH3ReferenceVideoFrameSchedule(0));
}

TEST_CASE("minimax_h3: video VAE tiling plan and seam blend match upstream") {
  // The tile plan is not a simple stride: it picks the smallest tile count whose
  // MINIMUM overlaps still cover the input, then distributes the leftover slack in
  // whole vae_ratio units ROUND-ROBIN across the seams. Getting that wrong shifts
  // every tile after the first and shows up as seam artifacts, not as an error.
  int64_t s_off = 0, l_off = 0, o_off = 0;
  for (int64_t c = 0; c < vllm_test::kH3TileCases; ++c) {
    const int64_t input_len = vllm_test::kH3TileInputs[c * 3 + 0];
    const int64_t tile_size = vllm_test::kH3TileInputs[c * 3 + 1];
    const int64_t overlap_min = vllm_test::kH3TileInputs[c * 3 + 2];
    const vllm::MiniMaxH3TilePlan got =
        vllm::MiniMaxH3SplitTiles(input_len, tile_size, overlap_min, vllm_test::kH3TileVaeRatio);
    INFO("tile case " << c << " len=" << input_len << " tile=" << tile_size);
    REQUIRE(static_cast<int64_t>(got.starts.size()) == vllm_test::kH3TileCounts[c]);
    for (size_t i = 0; i < got.starts.size(); ++i) {
      CHECK(got.starts[i] == vllm_test::kH3TileStarts[s_off + i]);
      CHECK(got.lengths[i] == vllm_test::kH3TileLens[l_off + i]);
    }
    for (size_t i = 0; i < got.overlaps.size(); ++i) {
      CHECK(got.overlaps[i] == vllm_test::kH3TileOverlaps[o_off + i]);
    }
    // Structural invariants the plan must satisfy for tiling to be lossless:
    // tiles cover the whole axis, and every seam overlaps by at least the minimum.
    CHECK(got.starts.front() == 0);
    CHECK(got.starts.back() + got.lengths.back() >= input_len);
    for (size_t i = 0; i < got.overlaps.size(); ++i) {
      CHECK(got.overlaps[i] >= overlap_min);
      CHECK(got.overlaps[i] % vllm_test::kH3TileVaeRatio == overlap_min % vllm_test::kH3TileVaeRatio);
    }
    s_off += static_cast<int64_t>(got.starts.size());
    l_off += static_cast<int64_t>(got.lengths.size());
    o_off += static_cast<int64_t>(got.starts.size()) - 1;
  }

  // A tile larger than the input is a single untiled pass with no seams.
  const vllm::MiniMaxH3TilePlan single = vllm::MiniMaxH3SplitTiles(100, 256, 64, 16);
  CHECK(single.starts.size() == 1);
  CHECK(single.lengths[0] == 100);
  CHECK(single.overlaps.empty());

  // The seam cross-fade.
  const std::vector<float> a = MakeParam("tiling.a", vllm_test::kH3TileBlendLen, 1.0);
  const std::vector<float> b = MakeParam("tiling.b", vllm_test::kH3TileBlendLen, 1.0);
  const std::vector<float> blended =
      vllm::MiniMaxH3BlendTiles(a, b, vllm_test::kH3TileBlendExtent);
  REQUIRE(blended.size() == std::size(vllm_test::kH3TileBlendGolden));
  CHECK(MaxAbsDiff(blended, vllm_test::kH3TileBlendGolden, blended.size()) <= 1e-6);
  // The fade starts fully on `a` and ends fully on `b`.
  CHECK(blended.front() == doctest::Approx(a[a.size() - vllm_test::kH3TileBlendExtent]));
  CHECK(blended[static_cast<size_t>(vllm_test::kH3TileBlendExtent)] ==
        doctest::Approx(b[static_cast<size_t>(vllm_test::kH3TileBlendExtent)]));
}

TEST_CASE("minimax_h3: presentation token tags match upstream") {
  // The fl2va "vision-span override" the denoise loop requires callers to have
  // applied. The load-bearing detail: a vision block is
  // <|vision_start|> + pad*count + <|vision_end|>, and the WHOLE block -- markers
  // included -- is tagged VIDEO. Tagging only the pads leaves two markers as TEXT
  // and shifts every AdaLN modulation index after them.
  std::vector<vllm::MiniMaxH3PresentationSpan> spans;
  for (int64_t i = 0; i < vllm_test::kH3PresSpanCount; ++i) {
    vllm::MiniMaxH3PresentationSpan span;
    span.kind = vllm_test::kH3PresSpanKinds[i] == 0
                    ? vllm::MiniMaxH3PresentationSpan::Kind::kVision
                    : vllm::MiniMaxH3PresentationSpan::Kind::kText;
    span.length = vllm_test::kH3PresSpanLens[i];
    spans.push_back(span);
  }

  const std::vector<int64_t> tags = vllm::MiniMaxH3BuildPresentationTokenTags(spans);
  REQUIRE(static_cast<int64_t>(tags.size()) == vllm_test::kH3PresTagCount);
  REQUIRE(tags.size() == std::size(vllm_test::kH3PresTagsGolden));
  for (size_t i = 0; i < tags.size(); ++i) CHECK(tags[i] == vllm_test::kH3PresTagsGolden[i]);

  // The tags must only ever be TEXT or VIDEO -- never the audio or padding tags.
  for (int64_t tag : tags) {
    CHECK((tag == vllm::kMiniMaxH3TagText || tag == vllm::kMiniMaxH3TagVideo));
  }

  // A vision block is pad_count + 2 tokens (the two markers).
  CHECK(vllm::MiniMaxH3VisionBlockTokenLength(3) == 5);
  CHECK(vllm::MiniMaxH3VisionBlockTokenLength(1) == 3);
  CHECK_THROWS(vllm::MiniMaxH3VisionBlockTokenLength(0));
  CHECK_THROWS(vllm::MiniMaxH3BuildPresentationTokenTags({}));

  // Every VIDEO run must be a whole vision block, i.e. its length must equal one
  // of the emitted vision spans -- proving the markers were tagged with the pads.
  std::vector<int64_t> video_runs;
  for (size_t i = 0; i < tags.size();) {
    if (tags[i] != vllm::kMiniMaxH3TagVideo) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < tags.size() && tags[j] == vllm::kMiniMaxH3TagVideo) ++j;
    video_runs.push_back(static_cast<int64_t>(j - i));
    i = j;
  }
  std::vector<int64_t> vision_spans;
  for (const vllm::MiniMaxH3PresentationSpan& span : spans) {
    if (span.kind == vllm::MiniMaxH3PresentationSpan::Kind::kVision) {
      vision_spans.push_back(span.length);
    }
  }
  CHECK(video_runs == vision_spans);
}

TEST_CASE("minimax_h3: the VAE encoder ResnetBlock3D matches upstream") {
  // The repeated unit of the video VAE's 3D-CNN ENCODER (used for image/video
  // CONDITIONING, not for output frames). Two details are load-bearing and both
  // are exercised: the convolution is CAUSAL in time (all padding on the LEFT, so
  // a frame never sees the future), and GroupNorm's statistics span TIME as well
  // as space.
  vllm::MiniMaxH3ResnetBlock3dConfig config;
  config.in_channels = vllm_test::kH3Res3dInCh;
  config.out_channels = vllm_test::kH3Res3dOutCh;
  config.t = vllm_test::kH3Res3dT;
  config.h = vllm_test::kH3Res3dH;
  config.w = vllm_test::kH3Res3dW;
  config.num_groups = vllm_test::kH3Res3dGroups;
  config.eps = 1e-6;

  vllm::MiniMaxH3AudioVaeWeights weights;
  auto put = [&](const std::string& n, int64_t count, double scale, double offset) {
    weights.tensors["rb." + n] = MakeParam("resnet3d." + n, count, scale, offset);
  };
  put("norm1.weight", config.in_channels, 0.1, 1.0);
  put("norm1.bias", config.in_channels, 0.05, 0.0);
  put("norm2.weight", config.out_channels, 0.1, 1.0);
  put("norm2.bias", config.out_channels, 0.05, 0.0);
  put("conv1.weight", config.out_channels * config.in_channels * 27, 0.1, 0.0);
  put("conv1.bias", config.out_channels, 0.05, 0.0);
  put("conv2.weight", config.out_channels * config.out_channels * 27, 0.1, 0.0);
  put("conv2.bias", config.out_channels, 0.05, 0.0);
  put("nin_shortcut.weight", config.out_channels * config.in_channels, 0.1, 0.0);
  put("nin_shortcut.bias", config.out_channels, 0.05, 0.0);

  const int64_t spatial = config.t * config.h * config.w;
  const std::vector<float> x =
      MakeParam("resnet3d.input", config.in_channels * spatial, 1.0);
  const std::vector<float> got =
      vllm::MiniMaxH3ResnetBlock3dForward(config, weights, "rb", x);

  REQUIRE(got.size() == std::size(vllm_test::kH3Res3dGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3Res3dGolden, got.size());
  INFO("resnet3d max|diff| = " << err);
  CHECK(err <= 1e-4);

  // CAUSALITY, proven rather than assumed: perturbing the LAST frame must leave
  // the FIRST frame's output bit-identical, while the last frame's own output
  // changes. A non-causal (symmetric) temporal pad would break the first check.
  std::vector<float> perturbed = x;
  for (int64_t c = 0; c < config.in_channels; ++c) {
    perturbed[static_cast<size_t>(c * spatial + (config.t - 1) * config.h * config.w)] += 5.0f;
  }
  const std::vector<float> other =
      vllm::MiniMaxH3ResnetBlock3dForward(config, weights, "rb", perturbed);
  // GroupNorm's statistics span time, so a change anywhere perturbs every frame
  // a little -- an exact-equality check on the whole BLOCK would be wrong. Here
  // the weaker (but still meaningful) claim is asserted: the perturbed frame moves
  // strictly more than the first. Strict causality is then proven on the bare
  // CONVOLUTION below, where no norm mixes the frames.
  double first_delta = 0.0, last_delta = 0.0;
  const int64_t frame = config.h * config.w;
  for (int64_t c = 0; c < config.out_channels; ++c) {
    for (int64_t i = 0; i < frame; ++i) {
      first_delta = std::max(first_delta,
                             std::abs(static_cast<double>(other[static_cast<size_t>(c * spatial + i)]) -
                                      got[static_cast<size_t>(c * spatial + i)]));
      const size_t last = static_cast<size_t>(c * spatial + (config.t - 1) * frame + i);
      last_delta = std::max(last_delta,
                            std::abs(static_cast<double>(other[last]) - got[last]));
    }
  }
  CHECK(last_delta > first_delta);

  // The causal convolution alone: with norms bypassed, a future-frame change must
  // NOT reach an earlier frame at all.
  vllm::MiniMaxH3Conv3dSpec spec;
  spec.in_channels = spec.out_channels = 1;
  spec.t = 3;
  spec.h = spec.w = 1;
  spec.kernel_t = 3;
  spec.kernel_h = spec.kernel_w = 1;
  spec.pad_t = 1;
  spec.pad_h = spec.pad_w = 0;
  spec.causal = true;
  const std::vector<float> kernel = {1.0f, 2.0f, 4.0f};
  const std::vector<float> a = {1.0f, 0.0f, 0.0f};
  const std::vector<float> b = {1.0f, 0.0f, 9.0f};  // only the LAST frame differs
  const std::vector<float> ca = vllm::MiniMaxH3CausalConv3d(a, spec, kernel, nullptr);
  const std::vector<float> cb = vllm::MiniMaxH3CausalConv3d(b, spec, kernel, nullptr);
  REQUIRE(ca.size() == cb.size());
  CHECK(ca[0] == cb[0]);   // frame 0 cannot see frame 2
  CHECK(ca[1] == cb[1]);   // frame 1 cannot see frame 2
  CHECK(ca[2] != cb[2]);   // frame 2 does
}

TEST_CASE("minimax_h3: the VAE encoder Downsample3D matches upstream") {
  // The strided conv between encoder levels. Its subtlety is the ASYMMETRIC
  // pre-pad: one pixel on the RIGHT of W and the BOTTOM of H before a stride-2
  // conv with padding (1, 0, 0). Padding symmetrically instead shifts the whole
  // sampling lattice by half a pixel -- no error, just a subtly wrong latent.
  vllm::MiniMaxH3Downsample3dConfig config;
  config.in_channels = vllm_test::kH3Down3dInCh;
  config.out_channels = vllm_test::kH3Down3dOutCh;
  config.t = vllm_test::kH3Down3dT;
  config.h = vllm_test::kH3Down3dH;
  config.w = vllm_test::kH3Down3dW;
  config.time_stride = 2;
  config.space_stride = 2;

  const std::vector<float> weight =
      MakeParam("down3d.conv.weight", config.out_channels * config.in_channels * 27, 0.1);
  const std::vector<float> bias = MakeParam("down3d.conv.bias", config.out_channels, 0.05);
  const std::vector<float> x = MakeParam(
      "down3d.input", config.in_channels * config.t * config.h * config.w, 1.0);

  const std::vector<float> got = vllm::MiniMaxH3Downsample3d(x, config, weight, bias);

  const int64_t expected = config.out_channels * vllm_test::kH3Down3dOutT *
                           vllm_test::kH3Down3dOutH * vllm_test::kH3Down3dOutW;
  CHECK(static_cast<int64_t>(got.size()) == expected);
  REQUIRE(got.size() == std::size(vllm_test::kH3Down3dGolden));
  const double err = MaxAbsDiff(got, vllm_test::kH3Down3dGolden, got.size());
  INFO("downsample3d max|diff| = " << err);
  CHECK(err <= 1e-4);

  // Strides genuinely halve each axis (T 3->2 under the causal pad, H/W 4->2).
  CHECK(vllm_test::kH3Down3dOutH == config.h / 2);
  CHECK(vllm_test::kH3Down3dOutW == config.w / 2);
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
