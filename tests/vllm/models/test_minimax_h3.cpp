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
#include <cstring>
#include <string>
#include <vector>

#include "minimax_h3_goldens.inc"
#include "minimax_h3_gguf_manifest.inc"
#include "minimax_h3_audio_vae_goldens.inc"
#include "minimax_h3_nvfp4_manifest.inc"
#include "minimax_h3_video_vae_manifest.inc"
#include "minimax_h3_video_vae_goldens.inc"

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
