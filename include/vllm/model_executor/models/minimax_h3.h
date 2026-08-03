// MiniMax-H3 (`MiniMaxAI/MiniMax-H3`) — the omni-modal video+audio DIFFUSION
// transformer, ported from vLLM-Omni (vllm-project/vllm-omni). This is the
// project's FIRST diffusion architecture: H3 is not an autoregressive decoder, so
// it has no KV cache, no sampler, and no logits. One request runs a fixed 50-step
// flow-matching denoise loop in which the DiT is forwarded ONCE per step over the
// WHOLE packed sequence, and the resulting latents are decoded to frames + a
// stereo waveform by two VAEs.
//
// ─── HONESTY (up front) ──────────────────────────────────────────────────────
// The real checkpoint is ~354 GB (33.1B DiT + a Qwen3-VL-32B-derived encoder + a
// video VAE + an audio VAE) and its validated serving config is 4x NVIDIA B300 at
// ~133 GB peak per rank. That does not fit this project's hardware (one GB10, 119
// GiB UNIFIED), so there is NO end-to-end token/frame gate for H3 on this box and
// none is claimed. What IS gated here is exact: the layout math, the scheduler,
// and the DiT forward are compared against the UPSTREAM vLLM-Omni modules
// executed at reduced dimensions (scripts/gen-minimax-h3-goldens.py). Structure
// and math are proven; end-to-end generation is hardware-blocked. Full lifecycle,
// component inventory, and the remaining bricks: .agents/specs/minimax-h3.md.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: vllm-project/vllm-omni, vllm_omni/diffusion/models/minimax_h3/
//   OURS                                  <-  UPSTREAM
//   MiniMaxH3DitParams                    <-  minimax_h3_transformer.py:47-78
//                                             (MiniMaxH3DiTArchConfig.from_mapping)
//   MiniMaxH3PatchifyVideoLatent          <-  packed_tokens.py:23-41
//   MiniMaxH3UnpatchifyVideoTokens        <-  packed_tokens.py:44-70
//   MiniMaxH3PackAudioLatent              <-  packed_tokens.py:73-85
//   MiniMaxH3UnpackAudioTokens            <-  packed_tokens.py:88-106
//   BuildMiniMaxH3PackedSequence          <-  packed_sequence.py:116-239
//   BuildMiniMaxH3PackedSequenceRef2va    <-  packed_sequence.py:290-557
//   MiniMaxH3RfVToX0                      <-  scheduling_...euler_ancestral.py:49-69
//   MiniMaxH3EulerEta0Step                <-  scheduling_...euler_ancestral.py:72-102
//   MiniMaxH3DitForward                   <-  minimax_h3_transformer.py:986-1102
//   EnumerateMiniMaxH3DitTensors          <-  minimax_h3_transformer.py:906-922
//   MiniMaxH3ReorderGroupedQkv            <-  minimax_h3_transformer.py:139-168
//   MiniMaxH3DenoiseLoop                  <-  denoise_loop.py:129-239
//   MiniMaxH3TimeShiftSigmas / shape plan <-  time_request.py:5-61,
//                                            pipeline_minimax_h3.py:121-122,
//                                            207-222, 374-434
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

// MiniMaxH3DiTArchConfig (minimax_h3_transformer.py:47-78). Defaults are the
// SHIPPED MiniMax-H3 geometry; `ParseMiniMaxH3DitParams` overrides any key the
// checkpoint's transformer config actually carries, exactly like `from_mapping`.
struct MiniMaxH3DitParams {
  int64_t num_layers = 50;
  int64_t token_refiner_num_layers = 2;
  int64_t hidden_size = 5376;
  int64_t num_attention_heads = 56;  // MHA: num_kv_heads == num_attention_heads
  int64_t attention_head_dim = 128;
  int64_t ffn_hidden_size = 14336;
  int64_t latents_dim = 24;        // video VAE latent channels
  int64_t audio_latents_dim = 32;  // audio VAE latent channels
  int64_t patch_size_t = 1;
  int64_t patch_size_h = 2;
  int64_t patch_size_w = 2;
  int64_t text_dim = 5120;  // H3-Encoder hidden width (Qwen3-VL layer 50)
  int64_t timestep_input_dim = 256;
  int64_t time_embed_hidden_size = 5376;
  int64_t time_embed_dim = 2688;
  int64_t adaln_out_features = 18 * 5376;      // 6 vectors x 3 modalities x H
  int64_t final_adaln_out_features = 2 * 5376;  // 2 vectors x 1 modality x H
  int64_t rope_inv_freq_len = 16;
  double norm_eps = 1e-5;
  double qk_norm_eps = 1e-5;
  double final_norm_eps = 1e-5;

  // Derived. video_row_width is the packed video token width
  // (latents_dim * patch volume; 24*1*2*2 = 96 at shipped scale).
  int64_t video_row_width() const {
    return latents_dim * patch_size_t * patch_size_h * patch_size_w;
  }
  // 3D RoPE rotates 6*rope_inv_freq_len of attention_head_dim dims
  // (minimax_h3_transformer.py:207-230; 96 of 128 at shipped scale).
  int64_t rope_rot_dim() const { return 6 * rope_inv_freq_len; }
};

// AdaLN modality count: token tags are -1 padding and 0/1/2 for video/text/audio
// (minimax_h3_transformer.py:103-106).
inline constexpr int64_t kMiniMaxH3AdalnModalityNum = 3;

// Packed-sequence token tags (packed_sequence.py:218-221).
inline constexpr int64_t kMiniMaxH3TagPadding = -1;
inline constexpr int64_t kMiniMaxH3TagVideo = 0;
inline constexpr int64_t kMiniMaxH3TagText = 1;
inline constexpr int64_t kMiniMaxH3TagAudio = 2;

// Packed-sequence placeholder input ids (packed_sequence.py:27-35).
inline constexpr int64_t kMiniMaxH3TextId = -5;
inline constexpr int64_t kMiniMaxH3ImgVidCondId = -11;
inline constexpr int64_t kMiniMaxH3AudioRefCondId = -17;
inline constexpr int64_t kMiniMaxH3AudioFirstId = -15;
inline constexpr int64_t kMiniMaxH3AudioId = -14;
inline constexpr int64_t kMiniMaxH3VideoFirstId = -3;
inline constexpr int64_t kMiniMaxH3VideoId = -2;
inline constexpr int64_t kMiniMaxH3VideoLastId = -4;
inline constexpr int64_t kMiniMaxH3PadId = -1;

// Condition anchor timesteps (denoise_loop.py:22-24).
inline constexpr double kMiniMaxH3ImgVidCondTimestep = 0.999;
inline constexpr double kMiniMaxH3AudioRefCondTimestep = 1.0;

// Parse the checkpoint transformer config. Mirrors `from_mapping`: unknown keys
// are ignored, present keys override, and `patch_size` must carry three values.
MiniMaxH3DitParams ParseMiniMaxH3DitParams(const nlohmann::json& config);

// ---------------------------------------------------------------------------
// Latent <-> packed-token conversion (packed_tokens.py). Host-side layout math on
// f32 rows; these move bytes, they do not compute, so they stay off the device.
// ---------------------------------------------------------------------------

// [B,C,T,H,W] -> [B*t*h*w, C*pt*ph*pw] (packed_tokens.py:23-41).
std::vector<float> MiniMaxH3PatchifyVideoLatent(const std::vector<float>& latent, int64_t batch,
                                                int64_t channels, int64_t full_t, int64_t full_h,
                                                int64_t full_w, int64_t patch_t, int64_t patch_h,
                                                int64_t patch_w);

// Inverse of the above (packed_tokens.py:44-70). `rows` is [N, C*pt*ph*pw].
std::vector<float> MiniMaxH3UnpatchifyVideoTokens(const std::vector<float>& rows, int64_t t,
                                                  int64_t h, int64_t w, int64_t channels,
                                                  int64_t patch_t, int64_t patch_h, int64_t patch_w);

// [audio_channel, latent_dim, T] -> [audio_channel*T, latent_dim]
// (packed_tokens.py:73-85).
std::vector<float> MiniMaxH3PackAudioLatent(const std::vector<float>& latent, int64_t audio_channel,
                                            int64_t latent_dim, int64_t steps);

// Inverse (packed_tokens.py:88-106). `rows` is [audio_t, latent_dim].
std::vector<float> MiniMaxH3UnpackAudioTokens(const std::vector<float>& rows, int64_t audio_t,
                                              int64_t audio_channel, int64_t latent_dim);

// ---------------------------------------------------------------------------
// Packed sequence (packed_sequence.py)
// ---------------------------------------------------------------------------

// The structural fields of one CFG branch's packed sequence. Layout:
//   [text L | imgvid_cond C | audio A | video_target V | pad P]
// The used length is padded up to a multiple of 64 and the padding becomes a
// SECOND attention document, so attention never crosses into it.
struct MiniMaxH3PackedSequence {
  int64_t seq_len = 0;
  std::vector<int64_t> input_ids;    // [seq_len] placeholder ids
  std::vector<uint8_t> image_mask;   // [seq_len]
  std::vector<uint8_t> audio_mask;   // [seq_len]
  std::vector<int64_t> img_pos;      // cond rows then target rows
  std::vector<int64_t> audio_pos;    // reference rows then target rows
  std::vector<int64_t> text_pos;     // [0, text_len)
  std::vector<uint8_t> update_mask;  // per img_pos row: is it a denoise target?
  std::vector<uint8_t> audio_update_mask;  // per audio_pos row (ref2va only)
  // [seq_len, 3] (t, h, w) grid in FP64. Kept as double on purpose: the grid is
  // built by fp64 accumulations whose last ulp feeds RoPE, and upstream keeps two
  // deliberately DIFFERENT summation orders for it (see packed_sequence.py:101-113).
  std::vector<double> img_position_ids;
  std::vector<int64_t> token_tags;   // [seq_len]
  std::vector<int32_t> cu_seqlens;   // {0, used, seq_len}
  std::vector<int32_t> document_id;  // [seq_len]; 1 on the padding document
};

// fl2va / t2va layout (packed_sequence.py:116-239). `keyframe_frame_indices` must
// be one of {}, {0}, {-1}, {0,-1} and requires `frame_count` when non-empty.
MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequence(int64_t text_len, int64_t latent_t,
                                                     int64_t latent_h, int64_t latent_w,
                                                     int64_t audio_t, int64_t audio_channel,
                                                     bool include_keyframe_cond,
                                                     const std::vector<int64_t>& keyframe_frame_indices,
                                                     int64_t frame_count);

// One ref2va reference block (packed_sequence.py:290-313).
struct MiniMaxH3RefBlock {
  enum class Kind { kImage, kAudio, kVideoAudio };
  Kind kind = Kind::kImage;
  int64_t ref_audio_t = 0;
  int64_t latent_t = 0;
  int64_t latent_h = 0;
  int64_t latent_w = 0;
};

// General ref2va-family layout (packed_sequence.py:290-557).
MiniMaxH3PackedSequence BuildMiniMaxH3PackedSequenceRef2va(
    int64_t text_len, int64_t latent_t, int64_t latent_h, int64_t latent_w, int64_t audio_t,
    const std::vector<MiniMaxH3RefBlock>& ref_blocks, int64_t audio_channel);

// ---------------------------------------------------------------------------
// Request planning (time_request.py + pipeline_minimax_h3.py shape resolution)
// ---------------------------------------------------------------------------

// Output frame rate is FIXED (pipeline_minimax_h3.py:83).
inline constexpr int64_t kMiniMaxH3Fps = 24;
// Reference-image rescale target (pipeline_minimax_h3.py:87-88).
inline constexpr int64_t kMiniMaxH3ReferenceImageShortEdge = 2048;
inline constexpr int64_t kMiniMaxH3ReferenceImageMultiple = 32;
// Default frame counts when the request names neither duration nor num_frames
// (pipeline_minimax_h3.py:405).
inline constexpr int64_t kMiniMaxH3DefaultFramesT2va = 209;
inline constexpr int64_t kMiniMaxH3DefaultFramesRef2va = 124;
// Default sigma shift scales (pipeline_minimax_h3.py:283-285).
inline constexpr double kMiniMaxH3DefaultVideoShift = 12.0;
inline constexpr double kMiniMaxH3DefaultAudioShift = 3.0;
inline constexpr int64_t kMiniMaxH3DefaultSteps = 50;

// Snap up to the 17n+5 frame boundary (time_request.py:5-12).
int64_t MiniMaxH3AlignFrameCount(int64_t frame_count);
// frame_count -> video latent T (time_request.py:15-18).
int64_t MiniMaxH3VideoLatentT(int64_t frame_count);
// Inverse; T must be 1 or match 5n+2 (time_request.py:21-26).
int64_t MiniMaxH3FrameCountFromVideoLatentT(int64_t out_t);
// Audio latents run at 40 Hz (time_request.py:29-31).
int64_t MiniMaxH3AudioLatentT(double duration_seconds);
// `max(multiple, round(value/multiple)*multiple)` (pipeline_minimax_h3.py:121-122).
int64_t MiniMaxH3AlignMultiple(double value, int64_t multiple);
// Reference image rescale to a 2048 short edge on a 32 grid
// (pipeline_minimax_h3.py:207-222). Returns {width, height}.
std::pair<int64_t, int64_t> MiniMaxH3ReferenceImageShape(int64_t width, int64_t height);

// The rectified-flow time-shifted sigma schedule (time_request.py:34-61):
// base = linspace(1, 0, num_steps); sigma = s*base / (1 + (s-1)*base); duplicate
// consecutive values are collapsed and a terminal 0 is appended when needed.
std::vector<double> MiniMaxH3TimeShiftSigmas(int64_t num_steps, double shift_scale);

// The resolved generation plan for one request.
struct MiniMaxH3ShapePlan {
  int64_t height = 0;
  int64_t width = 0;
  int64_t num_frames = 0;
  int64_t latent_t = 0;
  int64_t audio_t = 0;
};

// `_resolve_shape` (pipeline_minimax_h3.py:393-434). Pass `duration_seconds <= 0`
// and `requested_frames <= 1` to take the per-task default; pass
// `height`/`width` <= 0 to take the aspect-derived default (which needs the
// keyframe aspect for fl2va, hence `image_width`/`image_height`).
MiniMaxH3ShapePlan MiniMaxH3ResolveShape(const std::string& task, double duration_seconds,
                                         int64_t requested_frames, int64_t height, int64_t width,
                                         int64_t image_width, int64_t image_height);

// `_resolve_task` (pipeline_minimax_h3.py:374-391). `requested` may be empty.
std::string MiniMaxH3ResolveTask(const std::string& requested, const std::string& partition,
                                 bool has_image, const std::vector<std::string>& supported_tasks);

// ---------------------------------------------------------------------------
// Flow-matching scheduler (scheduling_minimax_h3_euler_ancestral.py)
// ---------------------------------------------------------------------------

// x0 = xt + (1 - t) * v (scheduling:49-69). Rectified-flow velocity -> clean sample.
std::vector<float> MiniMaxH3RfVToX0(const std::vector<float>& xt, const std::vector<float>& v,
                                    double timestep);

// ---------------------------------------------------------------------------
// Condition-noise augmentation (condition_noise.py) — fl2va / ref2va
// ---------------------------------------------------------------------------

// Packed row widths (denoise_loop.py:26-28).
inline constexpr int64_t kMiniMaxH3VideoRowWidth = 96;
inline constexpr int64_t kMiniMaxH3AudioRowWidth = 32;
// Channel-major packed audio condition rows are always stereo.
inline constexpr int64_t kMiniMaxH3AudioCondChannels = 2;

// out = noise_aug*clean + (1 - noise_aug)*noise, over packed condition rows.
// `condition_shapes` is a flat list of (latent_t, latent_h, latent_w) triples in
// packed visual-condition order. NOISE IS AN INPUT (see the .cpp for why).
std::vector<float> MiniMaxH3ImgvidCondNoiseAug(const std::vector<float>& clean_rows,
                                               const std::vector<int64_t>& condition_shapes,
                                               int64_t target_latent_t,
                                               int64_t imgvid_cond_num_frames, double noise_aug,
                                               const std::vector<float>& noise_rows);

// The audio-side equivalent; `condition_audio_t` is the latent T of each
// audio-bearing condition in request order.
std::vector<float> MiniMaxH3AudioCondNoiseAug(const std::vector<float>& clean_rows,
                                              const std::vector<int64_t>& condition_audio_t,
                                              double noise_aug,
                                              const std::vector<float>& noise_rows);

// Ancestral Euler with eta = 0 (scheduling:72-102):
//   out = r * state + (1 - r) * denoised,  r = sigma_next / sigma_curr.
// sigma_curr == 0 is the terminal step and returns `state` unchanged.
std::vector<float> MiniMaxH3EulerEta0Step(const std::vector<float>& state,
                                          const std::vector<float>& denoised, double sigma_curr,
                                          double sigma_next);

// ---------------------------------------------------------------------------
// DiT weights + forward
// ---------------------------------------------------------------------------

// Checkpoint tensor names in load order, with their expected shapes. The H3 DiT
// loads by EXACT checkpoint name (minimax_h3_transformer.py:906-922), so this
// enumeration IS the weight contract and is gated structurally without the
// checkpoint.
struct MiniMaxH3TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  // The 12 latent/timestep/output params and the RoPE buffer stay FP32 after load
  // (minimax_h3_transformer.py:85-101, 898-904); everything else is BF16.
  bool fp32 = false;
};

std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3DitTensors(const MiniMaxH3DitParams& params);

// --- GGUF arm (minimax_h3_gguf.cpp) ---
// The ComfyUI-format H3 GGUFs keep the checkpoint's own parameter names, so the
// name map is the IDENTITY against the contract above; only the SHAPES need
// resolving. GGUF `ne` is reversed relative to torch, EXCEPT where ComfyUI had to
// reshape a tensor for quant-block alignment, in which case the true torch shape
// is recorded in `comfy.gguf.orig_shape.<name>` and is used verbatim.
std::vector<int64_t> MiniMaxH3GgufLogicalShape(const std::vector<int64_t>& gguf_dims,
                                               const std::vector<int64_t>& orig_shape);

// A ComfyUI GGUF carries no transformer config, so the SHAPES are the config.
MiniMaxH3DitParams ParseMiniMaxH3DitParamsFromGgufManifest(
    const std::vector<MiniMaxH3TensorSpec>& manifest);

class GgufFile;
// Names + logical shapes + fp32-island flags read out of a GGUF.
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3GgufTensors(const GgufFile& file);

// ---------------------------------------------------------------------------
// Audio VAE decoder (minimax_h3_audio_vae.cpp)
//
// H3's two VAEs are checkpoint REMOTE CODE loaded under `trust_remote_code`
// (vae.py:41-53); vLLM-Omni only ADAPTS them, so a no-Python engine must
// REIMPLEMENT them. This is the audio side: a DAC-lineage BigVGAN vocoder at
// 32 kHz / 2 channels, gated against the checkpoint's own modules by
// scripts/gen-minimax-h3-audio-vae-goldens.py.
// ---------------------------------------------------------------------------

inline constexpr int64_t kMiniMaxH3AudioSampleRate = 32000;
inline constexpr int64_t kMiniMaxH3AudioChannels = 2;

struct MiniMaxH3AudioVaeConfig {
  int64_t num_mels = 2048;                 // == DacAudioVAE latent_dim
  int64_t upsample_initial_channel = 1024;  // decoder_dim
  std::vector<int64_t> upsample_rates = {5, 5, 2, 2, 2, 2, 2};
  std::vector<int64_t> upsample_kernel_sizes = {9, 9, 4, 4, 4, 4, 4};
  std::vector<int64_t> resblock_kernel_sizes = {3, 7, 11};
  std::vector<std::vector<int64_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  bool use_tanh_at_final = false;  // H3 CLAMPS instead
  bool use_bias_at_final = false;
  bool snake_logscale = true;
};

// Parameters keyed by their torch state_dict name, so the checkpoint's own
// naming IS the contract (`conv_pre.parametrizations.weight.original0`, ...).
struct MiniMaxH3AudioVaeWeights {
  std::map<std::string, std::vector<float>> tensors;

  const std::vector<float>& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return tensors.count(name) != 0; }
};

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60) — built at load time, never
// read from the checkpoint.
std::vector<float> MiniMaxH3KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size);

// torch weight_norm: w = g * v / ||v||, norm over every dim except dim 0. Every
// conv in this decoder is weight-normalized, so the checkpoint stores (g, v).
std::vector<float> MiniMaxH3MaterializeWeightNorm(const std::vector<float>& g,
                                                  const std::vector<float>& v,
                                                  int64_t out_channels);

// Decode one channel of audio latents to a waveform in [-1, 1]. When the weights
// carry `dec_in_proj` (the checkpoint's Conv1d k=1 from vae_latent_channels to
// num_mels, applied before BigVGAN — dac_audio_vae.py:218-231) the input is
// [vae_latent_channels, frames]; otherwise it is already [num_mels, frames].
std::vector<float> MiniMaxH3AudioVaeDecode(const MiniMaxH3AudioVaeConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t frames,
                                           int64_t* out_samples);

// ---------------------------------------------------------------------------
// Video VAE decoder (minimax_h3_video_vae.cpp)
//
// The real 560-tensor manifest splits the video VAE cleanly: the ENCODER is the
// 3D CNN, but the DECODER — the half generation needs — is a 36-block TRANSFORMER.
// This is that block, the repeated unit.
// ---------------------------------------------------------------------------

struct MiniMaxH3VideoVaeBlockConfig {
  int64_t dim = 0;       // embed_dim
  int64_t heads = 0;
  int64_t dim_head = 0;
  int64_t ff_inner = 0;  // w1 emits 2 * ff_inner ([gate | up])
  double eps = 1e-5;
};

// One decoder TransformerBlock (base_module.py:200-281), fp32:
//   h += scale1 * Attention(RMSNorm(h));  h += scale2 * GatedSiLU_FF(RMSNorm(h))
// `scale1`/`scale2` are learned PER-CHANNEL vectors. NOTE the qkv layout is
// PER-HEAD INTERLEAVED ([head][q|k|v]), unlike the DiT's [q_all|k_all|v_all].
// Parameters are looked up by their torch state_dict names under `prefix`.
// `rope_cos`/`rope_sin` are per-TOKEN [seq, rot_dim] (shared across heads) and may
// be null for the no-RoPE path.
std::vector<float> MiniMaxH3VideoVaeBlockForward(const MiniMaxH3VideoVaeBlockConfig& config,
                                                 const MiniMaxH3AudioVaeWeights& weights,
                                                 const std::string& prefix,
                                                 const std::vector<float>& hidden, int64_t seq,
                                                 const float* rope_cos = nullptr,
                                                 const float* rope_sin = nullptr,
                                                 int64_t rot_dim = 0);

// The whole ViT3D decoder (vae_vit.py:216-365). Real hyperparameters from the
// checkpoint's `vit_decoder_kwargs`: 36 layers, 32 heads x 64, RMS norms, qk RMS
// norm WITHOUT affine, gated SiLU, rope_theta 100.0, rope_dim_ratio 0.75.
struct MiniMaxH3VideoVaeDecoderConfig {
  MiniMaxH3VideoVaeBlockConfig block;
  int64_t num_layers = 36;
  int64_t in_channels = 24;   // video latent channels
  int64_t out_channels = 3;   // RGB
  int64_t patch_size = 16;
  int64_t patch_size_t = 4;
  int64_t num_register_tokens = 4;
  int64_t rope_apply_dim = 48;  // int(dim_head * rope_dim_ratio)
  double rope_theta = 100.0;
};

struct MiniMaxH3VideoFrameShape {
  int64_t channels = 0, t = 0, h = 0, w = 0;
};

// 3D RoPE tables for one latent grid (RotaryEmbeddingND + create_token_ids).
void MiniMaxH3VideoVaeRope(int64_t latent_t, int64_t latent_h, int64_t latent_w,
                           int64_t num_suffix, int64_t rope_apply_dim, double rope_theta,
                           std::vector<float>* cos_out, std::vector<float>* sin_out);

// Decode a video latent [in_channels, T, H, W] to frames [out_channels, T*pt, H*ps, W*ps].
std::vector<float> MiniMaxH3VideoVaeDecode(const MiniMaxH3VideoVaeDecoderConfig& config,
                                           const MiniMaxH3AudioVaeWeights& weights,
                                           const std::vector<float>& latent, int64_t latent_t,
                                           int64_t latent_h, int64_t latent_w,
                                           MiniMaxH3VideoFrameShape* out_shape);

// ---------------------------------------------------------------------------
// H3-Encoder text tower (minimax_h3_encoder.cpp)
//
// A Qwen3-VL text model with three H3-specific deltas: only the first
// `selected_layer` (50) decoder layers run, the output is UNNORMALIZED (no final
// RMSNorm), and DeepStack visual features are added at the visual token positions
// after each of the first `deepstack.size()` layers.
// ---------------------------------------------------------------------------

inline constexpr int64_t kMiniMaxH3EncoderSelectedLayer = 50;
inline constexpr int64_t kMiniMaxH3EncoderHiddenDim = 5120;

struct MiniMaxH3EncoderConfig {
  int64_t hidden_size = kMiniMaxH3EncoderHiddenDim;
  int64_t num_hidden_layers = 64;
  int64_t selected_layer = kMiniMaxH3EncoderSelectedLayer;
  int64_t num_attention_heads = 40;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t intermediate_size = 17408;
  double rms_norm_eps = 1e-6;
  double rope_theta = 5000000.0;
  std::vector<int64_t> mrope_section = {24, 20, 20};
};

// num_layers = min(config.num_hidden_layers, selected_layer).
int64_t MiniMaxH3EncoderNumLayers(int64_t config_num_hidden_layers, int64_t selected_layer);

// Interleaved M-RoPE cos/sin ([seq, head_dim]) from [3, seq] (t, h, w) positions.
void MiniMaxH3EncoderMrope(const int64_t* positions, int64_t seq, int64_t head_dim,
                           double rope_theta, const std::vector<int64_t>& mrope_section,
                           std::vector<float>* cos_out, std::vector<float>* sin_out);

// The truncated, UNNORMALIZED text tower. `deepstack[i]` is [num_visual, hidden]
// and may be empty; `visual_pos_mask` is [seq] and is required when it is not.
std::vector<float> MiniMaxH3EncoderTextForward(const MiniMaxH3EncoderConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::vector<float>& inputs_embeds,
                                               const int64_t* positions, int64_t seq,
                                               const uint8_t* visual_pos_mask,
                                               const std::vector<std::vector<float>>& deepstack);

// One vision-tower block (encoder.py:417-481) — the repeated unit of the ViT.
// Unlike the text tower it uses LayerNorm (with bias), a [q_all|k_all|v_all] qkv
// layout, fp32 rotary, NON-CAUSAL attention segmented by `cu_seqlens`, and the
// TANH-approximate GELU.
struct MiniMaxH3VisionBlockConfig {
  int64_t hidden_size = 1152;
  int64_t num_heads = 16;
  int64_t intermediate_size = 4304;
  double eps = 1e-6;
};

std::vector<float> MiniMaxH3VisionBlockForward(const MiniMaxH3VisionBlockConfig& config,
                                               const MiniMaxH3AudioVaeWeights& weights,
                                               const std::string& prefix,
                                               const std::vector<float>& hidden, int64_t seq,
                                               const float* cos, const float* sin,
                                               const int32_t* cu_seqlens, int64_t num_segments);

// The whole vision tower (encoder.py:483-600).
struct MiniMaxH3VisionTowerConfig {
  MiniMaxH3VisionBlockConfig block;
  int64_t depth = 27;
  int64_t patch_size = 16;
  int64_t temporal_patch_size = 2;
  int64_t in_channels = 3;
  int64_t spatial_merge_size = 2;
  int64_t out_hidden_size = 5120;
  int64_t num_position_embeddings = 2304;  // must be a perfect square
  double rope_theta = 10000.0;
  std::vector<int64_t> deepstack_visual_indexes;
};

struct MiniMaxH3VisionTowerResult {
  std::vector<float> merged;                    // [tokens/merge^2, out_hidden_size]
  std::vector<std::vector<float>> deepstack;    // one per deepstack index
};

// Bilinear resample of the learned position grid, then spatial-merge permute.
std::vector<float> MiniMaxH3VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_table,
                                                      int64_t num_grid_per_side, int64_t dim,
                                                      const std::vector<int64_t>& grid_thw,
                                                      int64_t merge_size);

// 2D rotary frequencies in spatial-merge order.
std::vector<float> MiniMaxH3VisionRotary(const std::vector<int64_t>& grid_thw, int64_t merge_size,
                                         int64_t rotary_dim, double theta);

// `patches` is [tokens, in_channels * temporal_patch * patch * patch]; `grid_thw`
// is a flat list of (t, h, w) triples, one per image/video.
MiniMaxH3VisionTowerResult MiniMaxH3VisionTowerForward(const MiniMaxH3VisionTowerConfig& config,
                                                       const MiniMaxH3AudioVaeWeights& weights,
                                                       const std::vector<float>& patches,
                                                       const std::vector<int64_t>& grid_thw);

// The checkpoint stores qkv GROUPED per query group as [q_per_group, k, v]; the
// fused qkv projection wants [q_all, k_all, v_all] (minimax_h3_transformer.py:
// 139-168). H3 is MHA, so heads_per_group == 1.
std::vector<float> MiniMaxH3ReorderGroupedQkv(const std::vector<float>& weight,
                                              int64_t num_query_groups, int64_t heads_per_group,
                                              int64_t head_dim, int64_t in_features);

// Non-owning views of every DiT parameter, in the shape the forward consumes.
struct MiniMaxH3DitBlockWeights {
  vt::Tensor norm1;      // [H]
  vt::Tensor norm2;      // [H]
  vt::Tensor qkv_proj;   // [3*heads*Dh, H]
  vt::Tensor q_norm;     // [Dh]
  vt::Tensor k_norm;     // [Dh]
  vt::Tensor out_proj;   // [H, heads*Dh]
  vt::Tensor fc1;        // [2*ffn, H] as [gate; up]
  vt::Tensor fc2;        // [H, ffn]
  vt::Tensor adaln_w;    // [expand*modality*H, time_embed_dim] (blocks only)
  vt::Tensor adaln_b;    // [expand*modality*H]
};

struct MiniMaxH3DitWeights {
  vt::Tensor video_patch_proj_w, video_patch_proj_b;
  vt::Tensor audio_patch_proj_w, audio_patch_proj_b;
  vt::Tensor condition_proj_w, condition_proj_b;
  vt::Tensor time_proj_in_w, time_proj_in_b;
  vt::Tensor time_proj_out_w, time_proj_out_b;
  vt::Tensor rope_inv_freq;  // [rope_inv_freq_len], fp32
  std::vector<MiniMaxH3DitBlockWeights> refiner;  // no adaln, no rope
  vt::Tensor refiner_final_norm;
  std::vector<MiniMaxH3DitBlockWeights> blocks;
  vt::Tensor final_norm;
  vt::Tensor final_adaln_w, final_adaln_b;
  vt::Tensor video_out_w, video_out_b;
  vt::Tensor audio_out_w, audio_out_b;
};

// A GGUF-loaded DiT: owned dequantized buffers plus the views the forward takes.
// `storage` must outlive `weights` (the views are non-owning).
struct MiniMaxH3GgufDit {
  MiniMaxH3DitParams params;
  std::map<std::string, std::vector<float>> storage;
  std::map<std::string, std::vector<int64_t>> shapes;
  MiniMaxH3DitWeights weights;
};

// Materialize the DiT from a ComfyUI-format GGUF: derive the geometry from the
// manifest, dequantize every tensor to f32 through the shared GGUF dequant path
// (so the Q2_K/Q3_K/Q4_K families the H3 GGUFs use are covered), and bind the
// forward's views. Missing tensors throw by name rather than reading as zeros.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromGguf(const GgufFile& file);

// Bind the forward's views onto a MiniMaxH3GgufDit's owned buffers. Shared by the
// GGUF and NVFP4 arms: both land on the SAME weight contract.
void BindMiniMaxH3DitViews(MiniMaxH3GgufDit* out);

class SafetensorsFile;
// Materialize the DiT from an NVFP4 safetensors checkpoint. Quantized projections
// carry the compressed-tensors triple (U8 packed FP4 + E4M3 group-16
// `weight_scale` + F32 scalar `weight_scale_2`) and are dequantized through the
// project's existing NVFP4 path; the fp32/bf16 islands are read as-is.
MiniMaxH3GgufDit LoadMiniMaxH3DitFromNvfp4(const SafetensorsFile& file);

// The per-step inputs of one denoise step (minimax_h3_transformer.py:986-1102).
// Row-major host buffers; the forward stages them onto `device` itself.
struct MiniMaxH3DitInputs {
  int64_t seq_len = 0;
  const float* x = nullptr;            // [seq_len, video_row_width]
  const float* audio_x = nullptr;      // [seq_len, audio_latents_dim]
  const double* img_position_ids = nullptr;  // [seq_len, 3]
  const float* unique_timesteps = nullptr;   // [M]
  int64_t num_unique_timesteps = 0;
  const int64_t* inverse_indices = nullptr;  // [seq_len] -> [0, M)
  const int64_t* token_tags = nullptr;       // [seq_len]
  const float* prompt_embeds = nullptr;      // [text_len, text_dim]
  const int64_t* img_pos = nullptr;
  int64_t num_img_pos = 0;
  const int64_t* audio_pos = nullptr;
  int64_t num_audio_pos = 0;
  const int64_t* text_pos = nullptr;
  int64_t num_text_pos = 0;
  const int64_t* infer_out_pos = nullptr;  // rows the video head reports
  int64_t num_infer_out_pos = 0;
  const uint8_t* update_mask = nullptr;        // [num_infer_out_pos]
  const uint8_t* audio_update_mask = nullptr;  // optional, [num_audio_pos]
  const int32_t* cu_seqlens = nullptr;         // {0, used, seq_len}
  int64_t num_cu_seqlens = 0;
  const int32_t* refiner_cu_seqlens = nullptr;
  int64_t num_refiner_cu_seqlens = 0;
  bool skip_mask_out_condition = false;
};

struct MiniMaxH3DitOutputs {
  std::vector<float> video_logits;  // [num_infer_out_pos, video_row_width]
  std::vector<float> audio_logits;  // [num_audio_pos, audio_latents_dim]
};

// One DiT forward = one denoise step's velocity prediction. `compute_dtype` picks
// the block-stream dtype: kBF16 is the production path (upstream's cast points are
// preserved), kF32 is the parity path the golden suite gates.
MiniMaxH3DitOutputs MiniMaxH3DitForward(vt::Device device, const MiniMaxH3DitParams& params,
                                        const MiniMaxH3DitWeights& weights,
                                        const MiniMaxH3DitInputs& inputs,
                                        vt::DType compute_dtype);

// ---------------------------------------------------------------------------
// Denoise loop (denoise_loop.py:129-239)
// ---------------------------------------------------------------------------

// Static per-branch state: the packed layout plus the fixed forward inputs.
struct MiniMaxH3DenoiseBranch {
  MiniMaxH3PackedSequence packed;
  std::vector<float> text_embeddings;  // [text_len, text_dim]
  std::vector<int64_t> token_tags;     // seq_len, with fl2va vision overrides applied
};

// Runs the CFG-distilled loop: one positive forward per step, video and audio
// target rows chained through the Euler-eta0 update while pinned condition rows
// are reset to their anchors every step. Returns the final (video, audio) rows.
struct MiniMaxH3DenoiseResult {
  std::vector<float> video_rows;  // [num_img_pos, video_row_width]
  std::vector<float> audio_rows;  // [num_audio_pos, audio_latents_dim]
};

MiniMaxH3DenoiseResult MiniMaxH3DenoiseLoop(
    vt::Device device, const MiniMaxH3DitParams& params, const MiniMaxH3DitWeights& weights,
    const MiniMaxH3DenoiseBranch& branch, const std::vector<float>& initial_video_rows,
    const std::vector<float>& initial_audio_rows, const std::vector<float>& keyframe_cond_rows,
    const std::vector<float>& audio_ref_rows, const std::vector<double>& sigmas_video,
    const std::vector<double>& sigmas_audio, vt::DType compute_dtype);


// ---------------------------------------------------------------------------
// t2va pipeline assembly (minimax_h3_pipeline.cpp)
//
// The wiring from prompt embeddings to frames + waveform. Every stage it calls is
// separately ported and gated; this composes them.
// ---------------------------------------------------------------------------

struct MiniMaxH3T2vaRequest {
  int64_t text_len = 0;
  int64_t latent_t = 0, latent_h = 0, latent_w = 0;
  int64_t audio_t = 0;
  int64_t audio_channel = kMiniMaxH3AudioChannels;
  int64_t num_steps = kMiniMaxH3DefaultSteps;
  double video_shift = kMiniMaxH3DefaultVideoShift;
  double audio_shift = kMiniMaxH3DefaultAudioShift;
  // Per-channel latent statistics from each VAE's config.json; empty skips the
  // denormalization (useful in unit tests).
  std::vector<float> video_latents_mean, video_latents_std;
  std::vector<float> audio_latents_mean, audio_latents_std;
};

struct MiniMaxH3T2vaResult {
  std::vector<float> frames;  // [C, T, H, W]
  MiniMaxH3VideoFrameShape frame_shape;
  std::vector<float> waveform;  // channel-major, audio_samples_per_channel each
  int64_t audio_channels = 0;
  int64_t audio_samples_per_channel = 0;
  int64_t sample_rate = 0;
};

// Run the whole t2va path. NOISE IS AN INPUT: upstream seeds a torch CPU
// generator, and reproducing torch's RNG bit-exactly decides WHICH sample you get
// rather than whether the pipeline is right, so the caller supplies it.
MiniMaxH3T2vaResult MiniMaxH3GenerateT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                          const MiniMaxH3DitParams& dit_params,
                                          const MiniMaxH3DitWeights& dit_weights,
                                          const MiniMaxH3VideoVaeDecoderConfig& video_config,
                                          const MiniMaxH3AudioVaeWeights& video_weights,
                                          const MiniMaxH3AudioVaeConfig& audio_config,
                                          const MiniMaxH3AudioVaeWeights& audio_weights,
                                          const std::vector<float>& prompt_embeds,
                                          const std::vector<float>& initial_video_rows,
                                          const std::vector<float>& initial_audio_rows,
                                          vt::DType compute_dtype);

}  // namespace vllm
