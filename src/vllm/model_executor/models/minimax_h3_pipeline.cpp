// MiniMax-H3 t2va pipeline assembly — the path from prompt embeddings to frames
// and a waveform.
//
// Everything this file calls was ported and gated separately; W6 is the WIRING:
//
//   prompt_embeds ──► packed sequence (fl2va/t2va layout)
//                     │
//                     ├─► denoise loop: 50 x DiT forward + euler-eta0 step
//                     │      (sigmas from the rectified-flow time-shift schedule)
//                     ▼
//              video rows / audio rows
//                     │
//        unpatchify ──┤── unpack audio
//                     ▼
//        video latent ──► video VAE ViT3D decoder ──► frames  [C, T, H, W]
//        audio latent ──► audio VAE BigVGAN       ──► waveform per channel
//
// Latents are DENORMALIZED before decode (`latent * std + mean`), mirroring
// vae.py:252-270 / :341-357.
//
// NOISE IS AN INPUT, deliberately. Upstream seeds a torch CPU generator
// (pipeline_minimax_h3.py:813-843); reproducing torch's RNG bit-exactly is a
// separate concern that decides WHICH sample you get, not whether the pipeline is
// correct, so the caller supplies the initial rows. See the spec's open items.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {

MiniMaxH3DenoiseResult MiniMaxH3DenoiseT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                            const MiniMaxH3DitParams& dit_params,
                                            const MiniMaxH3DitWeights& dit_weights,
                                            const std::vector<float>& prompt_embeds,
                                            const std::vector<float>& initial_video_rows,
                                            const std::vector<float>& initial_audio_rows,
                                            vt::DType compute_dtype,
                                            const MiniMaxH3DitDeviceWeights* prestaged) {
  VT_CHECK(request.text_len > 0, "minimax_h3 t2va: text_len must be positive");
  VT_CHECK(request.num_steps >= 1, "minimax_h3 t2va: num_steps must be >= 1");
  VT_CHECK(static_cast<int64_t>(prompt_embeds.size()) == request.text_len * dit_params.text_dim,
           "minimax_h3 t2va: prompt_embeds must be [text_len, text_dim]");

  // --- 1. packed layout (t2va: no keyframe conditioning) ---
  MiniMaxH3DenoiseBranch branch;
  branch.packed = BuildMiniMaxH3PackedSequence(
      request.text_len, request.latent_t, request.latent_h, request.latent_w, request.audio_t,
      request.audio_channel, /*include_keyframe_cond=*/false, {}, /*frame_count=*/0);
  branch.text_embeddings = prompt_embeds;
  branch.token_tags = branch.packed.token_tags;

  // --- 2. the rectified-flow sigma schedules (video and audio shift differently) ---
  const std::vector<double> sigmas_video =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.video_shift);
  const std::vector<double> sigmas_audio =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.audio_shift);
  VT_CHECK(sigmas_video.size() == sigmas_audio.size(),
           "minimax_h3 t2va: the two sigma schedules must have equal length");

  // --- 3. the denoise loop (one DiT forward per step) ---
  return MiniMaxH3DenoiseLoop(device, dit_params, dit_weights, branch, initial_video_rows,
                              initial_audio_rows, /*keyframe_cond_rows=*/{},
                              /*audio_ref_rows=*/{}, sigmas_video, sigmas_audio, compute_dtype,
                              prestaged);
}

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
                                          vt::DType compute_dtype,
                                          const MiniMaxH3DitDeviceWeights* prestaged) {
  const MiniMaxH3DenoiseResult denoised =
      MiniMaxH3DenoiseT2va(device, request, dit_params, dit_weights, prompt_embeds,
                           initial_video_rows, initial_audio_rows, compute_dtype, prestaged);

  // --- 4. rows -> latents ---
  const int64_t ph = request.latent_h / dit_params.patch_size_h;
  const int64_t pw = request.latent_w / dit_params.patch_size_w;
  std::vector<float> video_latent = MiniMaxH3UnpatchifyVideoTokens(
      denoised.video_rows, request.latent_t, ph, pw, dit_params.latents_dim,
      dit_params.patch_size_t, dit_params.patch_size_h, dit_params.patch_size_w);
  std::vector<float> audio_latent = MiniMaxH3UnpackAudioTokens(
      denoised.audio_rows, request.audio_t * request.audio_channel, request.audio_channel,
      dit_params.audio_latents_dim);

  // --- 5. denormalize (vae.py:252-270, :341-357) ---
  auto denormalize = [](std::vector<float>& latent, int64_t channels, int64_t per_channel,
                        const std::vector<float>& mean, const std::vector<float>& std_dev) {
    if (mean.empty() && std_dev.empty()) return;
    VT_CHECK(static_cast<int64_t>(mean.size()) == channels &&
                 static_cast<int64_t>(std_dev.size()) == channels,
             "minimax_h3 t2va: latents_mean/std must have one value per channel");
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t i = 0; i < per_channel; ++i) {
        float& value = latent[static_cast<size_t>(c * per_channel + i)];
        value = value * std_dev[static_cast<size_t>(c)] + mean[static_cast<size_t>(c)];
      }
    }
  };
  const int64_t video_per_channel = request.latent_t * request.latent_h * request.latent_w;
  denormalize(video_latent, dit_params.latents_dim, video_per_channel, request.video_latents_mean,
              request.video_latents_std);
  const int64_t audio_steps = request.audio_t / request.audio_channel;
  denormalize(audio_latent, dit_params.audio_latents_dim, audio_steps * request.audio_channel,
              request.audio_latents_mean, request.audio_latents_std);

  // --- 6. decode ---
  MiniMaxH3T2vaResult result;
  // post_quant_conv (the AutoencoderKL WRAPPER's Conv3d 1x1x1 channel mix) runs on
  // the latent BEFORE the decoder. It sits OUTSIDE ViT3DDecoder, which is why the
  // decoder's own 8.9e-8 gate never covered it. Applied only when the weights
  // carry it, so a synthetic/reduced-dimension weight set without it still runs --
  // the structural t2va test does not ship one.
  if (video_weights.Has("post_quant_conv.weight")) {
    video_latent = MiniMaxH3VideoVaePostQuantConv(video_weights, video_latent,
                                                  dit_params.latents_dim, video_per_channel);
  }
  // On a device, run the ViT3D decoder device-resident. The portable decoder is a
  // scalar reference; at real resolutions it is the stage that does not finish. It
  // stays the CPU path, and stays the thing the device path is gated against.
  if (device.type != vt::DeviceType::kCPU) {
    vt::Queue vq = vt::GetBackend(device.type).CreateQueue();
    const MiniMaxH3VideoVaeDeviceWeights staged_vae =
        StageMiniMaxH3VideoVaeWeights(vq, video_config, video_weights);
    result.frames = MiniMaxH3VideoVaeDecodeDevice(device, video_config, staged_vae, video_latent,
                                                  request.latent_t, request.latent_h,
                                                  request.latent_w, &result.frame_shape);
  } else {
    result.frames = MiniMaxH3VideoVaeDecode(video_config, video_weights, video_latent,
                                            request.latent_t, request.latent_h, request.latent_w,
                                            &result.frame_shape);
  }

  // The audio VAE decodes ONE channel at a time; the packed rows are channel-major.
  result.audio_channels = request.audio_channel;
  for (int64_t c = 0; c < request.audio_channel; ++c) {
    std::vector<float> channel(static_cast<size_t>(dit_params.audio_latents_dim * audio_steps));
    for (int64_t d = 0; d < dit_params.audio_latents_dim; ++d) {
      for (int64_t t = 0; t < audio_steps; ++t) {
        channel[static_cast<size_t>(d * audio_steps + t)] = audio_latent[static_cast<size_t>(
            (c * dit_params.audio_latents_dim + d) * audio_steps + t)];
      }
    }
    int64_t samples = 0;
    const std::vector<float> wave =
        MiniMaxH3AudioVaeDecode(audio_config, audio_weights, channel, audio_steps, &samples);
    if (c == 0) {
      result.audio_samples_per_channel = samples;
      result.waveform.reserve(static_cast<size_t>(samples * request.audio_channel));
    } else {
      VT_CHECK(samples == result.audio_samples_per_channel,
               "minimax_h3 t2va: audio channels decoded to different lengths");
    }
    result.waveform.insert(result.waveform.end(), wave.begin(), wave.end());
  }
  result.sample_rate = kMiniMaxH3AudioSampleRate;
  return result;
}

}  // namespace vllm
