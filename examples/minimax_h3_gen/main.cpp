// minimax-h3-gen: the ASSEMBLY driver — open the real checkpoints, run the whole
// t2va path, and write an MP4.
//
// Everything below this line was gated component by component (packed layout, DiT
// forward, both VAE decoders, the encoder towers, the loaders). This is the piece
// that puts them together over REAL files, which is the only way the remaining
// integration questions — config plumbing, latent statistics, shard assembly —
// actually get answered.
//
// It lives in examples/ for the same reason the muxer does: this is where the
// ffmpeg invocation is allowed (developer-ratified 2026-08-03). src/vllm/ builds
// artifacts and argv and spawns nothing.
//
// Usage:
//   minimax-h3-gen --dit <dit.gguf|dit.safetensors>
//                  --video-vae <video_vae.safetensors> --video-vae-config <config.json>
//                  --audio-vae <audio_vae.safetensors> --audio-vae-config <config.json>
//                  --prompt-embeds <f32.bin>   (rows of text_dim, little-endian f32)
//                  --out <out.mp4>
//                  [--keep-quant] [--steps N] [--frames N] [--height N] [--width N]
//                  [--workdir DIR] [--ffmpeg PATH] [--dry-run]
//
// PROMPT EMBEDDINGS are taken as a file rather than computed here, deliberately:
// the encoder tower needs a tokenizer + a 32B forward, which is its own driver.
// This keeps the assembly question ("do the checkpoints compose into a video?")
// separable from the encoding question.
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vt/backend.h"

namespace {

int RunFfmpeg(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) throw std::runtime_error("waitpid failed");
  if (WIFSIGNALED(status)) {
    throw std::runtime_error("ffmpeg died on signal " + std::to_string(WTERMSIG(status)));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  nlohmann::json j;
  in >> j;
  return j;
}

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = in.tellg();
  if (bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error(path + ": size is not a whole number of f32 values");
  }
  in.seekg(0);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

void WriteFile(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string Need(int argc, char** argv, int i, const std::string& flag) {
  if (i >= argc) throw std::runtime_error("missing value for " + flag);
  return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
  std::string dit_path, video_vae_path, video_cfg_path, audio_vae_path, audio_cfg_path;
  std::string embeds_path, out_path, workdir = "/tmp/minimax_h3_gen", ffmpeg = "ffmpeg";
  bool keep_quant = false, dry_run = false, dequant_bf16 = false, denoise_only = false;
  std::string device_name = "cpu";
  std::string encoder_path, prompt, tokenizer_path, save_embeds_path;
  int64_t encoder_max_layers = 0;
  int64_t steps = 0, frames = 0, height = 0, width = 0;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string f = argv[i];
      if (f == "--dit") dit_path = Need(argc, argv, ++i, f);
      else if (f == "--video-vae") video_vae_path = Need(argc, argv, ++i, f);
      else if (f == "--video-vae-config") video_cfg_path = Need(argc, argv, ++i, f);
      else if (f == "--audio-vae") audio_vae_path = Need(argc, argv, ++i, f);
      else if (f == "--audio-vae-config") audio_cfg_path = Need(argc, argv, ++i, f);
      else if (f == "--prompt-embeds") embeds_path = Need(argc, argv, ++i, f);
      else if (f == "--out") out_path = Need(argc, argv, ++i, f);
      else if (f == "--workdir") workdir = Need(argc, argv, ++i, f);
      else if (f == "--ffmpeg") ffmpeg = Need(argc, argv, ++i, f);
      else if (f == "--keep-quant") keep_quant = true;
      else if (f == "--dequant-bf16") dequant_bf16 = true;
      else if (f == "--dry-run") dry_run = true;
      else if (f == "--denoise-only") denoise_only = true;
      else if (f == "--device") device_name = Need(argc, argv, ++i, f);
      else if (f == "--encoder") encoder_path = Need(argc, argv, ++i, f);
      else if (f == "--prompt") prompt = Need(argc, argv, ++i, f);
      else if (f == "--tokenizer") tokenizer_path = Need(argc, argv, ++i, f);
      else if (f == "--save-embeds") save_embeds_path = Need(argc, argv, ++i, f);
      else if (f == "--encoder-max-layers") encoder_max_layers = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--steps") steps = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--frames") frames = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--height") height = std::stoll(Need(argc, argv, ++i, f));
      else if (f == "--width") width = std::stoll(Need(argc, argv, ++i, f));
      else throw std::runtime_error("unknown argument: " + f);
    }
    // --denoise-only stops after the DiT step loop, so it needs neither VAE nor an
    // output path -- and, just as importantly, does not spend their memory. On a
    // unified-memory box that headroom is the difference between a run and a reboot.
    const bool need_vaes = !denoise_only;
    if (dit_path.empty() || (need_vaes && (video_vae_path.empty() || audio_vae_path.empty())) ||
        (need_vaes && out_path.empty()) ||
        (embeds_path.empty() && (encoder_path.empty() || prompt.empty()))) {
      std::cerr << "usage: minimax-h3-gen --dit <f> --video-vae <f> --audio-vae <f> "
                   "--prompt-embeds <f32.bin> --out <out.mp4> [--video-vae-config <j>] "
                   "[--audio-vae-config <j>] [--keep-quant] [--steps N] [--frames N] "
                   "[--height N] [--width N] [--device cpu|cuda] [--workdir DIR] [--ffmpeg PATH] "
                   "[--dry-run] [--denoise-only]\n";
      return 2;
    }

    // --- 1. DiT ---
    std::cerr << "loading DiT " << dit_path << (keep_quant ? " (keep-quant)" : "") << "\n";
    vllm::MiniMaxH3GgufDit dit;
    vllm::MiniMaxH3DitDeviceWeights streamed;
    bool have_streamed = false;
    if (EndsWith(dit_path, ".gguf")) {
      const vllm::GgufFile f = vllm::GgufFile::Open(dit_path);
      // --dequant-bf16 loads STRAIGHT to bf16 (~33 GB). Keeping blocks would leave
      // the AdaLN projections ineligible (K=2688 is not a whole number of 256-element
      // Q3_K blocks) and dequantize them to ~52 GB of f32 — which is what does not fit.
      if (dequant_bf16 && device_name == "cuda") {
        // STREAM straight onto the device: dequantize + upload one tensor at a time
        // so the host copy never accumulates. Peak is what kills a unified-memory
        // box, and load-then-stage holds ~33 GB twice.
        vt::Queue sq = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
        const auto t0 = std::chrono::steady_clock::now();
        streamed = vllm::StreamMiniMaxH3DitToDeviceBf16(sq, f, &dit.params);
        have_streamed = true;
        std::cerr << "  streamed DiT -> device (bf16) in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s\n";
      } else {
        dit = dequant_bf16 ? vllm::LoadMiniMaxH3DitFromGgufBf16(f)
                           : vllm::LoadMiniMaxH3DitFromGguf(f, keep_quant);
      }
    } else {
      const vllm::SafetensorsFile f = vllm::SafetensorsFile::Open(dit_path);
      if (device_name == "cuda") {
        // Stream NVFP4 straight to the device. The host-f32 reference loader is
        // ~132 GB for this checkpoint and gets OOM-killed during load on a
        // unified-memory box; streaming keeps peak at the device copy plus one
        // tensor.
        vt::Queue sq = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
        const auto t0 = std::chrono::steady_clock::now();
        streamed = vllm::StreamMiniMaxH3Nvfp4ToDeviceBf16(sq, f, &dit.params);
        have_streamed = true;
        std::cerr << "  streamed NVFP4 DiT -> device (bf16) in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s\n";
      } else {
        dit = vllm::LoadMiniMaxH3DitFromNvfp4(f);
      }
    }
    std::cerr << "  layers=" << dit.params.num_layers << " hidden=" << dit.params.hidden_size
              << " heads=" << dit.params.num_attention_heads << "\n";

    // --- 1b. optional encoder probe. Loading the 32B tower keep-quant is the
    // precondition for real text conditioning; this reports the geometry it
    // recovered so the loader can be validated against the REAL file. ---
    std::vector<float> encoded_prompt;
    if (!encoder_path.empty()) {
      std::cerr << "loading encoder " << encoder_path << " (keep-quant)\n";
      const vllm::GgufFile ef = vllm::GgufFile::Open(encoder_path);
      const vllm::MiniMaxH3EncoderQuantWeights enc =
          vllm::LoadMiniMaxH3EncoderFromGguf(ef, encoder_max_layers);
      vllm::MiniMaxH3EncoderConfig ec = enc.config;
      std::cerr << "  encoder layers=" << ec.num_hidden_layers << " hidden=" << ec.hidden_size
                << " heads=" << ec.num_attention_heads << " kv_heads=" << ec.num_key_value_heads
                << " head_dim=" << ec.head_dim << " ffn=" << ec.intermediate_size << "\n";
      size_t quant_bytes = 0;
      for (const auto& kv : enc.quant_storage) quant_bytes += kv.second.size();
      std::cerr << "  encoder resident (keep-quant) = " << (quant_bytes / (1024.0 * 1024.0 * 1024.0))
                << " GiB\n";

      if (!prompt.empty()) {
        // The ComfyUI-style encoder GGUF is WEIGHTS ONLY — it carries no
        // `tokenizer.ggml.*` metadata, unlike a llama.cpp export — so the vocab
        // comes from the checkpoint's own tokenizer.json. `--tokenizer` is
        // therefore required with `--prompt` unless the GGUF happens to embed one.
        const vllm::tok::Tokenizer tokenizer =
            tokenizer_path.empty() ? vllm::tok::Tokenizer::FromGguf(ef)
                                   : vllm::tok::Tokenizer::FromHfJson(tokenizer_path);
        const std::vector<int32_t> ids = tokenizer.Encode(prompt);
        VT_CHECK(!ids.empty(), "minimax-h3-gen: the prompt tokenized to nothing");
        std::cerr << "  prompt tokens = " << ids.size() << "\n";
        const std::vector<float> embeds = vllm::MiniMaxH3EncoderEmbedTokens(enc, ids);
        // Text-only: all three M-RoPE axes are the token index.
        const int64_t seq = static_cast<int64_t>(ids.size());
        std::vector<int64_t> pos(static_cast<size_t>(3 * seq));
        for (int64_t a = 0; a < 3; ++a) {
          for (int64_t s = 0; s < seq; ++s) pos[static_cast<size_t>(a * seq + s)] = s;
        }
        vt::Device enc_dev{};
        if (device_name == "cuda") {
          enc_dev = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
        }
        vt::Queue eq{enc_dev, nullptr};
        vt::Backend& eb = vt::GetBackend(enc_dev.type);
        if (enc_dev.type != vt::DeviceType::kCPU) eq = eb.CreateQueue();
        const vllm::MiniMaxH3EncoderDeviceWeights staged =
            vllm::StageMiniMaxH3EncoderWeights(eq, enc);
        std::cerr << "  encoding prompt...\n";
        encoded_prompt =
            vllm::MiniMaxH3EncoderTextForwardDevice(eq, ec, staged, embeds, pos.data(), seq);
        std::cerr << "  conditioning = [" << seq << ", " << ec.hidden_size << "]\n";
        // Persisting the conditioning makes a checkpoint A/B CONTROLLED: two DiTs
        // can then be compared on byte-identical text conditioning instead of two
        // separate encoder runs. It also lets the second run skip the 13 GB tower
        // entirely, which is the difference between fitting and not on a
        // unified-memory box.
        if (!save_embeds_path.empty()) {
          std::ofstream out(save_embeds_path, std::ios::binary);
          if (!out) throw std::runtime_error("cannot write " + save_embeds_path);
          out.write(reinterpret_cast<const char*>(encoded_prompt.data()),
                    static_cast<std::streamsize>(encoded_prompt.size() * sizeof(float)));
          std::cerr << "  saved conditioning -> " << save_embeds_path << "\n";
        }
      }
    }

    // --- 2. VAEs + their configs (the configs carry the latent statistics) ---
    vllm::MiniMaxH3VideoVaeDecoderConfig video_cfg;
    vllm::MiniMaxH3LatentStats video_stats;
    if (!video_cfg_path.empty()) {
      video_cfg = vllm::ParseMiniMaxH3VideoVaeDecoderConfig(ReadJson(video_cfg_path), &video_stats);
    }
    vllm::MiniMaxH3AudioVaeConfig audio_cfg;
    vllm::MiniMaxH3LatentStats audio_stats;
    if (!audio_cfg_path.empty()) {
      audio_cfg = vllm::ParseMiniMaxH3AudioVaeConfig(ReadJson(audio_cfg_path), &audio_stats);
    }
    // The files stay in scope alongside the weights: the loaders read through the
    // mapping, so releasing it early would leave the weights pointing at nothing.
    std::optional<vllm::SafetensorsFile> video_file, audio_file;
    vllm::MiniMaxH3AudioVaeWeights video_weights, audio_weights;
    if (need_vaes) {
      std::cerr << "loading video VAE " << video_vae_path << "\n";
      video_file = vllm::SafetensorsFile::Open(video_vae_path);
      video_weights = vllm::LoadMiniMaxH3VideoVaeDecoderWeights(*video_file);
      std::cerr << "loading audio VAE " << audio_vae_path << "\n";
      audio_file = vllm::SafetensorsFile::Open(audio_vae_path);
      audio_weights = vllm::LoadMiniMaxH3AudioVaeWeights(*audio_file);
    }

    // --- 3. request shape ---
    vllm::MiniMaxH3T2vaRequest request;
    request.text_len = 0;  // set from the embeddings below
    if (steps > 0) request.num_steps = steps;
    // `_resolve_shape` decides frames/canvas/latent_t/audio_t; the latent GRID is
    // that canvas divided by the VAE's spatial ratio (prod(space_down) = 16).
    const vllm::MiniMaxH3ShapePlan plan = vllm::MiniMaxH3ResolveShape(
        "t2va", /*duration_seconds=*/0.0, frames, height, width,
        /*image_width=*/0, /*image_height=*/0);
    request.latent_t = plan.latent_t;
    request.latent_h = plan.height / vllm::kMiniMaxH3VaeRatio;
    request.latent_w = plan.width / vllm::kMiniMaxH3VaeRatio;
    request.audio_t = plan.audio_t;
    request.audio_channel = vllm::kMiniMaxH3AudioChannels;
    request.video_latents_mean = video_stats.mean;
    request.video_latents_std = video_stats.std_dev;
    request.audio_latents_mean = audio_stats.mean;
    request.audio_latents_std = audio_stats.std_dev;

    // Real conditioning when a prompt was encoded; otherwise the supplied file.
    const std::vector<float> prompt_embeds =
        !encoded_prompt.empty() ? encoded_prompt : ReadF32(embeds_path);
    if (dit.params.text_dim <= 0 || prompt_embeds.size() % static_cast<size_t>(dit.params.text_dim) != 0) {
      throw std::runtime_error("--prompt-embeds size is not a multiple of text_dim");
    }
    request.text_len = static_cast<int64_t>(prompt_embeds.size()) / dit.params.text_dim;
    std::cerr << "  text_len=" << request.text_len << " latent=" << request.latent_t << "x"
              << request.latent_h << "x" << request.latent_w << " steps=" << request.num_steps
              << "\n";

    if (dry_run) {
      std::cerr << "--dry-run: everything LOADED and planned; stopping before generation\n";
      return 0;
    }

    // --- 4. generate ---
    // NOISE IS AN INPUT (upstream seeds a torch CPU generator, and reproducing its
    // RNG bit-exactly decides WHICH sample you get, not whether the pipeline is
    // right), so it is drawn here from the shared deterministic stream.
    const int64_t frame_rows = (request.latent_h / dit.params.patch_size_h) *
                               (request.latent_w / dit.params.patch_size_w);
    const int64_t video_rows = request.latent_t * frame_rows;
    const int64_t audio_rows = request.audio_t * request.audio_channel;
    // A splitmix64 stream, seeded so a run is REPRODUCIBLE. This deliberately does
    // NOT reproduce torch's RNG: matching it bit-for-bit decides WHICH sample you
    // get, not whether the pipeline is correct, and pretending otherwise would
    // invite comparing our sample against upstream's as if they should match.
    auto fill = [](std::vector<float>& out, uint64_t seed) {
      uint64_t x = seed;
      for (float& v : out) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        v = static_cast<float>((z >> 11) * 0x1.0p-53 * 2.0 - 1.0);
      }
    };
    std::vector<float> noise_video(
        static_cast<size_t>(video_rows * dit.params.video_row_width()));
    std::vector<float> noise_audio(
        static_cast<size_t>(audio_rows * dit.params.audio_latents_dim));
    fill(noise_video, 0x5EED1234ULL);
    fill(noise_audio, 0x5EED5678ULL);

    std::cerr << "generating (" << request.num_steps << " steps)...\n";
    // On a device, the denoise loop stages the DiT weights ONCE and runs every step
    // device-resident. On CPU it uses the portable reference forward, which is a
    // correctness path, not a throughput one.
    vt::Device device{};
    if (device_name == "cuda") {
      device = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue().device;
    } else if (device_name != "cpu") {
      throw std::runtime_error("--device must be cpu or cuda");
    }
    // Stage ONCE here rather than inside the denoise loop. --dequant-bf16 trades
    // memory (DiT ~33 GB bf16 vs 15.6 GB kept-quant) for GEMM throughput: the
    // keep-quant path measured ~103 GFLOP/s, which is what makes a full-quality
    // render a multi-day job.
    vllm::MiniMaxH3DitDeviceWeights staged;
    const vllm::MiniMaxH3DitDeviceWeights* prestaged = nullptr;
    if (have_streamed) {
      prestaged = &streamed;
    } else if (device.type != vt::DeviceType::kCPU) {
      vt::Queue sq = vt::GetBackend(device.type).CreateQueue();
      const auto t0 = std::chrono::steady_clock::now();
      staged = vllm::StageMiniMaxH3DitWeights(sq, dit.params, dit.weights, vt::DType::kBF16);
      std::cerr << "  staged DiT (" << (dequant_bf16 ? "dequant-bf16" : "keep-quant") << ") in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                << " s\n";
      prestaged = &staged;
    }
    if (denoise_only) {
      // Time the DiT step loop by itself. Reported as an AVERAGE over the requested
      // steps rather than a single step: the first step pays one-off costs (RoPE
      // caches, allocator warm-up) that do not recur, so a one-step run overstates
      // the steady-state cost.
      const auto t0 = std::chrono::steady_clock::now();
      const vllm::MiniMaxH3DenoiseResult denoised = vllm::MiniMaxH3DenoiseT2va(
          device, request, dit.params, dit.weights, prompt_embeds, noise_video, noise_audio,
          vt::DType::kBF16, prestaged);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const int64_t seq = static_cast<int64_t>(request.text_len) + video_rows + audio_rows;
      std::cout << "denoise-only: seq=" << seq << " steps=" << request.num_steps << " total="
                << elapsed << " s  per_step=" << (elapsed / double(request.num_steps)) << " s\n";
      // Touch the result so the loop cannot be optimized away, and so an all-NaN
      // forward shows up here instead of passing as a fast run.
      double checksum = 0.0;
      for (const float v : denoised.video_rows) checksum += v;
      std::cout << "  video_rows=" << denoised.video_rows.size() << " checksum=" << checksum
                << "\n";
      return 0;
    }

    const vllm::MiniMaxH3T2vaResult result = vllm::MiniMaxH3GenerateT2va(
        device, request, dit.params, dit.weights, video_cfg, video_weights, audio_cfg,
        audio_weights, prompt_embeds, noise_video, noise_audio, vt::DType::kBF16, prestaged);

    // --- 5. artifacts (the LIBRARY builds these; nothing spawns) ---
    std::string mkdir_cmd = "mkdir -p '" + workdir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) throw std::runtime_error("cannot create " + workdir);
    for (int64_t f = 0; f < result.frame_shape.t; ++f) {
      char name[512];
      std::snprintf(name, sizeof(name), "%s/frame_%06lld.ppm", workdir.c_str(),
                    static_cast<long long>(f));
      WriteFile(name, vllm::MiniMaxH3WritePpmFrame(result.frames, result.frame_shape, f));
    }
    const std::string wav_path = workdir + "/audio.wav";
    WriteFile(wav_path, vllm::MiniMaxH3WriteWav(result.waveform, result.audio_channels,
                                                result.audio_samples_per_channel,
                                                result.sample_rate));
    std::cerr << "  wrote " << result.frame_shape.t << " frames + " << wav_path << "\n";

    // --- 6. mux (the ONE process spawn, and it is in examples/ by decision) ---
    vllm::MiniMaxH3MuxRequest mux;
    mux.frame_pattern = workdir + "/frame_%06d.ppm";
    mux.audio_path = wav_path;
    mux.output_path = out_path;
    std::vector<std::string> args = vllm::MiniMaxH3BuildMp4MuxArgs(mux);
    if (!args.empty()) args[0] = ffmpeg;
    const int status = RunFfmpeg(args);
    if (status != 0) {
      std::cerr << "ffmpeg exited " << status << "\n";
      return status;
    }
    std::cout << "wrote " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
