// Parakeet ASR end to end on real audio, real pretrained weights, CPU only.
//
// This exists to answer one question the unit gates deliberately do NOT answer:
// does the ported encoder actually TRANSCRIBE? The oracle in
// tests/vllm/models/test_parakeet_ctc_engine.cpp gates our forward against a
// real HF `ParakeetForCTC` module, but with RANDOM weights, so it proves the
// math and nothing about speech. The pretrained arm there feeds silence and
// only checks well-formedness. Neither produces a transcript, on purpose.
//
// So: 16 kHz mono WAV -> log-mel (ParakeetAudioProcessor) -> encoder + CTC head
// -> greedy argmax -> CTC collapse -> token ids on stdout. Decoding ids to text
// is left to the caller's tokenizer (the checkpoint ships tokenizer.json);
// printing ids keeps this example free of a JSON tokenizer dependency and makes
// the output directly diffable against HF's own `generate`.
//
// Usage:
//   parakeet-transcribe <ckpt-dir> <audio.wav>
// where <ckpt-dir> is an HF-format ParakeetForCTC (config.json +
// model.safetensors), e.g. a snapshot of nvidia/parakeet-ctc-0.6b.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vllm/multimodal/parakeet_audio_processor.h"
#include "vt/backend.h"

namespace {

// Minimal RIFF/WAVE reader: 16-bit PCM mono, which is what the LibriSpeech and
// NeMo sample clips are. Anything else is refused loudly rather than resampled,
// mirroring the extractor's own refusal to resample
// (feature_extraction_parakeet.py:195-201).
bool ReadWav16BitMono(const std::string& path, std::vector<float>* out, int* sample_rate) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  char riff[12];
  f.read(riff, 12);
  if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0) {
    std::fprintf(stderr, "%s is not a RIFF/WAVE file\n", path.c_str());
    return false;
  }
  int channels = 0;
  int bits = 0;
  while (f) {
    char id[4];
    uint32_t sz = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&sz), 4);
    if (!f) break;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      std::vector<char> fmt(sz);
      f.read(fmt.data(), sz);
      uint16_t ch = 0, bps = 0;
      uint32_t sr = 0;
      std::memcpy(&ch, fmt.data() + 2, 2);
      std::memcpy(&sr, fmt.data() + 4, 4);
      std::memcpy(&bps, fmt.data() + 14, 2);
      channels = ch;
      bits = bps;
      *sample_rate = static_cast<int>(sr);
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (channels != 1 || bits != 16) {
        std::fprintf(stderr, "need 16-bit mono, got %d-bit %d-channel\n", bits, channels);
        return false;
      }
      const size_t n = sz / 2;
      std::vector<int16_t> pcm(n);
      f.read(reinterpret_cast<char*>(pcm.data()), sz);
      out->resize(n);
      for (size_t i = 0; i < n; ++i) {
        (*out)[i] = static_cast<float>(pcm[i]) / 32768.0F;
      }
      return true;
    } else {
      f.seekg(sz, std::ios::cur);
    }
  }
  std::fprintf(stderr, "%s has no data chunk\n", path.c_str());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <hf-parakeet-ctc-dir> <audio.wav>\n", argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  const std::string wav = argv[2];

  std::vector<float> samples;
  int sample_rate = 0;
  if (!ReadWav16BitMono(wav, &samples, &sample_rate)) return 1;
  std::fprintf(stderr, "audio: %zu samples @ %d Hz (%.2f s)\n", samples.size(), sample_rate,
               static_cast<double>(samples.size()) / sample_rate);

  vllm::multimodal::ParakeetEncoderConfig cfg;
  const vllm::multimodal::ParakeetForCTCWeights w = vllm::multimodal::LoadParakeetForCTC(ckpt, &cfg);
  std::fprintf(stderr, "weights: %zu encoder layers, vocab %lld\n", w.encoder.layers.size(),
               static_cast<long long>(cfg.vocab_size));

  vllm::multimodal::ParakeetExtractorConfig ecfg;
  ecfg.feature_size = static_cast<int>(cfg.num_mel_bins);
  const vllm::multimodal::ParakeetAudioProcessor proc(ecfg);
  const vllm::multimodal::ParakeetAudioFeatures feats =
      proc.ProcessWaveform(samples.data(), static_cast<int64_t>(samples.size()), sample_rate);
  std::fprintf(stderr, "features: %lld frames (%lld valid)\n",
               static_cast<long long>(feats.num_frames),
               static_cast<long long>(feats.valid_frames));

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  const vllm::multimodal::ParakeetCTCOutput out = vllm::multimodal::ParakeetForCTCForward(
      feats.input_features, feats.num_frames, feats.valid_frames, w, cfg, cpu);
  std::fprintf(stderr, "encoder: %lld output frames (%lld valid)\n",
               static_cast<long long>(out.num_output_frames),
               static_cast<long long>(out.valid_output_frames));

  // Collapsed CTC ids, space separated, one line. Diff this against HF's
  // ParakeetForCTC.generate() on the same clip, or feed it to tokenizer.json.
  for (size_t i = 0; i < out.token_ids.size(); ++i) {
    std::printf("%s%d", i ? " " : "", out.token_ids[i]);
  }
  std::printf("\n");
  return 0;
}
