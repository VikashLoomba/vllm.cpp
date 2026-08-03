// MiniMax-H3 audio output: RIFF/WAVE serialization of the decoded waveform.
//
// The audio VAE emits a CHANNEL-MAJOR float waveform (all of channel 0, then all
// of channel 1) at 32 kHz. Every consumer wants interleaved PCM, so this does the
// interleave and the container in one place.
//
// WHY THIS EXISTS SEPARATELY FROM MP4 MUXING. `/v1/videos` ultimately returns MP4
// (H.264 video + audio), and how this project should obtain a muxer is an open
// DEPENDENCY DECISION (upstream vLLM-Omni shells out to ffmpeg; this tree has no
// subprocess precedent in the core library). WAV needs none of that: it is a
// 44-byte header plus samples, it is dependency-free, and it is required under
// EITHER outcome of that decision — a muxer will take PCM, and a standalone audio
// artifact is useful on its own. So it is implemented now and the container
// question stays open.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace {

void PutU32(std::string& out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

void PutU16(std::string& out, uint16_t value) {
  for (int i = 0; i < 2; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

}  // namespace

std::string MiniMaxH3WriteWav(const std::vector<float>& waveform, int64_t channels,
                              int64_t samples_per_channel, int64_t sample_rate) {
  VT_CHECK(channels > 0 && samples_per_channel > 0 && sample_rate > 0,
           "minimax_h3 wav: channels, samples and sample_rate must be positive");
  VT_CHECK(static_cast<int64_t>(waveform.size()) == channels * samples_per_channel,
           "minimax_h3 wav: waveform size does not match channels * samples_per_channel");

  const int64_t total = channels * samples_per_channel;
  const uint32_t data_bytes = static_cast<uint32_t>(total * 2);  // 16-bit PCM
  const uint16_t block_align = static_cast<uint16_t>(channels * 2);

  std::string out;
  out.reserve(44 + static_cast<size_t>(data_bytes));
  out += "RIFF";
  PutU32(out, 36u + data_bytes);  // chunk size = header remainder + payload
  out += "WAVE";
  out += "fmt ";
  PutU32(out, 16u);                                        // PCM fmt chunk size
  PutU16(out, 1u);                                         // format = PCM
  PutU16(out, static_cast<uint16_t>(channels));
  PutU32(out, static_cast<uint32_t>(sample_rate));
  PutU32(out, static_cast<uint32_t>(sample_rate) * block_align);  // byte rate
  PutU16(out, block_align);
  PutU16(out, 16u);                                        // bits per sample
  out += "data";
  PutU32(out, data_bytes);

  // The VAE's layout is CHANNEL-MAJOR; WAV is INTERLEAVED. Getting this backwards
  // produces audio that plays but with the channels time-smeared into each other.
  for (int64_t s = 0; s < samples_per_channel; ++s) {
    for (int64_t c = 0; c < channels; ++c) {
      float value = waveform[static_cast<size_t>(c * samples_per_channel + s)];
      value = std::min(1.0f, std::max(-1.0f, value));
      // Symmetric scaling by 32767 so +1.0 and -1.0 map to the extremes without
      // wrapping; -32768 is never produced.
      const int32_t quantized = static_cast<int32_t>(std::lround(value * 32767.0f));
      PutU16(out, static_cast<uint16_t>(static_cast<int16_t>(quantized)));
    }
  }
  return out;
}

}  // namespace vllm
