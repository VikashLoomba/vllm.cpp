// dots3-note IMAGE processor (W6a, #2512).
//
// Ported from `vllm/models/dots3_note/common/processor.py` read in `~/_git/vllm`
// at **`9035151d6`** — the merge of vllm#51255. `dots3_note` does not exist at
// our parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, so every anchor
// here names that SHA; upstream has already moved under this row once.
//
//   IMAGE_START / IMAGE_PAD / IMAGE_END      :41-43
//   Dots3NoteImageProcessor.__init__         :63-79
//   .factor                                  :83-84
//   ._round_by_factor / _ceil / _floor       :86-96
//   .resized_size                            :97-146
//   .preprocess                              :147-218
//
// WHY THIS IS NOT `Qwen3VLImageProcessor` WITH DIFFERENT NUMBERS. Three things
// differ and each is silent when wrong:
//
//   1. `resized_size` is NOT `smart_resize`. Upstream dots3 rounds each side
//      INDEPENDENTLY to a multiple of `factor` and only then applies the pixel
//      budget (`processed.py:139-146`), where `smart_resize`
//      (transformers `image_processing_qwen2_vl.py:62`) does the same rounding
//      but with different guards and a different min-pixel branch. The two
//      agree on many images and disagree on some, and a disagreement moves the
//      GRID, which moves the placeholder count, which changes the prompt.
//   2. `image_mean` / `image_std` are PER-CHANNEL lists, not the single scalar
//      Qwen3-VL's 0.5/0.5 collapses to.
//   3. The patch row order is selected by `pre_pixel_shuffle`. TRUE is the
//      2x2-grouped order (which happens to be byte-identical to Qwen3-VL's
//      merge-grouped patchify); FALSE is plain row-major. The released
//      checkpoint sets TRUE, and the tower's RoPE position builder reads the
//      SAME flag — so a processor and a tower that disagree on it produce a
//      well-shaped, wrong answer.
//
// The placeholder EXPANSION is shared rather than re-written:
// `multimodal::ExpandImagePlaceholders` already takes the image token id, the
// merge size and the grids, and dots3's rule is upstream's same
// `grid.prod(-1) // merge_size**2` (`multimodal.py:151-155` @ `9035151d6`).
#ifndef VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_
#define VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// The subset of `preprocessor_config.json` + `config.json` the image path
// needs. Defaults are upstream's own where upstream has one; the three token
// ids have none upstream (they come from `added_tokens.json`, read at
// `multimodal.py:82-90` @ `9035151d6`) and are therefore REQUIRED by the
// loader below rather than defaulted to a number this port invented.
struct Dots3NoteProcessorConfig {
  int patch_size = 14;
  int temporal_patch_size = 1;
  int merge_size = 2;  // == vision_config.spatial_merge_size
  bool pre_pixel_shuffle = true;
  // Per channel, in the checkpoint's own order (R, G, B).
  std::array<double, 3> image_mean{0.5, 0.5, 0.5};
  std::array<double, 3> image_std{0.5, 0.5, 0.5};
  double rescale_factor = 1.0 / 255.0;
  int64_t min_pixels = 3136;
  int64_t max_pixels = 12845056;

  // `<|img|>` / `<|imgpad|>` / `<|endofimg|>` in the checkpoint's tokenizer.
  int32_t image_token_id = -1;
  int32_t image_start_token_id = -1;
  int32_t image_end_token_id = -1;

  std::string model_id = "dots-studio/dots3-note-prev";  // for the mm-hash
};

// Load from the two HF json documents. THROWS BY NAME when the three image
// token ids cannot be resolved: a processor that guessed them would inject a
// marker the tokenizer maps to something else, and the request would be served
// as text with the image dropped.
Dots3NoteProcessorConfig LoadDots3NoteProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id);

// `Dots3NoteImageProcessor.resized_size` (`common/processor.py:97-146` @
// `9035151d6`) — the height/width both divisible by `factor` whose product lies
// in `[min_pixels, max_pixels]`. Throws upstream's two refusals: a side under
// `factor / 4`, and an aspect ratio over 200.
std::array<int64_t, 2> Dots3NoteResizedSize(int64_t height, int64_t width,
                                            int64_t factor, int64_t min_pixels,
                                            int64_t max_pixels);

class Dots3NoteImageProcessor {
 public:
  explicit Dots3NoteImageProcessor(Dots3NoteProcessorConfig cfg)
      : cfg_(std::move(cfg)) {}

  const Dots3NoteProcessorConfig& config() const { return cfg_; }

  int64_t factor() const {
    return static_cast<int64_t>(cfg_.patch_size) * cfg_.merge_size;
  }

  // Preprocess ONE RGB image (HWC uint8, height*width*3) into
  // `pixel_values [num_patches, channel*temporal*patch*patch]` +
  // `image_grid_thw`. A genuine bicubic RESIZE is a NAMED residual, exactly as
  // it is on the Qwen3-VL processor beside this one: an image whose dimensions
  // `Dots3NoteResizedSize` would change is REFUSED with both sizes in the
  // message rather than silently patchified at the wrong grid.
  ImageKwargs ProcessImage(const uint8_t* rgb, int64_t height,
                           int64_t width) const;

  std::string HashImage(const uint8_t* rgb, int64_t height,
                        int64_t width) const;

 private:
  Dots3NoteProcessorConfig cfg_;
};

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_DOTS3_NOTE_PROCESSOR_H_
