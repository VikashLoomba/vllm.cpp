// dots3-note IMAGE processor (W6a, #2512). Ported from
// `vllm/models/dots3_note/common/processor.py` read in `~/_git/vllm` at
// `9035151d6`. See the header for the full provenance and for the three ways
// this differs from the Qwen3-VL processor beside it.
#include "vllm/multimodal/dots3_note_processor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/hasher.h"
#include "vt/dtype.h"

namespace vllm::multimodal {

namespace {

nlohmann::json LoadJson(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open json: " + path);
  nlohmann::json j;
  f >> j;
  return j;
}

// `round(v / factor) * factor` with Python's round-half-to-EVEN
// (`processor.py:86-88` @ `9035151d6`). `std::round` is half-away-from-zero and
// would disagree at exactly .5, which is one grid row.
int64_t RoundByFactor(int64_t v, int64_t factor) {
  return static_cast<int64_t>(std::nearbyint(static_cast<double>(v) /
                                             static_cast<double>(factor))) *
         factor;
}
int64_t CeilByFactor(double v, int64_t factor) {
  return static_cast<int64_t>(
             std::ceil(v / static_cast<double>(factor))) * factor;
}
int64_t FloorByFactor(double v, int64_t factor) {
  return static_cast<int64_t>(
             std::floor(v / static_cast<double>(factor))) * factor;
}

// Resolve one of the three image token ids from `added_tokens.json` (upstream's
// own source, `multimodal.py:82-90` @ `9035151d6`) or, failing that, from the
// `config.json` key a converted checkpoint may carry. Returns -1 when neither
// answers; the caller refuses BY NAME rather than defaulting.
int32_t ResolveTokenId(const nlohmann::json& added, const nlohmann::json& cfg,
                       const char* marker, const char* config_key) {
  if (added.is_object()) {
    const auto it = added.find(marker);
    if (it != added.end() && it->is_number_integer())
      return it->get<int32_t>();
  }
  if (cfg.is_object()) {
    const auto it = cfg.find(config_key);
    if (it != cfg.end() && it->is_number_integer())
      return it->get<int32_t>();
  }
  return -1;
}

}  // namespace

Dots3NoteProcessorConfig LoadDots3NoteProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id) {
  Dots3NoteProcessorConfig cfg;
  cfg.model_id = model_id;

  const nlohmann::json pp = LoadJson(preprocessor_config_json_path);
  cfg.patch_size = pp.value("patch_size", cfg.patch_size);
  cfg.temporal_patch_size =
      pp.value("temporal_patch_size", cfg.temporal_patch_size);
  cfg.merge_size = pp.value("merge_size", cfg.merge_size);
  cfg.pre_pixel_shuffle = pp.value("pre_pixel_shuffle", cfg.pre_pixel_shuffle);
  cfg.min_pixels = pp.value("min_pixels", cfg.min_pixels);
  cfg.max_pixels = pp.value("max_pixels", cfg.max_pixels);
  // The `size` shorthand the HF image-processor family also writes.
  if (pp.contains("size") && pp["size"].is_object()) {
    const auto& sz = pp["size"];
    cfg.min_pixels = sz.value("shortest_edge", cfg.min_pixels);
    cfg.max_pixels = sz.value("longest_edge", cfg.max_pixels);
  }
  // PER CHANNEL. Reading only `[0]` — which is what the Qwen3-VL loader beside
  // this one can afford, because its mean and std are 0.5 on all three — would
  // silently normalize green and blue with red's statistics.
  const auto read3 = [](const nlohmann::json& j, const char* key,
                        std::array<double, 3>* out) {
    if (!j.contains(key)) return;
    const auto& a = j[key];
    if (!a.is_array()) {
      throw std::runtime_error(std::string("dots3-note processor: '") + key +
                               "' must be a 3-entry list, got " + a.dump());
    }
    if (a.size() == 1) {
      (*out) = {a[0].get<double>(), a[0].get<double>(), a[0].get<double>()};
      return;
    }
    if (a.size() != 3) {
      throw std::runtime_error(
          std::string("dots3-note processor: '") + key + "' has " +
          std::to_string(a.size()) +
          " entries; the RGB pipeline needs 1 or 3 (processor.py:76-77 @ "
          "9035151d6)");
    }
    for (int i = 0; i < 3; ++i) (*out)[static_cast<size_t>(i)] = a[i].get<double>();
  };
  read3(pp, "image_mean", &cfg.image_mean);
  read3(pp, "image_std", &cfg.image_std);
  cfg.rescale_factor = pp.value("rescale_factor", cfg.rescale_factor);

  const nlohmann::json cj = LoadJson(config_json_path);
  // `vision_config` is the AUTHORITY on the patch/merge geometry, exactly as it
  // is for the tower: a `preprocessor_config.json` that disagrees with the
  // model it belongs to would move the grid.
  if (cj.contains("vision_config") && cj["vision_config"].is_object()) {
    const auto& vc = cj["vision_config"];
    cfg.merge_size = vc.value("spatial_merge_size", cfg.merge_size);
    cfg.patch_size = vc.value("patch_size", cfg.patch_size);
    cfg.temporal_patch_size =
        vc.value("temporal_patch_size", cfg.temporal_patch_size);
    cfg.pre_pixel_shuffle = vc.value("pre_pixel_shuffle", cfg.pre_pixel_shuffle);
  }

  // The three ids. Upstream reads `added_tokens.json` and RAISES when
  // `<|imgpad|>` is absent (`multimodal.py:86-90` @ `9035151d6`); this mirrors
  // that, and extends it to the two markers around it, because injecting a
  // start/end marker the tokenizer does not know breaks the prompt just as
  // quietly.
  nlohmann::json added = nlohmann::json::object();
  {
    const std::string dir =
        config_json_path.substr(0, config_json_path.find_last_of("/\\") + 1);
    std::ifstream f(dir + "added_tokens.json");
    if (f) f >> added;
  }
  cfg.image_token_id =
      ResolveTokenId(added, cj, "<|imgpad|>", "image_token_id");
  cfg.image_start_token_id =
      ResolveTokenId(added, cj, "<|img|>", "image_start_token_id");
  cfg.image_end_token_id =
      ResolveTokenId(added, cj, "<|endofimg|>", "image_end_token_id");
  const auto require = [](int32_t id, const char* marker, const char* key) {
    if (id >= 0) return;
    throw std::runtime_error(
        std::string("dots3-note processor: the image token '") + marker +
        "' has no id. Upstream reads it from `added_tokens.json` and raises "
        "when it is missing (multimodal.py:86-90 @ 9035151d6); this port also "
        "accepts `config.json`'s `" + key +
        "`. Refusing rather than guessing an id: a marker the tokenizer does "
        "not know is injected as ordinary text and the image is dropped.");
  };
  require(cfg.image_token_id, "<|imgpad|>", "image_token_id");
  require(cfg.image_start_token_id, "<|img|>", "image_start_token_id");
  require(cfg.image_end_token_id, "<|endofimg|>", "image_end_token_id");
  return cfg;
}

std::array<int64_t, 2> Dots3NoteResizedSize(int64_t height, int64_t width,
                                            int64_t factor, int64_t min_pixels,
                                            int64_t max_pixels) {
  // `processor.py:131-146` @ `9035151d6`, in upstream's own order.
  if (std::min(height, width) < factor / 4) {
    throw std::runtime_error(
        "dots3-note processor: image height and width must be at least " +
        std::to_string(factor / 4) + ", got " + std::to_string(height) + "x" +
        std::to_string(width));
  }
  if (std::min(height, width) <= 0 ||
      static_cast<double>(std::max(height, width)) /
              static_cast<double>(std::min(height, width)) >
          200.0) {
    throw std::runtime_error(
        "dots3-note processor: image aspect ratio must be smaller than 200");
  }
  int64_t rh = std::max(factor, RoundByFactor(height, factor));
  int64_t rw = std::max(factor, RoundByFactor(width, factor));
  const double hw = static_cast<double>(height) * static_cast<double>(width);
  if (rh * rw > max_pixels) {
    const double beta = std::sqrt(hw / static_cast<double>(max_pixels));
    rh = std::max(factor, FloorByFactor(static_cast<double>(height) / beta, factor));
    rw = std::max(factor, FloorByFactor(static_cast<double>(width) / beta, factor));
  } else if (rh * rw < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / hw);
    rh = CeilByFactor(static_cast<double>(height) * beta, factor);
    rw = CeilByFactor(static_cast<double>(width) * beta, factor);
    if (rh * rw > max_pixels) {
      const double b2 = std::sqrt(static_cast<double>(rh) *
                                  static_cast<double>(rw) /
                                  static_cast<double>(max_pixels));
      rh = std::max(factor, FloorByFactor(static_cast<double>(rh) / b2, factor));
      rw = std::max(factor, FloorByFactor(static_cast<double>(rw) / b2, factor));
    }
  }
  return {rh, rw};
}

std::string Dots3NoteImageProcessor::HashImage(const uint8_t* rgb,
                                               int64_t height,
                                               int64_t width) const {
  return MultiModalHasher::HashImageRGB(cfg_.model_id, rgb, height, width);
}

ImageKwargs Dots3NoteImageProcessor::ProcessImage(const uint8_t* rgb,
                                                  int64_t height,
                                                  int64_t width) const {
  const int64_t patch = cfg_.patch_size;
  const int64_t merge = cfg_.merge_size;
  const int64_t tp = cfg_.temporal_patch_size;
  const int64_t f = factor();

  const std::array<int64_t, 2> rs =
      Dots3NoteResizedSize(height, width, f, cfg_.min_pixels, cfg_.max_pixels);
  const int64_t rh = rs[0], rw = rs[1];
  if (rh != height || rw != width) {
    // A genuine bicubic resize (`Image.Resampling.BICUBIC`,
    // `processor.py:174`). NAMED, exactly as the Qwen3-VL processor beside this
    // one names it: patchifying at the wrong grid would change the placeholder
    // count and serve a well-shaped wrong prompt.
    //
    // THIS IS A CAPABILITY GAP, NOT A CORNER. `factor` is
    // `patch_size * merge_size`, which on the released checkpoint is 28, and
    // upstream ALWAYS resizes — so once W6b lifts the MoE ViT refusal, almost
    // no real image clears this. It is owed by W8 (the MM front end brick that
    // owns the processor), recorded under `## Owed` in
    // `.agents/specs/dots3-note.md`, and tracked by issue #2537. Both this
    // comment and the message below claimed that record before it existed; the
    // fresh review of #2523 found the claim false and this is the repair.
    throw std::runtime_error(
        "Dots3NoteImageProcessor: image requires resize (" +
        std::to_string(width) + "x" + std::to_string(height) + " -> " +
        std::to_string(rw) + "x" + std::to_string(rh) +
        "); the bicubic resize path is not ported (W6a uses conformant "
        "images). Owed by W8 and tracked by issue #2537; see "
        ".agents/specs/dots3-note.md `## Owed`.");
  }

  const int64_t grid_h = rh / patch;
  const int64_t grid_w = rw / patch;
  const int64_t grid_t = 1;  // ONE image; video grids are W7's
  if (cfg_.pre_pixel_shuffle && (grid_h % merge != 0 || grid_w % merge != 0)) {
    throw std::runtime_error(
        "Dots3NoteImageProcessor: the " + std::to_string(grid_h) + "x" +
        std::to_string(grid_w) +
        " patch grid does not group into whole " + std::to_string(merge) +
        "x" + std::to_string(merge) +
        " blocks, which `pre_pixel_shuffle` requires (processor.py:185-197 @ "
        "9035151d6)");
  }
  const int64_t num_patches = grid_t * grid_h * grid_w;
  const int64_t feat = 3 * tp * patch * patch;

  // Fused rescale + normalize, PER CHANNEL:
  //   (raw * rescale - mean) / std  ==  (raw - mean/rescale) / (std/rescale)
  // `processor.py:166-167`.
  double shift[3], scale[3];
  for (int c = 0; c < 3; ++c) {
    shift[c] = cfg_.image_mean[static_cast<size_t>(c)] / cfg_.rescale_factor;
    scale[c] = cfg_.image_std[static_cast<size_t>(c)] / cfg_.rescale_factor;
  }

  ImageKwargs out;
  out.num_patches = num_patches;
  out.patch_feature_dim = feat;
  out.image_grid_thw = {grid_t, grid_h, grid_w};
  out.pixel_values_f32.resize(static_cast<size_t>(num_patches * feat));
  out.pixel_values_bf16.resize(static_cast<size_t>(num_patches * feat));

  const int64_t rowstride = width * 3;  // HWC uint8 source stride
  // The column index is the same under both row orders:
  //   k = ((c * tp + t) * patch + ph) * patch + pw
  // because both transposes end `..., C, tp, ph, pw` (`processor.py:196`,
  // `:207`). Only the ROW index differs.
  const auto emit = [&](int64_t r, int64_t src_h0, int64_t src_w0) {
    for (int64_t c = 0; c < 3; ++c) {
      for (int64_t t = 0; t < tp; ++t) {
        for (int64_t ph = 0; ph < patch; ++ph) {
          const int64_t H = src_h0 + ph;
          for (int64_t pw = 0; pw < patch; ++pw) {
            const int64_t W = src_w0 + pw;
            const uint8_t raw = rgb[H * rowstride + W * 3 + c];
            const float v = static_cast<float>(
                (static_cast<double>(raw) - shift[c]) / scale[c]);
            const int64_t k = ((c * tp + t) * patch + ph) * patch + pw;
            const size_t idx = static_cast<size_t>(r * feat + k);
            out.pixel_values_f32[idx] = v;
            out.pixel_values_bf16[idx] = vt::F32ToBF16(v);
          }
        }
      }
    }
  };

  if (cfg_.pre_pixel_shuffle) {
    // `reshape(t, tp, C, Gh, m, p, Gw, m, p).transpose(0,3,6,4,7,2,1,5,8)`
    // (`processor.py:185-197`): row index
    //   r = ((gh * Gw + gw) * m + mh) * m + mw
    const int64_t Gh = grid_h / merge, Gw = grid_w / merge;
    for (int64_t gh = 0; gh < Gh; ++gh) {
      for (int64_t gw = 0; gw < Gw; ++gw) {
        for (int64_t mh = 0; mh < merge; ++mh) {
          for (int64_t mw = 0; mw < merge; ++mw) {
            const int64_t r = ((gh * Gw + gw) * merge + mh) * merge + mw;
            emit(r, (gh * merge + mh) * patch, (gw * merge + mw) * patch);
          }
        }
      }
    }
  } else {
    // `reshape(t, tp, C, gh, p, gw, p).transpose(0,3,5,2,1,4,6)`
    // (`processor.py:199-208`): plain row-major, r = gh * grid_w + gw.
    for (int64_t gh = 0; gh < grid_h; ++gh) {
      for (int64_t gw = 0; gw < grid_w; ++gw) {
        emit(gh * grid_w + gw, gh * patch, gw * patch);
      }
    }
  }
  return out;
}

}  // namespace vllm::multimodal
