// dots3-note's multimodal CHAT seam, registered on its own architecture
// (W6a, #2512).
//
// This is the SECOND architecture to reach the registry #2481 built, and it is
// what that row was for: adding a multimodal model is a NEW translation unit
// plus one `REGISTER_VLLM_MM_CHAT` line, with ZERO edits to any shared table.
// Before #2481 the server decided whether it could serve images by asking
// whether `<model_dir>/preprocessor_config.json` existed and then built
// Qwen3-VL's processor unconditionally — so this model would have been served
// Qwen3-VL's patch geometry, its merge size and its token ids, against its own
// `vision_config`, and answered 200.
//
// Ported from vLLM read in `~/_git/vllm` at **`9035151d6`**:
//   `Dots3NoteForCausalLM.get_placeholder_str` (`nvidia/multimodal.py:65-72`)
//     image -> f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}"
//   `IMAGE_START` / `IMAGE_PAD` / `IMAGE_END` (`common/processor.py:41-43`)
//     "<|img|>" / "<|imgpad|>" / "<|endofimg|>"
//   `_process_image_input` (`nvidia/multimodal.py:144-155`)
//     the placeholder run is `grid.prod(-1) // merge_size**2`
//   `MULTIMODAL_REGISTRY.register_processor` (`nvidia/multimodal.py:44-48`)
//     the registration sits ON THE MODEL and edits no shared table — the
//     mechanism this file mirrors.
//
// WHY THE MARKER IS BUILT HERE AND NOT TAKEN FROM `chat_mm.h`. That header's
// `ImagePlaceholderString()` returns Qwen3-VL's
// "<|vision_start|><|image_pad|><|vision_end|>" (`qwen3_vl.py:1716`), and
// `BuildMarkerInjectedContent` dispatches through it. Those are one
// architecture's markers, which is exactly the coupling #2475 removed from the
// install path; reaching for them here would put it back one layer down. What
// IS shared is everything that is not per-architecture: `ValidateChatMmLimits`
// (the per-item limit walk), `DecodeImageUrlPart` (the data-URI decode),
// `BaseProcessingInfo` (the `--limit-mm-per-prompt` fold) and
// `multimodal::ExpandImagePlaceholders` (the expansion rule, which is upstream's
// same `prod(grid) // merge**2` for both models).
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/model_executor/models/dots3_note_vision.h"  // the tower refusal
#include "vllm/multimodal/qwen3vl_processor.h"  // ExpandImagePlaceholders
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm::entrypoints::openai {

namespace {

using vllm::Dots3NoteVisionRefusalFor;
using vllm::LoadHfConfig;

namespace fs = std::filesystem;

// TU-local, matching `mm_chat_qwen3vl.cpp:38-56`. A checkpoint path arrives as
// UTF-8 and `fs::path` is `wchar_t`-based on Windows.
fs::path NativeUtf8Path(const std::string& value) {
#if defined(_WIN32)
  const std::u8string utf8(reinterpret_cast<const char8_t*>(value.data()),
                           value.size());
  return fs::path(utf8);
#else
  return fs::path(value);
#endif
}

std::string PathUtf8(const fs::path& path) {
#if defined(_WIN32)
  const std::u8string utf8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.string();
#endif
}

// `get_placeholder_str` (`nvidia/multimodal.py:65-68` @ `9035151d6`), the image
// branch. The SINGLE `<|imgpad|>` in the middle is what the tokenizer maps to
// ONE `image_token_id`, which `ExpandImagePlaceholders` then expands to N.
std::string Dots3NoteImageMarker() {
  return "<|img|><|imgpad|><|endofimg|>";
}

// `BuildMarkerInjectedContent`'s dots3 twin: walk the parts IN ORDER, append
// each text part's text and each image part's marker at its position. A
// bare-string message is returned unchanged, so the text path is byte-identical.
std::string BuildDots3NoteMarkerContent(const ChatMessage& message) {
  if (!message.content_parts.has_value()) {
    return message.content.has_value() ? *message.content : std::string();
  }
  std::string out;
  for (const ChatContentPart& part : *message.content_parts) {
    if (part.type == "text") {
      out += part.text;
    } else if (part.type == "image_url") {
      out += Dots3NoteImageMarker();
    }
    // Every other part type contributes nothing here and is refused earlier by
    // ValidateChatMmLimits, whose supported-limit map below declares only
    // "image".
  }
  return out;
}

// This seam's OWN ceiling — the other operand of the `min()` fold
// (`context.py:392-405`). Upstream's `Dots3NoteProcessingInfo` handles N images
// and video and audio; THIS seam locates exactly one image part and has no
// video or audio arm at all, so the honest ceiling is `{"image": 1}` and every
// other modality is ABSENT, which `context.py:414-415` reads as limit 0. A user
// limit can only LOWER it, so `--limit-mm-per-prompt image=99` still refuses the
// second image. Video is W7's and audio is W8's; when they land they raise
// these numbers here and nothing else changes.
std::map<std::string, std::optional<int>> Dots3NoteChatSupportedMmLimits() {
  return {{"image", std::optional<int>(1)}};
}

// `RouteImageRgb`'s dots3 twin (`chat_mm.cpp:162-187`): preprocess, expand the
// single placeholder id to `prod(grid)/merge^2` copies, and build the
// `mm_features` handle the engine's multimodal generate overload carries onto
// `Request.mm_features`.
multimodal::MultiModalInputs RouteDots3NoteImageRgb(
    const multimodal::Dots3NoteImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width, const std::vector<int32_t>& prompt_ids) {
  const multimodal::Dots3NoteProcessorConfig& cfg = proc.config();
  multimodal::ImageKwargs kw = proc.ProcessImage(rgb, height, width);
  const std::array<int64_t, 3> grid = kw.image_grid_thw;

  std::vector<std::array<int64_t, 3>> grids{grid};
  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandImagePlaceholders(
      prompt_ids, cfg.image_token_id, cfg.merge_size, grids, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "image";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    spec.mm_hash = proc.HashImage(rgb, height, width);
    spec.data = std::make_shared<multimodal::ImageKwargs>(std::move(kw));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

MultiModalChatFn MakeDots3NoteImageChatFn(
    std::shared_ptr<const multimodal::Dots3NoteImageProcessor> proc,
    const vllm::tok::Tokenizer& tokenizer, ChatPromptRenderFn prompt_fn,
    ImageCodecFn codec,
    std::shared_ptr<const multimodal::BaseProcessingInfo> info) {
  return [proc, info, &tokenizer, prompt_fn = std::move(prompt_fn),
          codec = std::move(codec)](const std::vector<ChatMessage>& messages)
             -> std::optional<multimodal::MultiModalInputs> {
    // STEP 0: the per-item limit check, BEFORE anything is decoded or dropped
    // (`chat_utils.py:662` validates as it tracks, for the same reason).
    ValidateChatMmLimits(*info, messages);

    const ChatContentPart* image_part = nullptr;
    for (const ChatMessage& m : messages) {
      if (!m.content_parts.has_value()) continue;
      for (const ChatContentPart& part : *m.content_parts) {
        if (part.type == "image_url") {
          image_part = &part;
          break;
        }
      }
      if (image_part != nullptr) break;
    }
    if (image_part == nullptr) return std::nullopt;  // the text path, untouched

    // 1. Inject dots3-note's OWN marker at each image part's position and
    //    render the templated prompt.
    std::vector<ChatMessage> rendered = messages;
    for (ChatMessage& m : rendered) {
      if (m.content_parts.has_value()) {
        m.content = BuildDots3NoteMarkerContent(m);
        m.content_parts.reset();
      }
    }
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, {},
                  nlohmann::ordered_json::object());

    // 2. Tokenize WITH special tokens: the single `<|imgpad|>` becomes ONE
    //    `image_token_id` (added tokens match leftmost-longest).
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(prompt);

    // 3. Decode + route: expand that id to N and build the mm_features.
    const DecodedMedia media = DecodeImageUrlPart(*image_part);
    const DecodedImageRgb img = codec(media);
    return RouteDots3NoteImageRgb(*proc, img.rgb.data(), img.height, img.width,
                                  prompt_ids);
  };
}

MultiModalChatSeam MakeDots3NoteChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.mm_config == nullptr || !ctx.prompt_fn ||
      !ctx.codec) {
    // Refuse by name rather than dereference. The install's catch turns this
    // into a REFUSING seam, which is an HTTP 400 naming the architecture — never
    // a silent text answer.
    throw std::runtime_error(
        "dots3-note multimodal chat seam: the install context is incomplete "
        "(tokenizer, multimodal config, chat-prompt renderer and image codec "
        "are all required)");
  }

  const std::string preprocessor_config_path =
      PathUtf8(NativeUtf8Path(ctx.model_dir) / "preprocessor_config.json");
  if (!fs::exists(NativeUtf8Path(preprocessor_config_path))) {
    throw std::runtime_error(
        "dots3-note multimodal chat seam: '" + preprocessor_config_path +
        "' is missing; the image processor's patch/merge geometry, its "
        "per-channel normalization and its pixel bounds are read from it");
  }

  // THE TOWER'S OWN REFUSAL, ASKED HERE AND NOT IN THE ENGINE. W6a shipped the
  // dense blocks and W6b (#2613) the pyramid MoE ones, so the RELEASED
  // `dots-studio/dots3-note-prev` — 17 of whose 42 blocks are routed — is
  // ACCEPTED here now. What still refuses is `use_bias` (#2616), the softmax
  // and top-k-below-2 router arms (#2615), the blockwise-FP8 tower (W9) and
  // video (W7).
  //
  // Asking at INSTALL is not a preference. `EncodeMmDots3NoteForCausalLM`
  // refuses too, but it runs inside the engine's busy loop: throwing there stops
  // `AsyncLLM` and turns every LATER request, TEXT ONES INCLUDED, into a 500.
  // That was measured on this row's served-request gate before this check
  // existed. Refusing here installs a REFUSING seam instead, which is upstream's
  // own shape for "this server does not accept images for this model": HTTP 400
  // naming the architecture and the reason, with the text path untouched. The
  // encoder's check stays as defence in depth, on the same polarity Qwen3-VL's
  // carries ("reaching this point is a defect").
  {
    const std::string why =
        Dots3NoteVisionRefusalFor(LoadHfConfig(ctx.config_path));
    if (!why.empty()) {
      throw std::runtime_error(
          "dots3-note multimodal chat seam: this checkpoint's vision tower is "
          "not ported — " + why +
          ". See .agents/specs/dots3-note.md §4.11 and §4.12, issues #2512 and "
          "#2613.");
    }
  }

  auto proc = std::make_shared<const multimodal::Dots3NoteImageProcessor>(
      multimodal::LoadDots3NoteProcessorConfig(
          preprocessor_config_path, ctx.config_path, ctx.served_model_name));
  // The engine's limits (`--limit-mm-per-prompt`, `--language-model-only`)
  // folded by min() against this seam's own ceiling. The MultiModalConfig is
  // held BY REFERENCE (`context.h:105`); the engine owns it and outlives the
  // seam.
  auto info = std::make_shared<const multimodal::BaseProcessingInfo>(
      *ctx.mm_config, Dots3NoteChatSupportedMmLimits());

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail = "dots3-note processor from " + preprocessor_config_path +
                " (dense + pyramid MoE vision arm)";
  seam.chat_fn = MakeDots3NoteImageChatFn(proc, *ctx.tokenizer, ctx.prompt_fn,
                                          ctx.codec, info);
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(dots3_note, "Dots3NoteForCausalLM", &MakeDots3NoteChatSeam)

}  // namespace vllm::entrypoints::openai
