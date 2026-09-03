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
#include "vllm/model_executor/models/dots3_note_audio.h"   // the audio refusal
#include "vllm/model_executor/models/dots3_note_vision.h"  // the tower refusal
#include "vllm/multimodal/qwen3vl_processor.h"  // ExpandImagePlaceholders
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/validation_error.h"  // a bad upload is a 400, not a 500

namespace vllm::entrypoints::openai {

namespace {

using vllm::Dots3NoteAudioRefusalFor;
using vllm::Dots3NoteVisionRefusalFor;
using vllm::HfConfig;
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

// The AUDIO branch of `get_placeholder_str` (`nvidia/multimodal.py:68-69` @
// `9035151d6`), W7a (#2703): `f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}"`.
//
// THE STRINGS COME FROM THE CONFIG, not from a literal, because `audio_config`
// carries them (`audio_comp_start` / `audio_comp_span` / `audio_comp_end`,
// `nvidia/audio.py:37-39`) and the marker injected here must be the one whose
// ids the processor resolved from the tokenizer. The image marker above is a
// literal only because `common/processor.py:41-43` hard-codes those three.
std::string Dots3NoteAudioMarker(
    const multimodal::Dots3NoteAudioProcessorConfig& cfg) {
  return cfg.audio_comp_start + cfg.audio_comp_span + cfg.audio_comp_end;
}

// `BuildMarkerInjectedContent`'s dots3 twin: walk the parts IN ORDER, append
// each text part's text and each image or audio part's marker at its position.
// A bare-string message is returned unchanged, so the text path is
// byte-identical.
//
// `audio_marker` is EMPTY when this install has no audio tower, and then an
// audio part contributes nothing — which it cannot reach anyway, because
// `ValidateChatMmLimits` has already refused it against a ceiling that does not
// declare "audio".
std::string BuildDots3NoteMarkerContent(const ChatMessage& message,
                                        const std::string& audio_marker) {
  if (!message.content_parts.has_value()) {
    return message.content.has_value() ? *message.content : std::string();
  }
  std::string out;
  for (const ChatContentPart& part : *message.content_parts) {
    if (part.type == "text") {
      out += part.text;
    } else if (part.type == "image_url") {
      out += Dots3NoteImageMarker();
    } else if (part.type == "input_audio" || part.type == "audio_url") {
      out += audio_marker;
    }
    // Every other part type contributes nothing here and is refused earlier by
    // ValidateChatMmLimits, whose supported-limit map below declares only the
    // modalities this seam can build features for.
  }
  return out;
}

// This seam's OWN ceiling — the other operand of the `min()` fold
// (`context.py:392-405`). Upstream's `Dots3NoteProcessingInfo` declares
// `{"image": 512, "video": 1, "audio": 128}` (`common/processor.py:527-534` @
// `9035151d6`); THIS seam locates exactly ONE image part and, since W7a
// (#2703), exactly ONE audio part, so the honest ceiling is `{"image": 1,
// "audio": 1}` and every other modality is ABSENT, which `context.py:414-415`
// reads as limit 0. A user limit can only LOWER it, so
// `--limit-mm-per-prompt image=99` still refuses the second image.
//
// `has_audio` IS A PARAMETER AND NOT A CONSTANT, because a checkpoint with no
// `audio_config` has no audio tower and declaring the modality would promise a
// capability that install could not deliver. VIDEO is W8's; when it lands it
// raises these numbers and nothing else changes. (W7 IS AUDIO and W8 IS VIDEO —
// this comment used to say the reverse, against the loader's own deferral
// table and against #2703's own title.)
std::map<std::string, std::optional<int>> Dots3NoteChatSupportedMmLimits(
    bool has_audio) {
  std::map<std::string, std::optional<int>> limits{
      {"image", std::optional<int>(1)}};
  if (has_audio) limits.emplace("audio", std::optional<int>(1));
  return limits;
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

// `RouteDots3NoteImageRgb`'s AUDIO twin (W7a, #2703): decode the container,
// run the front end, expand the single `<|audio_comp_pad|>` id to
// `ceil(num_samples / token_stride)` copies, and build the `mm_features`
// handle.
//
// WHY THIS IS NOT `RouteAudioWav` (`chat_mm.cpp:131-160`). That function is
// DEAD in `src/` — nothing outside `tests/` calls it — and its test is the
// `ROAD-V1-MM` parse gate, so it belongs to another row. It is also WRONG for
// this model in a way no shape check would report: it takes the token count
// from `AudioProcessorConfig::max_source_positions`, Whisper's FIXED 1500, and
// dots3's count is `ceil(num_samples / 1280)` and depends on the waveform.
// Editing it would move another row's gate to serve this one; W7a therefore
// writes its own, exactly as W6a wrote `RouteDots3NoteImageRgb` rather than
// editing `RouteImageRgb`.
//
// THE CONTAINER REFUSAL IS HERE AND NOT IN THE PROCESSOR, because the container
// is a REQUEST property while the rate is a CONFIG one.
// `DecodeWavPcm16MeanToMono` already refuses a non-PCM16 or malformed buffer by
// name; what this adds is WHO OWES IT, so an operator learns that rather than
// only what failed.
//
// W7c-1 (#2813) NARROWED it. It used to say a multi-channel WAV was owed to
// W7c; any channel count is now served, mean-reduced as upstream reduces. And
// the container arm left this row entirely: it needs a demuxer this tree does
// not vendor, five surfaces refuse compressed media for that same missing
// brick, and #2814 owns it. The refusal stays at the SEAM and stays the SAME
// predicate as the route, because this throw is HTTP 400 for one request while
// the same throw from inside `encode_mm` would set `AsyncLLM`'s errored latch
// and 500 every later request, TEXT ones included.
multimodal::MultiModalInputs RouteDots3NoteAudioWav(
    const multimodal::Dots3NoteAudioProcessor& proc, const DecodedMedia& audio,
    const std::vector<int32_t>& prompt_ids) {
  const multimodal::Dots3NoteAudioProcessorConfig& cfg = proc.config();

  multimodal::DecodedAudio decoded;
  try {
    // W7c-1 (#2813): the MEAN-reducing sibling, so a multi-channel PCM16 WAV
    // at the target rate is SERVED instead of refused. This is the production
    // call site the reachability mutation deletes.
    decoded = multimodal::DecodeWavPcm16MeanToMono(audio.bytes.data(),
                                                   audio.bytes.size());
  } catch (const std::exception& e) {
    // `InputValidationError`, NOT `std::runtime_error`. The container is a
    // property of the REQUEST, so this is a client error and
    // `ApiServer::handle_chat_completions` maps this type to HTTP 400
    // ("BadRequestError") while a bare `runtime_error` falls through to the
    // generic 500 arm. Upstream reaches the same place: `create_error_response`
    // maps `ValueError` to `BadRequestError`
    // (serve/utils/error_response.py:62-65). Reporting a malformed upload as a
    // SERVER fault is what the 500 arm is for, and it is not this.
    throw vllm::v1::InputValidationError(
        std::string("dots3-note audio chat seam: this request's audio is not a "
                    "PCM16 RIFF/WAVE buffer (") + e.what() +
        "). Only that container is ported. The `mp3`/`flac`/`ogg` an "
        "`input_audio.format` may name needs a demuxer this tree does not "
        "vendor; five surfaces refuse compressed media for that same missing "
        "brick, and it is tracked by issue #2814 — a SHARED brick, not a "
        "dots3-note one. Any CHANNEL count is served since W7c-1 (#2813): the "
        "channels are mean-reduced to mono, as upstream's "
        "`load_audio(..., mono=True)` does "
        "(vllm/multimodal/media/audio.py:207-208, :220 @ 9035151d6). A "
        "sampling rate other than `audio_config.sampling_rate` is refused "
        "separately, by the front end, and is owed to W7c-2. See "
        ".agents/specs/dots3-note.md §4.16 and issue #2813.");
  }

  // The front end refuses a wrong rate (W7c-2) BY NAME, and — since W7b (#2797)
  // lifted the `chunk_seconds` ceiling — a waveform past ONE chunk on a
  // checkpoint whose `chunk_samples` is not a whole number of `token_stride`s,
  // where upstream's own per-segment row sum and its prompt-side
  // `ceil(total / stride)` disagree (spec §4.15.3). Both messages name the
  // reason and the numbers.
  //
  // THIS IS THE FRONT END AND NOT THE ENGINE LOOP, and that is the point of
  // refusing here: `InputValidationError` becomes HTTP 400 for THIS request,
  // where the same throw from inside `encode_mm` would set `AsyncLLM`'s errored
  // latch and 500 every later request, text ones included.
  multimodal::AudioKwargs kw = proc.ProcessWaveform(
      decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()),
      decoded.sampling_rate);

  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandAudioPlaceholders(
      prompt_ids, cfg.audio_token_id,
      {static_cast<int>(kw.num_tokens)}, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "audio";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    // The mm-hash is over the RAW waveform, before feature extraction, exactly
    // as the image hash is over the raw pixels — so the encoder cache keys on
    // what the request carried rather than on what the front end derived.
    spec.mm_hash = proc.HashAudio(
        decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()));
    spec.audio_data = std::make_shared<multimodal::AudioKwargs>(std::move(kw));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

// THE CHAT FN, over both modalities (W7a, #2703).
//
// `audio_proc` is NULL when this checkpoint carries no `audio_config`, and then
// this seam behaves exactly as it did before W7a: the supported-limit map does
// not declare "audio", so `ValidateChatMmLimits` refuses an audio part with
// upstream's own "At most 0 audio(s) may be provided in one prompt." and the
// marker builder is handed an empty audio marker it never reaches.
//
// ONE ITEM OF EACH, and the reason is the same one the image arm has carried
// since W6a: this function LOCATES a single part per modality, so the seam's
// declared ceiling is what it can actually build. A second part of either kind
// is refused at STEP 0 rather than silently dropped.
MultiModalChatFn MakeDots3NoteChatFn(
    std::shared_ptr<const multimodal::Dots3NoteImageProcessor> proc,
    std::shared_ptr<const multimodal::Dots3NoteAudioProcessor> audio_proc,
    const vllm::tok::Tokenizer& tokenizer, ChatPromptRenderFn prompt_fn,
    ImageCodecFn codec,
    std::shared_ptr<const multimodal::BaseProcessingInfo> info) {
  const std::string audio_marker =
      audio_proc != nullptr ? Dots3NoteAudioMarker(audio_proc->config())
                            : std::string();
  return [proc, audio_proc, info, audio_marker, &tokenizer,
          prompt_fn = std::move(prompt_fn),
          codec = std::move(codec)](const std::vector<ChatMessage>& messages)
             -> std::optional<multimodal::MultiModalInputs> {
    // STEP 0: the per-item limit check, BEFORE anything is decoded or dropped
    // (`chat_utils.py:662` validates as it tracks, for the same reason).
    ValidateChatMmLimits(*info, messages);

    const ChatContentPart* image_part = nullptr;
    const ChatContentPart* audio_part = nullptr;
    for (const ChatMessage& m : messages) {
      if (!m.content_parts.has_value()) continue;
      for (const ChatContentPart& part : *m.content_parts) {
        if (part.type == "image_url" && image_part == nullptr) {
          image_part = &part;
        } else if ((part.type == "input_audio" || part.type == "audio_url") &&
                   audio_part == nullptr) {
          audio_part = &part;
        }
      }
    }
    // The text path, untouched and byte-identical.
    if (image_part == nullptr && audio_part == nullptr) return std::nullopt;

    // ONE MODALITY PER REQUEST, refused BY NAME rather than served half. A
    // prompt carrying both would need TWO `mm_features` entries whose
    // placeholder spans are expanded in ONE pass over the same id stream, and
    // `ExpandImagePlaceholders` and `ExpandAudioPlaceholders` are separate
    // functions over separate ids that each rebuild the whole vector — so
    // running them in sequence would leave the second one's offsets measured
    // against the first one's un-expanded input. Upstream applies all of its
    // `PromptReplacement`s in one pass (`common/processor.py:749-783` @
    // `9035151d6`); this port does not have that machinery yet, and a mixed
    // request is owed to W8 with the rest of the front end.
    if (image_part != nullptr && audio_part != nullptr) {
      // A client error, so HTTP 400 rather than 500 — see the container
      // refusal's note in `RouteDots3NoteAudioWav`.
      throw vllm::v1::InputValidationError(
          "dots3-note multimodal chat seam: this request carries BOTH an image "
          "and an audio part. Each is served on its own (W6a-W6c and W7a); "
          "serving them together needs the single-pass placeholder expansion "
          "upstream applies to all modalities at once "
          "(common/processor.py:749-783 @ 9035151d6), and that is owed to W8. "
          "Refused by name rather than expanding one and dropping the other. "
          "See .agents/specs/dots3-note.md §4.14.5 and issue #2703.");
    }
    if (audio_part != nullptr && audio_proc == nullptr) {
      // Unreachable while the limit map and this pointer agree; kept because
      // reaching it would otherwise be a null dereference rather than an
      // answer.
      throw std::runtime_error(
          "dots3-note multimodal chat seam: an audio part reached the route on "
          "an install with no audio processor. The supported-limit map and the "
          "processor must be built from the same `audio_config`.");
    }

    // 1. Inject dots3-note's OWN markers at each part's position and render the
    //    templated prompt.
    std::vector<ChatMessage> rendered = messages;
    for (ChatMessage& m : rendered) {
      if (m.content_parts.has_value()) {
        m.content = BuildDots3NoteMarkerContent(m, audio_marker);
        m.content_parts.reset();
      }
    }
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, {},
                  nlohmann::ordered_json::object());

    // 2. Tokenize WITH special tokens: the single `<|imgpad|>` or
    //    `<|audio_comp_pad|>` becomes ONE id (added tokens match
    //    leftmost-longest).
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(prompt);

    // 3. Decode + route: expand that id to N and build the mm_features.
    if (audio_part != nullptr) {
      const DecodedMedia media = audio_part->type == "input_audio"
                                     ? DecodeInputAudioPart(*audio_part)
                                     : DecodeDataUri(audio_part->url);
      return RouteDots3NoteAudioWav(*audio_proc, media, prompt_ids);
    }
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
  const HfConfig hf = LoadHfConfig(ctx.config_path);
  {
    const std::string why = Dots3NoteVisionRefusalFor(hf);
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

  // ── THE AUDIO PROCESSOR (W7a, #2703) ────────────────────────────────────
  //
  // A checkpoint with NO `audio_config` gets a null processor and a
  // ceiling that does not declare "audio", which is the state every dots3-note
  // checkpoint was in before this brick. That is not a refusal: upstream builds
  // no audio tower either (`nvidia/multimodal.py:119-126` @ `9035151d6`), so
  // there is nothing owed and nothing to name.
  //
  // A checkpoint WITH one whose arms are not ported REFUSES at INSTALL, for the
  // reason the vision refusal above records: throwing from inside `encode_mm`
  // stops `AsyncLLM` and 500s every LATER request, text ones included.
  std::shared_ptr<const multimodal::Dots3NoteAudioProcessor> audio_proc;
  multimodal::Dots3NoteAudioProcessorConfig audio_cfg =
      multimodal::LoadDots3NoteAudioProcessorConfig(ctx.config_path,
                                                    ctx.served_model_name);
  if (audio_cfg.present) {
    const std::string why = Dots3NoteAudioRefusalFor(hf);
    if (!why.empty()) {
      throw std::runtime_error(
          "dots3-note multimodal chat seam: this checkpoint's audio tower is "
          "not ported — " + why +
          ". See .agents/specs/dots3-note.md §4.14 and issue #2703.");
    }
    // The three marker ids, resolved FROM THE TOKENIZER BY STRING
    // (`common/processor.py:757-760` @ `9035151d6` reads `vocab[AUDIO_START]`
    // and friends). Doing it here rather than from `config.json` is what makes
    // "the marker string this seam injects encodes to this id" true by
    // construction: the object that resolves the id is the object that will
    // encode the prompt. It THROWS BY NAME when one does not resolve — the
    // released checkpoint's three are 151718 / 151719 / 151720 in
    // start / END / PAD order, so a default would be a guess that a shape check
    // could never catch.
    multimodal::Dots3NoteResolveAudioTokenIds(
        &audio_cfg, [&](const std::string& marker) -> int32_t {
          for (const vllm::tok::SpecialToken& t : ctx.tokenizer->AddedTokens()) {
            if (t.text == marker) return t.id;
          }
          return -1;
        });
    audio_proc = std::make_shared<const multimodal::Dots3NoteAudioProcessor>(
        std::move(audio_cfg));
  }

  // The engine's limits (`--limit-mm-per-prompt`, `--language-model-only`)
  // folded by min() against this seam's own ceiling. The MultiModalConfig is
  // held BY REFERENCE (`context.h:105`); the engine owns it and outlives the
  // seam.
  auto info = std::make_shared<const multimodal::BaseProcessingInfo>(
      *ctx.mm_config,
      Dots3NoteChatSupportedMmLimits(/*has_audio=*/audio_proc != nullptr));

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail = "dots3-note processor from " + preprocessor_config_path +
                " (dense + pyramid MoE vision arm" +
                (audio_proc != nullptr ? ", `dots` audio tower)" : ")");
  seam.chat_fn = MakeDots3NoteChatFn(proc, audio_proc, *ctx.tokenizer,
                                     ctx.prompt_fn, ctx.codec, info);
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(dots3_note, "Dots3NoteForCausalLM", &MakeDots3NoteChatSeam)

}  // namespace vllm::entrypoints::openai
