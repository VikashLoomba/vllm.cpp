// dots3-note W6a (#2512) and W6b (#2613) — THE SERVED IMAGE REQUEST, end to
// end, over a DENSE tower and over a PYRAMID one.
//
// One OpenAI `image_url` chat request travels the whole production chain on a
// CPU queue over the real serving stack:
//
//   ApiServer::handle_chat_completions
//     -> OpenAIServingChat (the ARCHITECTURE-dispatched multimodal chat seam:
//        `<|img|><|imgpad|><|endofimg|>` marker injection, tokenize,
//        Dots3NoteImageProcessor, placeholder EXPANSION, mm_features)
//     -> AsyncLLM::generate(MultiModalInputs) -> EngineCore
//     -> Scheduler::schedule                  (encoder admission + budget)
//     -> Executor -> GPUModelRunner::execute_model
//        (the encoder step runs the VISION TOWER — dense blocks and, since
//         W6b, pyramid MoE blocks — the gather slices its
//         rows, `EmbedMmDots3NoteForCausalLM` scatters them, `.mm` is set)
//     -> ModelRegistry::Forward -> ForwardDots3NoteForCausalLM
//     -> Dots3NoteModel::ForwardDevice, which reads `mm->inputs_embeds`
//
// WHY THIS FILE EXISTS RATHER THAN A UNIT TEST (AGENTS.md, "Nothing lands
// dead"). `test_dots3_note_vision.cpp` proves the tower computes the right
// numbers. It would pass just as well on a tree where nothing calls the tower:
// it constructs the weights and calls the forward itself. This file cannot. It
// enters through the HTTP dispatch on the server's default configuration and
// asserts that the pixels reached the model.
//
// THE WEIGHTS ARE SYNTHETIC AND THE TOKENS ARE NOT CHECKED, and on this row
// they never can be: `.agents/specs/dots3-note.md` §6.4 records option B — the
// checkpoint is 298.67 GB fp8 / 576.89 GB bf16 against 119-122 GiB hosts, so
// vLLM cannot be run on this model here and NO token-exact denominator exists.
// What this file gates is REACHABILITY and the shape of what flows: which stage
// ran, on how many rows, and that a request whose tower is owed is refused BY
// NAME rather than answered from the text path.
//
// THE LOAD-BEARING CASE IS THE TWO-IMAGE LOGPROB ONE. Status 200,
// `prompt_tokens` and `completion_tokens` all pass on a tree where the tower is
// replaced by a correctly SHAPED constant. The logprobs of the first generated
// token do not.
#include "vllm/entrypoints/openai/api_server.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "dots3_note_tiny_fixture.h"
#include "vllm/config/multimodal.h"
#include "vllm/multimodal/dots3_note_processor.h"
#include "vllm/multimodal/pil_resize.h"
#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/dtype.h"

namespace oai = vllm::entrypoints::openai;
using nlohmann::json;
using dots3_tiny::TinyCheckpoint;
using dots3_tiny::TinySpec;
using oai::ApiServer;
using oai::ChatMessage;
using oai::OpenAIServingChat;
using oai::OpenAIServingCompletion;
using oai::OpenAIServingModels;
using vllm::HfConfig;
using vllm::SchedulerConfig;
using vllm::tok::Tokenizer;
using vllm::v1::AsyncLLM;
using vllm::v1::Executor;
using vllm::v1::get_request_block_hasher;
using vllm::v1::GPUModelRunner;
using vllm::v1::init_none_hash;
using vllm::v1::InputProcessor;
using vllm::v1::KVCacheConfig;
using vllm::v1::OutputProcessor;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;

namespace {

std::string FixtureDir() { return DOTS3_NOTE_CKPT_FIXTURE_DIR; }

constexpr int kBlockSize = 16, kNumBlocks = 64, kMaxModelLen = 128;
// The architecture the model registry resolves this fixture to, and the one
// dots3-note's chat factory is registered under.
constexpr const char* kDots3Arch = "Dots3NoteForCausalLM";
// An architecture NOTHING registers a chat seam for; the premise of the
// refusal case is asserted rather than assumed.
constexpr const char* kUnregisteredMmArch = "Dots3NoteNotRegisteredForCausalLM";

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// A BPE fixture whose ADDED tokens are dots3-note's three image markers and,
// since W7a (#2703), its three AUDIO markers — so the string the chat seam
// injects tokenizes to exactly [14, 15, 16] or [17, 19, 18].
//
// NOTE THE AUDIO ORDER: start 17, END 18, pad 19. That is the RELEASED
// checkpoint's own (`<|audio_comp_start|>` 151718, `<|audio_comp_end|>` 151719,
// `<|audio_comp_pad|>` 151720), and the fixture reproduces it so a port that
// guessed "start, start+1, start+2" for the pad id is wrong HERE too. The seam
// resolves all three from this tokenizer BY STRING, which is what makes the
// injected marker and the expanded id the same thing by construction.
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_dots3mm_tok_" + std::to_string(counter++) + ".json")).string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", dots3_tiny::kImgStartId}, {"content", "<|img|>"}, {"special", true}},
       {{"id", dots3_tiny::kImgPadId}, {"content", "<|imgpad|>"}, {"special", true}},
       {{"id", dots3_tiny::kImgEndId}, {"content", "<|endofimg|>"}, {"special", true}},
       {{"id", dots3_tiny::kAudStartId},
        {"content", "<|audio_comp_start|>"},
        {"special", true}},
       {{"id", dots3_tiny::kAudEndId},
        {"content", "<|audio_comp_end|>"},
        {"special", true}},
       {{"id", dots3_tiny::kAudPadId},
        {"content", "<|audio_comp_pad|>"},
        {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},    {"o", 3},    {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},    {"1", 8},    {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12}, {"hello", 13}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array({json::array({"l", "l"}),
                                          json::array({"h", "e"}),
                                          json::array({"ll", "o"}),
                                          json::array({"he", "llo"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

// The chat prompt seam: concatenate the rendered contents. The mm seam has
// already replaced each image part with the marker string, so this is where the
// markers enter the prompt.
std::string ConcatChatPrompt(
    const std::vector<ChatMessage>& messages, bool,
    const std::vector<oai::ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&) {
  std::string p;
  for (const ChatMessage& m : messages)
    if (m.content.has_value()) p += *m.content;
  return p;
}

// The raw-RGB passthrough codec, the same shape `server_main.cpp` installs.
//
// TWO media types, because W6c (#2537) needs a NON-SQUARE image and the square
// one cannot carry its own dimensions. `image/x-raw-rgb` keeps W6a's shape — a
// perfect square inferred from the byte count — and `image/x-raw-rgb-hw`
// prefixes the pixels with height and width as two big-endian `uint16`s. The
// codec is the only place that changes: what it hands the seam is the same
// `DecodedImageRgb` either way, so the served path past it is byte-identical
// between the two cases.
oai::ImageCodecFn RawRgbCodec() {
  return [](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    oai::DecodedImageRgb out;
    if (media.media_type == "image/x-raw-rgb-hw") {
      REQUIRE(media.bytes.size() >= 4);
      out.height = (static_cast<int64_t>(media.bytes[0]) << 8) | media.bytes[1];
      out.width = (static_cast<int64_t>(media.bytes[2]) << 8) | media.bytes[3];
      REQUIRE(static_cast<std::size_t>(out.height * out.width * 3) ==
              media.bytes.size() - 4);
      out.rgb.assign(media.bytes.begin() + 4, media.bytes.end());
      return out;
    }
    REQUIRE(media.media_type == "image/x-raw-rgb");
    const std::size_t px = media.bytes.size() / 3;
    const auto side = static_cast<int64_t>(
        std::llround(std::sqrt(static_cast<double>(px))));
    REQUIRE(static_cast<std::size_t>(side * side * 3) == media.bytes.size());
    out.rgb = media.bytes;
    out.height = side;
    out.width = side;
    return out;
  };
}

std::string EncodeBase64(const std::vector<uint8_t>& raw) {
  static const char* kAlpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  size_t i = 0;
  for (; i + 2 < raw.size(); i += 3) {
    const uint32_t v = (uint32_t(raw[i]) << 16) | (uint32_t(raw[i + 1]) << 8) |
                       uint32_t(raw[i + 2]);
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += kAlpha[(v >> 6) & 63];
    out += kAlpha[v & 63];
  }
  if (i + 1 == raw.size()) {
    const uint32_t v = uint32_t(raw[i]) << 16;
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == raw.size()) {
    const uint32_t v = (uint32_t(raw[i]) << 16) | (uint32_t(raw[i + 1]) << 8);
    out += kAlpha[(v >> 18) & 63];
    out += kAlpha[(v >> 12) & 63];
    out += kAlpha[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

std::string ImageDataUri(int variant) {
  return "data:image/x-raw-rgb;base64," +
         EncodeBase64(dots3_tiny::FixtureImage(variant));
}

// Any HWC uint8 RGB buffer as a dimension-carrying data URI.
std::string RawImageDataUri(int64_t h, int64_t w,
                            const std::vector<uint8_t>& px) {
  std::vector<uint8_t> raw{static_cast<uint8_t>((h >> 8) & 0xFF),
                           static_cast<uint8_t>(h & 0xFF),
                           static_cast<uint8_t>((w >> 8) & 0xFF),
                           static_cast<uint8_t>(w & 0xFF)};
  raw.insert(raw.end(), px.begin(), px.end());
  return "data:image/x-raw-rgb-hw;base64," + EncodeBase64(raw);
}

// The NON-CONFORMANT image, dimensions carried in the payload (W6c, #2537).
std::string OddImageDataUri(int64_t h, int64_t w, int variant) {
  return RawImageDataUri(h, w, dots3_tiny::FixtureImageHW(h, w, variant));
}

json ChatBodyWithImage(int max_tokens, const std::string& data_uri,
                       bool logprobs);

json ChatBody(int max_tokens, int variant, bool logprobs) {
  return ChatBodyWithImage(max_tokens, ImageDataUri(variant), logprobs);
}

json ChatBodyWithImage(int max_tokens, const std::string& data_uri,
                       bool logprobs) {
  json body = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url", {{"url", data_uri}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", max_tokens},
      {"temperature", 0.0}};
  if (logprobs) {
    body["logprobs"] = true;
    body["top_logprobs"] = 3;
  }
  return body;
}

// ── W7a (#2703): the AUDIO request ──────────────────────────────────────────

// The audio-capable fixture spec. `vocab` is raised to 20 because the three
// AUDIO marker ids are 17, 18 and 19, and the embedding table is `[vocab,
// hidden]` — an id past it is an out-of-bounds gather, not a wrong answer. The
// image-only specs keep 17 so nothing about the existing cases moves.
dots3_tiny::TinySpec AudioSpec() {
  dots3_tiny::TinySpec s;
  s.with_audio = true;
  s.vocab = 20;
  return s;
}

// `data:` is NOT used: an `input_audio` part carries a BARE base64 payload and
// a `format`, which is what `DecodeInputAudioPart` (`chat_mm.cpp:122-129`)
// expects. `audio_url` would carry a data URI instead, and the seam handles
// both; this is the shape the OpenAI API documents.
json ChatBodyWithAudio(int max_tokens, const std::vector<uint8_t>& wav,
                       bool logprobs, const char* format = "wav") {
  json body = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "input_audio"},
                                    {"input_audio",
                                     {{"data", EncodeBase64(wav)},
                                      {"format", format}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", max_tokens},
      {"temperature", 0.0}};
  if (logprobs) {
    body["logprobs"] = true;
    body["top_logprobs"] = 3;
  }
  return body;
}

json ChatBodyAudio(int max_tokens, int variant, bool logprobs) {
  return ChatBodyWithAudio(max_tokens, dots3_tiny::FixtureAudioWav(variant),
                           logprobs);
}

// The whole production serving stack over the tiny dots3-note, on a CPU queue.
struct MmServerHarness {
  MmServerHarness(const HfConfig& c, vllm::LoadedModel& model,
                  const Tokenizer& tok)
      : scheduler(MakeSchedulerConfig(), MakeKv(c), kBlockSize,
                  /*enable_caching=*/true),
        runner(c, model, MakeKv(c), Q(), /*max_num_reqs=*/1, kMaxModelLen,
               kMaxModelLen * 4),
        executor(runner),
        input_processor(tok, c),
        output_processor(&tok),
        async_engine(input_processor, scheduler, executor, output_processor,
                     Hasher()),
        models("test-model"),
        completion(async_engine, "test-model"),
        chat(async_engine, "test-model", &ConcatChatPrompt, "hermes"),
        server(completion, chat, models, "9.9.9") {}

  // The SAME topology `MakeDots3NoteKVCache` publishes for this architecture:
  // ONE MLA group at the padded physical latent row.
  static KVCacheConfig MakeKv(const HfConfig& c) {
    return vllm::MakeDots3NoteKVCache(c, kBlockSize, kNumBlocks);
  }
  static SchedulerConfig MakeSchedulerConfig() {
    SchedulerConfig cfg;
    cfg.max_num_seqs = 1;
    cfg.max_num_batched_tokens = kMaxModelLen * 4;
    // Chunked prefill is ON because it is the PRODUCTION value, not because
    // this file measures it: the budget is 512 against a seven-token prompt, so
    // nothing ever chunks here and the straddling-item clamp is never reached.
    cfg.enable_chunked_prefill = true;
    cfg.max_model_len = kMaxModelLen;
    cfg.watermark = 0.0;
    cfg.max_num_encoder_input_tokens = kMaxModelLen * 4;
    cfg.encoder_cache_size = kMaxModelLen * 4;
    return cfg;
  }
  static vllm::v1::BlockHasher Hasher() {
    static bool init = false;
    if (!init) {
      init_none_hash(sha256_cbor);
      init = true;
    }
    return get_request_block_hasher(kBlockSize, sha256_cbor);
  }

  // THE PRODUCTION INSTALL, not a copy of it. `server_main.cpp` builds the same
  // context and calls the same function; what it adds on top is reading the
  // architecture and the multimodal declaration off `LoadedEngine`.
  oai::MultiModalChatInstall install(std::string_view architecture,
                                     bool is_multimodal_model,
                                     const TinyCheckpoint& ckpt,
                                     std::ostream& log) {
    oai::MultiModalChatContext ctx;
    ctx.architecture = architecture;
    ctx.model_dir = ckpt.dir();
    ctx.config_path = ckpt.config_path();
    ctx.served_model_name = "tiny-dots3-note";
    ctx.tokenizer = &Fixture();
    ctx.prompt_fn = &ConcatChatPrompt;
    ctx.codec = RawRgbCodec();
    ctx.mm_config = &mm_cfg;
    return oai::InstallMultiModalChatSeam(chat, is_multimodal_model, ctx, log);
  }

  // Declared FIRST so it outlives `chat`: the seam's `BaseProcessingInfo` holds
  // it by reference, exactly as the engine's own config is held in production.
  vllm::MultiModalConfig mm_cfg;
  Scheduler scheduler;
  GPUModelRunner runner;
  Executor executor;
  InputProcessor input_processor;
  OutputProcessor output_processor;
  AsyncLLM async_engine;
  OpenAIServingModels models;
  OpenAIServingCompletion completion;
  OpenAIServingChat chat;
  ApiServer server;
};

// Everything a served case needs: the tiny checkpoint on disk, the config, and
// the model the REAL registry loader returned from it.
struct Served {
  TinySpec spec;
  TinyCheckpoint ckpt;
  HfConfig config;
  std::unique_ptr<vllm::LoadedModel> model;

  explicit Served(TinySpec s = TinySpec{})
      : spec(s), ckpt(FixtureDir(), s), config(vllm::LoadHfConfig(ckpt.config_path())) {
    const std::vector<std::string> arch{kDots3Arch};
    const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(ckpt.weights_path()));
    const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
    model = reg.factory->load_weights(reg, config, source);
    vt::Queue q = Q();
    vllm::ModelRegistry::Prepare(*model, config, q);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE TWO REGISTRATIONS. The model registration says this model can EMBED
//    image features; the chat registration says the server can BUILD them from
//    an `image_url` part. Both are keyed on the architecture, and both are
//    reached through the static library's `--whole-archive` INTERFACE — so a
//    link that dropped either translation unit reads as an ABSENT registration
//    rather than as a subtly wrong one.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: both halves of the multimodal seam are registered on the architecture") {
  const std::vector<std::string> arch{kDots3Arch};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(arch);
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.info.supports_multimodal);
  CHECK(reg.factory->encode_mm != nullptr);
  CHECK(reg.factory->embed_mm != nullptr);
  // NOT M-RoPE: upstream's Dots3NoteForCausalLM is `SupportsMultiModal,
  // SupportsPP` and not `SupportsMRoPE` (nvidia/multimodal.py:49 @ 9035151d6).
  CHECK(reg.factory->mrope_prompt_positions == nullptr);

  const oai::MultiModalChatRegistration* seam =
      oai::MultiModalChatRegistry::Find(kDots3Arch);
  REQUIRE(seam != nullptr);
  CHECK(seam->architecture == kDots3Arch);
  CHECK(seam->make_seam != nullptr);
  const std::vector<std::string_view> archs =
      oai::MultiModalChatRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), std::string_view(kDots3Arch)) !=
        archs.end());
  // TWO architectures now hold seams, which is the whole point of #2481: the
  // registry is keyed, not a single entry with a filename check in front of it.
  CHECK(std::find(archs.begin(), archs.end(),
                  std::string_view("Qwen3VLForConditionalGeneration")) !=
        archs.end());
  // The premise of the refusal case below, asserted rather than assumed.
  CHECK(oai::MultiModalChatRegistry::Find(kUnregisteredMmArch) == nullptr);

  // And the predicate the runner reads is DERIVED from the two hooks, never
  // stored.
  const Served s;
  CHECK(vllm::ModelRegistry::SupportsMmInputs(*s.model));
  CHECK_FALSE(vllm::ModelRegistry::UsesMrope(*s.model));
}

// ---------------------------------------------------------------------------
// 2. THE SERVED REQUEST. One image chat request, through HTTP dispatch.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: a served image chat request reaches the model forward") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, /*is_multimodal_model=*/true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  INFO("install log: ", log.str());
  // The install announced WHICH processor it built and from where.
  CHECK(log.str().find("dots3-note processor") != std::string::npos);
  // ...and no longer says the pyramid half is owed, because W6b landed it.
  CHECK(log.str().find("is W6b") == std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(/*max_tokens=*/3, 0, false).dump());
  // A 500 here is the interesting failure and it is what this brick is about:
  // before W6a the factory had no `encode_mm`, so `SupportsMmInputs` was false
  // and the runner's whole multimodal arm was never entered.
  INFO("body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 3);
  // The prompt the engine actually ran is the EXPANDED one: `<|img|>` + FOUR
  // image tokens + `<|endofimg|>` + "hello". A seam that dropped the expansion
  // would report 3 prompt tokens and still answer 200.
  CHECK(j.at("usage").at("prompt_tokens") == 3 + dots3_tiny::kExpectedImageTokens);
}

// ---------------------------------------------------------------------------
// 2b. THE NON-CONFORMANT IMAGE IS SERVED, NOT REFUSED (W6c, #2537).
//
//     THIS IS THE RED-BEFORE FOR THE WHOLE BRICK. `factor` is
//     `patch_size * merge_size`, 4 on this fixture and 28 on the released
//     checkpoint, and before W6c `Dots3NoteImageProcessor::ProcessImage` threw
//     for any image whose sides were not already multiples of it. The throw
//     surfaced here as HTTP 400 with both sizes in the message — a refusal by
//     name, never a silent skip — so this case read 400 on the tree this brick
//     started from and reads 200 on the tree it leaves.
//
//     The image is 6x14. It is NON-SQUARE on purpose: 8x16 out of 6x14 keeps
//     the two axes distinguishable on both sides of the resample, so a
//     transposed loop or a swapped bound cannot pass here. And the token count
//     is the grid the RESIZED size implies (32 patches / 2² = 8) rather than
//     the four the square fixture produces, so a resize that silently kept the
//     original geometry would answer 200 with the wrong prompt length.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6c: a NON-CONFORMANT image is resized and served, not refused") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, /*is_multimodal_model=*/true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  // The premise, asserted rather than assumed: neither side of 6x14 is a
  // multiple of this fixture's `factor`, and the resolved target is 8x16.
  const int64_t factor = dots3_tiny::TinySpec{}.v_patch * dots3_tiny::TinySpec{}.v_merge;
  REQUIRE(dots3_tiny::kOddImageH % factor != 0);
  REQUIRE(dots3_tiny::kOddImageW % factor != 0);
  REQUIRE(dots3_tiny::kOddImageH != dots3_tiny::kOddImageW);
  const std::array<int64_t, 2> rs = vllm::multimodal::Dots3NoteResizedSize(
      dots3_tiny::kOddImageH, dots3_tiny::kOddImageW, factor,
      dots3_tiny::TinySpec{}.p_min_pixels, dots3_tiny::TinySpec{}.p_max_pixels);
  REQUIRE(rs[0] == dots3_tiny::kOddResizedH);
  REQUIRE(rs[1] == dots3_tiny::kOddResizedW);

  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyWithImage(/*max_tokens=*/3,
                        OddImageDataUri(dots3_tiny::kOddImageH,
                                        dots3_tiny::kOddImageW, /*variant=*/0),
                        /*logprobs=*/false)
          .dump());
  INFO("body: ", r.body);
  // A 500 here IS the pre-W6c behaviour: the processor's throw reaches the
  // dispatcher, which reports it with the message that names the missing path.
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 3);
  // `<|img|>` + EIGHT image tokens + `<|endofimg|>` + "hello". Eight, not four:
  // the placeholder run follows the RESIZED grid.
  CHECK(j.at("usage").at("prompt_tokens") ==
        3 + dots3_tiny::kOddExpectedImageTokens);
}

// ---------------------------------------------------------------------------
// 2c. THE RESIZED PIXELS REACH THE MODEL, and two non-conformant images that
//     resize to the SAME grid still give different forwards.
//
//     Case 2b would pass on a tree whose resampler returned a constant of the
//     right shape: the status, the grid and the token count are all properties
//     of the GEOMETRY, which `Dots3NoteResizedSize` already owned before W6c.
//     This case compares the served logprobs of two different 6x14 images. It
//     is the served counterpart of the resampler's numeric gate, and it is what
//     the "delete the resize call" mutation has to break.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6c: two different NON-CONFORMANT images give two different forwards") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult a = h.server.handle_chat_completions(
      ChatBodyWithImage(4,
                        OddImageDataUri(dots3_tiny::kOddImageH,
                                        dots3_tiny::kOddImageW, 0),
                        true)
          .dump());
  const ApiServer::DispatchResult b = h.server.handle_chat_completions(
      ChatBodyWithImage(4,
                        OddImageDataUri(dots3_tiny::kOddImageH,
                                        dots3_tiny::kOddImageW, 1),
                        true)
          .dump());
  INFO("a: ", a.body);
  INFO("b: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);

  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  const json& la = ja.at("choices").at(0).at("logprobs").at("content").at(0);
  const json& lb = jb.at("choices").at(0).at("logprobs").at("content").at(0);
  MESSAGE("odd image A logprob0 ", la.dump());
  MESSAGE("odd image B logprob0 ", lb.dump());
  CHECK(la.at("logprob").get<double>() != lb.at("logprob").get<double>());
  // Both legs still ran the RESIZED grid.
  CHECK(ja.at("usage").at("prompt_tokens") ==
        3 + dots3_tiny::kOddExpectedImageTokens);
  CHECK(jb.at("usage").at("prompt_tokens") ==
        3 + dots3_tiny::kOddExpectedImageTokens);
}

// ---------------------------------------------------------------------------
// 2d. THE SERVED PATH REALLY RESAMPLES, and this is the case that says so.
//
//     Cases 2b and 2c both survive a tree with the resize call DELETED. 2b
//     asserts geometry, which `Dots3NoteResizedSize` owned before W6c; 2c
//     asserts that two different images differ, which they do whether or not
//     either was resampled. Measured: with the call site disabled,
//     `test_openai_api_server_dots3_mm_forward` still read 14/14 and 199/199.
//     A gate that stays green without the call site measures a class, not a
//     capability (`.agents/reachability.md`).
//
//     This case closes that. It serves the 6x14 image and then serves the 8x16
//     image `PilResizeBicubicRgb` produces from it, and requires the two
//     logprob vectors to be IDENTICAL. The second request takes the processor's
//     identity path — 8x16 is already conformant — so the two agree only if the
//     served 6x14 request resampled to exactly those bytes. The numeric
//     correctness of those bytes is `test_dots3_note_vision`'s to prove against
//     the independent reference; what this asserts is that the production path
//     produced them.
//
//     SAY THE LIMIT OUT LOUD, because it is the same shape as the defect this
//     case repairs. Both legs run the SAME resampler, so a defect INSIDE it
//     cancels: with a half-pixel centre, with the support scaling dropped and
//     with the weight normalization skipped, this suite reads 16/16 and 240/240
//     while `test_dots3_note_vision` reads 182, 121 and 214 failed assertions.
//     That division is deliberate -- the served suite answers "was it called",
//     the reference gate answers "was it right" -- and it is written here so a
//     reader does not mistake a green served suite for a numeric verdict.
//
//     AND THE GEOMETRY HERE IS AN UPSCALE, which case 2e is about: 6x14 ->
//     8x16 puts `filterscale = max(1, in/out)` at 1, so this case is
//     BYTE-IDENTICAL with the support scaling deleted.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6c: a served NON-CONFORMANT image equals its pre-resized twin") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const std::vector<uint8_t> odd = dots3_tiny::FixtureImageHW(
      dots3_tiny::kOddImageH, dots3_tiny::kOddImageW, /*variant=*/0);
  const std::vector<uint8_t> pre = vllm::multimodal::PilResizeBicubicRgb(
      odd.data(), dots3_tiny::kOddImageH, dots3_tiny::kOddImageW,
      dots3_tiny::kOddResizedH, dots3_tiny::kOddResizedW);
  REQUIRE(pre.size() == static_cast<std::size_t>(dots3_tiny::kOddResizedH *
                                                 dots3_tiny::kOddResizedW * 3));
  // The premise: the pre-resized twin is CONFORMANT, so its own request takes
  // the identity path and cannot be resampled a second time.
  REQUIRE(dots3_tiny::kOddResizedH %
              (dots3_tiny::TinySpec{}.v_patch * dots3_tiny::TinySpec{}.v_merge) == 0);
  REQUIRE(dots3_tiny::kOddResizedW %
              (dots3_tiny::TinySpec{}.v_patch * dots3_tiny::TinySpec{}.v_merge) == 0);

  const ApiServer::DispatchResult a = h.server.handle_chat_completions(
      ChatBodyWithImage(4,
                        OddImageDataUri(dots3_tiny::kOddImageH,
                                        dots3_tiny::kOddImageW, 0),
                        true)
          .dump());
  const ApiServer::DispatchResult b = h.server.handle_chat_completions(
      ChatBodyWithImage(4, RawImageDataUri(dots3_tiny::kOddResizedH,
                                           dots3_tiny::kOddResizedW, pre),
                        true)
          .dump());
  INFO("6x14: ", a.body);
  INFO("8x16: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);
  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  REQUIRE(ja.at("usage").at("prompt_tokens") ==
          jb.at("usage").at("prompt_tokens"));
  const json& ca = ja.at("choices").at(0).at("logprobs").at("content");
  const json& cb = jb.at("choices").at(0).at("logprobs").at("content");
  REQUIRE(ca.size() == cb.size());
  REQUIRE(ca.size() > 0);
  for (std::size_t i = 0; i < ca.size(); ++i) {
    CAPTURE(i);
    // Bit-for-bit: the two requests run the same pixel bytes through the same
    // tower on the same queue, so anything but equality means the served 6x14
    // leg patchified something other than the resample.
    CHECK(ca.at(i).at("logprob").get<double>() ==
          cb.at(i).at("logprob").get<double>());
  }
  MESSAGE("6x14 logprob0 ", ca.at(0).dump());
  MESSAGE("8x16 logprob0 ", cb.at(0).dump());
}

// ---------------------------------------------------------------------------
// 2e. THE SERVED REQUEST THAT ACTUALLY DOWNSCALES (W6c, #2537).
//
//     Cases 2b, 2c and 2d all resize 6x14 to 8x16, which is an UPSCALE on both
//     axes. `filterscale = max(1, in/out)` is 1 there, the support stays 2.0,
//     and PIL's resampler is bit-for-bit the textbook four-tap cubic: 6x14 ->
//     8x16 is BYTE-IDENTICAL with the support scaling deleted. So the served
//     suite exercised none of what `pil_resize.cpp` exists for, while
//     `factor = 28` on the released checkpoint means essentially every real
//     request downscales.
//
//     `kBudgetMaxPixels` is what forces the other regime, and it forces it the
//     way production does: `max_pixels` comes off `preprocessor_config.json`,
//     so the served chain resolves it itself. 24x96 under a 64-pixel budget is
//     4x16 -- a 6x downscale on both axes, a 25-tap support-scaled window per
//     output pixel, and FOUR placeholder tokens rather than the eight the 6x14
//     cases produce.
//
//     THE SAME LIMIT AS 2d, and for the same reason: both legs run the same
//     resampler, so a defect INSIDE it cancels here. What this case adds is the
//     REGIME -- the served path now reaches the support-scaled window at all --
//     and the reachability arm inside it. The numeric verdict on the downscale
//     is `test_dots3_note_vision`'s "ProcessImage DOWNSCALES through the
//     support-scaled window", which compares it to the independent reference.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6c: a served image the PIXEL BUDGET downscales 6x is resized and served") {
  TinySpec spec;
  spec.p_max_pixels = dots3_tiny::kBudgetMaxPixels;
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  // The premise, asserted rather than assumed. This is a DOWNSCALE by 6 on both
  // axes, which is the only thing that puts `filterscale` above 1, and the two
  // sides stay unequal on both ends so an axis swap cannot survive it.
  const int64_t factor = TinySpec{}.v_patch * TinySpec{}.v_merge;
  const std::array<int64_t, 2> rs = vllm::multimodal::Dots3NoteResizedSize(
      dots3_tiny::kBigImageH, dots3_tiny::kBigImageW, factor, spec.p_min_pixels,
      spec.p_max_pixels);
  REQUIRE(rs[0] == dots3_tiny::kBigResizedH);
  REQUIRE(rs[1] == dots3_tiny::kBigResizedW);
  REQUIRE(dots3_tiny::kBigImageH / rs[0] == 6);
  REQUIRE(dots3_tiny::kBigImageW / rs[1] == 6);
  REQUIRE(rs[0] != rs[1]);

  const std::vector<uint8_t> big = dots3_tiny::FixtureImageHW(
      dots3_tiny::kBigImageH, dots3_tiny::kBigImageW, /*variant=*/0);
  const std::vector<uint8_t> pre = vllm::multimodal::PilResizeBicubicRgb(
      big.data(), dots3_tiny::kBigImageH, dots3_tiny::kBigImageW,
      dots3_tiny::kBigResizedH, dots3_tiny::kBigResizedW);
  REQUIRE(pre.size() == static_cast<std::size_t>(dots3_tiny::kBigResizedH *
                                                 dots3_tiny::kBigResizedW * 3));
  // The pre-resized twin is CONFORMANT and inside the same budget, so its own
  // request takes the processor's identity path and is not resampled again.
  REQUIRE(dots3_tiny::kBigResizedH % factor == 0);
  REQUIRE(dots3_tiny::kBigResizedW % factor == 0);
  REQUIRE(dots3_tiny::kBigResizedH * dots3_tiny::kBigResizedW <=
          spec.p_max_pixels);

  const ApiServer::DispatchResult a = h.server.handle_chat_completions(
      ChatBodyWithImage(4,
                        RawImageDataUri(dots3_tiny::kBigImageH,
                                        dots3_tiny::kBigImageW, big),
                        true)
          .dump());
  const ApiServer::DispatchResult b = h.server.handle_chat_completions(
      ChatBodyWithImage(4,
                        RawImageDataUri(dots3_tiny::kBigResizedH,
                                        dots3_tiny::kBigResizedW, pre),
                        true)
          .dump());
  INFO("24x96: ", a.body);
  INFO("4x16: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);

  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  // `<|img|>` + FOUR image tokens + `<|endofimg|>` + "hello": the placeholder
  // run follows the DOWNSCALED grid (1, 2, 8), not the 24x96 one, which would
  // be a (1, 12, 48) grid, 144 tokens, and would not fit `kMaxModelLen` at
  // all.
  CHECK(ja.at("usage").at("prompt_tokens") ==
        3 + dots3_tiny::kBigExpectedImageTokens);
  REQUIRE(ja.at("usage").at("prompt_tokens") ==
          jb.at("usage").at("prompt_tokens"));

  const json& ca = ja.at("choices").at(0).at("logprobs").at("content");
  const json& cb = jb.at("choices").at(0).at("logprobs").at("content");
  REQUIRE(ca.size() == cb.size());
  REQUIRE(ca.size() > 0);
  for (std::size_t i = 0; i < ca.size(); ++i) {
    CAPTURE(i);
    // Bit-for-bit, exactly as in 2d: anything but equality means the served
    // 24x96 leg patchified something other than the resample.
    CHECK(ca.at(i).at("logprob").get<double>() ==
          cb.at(i).at("logprob").get<double>());
  }
  MESSAGE("24x96 logprob0 ", ca.at(0).dump());
  MESSAGE("4x16 logprob0 ", cb.at(0).dump());
}

// ---------------------------------------------------------------------------
// 3. TWO DIFFERENT IMAGES GIVE TWO DIFFERENT FORWARDS.
//
//    Case 2 proves the request arrives. It does NOT prove the pixels do: an
//    encoder hook returning a correctly SHAPED constant satisfies every
//    assertion there, and so does a scatter that wrote the token embedding it
//    was about to overwrite. This case is the one that separates them, and it
//    is why deleting the `Dots3NoteVisionForward` call is a mutation with
//    something to detect rather than only a shape check to trip.
//
//    Compared on LOGPROBS rather than on the sampled text: on a random tiny
//    checkpoint the argmax over 17 vocabulary entries is saturated and does not
//    have to move, while the float logprobs move for ANY change in the hidden
//    state. Different `mm_hash` on each leg, so the encoder cache cannot answer
//    the second from the first.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: two DIFFERENT images give two different forwards") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult a =
      h.server.handle_chat_completions(ChatBody(4, /*variant=*/0, true).dump());
  const ApiServer::DispatchResult b =
      h.server.handle_chat_completions(ChatBody(4, /*variant=*/1, true).dump());
  INFO("a: ", a.body);
  INFO("b: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);

  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  const json& la = ja.at("choices").at(0).at("logprobs").at("content").at(0);
  const json& lb = jb.at("choices").at(0).at("logprobs").at("content").at(0);
  MESSAGE("image A logprob0 ", la.dump());
  MESSAGE("image B logprob0 ", lb.dump());
  // Same prompt, same weights, greedy sampling: the ONLY difference between the
  // two forwards is the pixels. If the logprobs match, the pixels did not reach
  // the forward.
  CHECK(la != lb);
}

// ---------------------------------------------------------------------------
// 4. THE SAME IMAGE TWICE hits the encoder cache by mm_hash and still answers
//    with the same shape — the tower runs once, both legs are served.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the same image twice is served twice, and the second hits the encoder cache") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  for (int i = 0; i < 2; ++i) {
    const ApiServer::DispatchResult r =
        h.server.handle_chat_completions(ChatBody(2, 0, false).dump());
    INFO("attempt ", i, " body: ", r.body);
    REQUIRE(r.status == 200);
    const json j = json::parse(r.body);
    CHECK(j.at("usage").at("completion_tokens") == 2);
    CHECK(j.at("usage").at("prompt_tokens") ==
          3 + dots3_tiny::kExpectedImageTokens);
  }
}

// ---------------------------------------------------------------------------
// 5. A CHECKPOINT WITH A PYRAMID VISION BLOCK IS **SERVED** (W6b, #2613).
//
//    THIS IS THE RELEASED CHECKPOINT'S CASE, at tiny scale, and it is the
//    inverse of the one W6a landed here. 17 of the released tower's 42 blocks
//    are pyramid MoE, so before this brick the install reported `kRefusing` and
//    every image came back 400. Now the same checkpoint installs and answers.
//
//    THE LOAD-BEARING ASSERTION IS THE TWO-IMAGE LOGPROB ONE, and what it is
//    load-bearing FOR was measured rather than assumed. It separates a tower
//    that ran from a tower replaced by a correctly shaped constant: status 200,
//    `prompt_tokens` and `completion_tokens` all pass on the second, and the
//    logprobs of the first generated token do not, because they move for any
//    change in the hidden state and the two images differ in every patch.
//
//    It does NOT separate a tower that ROUTED correctly from one that did not,
//    because a dense block still processes the pixels on the way. Three of
//    spec §4.12.9's mutations left this case green. Case 5c below is the one
//    that covers the router, and it exists because those three measurements
//    said this one could not.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: a checkpoint with a PYRAMID vision block is SERVED") {
  TinySpec spec;
  spec.v_pyramid = {-1, 4};  // block 1 is routed
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  INFO("install log: ", log.str());
  // The install no longer announces an owed pyramid, because nothing is owed.
  CHECK(log.str().find("W6b") == std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(3, 0, false).dump());
  INFO("body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 3);
  CHECK(j.at("usage").at("prompt_tokens") == 3 + dots3_tiny::kExpectedImageTokens);
}

TEST_CASE("dots3-note W6b: two DIFFERENT images through a PYRAMID tower give two different forwards") {
  TinySpec spec;
  spec.v_pyramid = {-1, 4};
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult a =
      h.server.handle_chat_completions(ChatBody(4, /*variant=*/0, true).dump());
  const ApiServer::DispatchResult b =
      h.server.handle_chat_completions(ChatBody(4, /*variant=*/1, true).dump());
  INFO("a: ", a.body);
  INFO("b: ", b.body);
  REQUIRE(a.status == 200);
  REQUIRE(b.status == 200);

  const json ja = json::parse(a.body);
  const json jb = json::parse(b.body);
  const json& la = ja.at("choices").at(0).at("logprobs").at("content").at(0);
  const json& lb = jb.at("choices").at(0).at("logprobs").at("content").at(0);
  MESSAGE("pyramid image A logprob0 ", la.dump());
  MESSAGE("pyramid image B logprob0 ", lb.dump());
  // Same prompt, same weights, greedy sampling: the ONLY difference between the
  // two forwards is the pixels, and they travel through a ROUTED block on the
  // way. If the logprobs match, the pixels did not reach the forward.
  CHECK(la != lb);
}

// ---------------------------------------------------------------------------
// 5c. THE ROUTING DECISION REACHES THE SERVED ANSWER (W6b, #2613).
//
//     MEASURED, NOT ASSUMED, AND THE MEASUREMENT IS WHY THIS CASE EXISTS. The
//     two-different-images case above is the only assertion in this file that
//     survives a tower replaced by a correctly shaped constant — but it does
//     NOT survive as a router gate. Three mutations recorded in spec §4.12.9
//     left it GREEN: routing every token to expert 0, dropping the router bias,
//     and replacing the routed FFN output with a constant. All three are
//     defects that change WHICH expert runs, and none of them changed what this
//     server answered, because two different images still give two different
//     logprobs however wrong the routed block is. A fixture whose block 0 is
//     ROUTED rather than dense does not rescue it either — that was built and
//     run, and spec §4.12.9 has the result.
//
//     So the router needs its own served assertion, and this is its shape: two
//     checkpoints that differ in `mlp.router_bias` AND IN NOTHING ELSE — every
//     other tensor is drawn from the same seed stream — must produce different
//     logprobs.
//
//     WHAT IT DETECTS, EXACTLY, AND WHAT NO CASE IN THIS FILE DETECTS. It reds
//     when the router ignores its bias, ignores its input, or feeds a constant
//     FFN — three defects that all change WHICH expert runs, and therefore
//     change the answer once the two checkpoints stop disagreeing about the
//     selection. It is NOT a gate on routed ARITHMETIC. Spec §4.12.9's M6
//     measures that limit: swapping which selected slot each expert's output is
//     written into multiplies every expert output by the OTHER selected
//     expert's routing weight — a genuine routed-path arithmetic defect that
//     leaves the selection SET intact — and it left this ENTIRE suite green,
//     12/12 with 177 assertions, while reding the tower gate on tolerance alone
//     at 2.3x to 15.6x its bound and firing ZERO set assertions.
//
//     So this file gates REACHABILITY and BIAS-DEPENDENCE of the routed block,
//     and nothing about the arithmetic inside it. The tower gate
//     (`test_dots3_note_vision`) is where routed arithmetic is caught, and spec
//     §4.12.6 and §4.12.9 record why no cheap served case can be made to catch
//     it: every served assertion available here is "two answers differ", and a
//     deterministic arithmetic defect leaves both answers well-defined and
//     distinct.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: the router BIAS changes what the server answers") {
  TinySpec biased;
  biased.v_pyramid = {-1, 4};
  biased.v_router_bias_amp = 0.4;
  TinySpec unbiased = biased;
  // Amplitude zero makes every `router_bias` entry exactly 0.0 while still
  // consuming the same value from the seed stream, so the two checkpoints
  // differ in ONE tensor family and agree byte-for-byte everywhere else.
  unbiased.v_router_bias_amp = 0.0;

  // THE PREMISE, asserted rather than assumed: the checkpoints really do differ
  // in exactly the router bias. Without this the case could be comparing two
  // identical models and reporting a router that works.
  const std::vector<dots3_tiny::StOut> a_ent = dots3_tiny::TinyEntries(biased);
  const std::vector<dots3_tiny::StOut> b_ent = dots3_tiny::TinyEntries(unbiased);
  REQUIRE(a_ent.size() == b_ent.size());
  int differing = 0;
  for (size_t i = 0; i < a_ent.size(); ++i) {
    REQUIRE(a_ent[i].name == b_ent[i].name);
    if (a_ent[i].values != b_ent[i].values) {
      ++differing;
      CHECK(a_ent[i].name == "vision_encoder.blocks.1.mlp.router_bias");
    }
  }
  CHECK(differing == 1);

  auto answer = [](const TinySpec& spec) {
    Served s(spec);
    MmServerHarness h(s.config, *s.model, Fixture());
    std::ostringstream log;
    REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
            oai::MultiModalChatInstall::kInstalled);
    const ApiServer::DispatchResult r =
        h.server.handle_chat_completions(ChatBody(4, 0, true).dump());
    INFO("body: ", r.body);
    REQUIRE(r.status == 200);
    return json::parse(r.body)
        .at("choices").at(0).at("logprobs").at("content").at(0);
  };
  const json la = answer(biased);
  const json lb = answer(unbiased);
  MESSAGE("biased router logprob0   ", la.dump());
  MESSAGE("unbiased router logprob0 ", lb.dump());
  CHECK(la != lb);
}

// ---------------------------------------------------------------------------
// 5b. A CHECKPOINT WHOSE VISION TOWER IS STILL OWED REFUSES BY NAME, and the
//     message names the key and the issue.
//
//     The shape W6a measured here has not gone away — it moved to the arm that
//     is still owed. `use_bias` is a config the shared MLP seam cannot express
//     (issue #2616), so the chat FACTORY refuses at install rather than letting
//     the throw land in the engine's busy loop, where it would stop `AsyncLLM`
//     and turn every LATER request — text ones included — into a 500.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6b: a checkpoint whose vision arm is OWED refuses the image BY NAME") {
  TinySpec spec;
  spec.v_use_bias = true;
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kRefusing);
  INFO("install log: ", log.str());
  CHECK(log.str().find("use_bias") != std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(2, 0, false).dump());
  INFO("body: ", r.body);
  // A 200 here is the defect: it would mean the tower was skipped and the
  // placeholder rows kept whatever the embedding table gave them, which is a
  // fluent wrong answer. A 500 is the OTHER defect, and it is the one this
  // ordering removes: an engine that died takes the text path with it.
  CHECK(r.status == 400);
  CHECK(r.body.find("use_bias") != std::string::npos);
  CHECK(r.body.find("#2616") != std::string::npos);
  // ...and the TEXT path over the SAME server, AFTER the refused image request,
  // still answers. This is the assertion the engine-fatal shape could not pass.
  const json text = {{"model", "test-model"},
                     {"messages", json::array({{{"role", "user"},
                                                {"content", "hello"}}})},
                     {"max_completion_tokens", 2},
                     {"temperature", 0.0}};
  const ApiServer::DispatchResult t =
      h.server.handle_chat_completions(text.dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
  CHECK(t.body.find("use_bias") == std::string::npos);
}

// ---------------------------------------------------------------------------
// 6. THE WRONG ARCHITECTURE GETS NOTHING (#2475), through the same dispatch.
//    This model directory carries a `preprocessor_config.json`, so an install
//    that keyed on the FILE — which is what the server did before #2481 — would
//    build a processor here and answer 200. Only a lookup that reads the
//    architecture refuses.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: an architecture with no registered seam REFUSES the image by name") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kUnregisteredMmArch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kRefusing);
  INFO("install log: ", log.str());
  CHECK(log.str().find(kUnregisteredMmArch) != std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(2, 0, false).dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  const json j = json::parse(r.body);
  const std::string message = j.at("error").at("message").get<std::string>();
  CHECK(message.find(kUnregisteredMmArch) != std::string::npos);
  CHECK(message.find("multimodal input is not available") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 7. THE SEAM'S CEILING IS ONE IMAGE, and a second one is REFUSED with
//    upstream's own message rather than silently dropped (#686). This seam
//    locates a single image part, so `{"image": 1}` is its implemented arm
//    stated as a number — and VIDEO and AUDIO are ABSENT from the map, which
//    `context.py:414-415` reads as unsupported.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: the chat seam declares ONE image, and refuses a second") {
  Served s;
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const json two = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url", {{"url", ImageDataUri(0)}}}},
                                   {{"type", "image_url"},
                                    {"image_url", {{"url", ImageDataUri(1)}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", 2},
      {"temperature", 0.0}};
  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(two.dump());
  INFO("body: ", r.body);
  // 200 here would be the #686 defect: the second image dropped without a word.
  CHECK(r.status == 400);
  CHECK(r.body.find("image") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 8 and 9. THE REFUSAL PREDICATE AND THE ROUTE PREDICATE ARE THE SAME
//    PREDICATE, or the engine-fatal cascade case 5 removed is still reachable
//    (fresh review of #2523).
//
//    Case 5 proves the shape works for ONE condition — a pyramid block. These
//    two prove it holds for the conditions the ENCODER asserts on. Before this
//    repair `Dots3NoteVisionRefusal` was a strict SUBSET of
//    `EncodeMmDots3NoteForCausalLM`'s `VT_CHECK`s: an all-dense checkpoint whose
//    `adapter_out_dim` is not the text width, or whose `adapter_merge_size` is
//    not `spatial_merge_size`, INSTALLED cleanly, served text, and then threw
//    inside the engine's busy loop on the first image — after which
//    `AsyncLLM::errored_` is set for the life of the process
//    (`async_llm.cpp:584-601`) and every later request, text included, is dead.
//
//    Each case therefore asserts BOTH halves, exactly as case 5 does: HTTP 400
//    on the image, and HTTP 200 on a TEXT request sent afterwards on the SAME
//    server. The second assertion is the one the pre-repair tree cannot pass.
// ---------------------------------------------------------------------------
TEST_CASE("dots3-note W6a: an adapter that does not land in the TEXT hidden space refuses at INSTALL") {
  TinySpec spec;
  // Every block is dense and every other key is conformant, so this checkpoint
  // clears every refusal case 5 exercises. What it gets wrong is the ONE thing
  // the encoder compares `adapter_out_dim` against: the text tower's width.
  spec.v_adapter_out_override = spec.hidden + 8;
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kRefusing);
  INFO("install log: ", log.str());
  CHECK(log.str().find("adapter_out_dim") != std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(2, 0, false).dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("adapter_out_dim") != std::string::npos);

  // THE ASSERTION THE PRE-REPAIR TREE FAILS. A 500 here — "request submitted to
  // a stopped AsyncLLM" — is the cascade: the image request threw inside the
  // busy loop and took the text path down with it.
  const json text = {{"model", "test-model"},
                     {"messages", json::array({{{"role", "user"},
                                                {"content", "hello"}}})},
                     {"max_completion_tokens", 2},
                     {"temperature", 0.0}};
  const ApiServer::DispatchResult t =
      h.server.handle_chat_completions(text.dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
  CHECK(t.body.find("stopped") == std::string::npos);
}

TEST_CASE("dots3-note W6a: an adapter merge that is not the PROMPT's merge refuses at INSTALL") {
  TinySpec spec;
  // `spatial_merge_size` stays 2, so the prompt side expands the placeholder to
  // prod(grid)/4 = FOUR tokens; the adapter folds `adapter_merge_size**2` = ONE
  // trunk token per row and emits SIXTEEN. Upstream keeps the two as
  // independent keys with independent defaults, so this is a config a
  // checkpoint can carry — not a shape the parse can rule out.
  spec.v_adapter_merge_override = 1;
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kRefusing);
  INFO("install log: ", log.str());
  CHECK(log.str().find("adapter_merge_size") != std::string::npos);

  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(ChatBody(2, 0, false).dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("adapter_merge_size") != std::string::npos);
  CHECK(r.body.find("spatial_merge_size") != std::string::npos);

  const json text = {{"model", "test-model"},
                     {"messages", json::array({{{"role", "user"},
                                                {"content", "hello"}}})},
                     {"max_completion_tokens", 2},
                     {"temperature", 0.0}};
  const ApiServer::DispatchResult t =
      h.server.handle_chat_completions(text.dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
  CHECK(t.body.find("stopped") == std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. W7a (#2703) — THE SERVED AUDIO REQUEST.
//
//    THE RED-BEFORE FOR THIS WHOLE BRICK. Before W7a
//    `Dots3NoteChatSupportedMmLimits()` returned `{{"image", 1}}` and nothing
//    else, so `ValidateChatMmLimits` refused an `input_audio` part at the
//    entrypoint with upstream's own message: HTTP 400, "At most 0 audio(s) may
//    be provided in one prompt." Every case in this section reads that on the
//    tree this brick started from.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dots3-note W7a: a served input_audio chat request reaches the model forward") {
  Served s(AudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, /*is_multimodal_model=*/true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  INFO("install log: ", log.str());
  // The install announced that it built an AUDIO tower as well as a vision one.
  CHECK(log.str().find("audio tower") != std::string::npos);

  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyAudio(/*max_tokens=*/3, 0, false).dump());
  INFO("body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 3);
  // The prompt the engine actually ran is the EXPANDED one:
  // `<|audio_comp_start|>` + SEVEN pad tokens + `<|audio_comp_end|>` + "hello".
  // A seam that dropped the expansion would report 3 and still answer 200.
  CHECK(j.at("usage").at("prompt_tokens") == 3 + dots3_tiny::kAudioTokens);
}

TEST_CASE("dots3-note W7a: two DIFFERENT waveforms give two different forwards") {
  // THE LOAD-BEARING CASE. Status, `prompt_tokens` and `completion_tokens` all
  // pass on a tree where the audio tower is replaced by a correctly SHAPED
  // constant — the two waveforms have the same length, so they expand to the
  // same seven placeholders and produce the same counts. The LOGPROBS of the
  // first generated token do not.
  const auto logprobs_of = [](int variant) {
    Served s(AudioSpec());
    MmServerHarness h(s.config, *s.model, Fixture());
    std::ostringstream log;
    REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
            oai::MultiModalChatInstall::kInstalled);
    const ApiServer::DispatchResult r = h.server.handle_chat_completions(
        ChatBodyAudio(/*max_tokens=*/1, variant, /*logprobs=*/true).dump());
    INFO("body: ", r.body);
    REQUIRE(r.status == 200);
    const json j = json::parse(r.body);
    std::vector<double> out;
    for (const json& t :
         j.at("choices")[0].at("logprobs").at("content")[0].at("top_logprobs")) {
      out.push_back(t.at("logprob").get<double>());
    }
    return out;
  };
  const std::vector<double> a = logprobs_of(0);
  const std::vector<double> b = logprobs_of(1);
  REQUIRE(a.size() == b.size());
  REQUIRE(!a.empty());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  MESSAGE("two waveforms move the first token's logprobs by up to " << worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("dots3-note W7a: the chat seam declares ONE audio, and refuses a second") {
  Served s(AudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  json body = ChatBodyAudio(/*max_tokens=*/1, 0, false);
  // A SECOND audio part in the same message.
  body["messages"][0]["content"].push_back(
      {{"type", "input_audio"},
       {"input_audio",
        {{"data", EncodeBase64(dots3_tiny::FixtureAudioWav(1))},
         {"format", "wav"}}}});
  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(body.dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  // Upstream's own message shape (`context.py:414-415`), with the seam's own
  // ceiling of 1 — a user `--limit-mm-per-prompt` can only LOWER it.
  CHECK(r.body.find("At most 1 audio(s)") != std::string::npos);
}

TEST_CASE("dots3-note W7a: a checkpoint with NO audio_config refuses the audio part by name") {
  // The state every dots3-note checkpoint was in before this brick, and the
  // state an image-only one is still in: the seam's ceiling does not declare
  // "audio" at all, which `context.py:414-415` reads as limit 0.
  //
  // THIS IS THE EXACT RED THIS BRICK STARTED FROM. Running this body against a
  // WITH-audio checkpoint on the pre-W7a tree produced the same 400.
  Served s;  // the default spec: `with_audio` is false
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  // The install did NOT announce an audio tower.
  CHECK(log.str().find("audio tower") == std::string::npos);

  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyAudio(/*max_tokens=*/1, 0, false).dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("At most 0 audio(s) may be provided in one prompt.") !=
        std::string::npos);
  // ...and the TEXT path on the same server is untouched, which is the whole
  // reason the refusal is at the entrypoint and not inside the engine loop.
  const ApiServer::DispatchResult t = h.server.handle_chat_completions(
      json{{"model", "test-model"},
           {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
           {"max_completion_tokens", 1},
           {"temperature", 0.0}}
          .dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
}

TEST_CASE("dots3-note W7a+W7c-1: a non-PCM16 container and a wrong rate refuse BY NAME, and the container refusal is NOT this row's") {
  Served s(AudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  SUBCASE("a compressed container is refused, and it names #2814 rather than this row") {
    // W7c-1 NARROWED this message. It used to say a `.mp3` was owed to W7c, a
    // dots3-note brick. It needs a demuxer this tree does not vendor, five
    // surfaces refuse compressed media for the same reason, and #2814 owns it.
    const std::vector<uint8_t> junk(2048, 0x42);
    const ApiServer::DispatchResult r = h.server.handle_chat_completions(
        ChatBodyWithAudio(1, junk, false, "flac").dump());
    INFO("body: ", r.body);
    CHECK(r.status == 400);
    CHECK(r.body.find("#2814") != std::string::npos);
    // ...and it no longer claims multi-channel WAV is owed to anyone.
    CHECK(r.body.find("PCM16 MONO") == std::string::npos);
    CHECK(r.body.find("multi-channel") == std::string::npos);
    // ...nor names librosa at all: the decode chain is soundfile/libsndfile
    // with a PyAV fallback, and nothing under `vllm/` imports librosa
    // (spec 4.16.4).
    CHECK(r.body.find("librosa") == std::string::npos);
  }
  SUBCASE("a 22050 Hz WAV names the resampler refusal and W7c-2") {
    const ApiServer::DispatchResult r = h.server.handle_chat_completions(
        ChatBodyWithAudio(1, dots3_tiny::FixtureAudioWav(0, /*sample_rate=*/22050),
                          false)
            .dump());
    INFO("body: ", r.body);
    CHECK(r.status == 400);
    CHECK(r.body.find("W7c-2") != std::string::npos);
    CHECK(r.body.find("RESAMPLING IS NOT PORTED") != std::string::npos);
    // The rate message CLAIMED librosa does the resample. libswresample does,
    // and no vLLM path imports librosa at all (spec 4.16.4). The message now
    // says so explicitly, so the assertion is on the CLAIM and not on the word.
    CHECK(r.body.find("with librosa") == std::string::npos);
    CHECK(r.body.find("NOT librosa") != std::string::npos);
    CHECK(r.body.find("libswresample") != std::string::npos);
  }
  SUBCASE("a payload that is not a RIFF/WAVE buffer at all is refused, not decoded") {
    const std::vector<uint8_t> junk(2048, 0x41);
    const ApiServer::DispatchResult r = h.server.handle_chat_completions(
        ChatBodyWithAudio(1, junk, false, "mp3").dump());
    INFO("body: ", r.body);
    CHECK(r.status == 400);
    CHECK(r.body.find("RIFF/WAVE") != std::string::npos);
  }
  // ...and the server still answers TEXT after all three, which is what "the
  // refusal is at the entrypoint" buys.
  const ApiServer::DispatchResult t = h.server.handle_chat_completions(
      json{{"model", "test-model"},
           {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
           {"max_completion_tokens", 1},
           {"temperature", 0.0}}
          .dump());
  CHECK(t.status == 200);
}

// ── W7c-1 (#2813): the STEREO inversion ─────────────────────────────────────
//
// THIS IS THE OWNERSHIP PROOF FOR THIS SLICE, and it is an INVERSION rather
// than an addition. The subcase above it used to be
// *"a STEREO WAV names the container refusal and W7c"* and asserted HTTP 400
// on this very request. Nothing about the model, the front end or the tower was
// missing: the file was refused because `DecodeWavPcm16Mono` threw "not mono".
//
// The fixture makes the mean CHECKABLE rather than approximable. Left is
// `m + d` and right is `m - d` over the fixture's own two variants, so the
// per-sample mean is EXACTLY `m` — `FixtureAudioPcm16(0)`, the clip every other
// audio case serves. The test recomputes that mean itself, in int, before it
// trusts anything the server did.
TEST_CASE("dots3-note W7c-1: a STEREO WAV at 16 kHz is SERVED, and its answer is the MEAN's") {
  // (a) The independent mean, computed here and not read out of `src/`.
  std::vector<int16_t> left, right;
  dots3_tiny::FixtureAudioPcm16StereoChannels(&left, &right);
  const std::vector<int16_t> mono = dots3_tiny::FixtureAudioPcm16(0);
  REQUIRE(left.size() == mono.size());
  REQUIRE(right.size() == mono.size());
  std::size_t differ_l = 0, differ_r = 0;
  for (std::size_t i = 0; i < mono.size(); ++i) {
    // Integer, exact: L + R is 2m, so the halving cannot round.
    const int sum = static_cast<int>(left[i]) + static_cast<int>(right[i]);
    REQUIRE(sum % 2 == 0);
    REQUIRE(sum / 2 == static_cast<int>(mono[i]));
    if (left[i] != mono[i]) ++differ_l;
    if (right[i] != mono[i]) ++differ_r;
  }
  // ...and the two channels are GENUINELY different from the mean and from each
  // other, so (d) below has something to measure. A fixture whose channels were
  // equal would make every arm of this case pass on a port that picked one.
  MESSAGE("stereo channels differ from their mean in " << differ_l << " and "
          << differ_r << " of " << mono.size() << " samples");
  CHECK(differ_l > mono.size() / 2);
  CHECK(differ_r > mono.size() / 2);

  const auto serve = [](const std::vector<uint8_t>& wav, bool logprobs) {
    Served s(AudioSpec());
    MmServerHarness h(s.config, *s.model, Fixture());
    std::ostringstream log;
    REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
            oai::MultiModalChatInstall::kInstalled);
    return h.server.handle_chat_completions(
        ChatBodyWithAudio(1, wav, logprobs).dump());
  };
  const auto logprobs_of = [&serve](const std::vector<uint8_t>& wav) {
    const ApiServer::DispatchResult r = serve(wav, /*logprobs=*/true);
    INFO("body: ", r.body);
    REQUIRE(r.status == 200);
    const json j = json::parse(r.body);
    std::vector<double> out;
    for (const json& t :
         j.at("choices")[0].at("logprobs").at("content")[0].at("top_logprobs")) {
      out.push_back(t.at("logprob").get<double>());
    }
    return out;
  };
  const auto worst_gap = [](const std::vector<double>& a,
                            const std::vector<double>& b) {
    REQUIRE(a.size() == b.size());
    REQUIRE(!a.empty());
    double w = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
      w = std::max(w, std::fabs(a[i] - b[i]));
    return w;
  };

  // (b) It SERVES. This is the true-before / false-after line: the same request
  // answered 400 before this slice.
  const ApiServer::DispatchResult r =
      serve(dots3_tiny::FixtureAudioWavStereo(), /*logprobs=*/false);
  INFO("stereo body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 1);
  // The placeholder span did not move: a 2-channel file of N FRAMES carries the
  // same N samples of waveform as the mono clip, not 2N and not N/2.
  const ApiServer::DispatchResult m =
      serve(dots3_tiny::FixtureAudioWav(0), /*logprobs=*/false);
  REQUIRE(m.status == 200);
  CHECK(j.at("usage").at("prompt_tokens") ==
        json::parse(m.body).at("usage").at("prompt_tokens"));

  // (c) The ANSWER is the mean's. Not a tolerance on features — the logprobs of
  // the served first token, which is the whole chain.
  const std::vector<double> stereo =
      logprobs_of(dots3_tiny::FixtureAudioWavStereo());
  const std::vector<double> mean =
      logprobs_of(dots3_tiny::FixtureAudioWav(0));
  const double gap_mean = worst_gap(stereo, mean);
  MESSAGE("stereo vs its own per-sample mean: worst logprob gap " << gap_mean);
  CHECK(gap_mean == 0.0);

  // (d) ...and NOT either channel's, which is what makes (c) load-bearing. A
  // port that took channel 0 would serve `left`; one that summed without
  // dividing would serve `2m`, whose amplitude the log-mel sees. Both are
  // separated here, and both are driven as mutations in spec 4.16.7.
  const double gap_left =
      worst_gap(stereo, logprobs_of(dots3_tiny::FixtureWavFromPcm16(left)));
  const double gap_right =
      worst_gap(stereo, logprobs_of(dots3_tiny::FixtureWavFromPcm16(right)));
  MESSAGE("stereo vs channel 0: " << gap_left << ", vs channel 1: "
          << gap_right);
  CHECK(gap_left > 1e-4);
  CHECK(gap_right > 1e-4);

  // A SUM WITHOUT THE DIVIDE IS NOT DRIVEN FROM A FIXTURE HERE, deliberately:
  // 2 * m leaves int16 (|m| reaches 19660) and could only be fed back through a
  // WAV saturated, which would measure clipping rather than the missing divide.
  // M2 in spec 4.16.7 mutates the production divide instead.
}

TEST_CASE("dots3-note W7a: an image and an audio part in ONE request refuse BY NAME, to W8") {
  Served s(AudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  json body = ChatBodyAudio(/*max_tokens=*/1, 0, false);
  body["messages"][0]["content"].push_back(
      {{"type", "image_url"}, {"image_url", {{"url", ImageDataUri(0)}}}});
  const ApiServer::DispatchResult r =
      h.server.handle_chat_completions(body.dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("BOTH an image") != std::string::npos);
  CHECK(r.body.find("W8") != std::string::npos);
  // Each on its OWN is still served — the refusal is about the COMBINATION.
  CHECK(h.server.handle_chat_completions(ChatBodyAudio(1, 0, false).dump())
            .status == 200);
  CHECK(h.server.handle_chat_completions(ChatBody(1, 0, false).dump()).status ==
        200);
}

TEST_CASE("dots3-note W7a: an audio checkpoint whose arms are OWED refuses at INSTALL") {
  // Not inside the engine loop. `mm_chat_dots3note.cpp:232-240` records why:
  // throwing from `encode_mm` stops `AsyncLLM` and turns every LATER request,
  // TEXT ones included, into a 500. A refusing SEAM answers 400 and leaves the
  // text path alone, and this case asserts both halves.
  dots3_tiny::TinySpec spec = AudioSpec();
  spec.a_use_causal = true;  // an arm W7a refuses by name
  Served s(spec);
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  const oai::MultiModalChatInstall got =
      h.install(kDots3Arch, true, s.ckpt, log);
  INFO("install log: ", log.str());
  CHECK(got == oai::MultiModalChatInstall::kRefusing);
  CHECK(log.str().find("use_causal") != std::string::npos);

  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyAudio(/*max_tokens=*/1, 0, false).dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);

  const ApiServer::DispatchResult t = h.server.handle_chat_completions(
      json{{"model", "test-model"},
           {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
           {"max_completion_tokens", 1},
           {"temperature", 0.0}}
          .dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. W7b (#2797) — THE SERVED MULTI-CHUNK AUDIO REQUEST.
//
//    THE RED-BEFORE FOR THIS BRICK. On W7a's head every case below reads HTTP
//    400 and "SEGMENTATION IS NOT PORTED": `Dots3NoteAudioProcessor::
//    ProcessWaveform` refused any waveform past `chunk_samples` BY NAME, and
//    the refusal named W7b. What this section proves is REACH — that a clip of
//    2.5 chunks travels `ApiServer::handle_chat_completions` on the default
//    configuration and lands 63 audio rows in the prompt embeddings.
//
//    It does NOT gate the chunk seams. §4.14.12 measured that this suite is
//    green under four separate tower-only defects, and W7b's four are of the
//    same kind: `test_dots3_note_audio` gates them against `ref_chunks` and
//    `RefTower`, and this file gates that anything reaches them at all.
// ═══════════════════════════════════════════════════════════════════════════

// The MULTI-CHUNK fixture spec. `a_chunk_seconds = 2` is not decoration: 16000
// is 12.5 token strides, so the DEFAULT `a_chunk_seconds = 1` is exactly a
// geometry whose per-segment token sum disagrees with the prompt side's single
// `ceil(total / stride)` past one chunk, and the port refuses it BY NAME
// (spec §4.15.3). Two seconds is the smallest chunk that divides.
namespace {

dots3_tiny::TinySpec LongAudioSpec() {
  dots3_tiny::TinySpec s = AudioSpec();
  s.a_chunk_seconds = dots3_tiny::kAudioLongChunkSeconds;
  // 63 audio placeholders plus the two markers and "hello" is a 66-token
  // prompt, and the fixture's default `max_pos` is 64 — the engine clamps
  // `max_model_len` to `max_position_embeddings` and answers 400 with "The
  // decoder prompt (length 66) is longer than the maximum model length of 64",
  // which is a REAL refusal and not a harness bug. Raised to the harness's own
  // `kMaxModelLen` for this spec ALONE, so no existing case moves. It changes
  // the rope cache length and nothing else about the weights.
  s.max_pos = kMaxModelLen;
  return s;
}

// The 5 s clip — 80000 samples = 2.5 chunks — through the SAME request writer
// and the SAME WAV writer as every W7a case.
json ChatBodyLongAudio(int max_tokens, int variant, bool logprobs) {
  return ChatBodyWithAudio(
      max_tokens,
      dots3_tiny::FixtureAudioWavLong(variant, dots3_tiny::kAudioLongSamples),
      logprobs);
}

}  // namespace

TEST_CASE("dots3-note W7b: a served input_audio request LONGER than chunk_seconds reaches the model forward") {
  Served s(LongAudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, /*is_multimodal_model=*/true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);
  INFO("install log: ", log.str());
  CHECK(log.str().find("audio tower") != std::string::npos);

  // ONE completion token, and that is a property of the FIXTURE rather than a
  // shortcut. Its `index_topk` is 32, so a SECOND step would resume request 0
  // from 66 already-computed tokens past that threshold and meet W4b-3c's
  // step-level DSA refusal (KV-DSV4-MULTICACHE, #1925) — a real refusal this
  // brick neither owns nor lifts. A single-shot prefill is served sparsely, and
  // it is the step that carries the 63 audio rows.
  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyLongAudio(/*max_tokens=*/1, 0, false).dump());
  INFO("body: ", r.body);
  REQUIRE(r.status == 200);
  const json j = json::parse(r.body);
  CHECK(j.at("object") == "chat.completion");
  CHECK(j.at("usage").at("completion_tokens") == 1);
  // The EXPANDED prompt: `<|audio_comp_start|>` + SIXTY-THREE pad tokens +
  // `<|audio_comp_end|>` + "hello". 63 is `ceil(80000/1280)` and also the
  // 25 + 25 + 13 the three chunks contribute, which is the whole point of
  // §4.15.3's invariant — a scatter that did not balance would throw inside the
  // engine, not answer 200.
  CHECK(j.at("usage").at("prompt_tokens") == 3 + dots3_tiny::kAudioLongTokens);
  // ...and the clip really is longer than one chunk on this config, so this is
  // not the W7a case wearing a different name.
  CHECK(dots3_tiny::kAudioLongSamples >
        dots3_tiny::kAudioLongChunkSeconds * 16000);
  CHECK(dots3_tiny::kAudioLongTokens > dots3_tiny::kAudioTokens);
}

TEST_CASE("dots3-note W7b: two DIFFERENT long waveforms give two different forwards") {
  // THE LOAD-BEARING CASE, as it is for every brick on this row. Both clips are
  // 80000 samples, so they expand to the same 63 placeholders and produce the
  // same status and the same token counts on a tree whose tower is a correctly
  // SHAPED constant. The LOGPROBS of the first generated token do not.
  const auto logprobs_of = [](int variant) {
    Served s(LongAudioSpec());
    MmServerHarness h(s.config, *s.model, Fixture());
    std::ostringstream log;
    REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
            oai::MultiModalChatInstall::kInstalled);
    const ApiServer::DispatchResult r = h.server.handle_chat_completions(
        ChatBodyLongAudio(/*max_tokens=*/1, variant, /*logprobs=*/true).dump());
    INFO("body: ", r.body);
    REQUIRE(r.status == 200);
    const json j = json::parse(r.body);
    std::vector<double> out;
    for (const json& t :
         j.at("choices")[0].at("logprobs").at("content")[0].at("top_logprobs")) {
      out.push_back(t.at("logprob").get<double>());
    }
    return out;
  };
  const std::vector<double> a = logprobs_of(0);
  const std::vector<double> b = logprobs_of(1);
  REQUIRE(a.size() == b.size());
  REQUIRE(!a.empty());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  MESSAGE("two long waveforms move the first token's logprobs by up to "
          << worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("dots3-note W7b: a NON-DIVISIBLE chunk geometry refuses a LONG clip at the entrypoint and serves everything else") {
  // §4.15.3, from the OUTSIDE. The tiny fixture's default `a_chunk_seconds = 1`
  // is a real non-divisible geometry rather than one invented to be refused, so
  // this is the shape a user meets. The refusal is a 400 from the chat seam —
  // NOT a throw from inside the engine loop, which would set `AsyncLLM`'s
  // errored latch and turn every later request, TEXT ones included, into a 500.
  // The three requests after it are what measures that difference.
  Served s(AudioSpec());
  MmServerHarness h(s.config, *s.model, Fixture());
  std::ostringstream log;
  REQUIRE(h.install(kDots3Arch, true, s.ckpt, log) ==
          oai::MultiModalChatInstall::kInstalled);

  const ApiServer::DispatchResult r = h.server.handle_chat_completions(
      ChatBodyWithAudio(/*max_tokens=*/1,
                        dots3_tiny::FixtureAudioWavLong(0, 40000),
                        /*logprobs=*/false)
          .dump());
  INFO("body: ", r.body);
  CHECK(r.status == 400);
  CHECK(r.body.find("#2797") != std::string::npos);
  CHECK(r.body.find("not a whole number of 1280") != std::string::npos);
  // The refusal carries BOTH numbers, so a reader can see the divergence
  // instead of being told there is one.
  CHECK(r.body.find("33 rows") != std::string::npos);
  CHECK(r.body.find("span of 32") != std::string::npos);
  // It no longer says "SEGMENTATION IS NOT PORTED", because it is.
  CHECK(r.body.find("SEGMENTATION IS NOT PORTED") == std::string::npos);

  // A clip INSIDE one chunk is still served on this very config — a one-segment
  // sum IS `ceil(n / stride)` — which is why the refusal is per request and not
  // an install-time capability refusal.
  const ApiServer::DispatchResult ok =
      h.server.handle_chat_completions(ChatBodyAudio(1, 0, false).dump());
  INFO("short body: ", ok.body);
  CHECK(ok.status == 200);

  // ...and TEXT still works, which is the property a throw inside the engine
  // loop would have destroyed.
  const ApiServer::DispatchResult t = h.server.handle_chat_completions(
      json{{"model", "test-model"},
           {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
           {"max_completion_tokens", 1},
           {"temperature", 0.0}}
          .dump());
  INFO("text body: ", t.body);
  CHECK(t.status == 200);
}
