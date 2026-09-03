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
#include <cmath>
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

// A BPE fixture whose ADDED tokens are dots3-note's three image markers, so the
// string the chat seam injects tokenizes to exactly [14, 15, 16].
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
       {{"id", dots3_tiny::kImgEndId}, {"content", "<|endofimg|>"}, {"special", true}}});
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
oai::ImageCodecFn RawRgbCodec() {
  return [](const oai::DecodedMedia& media) -> oai::DecodedImageRgb {
    REQUIRE(media.media_type == "image/x-raw-rgb");
    const std::size_t px = media.bytes.size() / 3;
    const auto side = static_cast<int64_t>(
        std::llround(std::sqrt(static_cast<double>(px))));
    REQUIRE(static_cast<std::size_t>(side * side * 3) == media.bytes.size());
    oai::DecodedImageRgb out;
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

json ChatBody(int max_tokens, int variant, bool logprobs) {
  json body = {
      {"model", "test-model"},
      {"messages",
       json::array({{{"role", "user"},
                     {"content",
                      json::array({{{"type", "image_url"},
                                    {"image_url", {{"url", ImageDataUri(variant)}}}},
                                   {{"type", "text"}, {"text", "hello"}}})}}})},
      {"max_completion_tokens", max_tokens},
      {"temperature", 0.0}};
  if (logprobs) {
    body["logprobs"] = true;
    body["top_logprobs"] = 3;
  }
  return body;
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
//     NOT survive as a router gate, because a dense block still processes the
//     pixels on the way. Three mutations recorded in spec §4.12.9 left it
//     GREEN: routing every token to expert 0, dropping the router bias, and
//     replacing the routed FFN output with a constant. All three are defects
//     that change WHICH expert runs, and none of them changed what this server
//     answered.
//
//     So the router needs its own served assertion, and this is its shape: two
//     checkpoints that differ in `mlp.router_bias` AND IN NOTHING ELSE — every
//     other tensor is drawn from the same seed stream — must produce different
//     logprobs. A router that ignores its bias, ignores its input, or feeds a
//     constant FFN gives the two identical answers and this case reds.
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
