// ENG-MM-EMBED-DEVICE-IDS (#2730) — the MULTIMODAL embed reads the DEVICE
// identifiers, and a hook that does not is refused.
//
// `MmEmbedInputs` carried a host `const std::vector<int32_t>* token_ids` and
// nothing else for identifiers, while the asynchronous runner splices each
// decode row's sampled token into the DEVICE buffer on the main queue and
// deliberately never writes it back. `token_ids_cpu` is zero-initialised, so
// `ModelRegistry::EmbedMm` built `inputs_embeds` from TOKEN ID 0 on every decode
// row of an image request — at rc=0, with plausible-looking output.
// `batch_carries_mm()` returns true on those decode steps BY DESIGN and says so
// in its own comment, so the path is reached rather than theoretical, and
// `async_device_mirror()` is the DEFAULT on CUDA.
//
// WHY EVERY CASE GOES THROUGH `ModelRegistry::EmbedMm`. AGENTS.md
// `## Nothing lands dead` asks for the smallest failing test to enter through a
// production entry point, and this is the one the runner calls
// (`GPUModelRunner::execute_model`, under
// `supports_mm_inputs() && batch_carries_mm()`). Calling
// `EmbedMmQwen3VLForConditionalGeneration` directly would measure a function; the
// seam is what carries the channel and the guard.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"     // Qwen3DenseWeights, OwnedTensor
#include "vllm/model_executor/models/qwen3_vl.h"  // Qwen3VLWeights, BorrowQwen3VLLoadedModel
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

// The smallest geometry that can still distinguish the rows this test asserts
// on: a vocabulary wide enough that the STALE id (0), the two prefill ids and
// the SPLICED id are four different rows, and a hidden width > 1 so a row copied
// from the wrong place cannot coincide on its single element.
constexpr int64_t kHidden = 8;
constexpr int64_t kVocab = 16;
constexpr int64_t kTokens = 3;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// A distinct value per (row, column), so a row read from the wrong table index
// is a value mismatch and not a near-tie.
//
// ROUNDED THROUGH BF16, which is not cosmetic: the table is stored bf16 and bf16
// carries 8 mantissa bits, so `6 + 3/64` lands on 6.0625 in the table while the
// f32 expression says 6.046875. Comparing the download against the unrounded
// expression fails on the STORAGE dtype rather than on the identifiers, which
// would make this suite red for a reason it is not about.
float TableValue(int64_t row, int64_t col) {
  return vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(row) +
                                     static_cast<float>(col) / 64.0f));
}

vllm::OwnedTensor MakeEmbedTable() {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = 2;
  t.shape[0] = kVocab;
  t.shape[1] = kHidden;
  t.bytes.resize(static_cast<size_t>(kVocab * kHidden) * 2);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t r = 0; r < kVocab; ++r)
    for (int64_t c = 0; c < kHidden; ++c)
      p[static_cast<size_t>(r * kHidden + c)] =
          vt::F32ToBF16(TableValue(r, c));
  return t;
}

vllm::HfConfig MakeConfig() {
  vllm::HfConfig c;
  c.architectures = {"Qwen3VLForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.vocab_size = kVocab;
  return c;
}

// The TEXT half only. A decode step of an image request covers NO placeholder
// row — that is exactly the step this defect lives on, and it is why the vision
// tower is not built here: `mm_embeds` is empty, the mask is all false, and the
// hook reduces to the token lookup the defect corrupts. `deepstack_visual_indexes`
// stays empty so `levels == 0` and the gathered-slice width is `hidden`.
vllm::Qwen3VLWeights MakeTextOnlyVlWeights() {
  vllm::Qwen3VLWeights w;
  w.text.tie_word_embeddings = true;
  w.text.embed_tokens = MakeEmbedTable();
  return w;
}

// One step's worth of borrowed channels, held by value so they outlive the call.
struct StepChannels {
  // The HOST vector as the asynchronous runner leaves it: rows 0 and 1 belong to
  // a PREFILL request and are correct; row 2 is an image request's DECODE row,
  // and `token_ids_cpu` zero-initialisation is why it reads 0.
  std::vector<int32_t> host_ids{5, 6, 0};
  std::vector<vt::Tensor> mm_embeds;
  std::vector<char> is_mm_embed = std::vector<char>(kTokens, 0);
  std::vector<int32_t> mrope = std::vector<int32_t>(3 * kTokens, 0);

  vllm::MmEmbedInputs Make() const {
    vllm::MmEmbedInputs in;
    in.token_ids = &host_ids;
    in.mm_embeds = &mm_embeds;
    in.is_mm_embed = &is_mm_embed;
    in.mrope_positions = &mrope;
    return in;
  }
};

// The identifiers the combine actually wrote: row 2's sampled token is 9, and 9
// is neither 0 (what the host vector holds) nor either prefill id.
constexpr int32_t kDeviceIds[kTokens] = {5, 6, 9};

// An owned "device" allocation through the backend, which on CPU is host memory
// the backend still reaches through `Copy` — so the seam under test is exercised
// with the same calls a CUDA step makes.
class DeviceIds {
 public:
  DeviceIds(vt::Backend& b, vt::Queue& q, const int32_t* src, int64_t n)
      : b_(b), n_(n) {
    p_ = b_.Alloc(static_cast<size_t>(n) * sizeof(int32_t));
    b_.Copy(q, p_, src, static_cast<size_t>(n) * sizeof(int32_t));
    b_.Synchronize(q);
  }
  ~DeviceIds() { b_.Free(p_); }
  DeviceIds(const DeviceIds&) = delete;
  DeviceIds& operator=(const DeviceIds&) = delete;
  const int32_t* get() const { return reinterpret_cast<const int32_t*>(p_); }

 private:
  vt::Backend& b_;
  void* p_ = nullptr;
  int64_t n_ = 0;
};

std::vector<float> DownloadEmbeds(vt::Backend& b, vt::Queue& q,
                                  const vt::Tensor& t) {
  const int64_t n = t.Numel();
  std::vector<uint16_t> bits(static_cast<size_t>(n));
  b.Copy(q, bits.data(), t.data, static_cast<size_t>(n) * sizeof(uint16_t));
  b.Synchronize(q);
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    out[static_cast<size_t>(i)] = vt::BF16ToF32(bits[static_cast<size_t>(i)]);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE DEFECT, as a number, through the production seam and the REAL
//    Qwen3-VL registration.
// ---------------------------------------------------------------------------
TEST_CASE("mm embed: the merged embeds come from the DEVICE identifiers, not the stale host vector") {
  const vllm::HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeTextOnlyVlWeights();
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(q.device.type);

  StepChannels step;
  const DeviceIds device_ids(b, q, kDeviceIds, kTokens);

  vllm::MmEmbedInputs in = step.Make();
  in.device_token_ids = device_ids.get();
  // TWO REQUESTS, MIXED. Rows 0-1 are a prefill whose host ids are perfectly
  // good; row 2 is a decode row the combine spliced. A per-REQUEST reading of
  // this step finds a prefill request and concludes the host vector is fine; the
  // step's own answer is that it is stale. The two readings disagree here, which
  // is the asymmetry every earlier test in this family missed by using
  // `num_reqs == 1`.
  in.host_token_ids_stale = true;

  const vllm::MmForwardBuffers out =
      vllm::ModelRegistry::EmbedMm(*model, c, q, in);
  REQUIRE(out.mm.inputs_embeds.data != nullptr);
  REQUIRE(out.mm.inputs_embeds.Numel() == kTokens * kHidden);

  const std::vector<float> embeds = DownloadEmbeds(b, q, out.mm.inputs_embeds);

  // THE DECODE ROW. Before this row it embedded table row 0 — the value
  // `token_ids_cpu` zero-initialisation leaves in the host vector — which is the
  // defect stated as a number rather than as prose.
  for (int64_t col = 0; col < kHidden; ++col) {
    const float got = embeds[static_cast<size_t>(2 * kHidden + col)];
    CHECK(got == doctest::Approx(TableValue(kDeviceIds[2], col)));
    // Named separately so a failure says WHICH wrong row was read. A silent
    // fallback to the host vector reads table row 0.
    CHECK(got != doctest::Approx(TableValue(0, col)));
  }

  // THE PREFILL ROWS ARE UNTOUCHED. A splice that wrote the wrong extent — one
  // row, or past the end — would move these, and a gate that only looked at the
  // decode row could not see it.
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < kHidden; ++col) {
      CHECK(embeds[static_cast<size_t>(row * kHidden + col)] ==
            doctest::Approx(TableValue(kDeviceIds[row], col)));
    }
  }
}

// ---------------------------------------------------------------------------
// 1b. THE CHANNEL IS NOT MANDATORY. Every non-mirror step — every CPU step and
//     every run with VT_ASYNC_DEVICE_MIRROR=0 — passes a null pointer, and the
//     hook must then embed the host vector exactly as it did before this row.
// ---------------------------------------------------------------------------
TEST_CASE("mm embed: a null device channel leaves the host lookup byte-identical") {
  const vllm::HfConfig c = MakeConfig();
  const vllm::Qwen3VLWeights w = MakeTextOnlyVlWeights();
  std::unique_ptr<vllm::LoadedModel> model = vllm::BorrowQwen3VLLoadedModel(w);
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(q.device.type);

  StepChannels step;
  const vllm::MmForwardBuffers out =
      vllm::ModelRegistry::EmbedMm(*model, c, q, step.Make());
  const std::vector<float> embeds = DownloadEmbeds(b, q, out.mm.inputs_embeds);

  for (int64_t row = 0; row < kTokens; ++row)
    for (int64_t col = 0; col < kHidden; ++col)
      CHECK(embeds[static_cast<size_t>(row * kHidden + col)] ==
            doctest::Approx(TableValue(step.host_ids[static_cast<size_t>(row)],
                                       col)));
}

// ---------------------------------------------------------------------------
// 2. THE GUARD. A hook that does NOT declare the capability is refused rather
//    than handed a stale host vector — the shape `ModelRegistry::Forward`
//    already has, on the same three-term predicate.
// ---------------------------------------------------------------------------
namespace {

bool g_embed_ran = false;

vllm::MmForwardBuffers RecordingEmbed(vllm::LoadedModel& model,
                                      const vllm::HfConfig& config,
                                      vt::Queue& queue,
                                      const vllm::MmEmbedInputs& inputs) {
  (void)model;
  (void)config;
  (void)queue;
  (void)inputs;
  g_embed_ran = true;
  return {};
}

class FakeLoadedModel : public vllm::LoadedModel {
 public:
  explicit FakeLoadedModel(const vllm::ModelRegistration& reg)
      : vllm::LoadedModel(reg) {}
};

// The guard fires BEFORE dispatch, so "did the hook run" is exactly "did the
// guard let the step through", which is the observable a seam case needs.
bool EmbedReached(bool claimed, const int32_t* device_ids, bool stale,
                  std::string* message) {
  vllm::ModelFactory factory{};
  factory.embed_mm = &RecordingEmbed;
  factory.embed_mm_consumes_device_token_ids = claimed;
  const vllm::ModelRegistration reg{"FakeMmForConditionalGeneration", &factory,
                                    {}};
  FakeLoadedModel model(reg);

  const vllm::HfConfig c = MakeConfig();
  vt::Queue q = Q();
  StepChannels step;
  vllm::MmEmbedInputs in = step.Make();
  in.device_token_ids = device_ids;
  in.host_token_ids_stale = stale;

  g_embed_ran = false;
  try {
    (void)vllm::ModelRegistry::EmbedMm(model, c, q, in);
  } catch (const std::exception& e) {
    if (message != nullptr) *message = e.what();
    return false;
  }
  return g_embed_ran;
}

}  // namespace

TEST_CASE("ModelRegistry::EmbedMm REFUSES a stale step for a hook that does not claim the channel") {
  vt::Queue q = Q();
  vt::Backend& b = vt::GetBackend(q.device.type);
  const DeviceIds device_ids(b, q, kDeviceIds, kTokens);

  // THE DEFECT'S SHAPE: the mirror is live, a row was spliced, and the hook does
  // not read the device buffer. Before this row it was not a refusal at all — it
  // was `inputs_embeds` built from token id 0.
  std::string message;
  CHECK_FALSE(EmbedReached(/*claimed=*/false, device_ids.get(), /*stale=*/true,
                           &message));
  // The refusal names the ARRIVING architecture, because an operator reading it
  // has a model directory and not a call stack.
  CHECK(message.find("FakeMmForConditionalGeneration") != std::string::npos);
  CHECK(message.find("embed_mm_consumes_device_token_ids") != std::string::npos);

  // THE SIDE THAT, IF WRONG, MAKES THE CAPABILITY UNREACHABLE while every other
  // gate stays green: a hook that HAS wired itself to the device identifiers must
  // proceed.
  CHECK(EmbedReached(/*claimed=*/true, device_ids.get(), /*stale=*/true,
                     nullptr));

  // THE TERM THAT KEEPS A CORRECT PATH ALIVE, and the reason this is not a
  // nullness test. The runner sets the pointer on EVERY step once the mirror is
  // engaged, and an image request's PREFILL step is all-prefill: the combine
  // splices nothing and the host vector is perfectly good. Every image request in
  // this tree reaches its first token through that step, so a guard that refused
  // here would take multimodal serving away entirely — the
  // `ForwardLlamaModelEmbedding` mistake (#2710 D1), in the one place it cannot
  // be survived.
  CHECK(EmbedReached(/*claimed=*/false, device_ids.get(), /*stale=*/false,
                     nullptr));
  CHECK(EmbedReached(/*claimed=*/true, device_ids.get(), /*stale=*/false,
                     nullptr));

  // NO MIRROR, EITHER WAY. Null on every CPU step and every
  // VT_ASYNC_DEVICE_MIRROR=0 run; a guard that fired here would break every
  // multimodal hook in the tree at once. The stale flag must not be able to
  // refuse on its own.
  CHECK(EmbedReached(/*claimed=*/false, nullptr, /*stale=*/true, nullptr));
  CHECK(EmbedReached(/*claimed=*/true, nullptr, /*stale=*/true, nullptr));
  CHECK(EmbedReached(/*claimed=*/false, nullptr, /*stale=*/false, nullptr));
  CHECK(EmbedReached(/*claimed=*/true, nullptr, /*stale=*/false, nullptr));
}

// ---------------------------------------------------------------------------
// 3. THE REGISTRATIONS SAY WHAT THEY DO, and the ASYMMETRY between them is the
//    design rather than an oversight.
// ---------------------------------------------------------------------------
TEST_CASE("the two embed_mm registrations declare the hook capability, and only one declares the forward's") {
  const vllm::ModelRegistration& vl = vllm::ModelRegistry::Resolve(
      std::vector<std::string>{"Qwen3VLForConditionalGeneration"});
  REQUIRE(vl.factory != nullptr);
  REQUIRE(vl.factory->embed_mm != nullptr);
  CHECK(vl.factory->embed_mm_consumes_device_token_ids);
  // `ForwardQwen3VLForConditionalGeneration` reads NO token identifiers: it
  // refuses a step without `input.mm` by name and then embeds `mm.inputs_embeds`.
  // Through `ModelRegistry::Forward` it is reachable only from the runner branch
  // that has already called `EmbedMm` with the same device pointer on the same
  // step, so after this row the registration's whole reachable path resolves its
  // identifiers from the device buffer. Without this bit the fix would be
  // computed and then thrown away by the forward's own guard.
  CHECK(vl.factory->consumes_device_token_ids);

  const vllm::ModelRegistration& d3 = vllm::ModelRegistry::Resolve(
      std::vector<std::string>{"Dots3NoteForCausalLM"});
  REQUIRE(d3.factory != nullptr);
  REQUIRE(d3.factory->embed_mm != nullptr);
  CHECK(d3.factory->embed_mm_consumes_device_token_ids);
  // AND NOT the forward's, which is the asymmetry one bit could not express.
  // `ForwardDots3NoteForCausalLM` reaches
  // `Dots3NoteModel::ForwardDevice(input.token_ids, ...)`, and the runner takes
  // the multimodal branch when ANY request in the batch carries multimodal items
  // — so a dots3-note batch of TEXT-ONLY requests takes the text arm and embeds
  // the stale host vector. It stays refused until #2732 ports that arm.
  CHECK_FALSE(d3.factory->consumes_device_token_ids);
}
