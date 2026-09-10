// vllm.cpp original. Native ROCm BF16 MoE production gate (#3094).
// The checkpoint enters through ModelRegistry::Load and every step through Forward.
#include <doctest/doctest.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "rocm_moe_fixture.h"
#include "support/rocm_moe_reference_set.h"
#include "support/residual_norm_fixture.h"
#include "support/residual_norm_test.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {
using namespace rocm_moe_fixture;
vt::MoeRouterTopKFn native_router = nullptr;
vt::RmsNormFn native_norm = nullptr;
vt::ResidualRmsNormFn native_expression = nullptr;
bool observe_residual = false;
int observed_norm_calls = 0;
std::vector<uint16_t> observed_attention, observed_residual, observed_gamma, observed_post;
std::vector<uint16_t> observed_first;
struct ExpressionRecord {
  vt::ResidualRmsNormArgs args;
  std::vector<uint16_t> a, base, delta, gamma, norm, residual;
  uintptr_t attention_pointer = 0;
};
std::vector<ExpressionRecord> observed_expressions;
std::vector<uint16_t> ReadRow(vt::Queue& q, const vt::Tensor& tensor) {
  REQUIRE(tensor.dtype == vt::DType::kBF16);
  REQUIRE(tensor.Numel() >= 128);
  std::vector<uint16_t> row(128);
  auto& backend = vt::GetBackend(q.device);
  backend.Copy(q, row.data(), tensor.data, row.size() * sizeof(uint16_t));
  backend.Synchronize(q);
  return row;
}
void CaptureResidualNorm(vt::Queue& q, vt::Tensor& out, const vt::Tensor& x,
                         const vt::Tensor& weight, const vt::RmsNormArgs& args,
                         vt::Tensor* residual) {
  const bool legacy = vt::GetBackend(q.device).GetResidualNormPolicy() ==
                      vt::ResidualNormPolicy::kMaterialized;
  const bool capture = observe_residual && legacy && observed_norm_calls++ == 3;
  if (capture) {
    REQUIRE(residual != nullptr);
    observed_attention = ReadRow(q, x);
    observed_residual = ReadRow(q, *residual);
    observed_gamma = ReadRow(q, weight);
  }
  native_norm(q, out, x, weight, args, residual);
  if (capture) observed_post = ReadRow(q, out);
  if (observe_residual && !legacy && observed_first.empty()) observed_first = ReadRow(q, out);
}
void CaptureExpression(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a,
                        const vt::Tensor& base, const vt::Tensor* delta,
                        const vt::Tensor& weight, const vt::ResidualRmsNormArgs& args,
                        vt::Tensor* residual_out) {
  const bool first = observe_residual && observed_norm_calls++ == 0;
  ExpressionRecord record;
  if (observe_residual) {
    record.args = args;
    record.a = ReadRow(q, a); record.base = ReadRow(q, base);
    record.gamma = ReadRow(q, weight);
    record.attention_pointer = reinterpret_cast<uintptr_t>(a.data);
    if (delta != nullptr) record.delta = ReadRow(q, *delta);
  }
  if (first) {
    observed_attention = ReadRow(q, a);
    observed_residual = ReadRow(q, base);
    observed_gamma = ReadRow(q, weight);
  }
  native_expression(q, out, a, base, delta, weight, args, residual_out);
  if (first) observed_post = ReadRow(q, out);
  if (observe_residual) {
    record.norm = ReadRow(q, out);
    if (residual_out != nullptr) record.residual = ReadRow(q, *residual_out);
    CHECK(ReadRow(q, a) == record.a);
    CHECK(ReadRow(q, weight) == record.gamma);
    if (!args.descriptor.residual_alias_base) CHECK(ReadRow(q, base) == record.base);
    if (delta != nullptr && !args.descriptor.output_alias_delta) CHECK(ReadRow(q, *delta) == record.delta);
    observed_expressions.push_back(std::move(record));
  }
}
nlohmann::json* current_routes = nullptr;
void CaptureRoutes(vt::Queue& q, vt::Tensor& weights, vt::Tensor& indices,
                    const vt::Tensor& logits, const vt::MoeRouterTopKArgs& args,
                    const vt::Tensor* bias) {
  native_router(q, weights, indices, logits, args, bias);
  if (current_routes == nullptr) return;
  std::vector<int32_t> ids(static_cast<size_t>(indices.Numel()));
  auto& backend = vt::GetBackend(q.device);
  backend.Copy(q, ids.data(), indices.data, ids.size() * sizeof(int32_t));
  backend.Synchronize(q);
  current_routes->push_back(ids);
}
vt::Tensor Tensor(void* data, vt::DType dtype, vt::Device device,
                  std::initializer_list<int64_t> dimensions) {
  vt::Tensor tensor;
  tensor.data = data;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.rank = static_cast<int>(dimensions.size());
  int axis = 0;
  for (int64_t size : dimensions) tensor.shape[axis++] = size;
  int64_t stride = 1;
  for (int i = tensor.rank - 1; i >= 0; --i) {
    tensor.stride[i] = stride;
    stride *= tensor.shape[i];
  }
  return tensor;
}
struct Buffer {
  vt::Backend& backend;
  void* data;
  Buffer(vt::Backend& b, size_t bytes) : backend(b), data(b.Alloc(bytes)) {}
  ~Buffer() { backend.Free(data); }
};
struct QueueGuard {
  vt::Backend& backend;
  vt::Queue queue;
  explicit QueueGuard(vt::Backend& b) : backend(b), queue(b.CreateQueue()) {}
  ~QueueGuard() { backend.DestroyQueue(queue); }
};

nlohmann::json Generate(const std::filesystem::path& fixture, int length, int concurrency,
                        int steps = 8) {
  auto config = vllm::LoadHfConfig((fixture / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((fixture / "model.safetensors").string()));
  auto model = vllm::ModelRegistry::Load(config, vllm::ModelSource::FromSafetensors(shards));
  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard queue(backend);
  auto& q = queue.queue;
  constexpr int64_t block = 64;
  std::vector<std::unique_ptr<Buffer>> allocations;
  std::vector<vllm::PagedKvCache> caches;
  for (int layer = 0; layer < kL; ++layer) {
    const size_t bytes = static_cast<size_t>(concurrency * 2 * block * kHkv * kDh) * 2;
    allocations.push_back(std::make_unique<Buffer>(backend, bytes));
    std::vector<uint16_t> zeros(bytes / 2, 0);
    backend.Copy(q, allocations.back()->data, zeros.data(), bytes);
    vllm::PagedKvCache kv;
    kv.data = allocations.back()->data;
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = concurrency;
    kv.block_size = block;
    kv.num_kv_heads = kHkv;
    kv.head_size = kDh;
    caches.push_back(kv);
    backend.Synchronize(q);
  }
  std::vector<vllm::GdnStateCache> gdn;
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<std::vector<int32_t>> generated(static_cast<size_t>(concurrency));
  nlohmann::json logits = nlohmann::json::array();
  nlohmann::json routes = nlohmann::json::array();
  current_routes = &routes;
  Buffer sampled(backend, static_cast<size_t>(concurrency) * sizeof(int64_t));
  auto ids = Tensor(sampled.data, vt::DType::kI64, q.device, {concurrency});
  for (int step = 0; step < steps; ++step) {
    const int count = step == 0 ? length : 1;
    const int previous = step == 0 ? 0 : length + step - 1;
    std::vector<int32_t> tokens, positions, gather;
    vllm::v1::CommonAttentionMetadata attention;
    attention.num_reqs = concurrency;
    attention.num_actual_tokens = count * concurrency;
    attention.query_start_loc = {0};
    for (int r = 0; r < concurrency; ++r) {
      const auto prompt = Prompt(length, r);
      for (int i = 0; i < count; ++i) {
        tokens.push_back(step == 0 ? prompt[static_cast<size_t>(i)] : generated[r].back());
        positions.push_back(previous + i);
        attention.slot_mapping.push_back(r * block + previous + i);
      }
      attention.query_start_loc.push_back((r + 1) * count);
      attention.seq_lens.push_back(previous + count);
      attention.block_table_tensor.push_back(r);
      gather.push_back((r + 1) * count - 1);
    }
    attention.query_start_loc_cpu = attention.query_start_loc;
    attention.seq_lens_cpu = attention.seq_lens;
    attention.max_query_len = count;
    attention.max_seq_len = previous + count;
    attention.block_table_num_cols = 1;
    attention.causal = true;
    vllm::ModelForwardInput input{tokens, positions, attention, gdn_meta, caches, gdn,
                                 config, q, gather};
    input.num_reqs = concurrency;
    input.gdn_state_slots = concurrency;
    input.pure_decode = step > 0;
    input.uniform_query_len = count;
    const auto result = vllm::ModelRegistry::Forward(*model, input);
    REQUIRE(result.on_device());
    REQUIRE(result.device_tensor.dtype == vt::DType::kF32);
    vt::GreedyArgmax(q, ids, result.device_tensor);
    std::vector<int64_t> host_ids(static_cast<size_t>(concurrency));
    std::vector<float> host_logits(static_cast<size_t>(concurrency * kV));
    backend.Copy(q, host_ids.data(), sampled.data, host_ids.size() * sizeof(int64_t));
    backend.Copy(q, host_logits.data(), result.device_tensor.data,
                 host_logits.size() * sizeof(float));
    backend.Synchronize(q);
    for (float value : host_logits) REQUIRE(std::isfinite(value));
    logits.push_back(host_logits);
    for (int r = 0; r < concurrency; ++r)
      generated[r].push_back(static_cast<int32_t>(host_ids[r]));
  }
  current_routes = nullptr;
  return {{"length", length}, {"concurrency", concurrency}, {"tokens", generated},
          {"logits", logits}, {"expert_ids", routes}};
}
}  // namespace

TEST_CASE("ROCm residual row zero matches the compiled primary through production forward") {
  using namespace residual_norm_fixture;
  const char* directory = std::getenv("VT_ROCM_MOE_FIXTURE");
  REQUIRE(directory != nullptr);
  native_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kRmsNorm, vt::DeviceType::kROCM,
      {"test-residual-row-observer", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::RmsNormFn>(&CaptureResidualNorm))});
  native_expression = reinterpret_cast<vt::ResidualRmsNormFn>(
      vt::GetOp(vt::OpId::kResidualRmsNorm, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kResidualRmsNorm, vt::DeviceType::kROCM,
      {"test-residual-expression-observer", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::ResidualRmsNormFn>(&CaptureExpression))});
  observe_residual = true;
  observed_norm_calls = 0;
  observed_first.clear(); observed_expressions.clear();
  Generate(directory, 33, 2, 1);
  observe_residual = false;
  CHECK(observed_attention == std::vector<uint16_t>(kAttention.begin(), kAttention.end()));
  CHECK(observed_residual == std::vector<uint16_t>(kResidual.begin(), kResidual.end()));
  CHECK(observed_gamma == std::vector<uint16_t>(kGamma.begin(), kGamma.end()));
  REQUIRE(observed_post.size() == kPostNorm.size());
  int different = 0;
  for (size_t j = 0; j < kPostNorm.size(); ++j) {
    CAPTURE(j);
    different += observed_post[j] != kPostNorm[j];
    CHECK(observed_post[j] == kPostNorm[j]);
  }
  MESSAGE("Compared 128 production post-attention BF16 words with the compiled primary; differences: ", different);
  CHECK(observed_first == std::vector<uint16_t>(kFirstNorm.begin(), kFirstNorm.end()));
  REQUIRE(observed_expressions.size() == 4);
  auto checkpoint = vllm::SafetensorsFile::Open((std::filesystem::path(directory) / "model.safetensors").string());
  const std::vector<std::string> weights{
      "model.layers.0.post_attention_layernorm.weight", "model.layers.1.input_layernorm.weight",
      "model.layers.1.post_attention_layernorm.weight", "model.norm.weight"};
  for (size_t i = 0; i < observed_expressions.size(); ++i) {
    CAPTURE(i);
    const auto& record = observed_expressions[i];
    const bool triple = i % 2 == 1;
    CHECK(record.args.descriptor.expression == (triple ? vt::ResidualNormExpr::kDeltaPlusAdd : vt::ResidualNormExpr::kAdd));
    CHECK(record.args.descriptor.materialize_residual == (i == 1));
    const auto& gamma = checkpoint.Get(weights[i]);
    REQUIRE(gamma.dtype == "BF16");
    const auto* words = reinterpret_cast<const uint16_t*>(gamma.data);
    CHECK(record.gamma == std::vector<uint16_t>(words, words + 128));
    std::vector<uint16_t> expected_residual;
    const auto expected = residual_norm_test::Reference(record.a, record.base,
        triple ? &record.delta : nullptr, record.gamma, 1, 128, 128,
        record.args.eps, &expected_residual);
    CHECK(record.norm == expected);
    if (i == 1) CHECK(record.residual == expected_residual);
    else CHECK(record.residual.empty());
    if (triple) {
      CHECK(record.a == observed_expressions[i - 1].a);
      CHECK(record.base == observed_expressions[i - 1].base);
      CHECK(record.attention_pointer == observed_expressions[i - 1].attention_pointer);
    }
  }
  CHECK(observed_expressions[2].base == observed_expressions[1].residual);
}

TEST_CASE("ROCm BF16 MoE enters native providers through the production registry") {
  const char* directory = std::getenv("VT_ROCM_MOE_FIXTURE");
  if (directory == nullptr) {
    MESSAGE("Set VT_ROCM_MOE_FIXTURE to the exported production checkpoint");
    std::exit(77);
  }
  const std::filesystem::path fixture(directory);
  if (std::getenv("VT_ROCM_MOE_EXPORT_ONLY") != nullptr) {
    Export(fixture);
    REQUIRE(std::filesystem::file_size(fixture / "model.safetensors") > 0);
    return;
  }
  const char* oracle_path = std::getenv("VT_ROCM_MOE_ORACLE");
  REQUIRE(oracle_path != nullptr);
  nlohmann::json oracle;
  std::ifstream(oracle_path) >> oracle;
  // Test-only observation of the real device router. Expert providers retain
  // their native registry entries and per-call selection accounting.
  native_router = reinterpret_cast<vt::MoeRouterTopKFn>(
      vt::GetOp(vt::OpId::kMoeRouterTopK, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kMoeRouterTopK, vt::DeviceType::kROCM,
      {"test-moe-route-capture", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::MoeRouterTopKFn>(&CaptureRoutes))});
  vt::EnableOpProviderCallStats(true);
  for (auto op : {vt::OpId::kMoeGroupedGemmBf16GateUpSiluNative,
                  vt::OpId::kMoeGroupedGemmBf16Weighted, vt::OpId::kMoeCombinePreweighted})
    vt::ResetOpProviderStats(op, vt::DeviceType::kROCM);
  nlohmann::json runs = nlohmann::json::array();
  std::set<std::vector<int32_t>> selected_pairs;
  for (int length : {1, 3, 33}) {
    for (int concurrency : {1, 2}) {
      nlohmann::json reference;
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto run = Generate(fixture, length, concurrency);
        if (repeat == 0) reference = run["tokens"];
        CHECK(run["tokens"] == reference);
        bool compared = false;
        for (const auto& expected : oracle.at("runs")) {
          if (expected["length"] != length || expected["concurrency"] != concurrency ||
              expected["repeat"] != repeat)
            continue;
          CAPTURE(length);
          CAPTURE(concurrency);
          CAPTURE(repeat);
          compared = true;
          const auto native = run["tokens"].get<std::vector<std::vector<int32_t>>>();
          // The compared request count is the captured record's concurrency, not
          // whatever the native run happened to return.
          REQUIRE(native.size() == static_cast<size_t>(concurrency));
          for (size_t request = 0; request < native.size(); ++request) {
            CAPTURE(request);
            // Every captured configuration of this workload forms the reference
            // set for this request. Request 1 exists only in the records at
            // concurrency 2, so a record without it contributes nothing.
            const auto reference_set = rocm_moe_reference_set::Collect(
                oracle.at("runs"), length, repeat, request);
            REQUIRE(!reference_set.empty());
            const auto comparison = rocm_moe_reference_set::Compare(
                reference_set, native[request], concurrency);
            const int matched = comparison.pass() ? comparison.matched : -1;
            // The gate reports the reference set, the matched configuration, the
            // same-configuration outcome, and the reference's own disagreement
            // positions. The same-configuration outcome is a report and not an
            // assertion: the pinned primary disagrees with itself at length 33.
            std::cout << "[production tokens] length " << length << " concurrency "
                      << concurrency << " repeat " << repeat << " request " << request
                      << ": reference set "
                      << rocm_moe_reference_set::Describe(reference_set)
                      << "; matched configuration "
                      << (matched < 0
                              ? std::string("none")
                              : std::to_string(reference_set[static_cast<size_t>(matched)]
                                                   .concurrency))
                      << "; same-configuration match "
                      << (comparison.same_configuration_match ? "true" : "false")
                      << "; reference disagreement positions "
                      << rocm_moe_reference_set::Describe(comparison.disagreements)
                      << std::endl;
            // Whole-sequence membership. A per-position mix of two reference
            // sequences matches no member and fails here.
            CHECK(comparison.pass());
          }
        }
        REQUIRE(compared);
        REQUIRE(run["expert_ids"].size() == static_cast<size_t>(8 * kL));
        for (const auto& call : run["expert_ids"]) {
          const auto ids = call.get<std::vector<int32_t>>();
          for (size_t pair = 0; pair < ids.size(); pair += 2)
            selected_pairs.insert({ids[pair], ids[pair + 1]});
        }
        run["repeat"] = repeat;
        runs.push_back(std::move(run));
      }
    }
  }
  if (const char* output = std::getenv("VT_ROCM_MOE_OUTPUT"))
    std::ofstream(output) << runs.dump(2) << '\n';
  CHECK(selected_pairs.size() > 1);
  CHECK(vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16, vt::DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16GateUpSilu, vt::DeviceType::kROCM));
  for (auto op : {vt::OpId::kMoeGroupedGemmBf16GateUpSiluNative,
                  vt::OpId::kMoeGroupedGemmBf16Weighted, vt::OpId::kMoeCombinePreweighted}) {
    CAPTURE(static_cast<int>(op));
    const auto stats = vt::GetOpProviderStats(op, vt::DeviceType::kROCM);
    CHECK(vt::OpRegistered(op, vt::DeviceType::kROCM));
    CHECK(stats.selections > 0);
    REQUIRE(stats.last_selected != nullptr);
    CHECK(std::string(stats.last_selected) == vt::kNativeProviderName);
    CHECK(stats.declines == 0);
    CHECK(stats.fallbacks == 0);
    CHECK(vt::GetOpProviderStats(op, vt::DeviceType::kCPU).selections == 0);
  }
  vt::EnableOpProviderCallStats(false);
}
