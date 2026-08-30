// GLM-5.3 (`GlmMoeDsaForCausalLM`) — W7's load gate, on a complete synthetic
// `glm-dsa` GGUF.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHY A SYNTHETIC MODEL AND NOT THE ARTIFACT ──────────────────────────────
// The real gate is a 201.83 GiB load under an `rc` lease, and it is not
// something CI can run. What CI can run is the same loader over a model with
// the same STRUCTURE at 1/1000th the size: the same tensor names, the same
// heterogeneous indexer schedule with `full` and `shared` layers interleaved,
// the same leading dense block, the same stacked keep-quant towers, and the
// same multi-token-prediction tail to drop. Every refusal and every accounting
// rule this loader has is reachable here.
//
// This is the case the reachability rule asks for. It enters through
// `LoadedEngine::FromModelDir` — the production entry point a user arrives
// through — and not through `LoadGlmMoeDsaFromGguf`, so deleting the
// `kGgufArchArms` row, the `REGISTER_VLLM_MODEL` line, or the GGUF branch of
// `LoadGlmMoeDsaForCausalLM` reds it. A unit test that called the loader
// directly would prove the loader works and nothing about whether anything
// reaches it.
//
// ─── THE TOWERS ARE Q8_0, AND THAT IS NOT COSMETIC ───────────────────────────
// A routed-expert tower that routes to an EXPAND residency leaves the streaming
// lane, and `gguf_device_fit.cpp`'s admission rule is all-or-nothing across a
// model's `*_exps.weight` set — one expanded tower disqualifies all of them. On
// the real arm that is the difference between 6.375 GiB of blocks and 24.000
// GiB of bf16 for four tensors, and between streaming and not streaming for all
// 228. F32 has no keep-quant residency, so a fixture whose towers were F32
// would exercise the expand path and never the one production takes. The blocks
// below are BUILT rather than filled with noise: random bytes in a Q8_0 fp16
// scale produce inf and NaN, which propagate and make every later comparison
// vacuous.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/model_registry.h"

#include "../gguf_builder.h"

namespace {

// The tiny model's geometry. Every number is chosen so the shapes are legal for
// the real code path: `K` is a whole number of Q8_0 blocks on every tower, the
// MLA latent is stated consistently three times, and the indexer schedule has
// both kinds of layer with a `full` one first.
constexpr int64_t kHidden = 64;
constexpr int64_t kVocab = 32;
constexpr int64_t kBackbone = 4;   // block_count 5 minus one MTP block
constexpr int64_t kHeads = 4;
constexpr int64_t kQLora = 48;
constexpr int64_t kKvLora = 32;
constexpr int64_t kQkRope = 8;
constexpr int64_t kQkHead = 24;    // key_length_mla
constexpr int64_t kQkNope = kQkHead - kQkRope;  // 16
constexpr int64_t kVHead = 24;
constexpr int64_t kInter = 128;    // the leading dense block's MLP
constexpr int64_t kMoeInter = 32;
constexpr int64_t kExperts = 8;
constexpr int64_t kIdxHeads = 4;
constexpr int64_t kIdxHead = 16;
constexpr int64_t kLeadingDense = 1;
// `indexer.types` = full, shared, full, shared. Layer 0 is full, which
// `GlmMoeDsaMlaSchedule` requires: a `shared` layer 0 would attend through a
// buffer nothing had written.
constexpr int64_t kFullLayers = 2;
constexpr int64_t kSharedLayers = kBackbone - kFullLayers;

// ─── Q8_0 blocks, built ──────────────────────────────────────────────────────
// `block_q8_0` is `{ ggml_fp16_t d; int8_t qs[32]; }` = 34 bytes for 32
// elements, which `gguf_reader.cpp` case 8 sizes as `{32, 34}`.
constexpr uint16_t kFp16One = 0x3C00;  // 1.0 in IEEE half

std::string Q8_0Blocks(int64_t numel, uint32_t seed) {
  REQUIRE(numel % 32 == 0);
  std::string out;
  out.reserve(static_cast<size_t>(numel) / 32 * 34);
  uint32_t s = seed | 1u;
  for (int64_t b = 0; b < numel / 32; ++b) {
    char d[2];
    std::memcpy(d, &kFp16One, 2);
    out.append(d, 2);
    for (int i = 0; i < 32; ++i) {
      s = s * 1664525u + 1013904223u;
      // A small signed value, deterministic and finite. The scale is exactly
      // 1.0, so the dequantized weight is this integer.
      out.push_back(static_cast<char>(static_cast<int8_t>((s >> 24) % 7) - 3));
    }
  }
  return out;
}

// F32 payload for a tensor of `numel` elements.
std::string F32Zeros(int64_t numel) {
  return std::string(static_cast<size_t>(numel) * 4, '\0');
}

int64_t Prod(const std::vector<uint64_t>& dims) {
  int64_t n = 1;
  for (uint64_t d : dims) n *= static_cast<int64_t>(d);
  return n;
}

// `GgufModelBuilder::AddTensor` takes dims in the FILE's ggml `ne` order, which
// is the reverse of the `[N, K]` torch order `GgufTensorInfo::shape` reports.
// So a `[N, K]` matmul weight is added as `{K, N}`.
void AddF32(gguf_test::GgufModelBuilder& b, const std::string& name,
            const std::vector<uint64_t>& ne) {
  b.AddTensor(name, ne, /*ggml_type=*/0, F32Zeros(Prod(ne)));
}

void AddQ8(gguf_test::GgufModelBuilder& b, const std::string& name,
           const std::vector<uint64_t>& ne, uint32_t seed) {
  b.AddTensor(name, ne, /*ggml_type=*/8, Q8_0Blocks(Prod(ne), seed));
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// A COMPLETE `glm-dsa` file: header, tokenizer, and every tensor this port
// claims plus the multi-token-prediction tail it drops.
//
// `extra` and `omit` exist for the negative cases, which are the half of this
// suite that proves the accounting is a gate rather than a decoration.
std::string BuildCompleteGlmDsa(const std::string& extra_tensor = "",
                                bool expand_one_tower = false) {
  gguf_test::GgufModelBuilder b;
  const std::string p = "glm-dsa.";
  b.AddKv(gguf_test::StrKv("general.architecture", "glm-dsa"));
  b.AddKv(gguf_test::U32Kv(p + "block_count", kBackbone + 1));
  b.AddKv(gguf_test::U32Kv(p + "nextn_predict_layers", 1));
  b.AddKv(gguf_test::U32Kv(p + "embedding_length", kHidden));
  // Small, unlike the real file's 1,048,576: the rope cos/sin cache is
  // `[max_position_embeddings, qk_rope_head_dim]` and the real one is 134 MB.
  b.AddKv(gguf_test::U32Kv(p + "context_length", 256));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count", kHeads));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(gguf_test::U32Kv(p + "feed_forward_length", kInter));
  b.AddKv(gguf_test::F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::F32Kv(p + "rope.freq_base", 8e6f));
  b.AddKv(gguf_test::U32Kv(p + "attention.kv_lora_rank", kKvLora));
  b.AddKv(gguf_test::U32Kv(p + "rope.dimension_count", kQkRope));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length", kKvLora + kQkRope));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length_mla", kQkHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.value_length_mla", kVHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.q_lora_rank", kQLora));
  b.AddKv(gguf_test::U32Kv(p + "expert_count", kExperts));
  b.AddKv(gguf_test::U32Kv(p + "expert_used_count", 2));
  b.AddKv(gguf_test::U32Kv(p + "expert_feed_forward_length", kMoeInter));
  b.AddKv(gguf_test::U32Kv(p + "expert_shared_count", 1));
  b.AddKv(gguf_test::U32Kv(p + "leading_dense_block_count", kLeadingDense));
  b.AddKv(gguf_test::F32Kv(p + "expert_weights_scale", 2.5f));
  b.AddKv(gguf_test::BoolKv(p + "expert_weights_norm", true));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.head_count", kIdxHeads));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.key_length", kIdxHead));
  b.AddKv(gguf_test::U32Kv(p + "attention.indexer.top_k", 64));
  // full, shared, full, shared.
  b.AddKv(gguf_test::BoolArrayKv(p + "attention.indexer.types",
                                 std::vector<bool>{true, false, true, false}));
  {
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.model", "gpt2"));
    b.AddKv(gguf_test::StrKv("tokenizer.ggml.pre", "glm4"));
    std::vector<std::string> toks;
    std::vector<int32_t> types;
    for (int i = 0; i < kVocab; ++i) {
      toks.push_back("t" + std::to_string(i));
      types.push_back(1);
    }
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens", toks));
    b.AddKv(gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.merges",
                                  std::vector<std::string>{}));
  }

  // ── model level ──
  AddF32(b, "token_embd.weight", {kHidden, kVocab});
  AddF32(b, "output_norm.weight", {kHidden});
  AddF32(b, "output.weight", {kHidden, kVocab});

  // ── every block, backbone and MTP tail ──
  for (int64_t l = 0; l <= kBackbone; ++l) {
    AddF32(b, Blk(l, "attn_norm.weight"), {kHidden});
    AddF32(b, Blk(l, "ffn_norm.weight"), {kHidden});
    AddF32(b, Blk(l, "attn_q_a.weight"), {kHidden, kQLora});
    AddF32(b, Blk(l, "attn_q_a_norm.weight"), {kQLora});
    AddF32(b, Blk(l, "attn_q_b.weight"), {kQLora, kHeads * kQkHead});
    AddF32(b, Blk(l, "attn_kv_a_mqa.weight"), {kHidden, kKvLora + kQkRope});
    AddF32(b, Blk(l, "attn_kv_a_norm.weight"), {kKvLora});
    AddF32(b, Blk(l, "attn_k_b.weight"), {kQkNope, kKvLora, kHeads});
    AddF32(b, Blk(l, "attn_v_b.weight"), {kKvLora, kVHead, kHeads});
    AddF32(b, Blk(l, "attn_output.weight"), {kHeads * kVHead, kHidden});
    // The conversion broadcasts these onto every block; the loader reads the
    // schedule and drops the surplus.
    AddF32(b, Blk(l, "indexer.attn_q_b.weight"), {kQLora, kIdxHeads * kIdxHead});
    AddF32(b, Blk(l, "indexer.attn_k.weight"), {kHidden, kIdxHead});
    AddF32(b, Blk(l, "indexer.k_norm.weight"), {kIdxHead});
    AddF32(b, Blk(l, "indexer.k_norm.bias"), {kIdxHead});
    AddF32(b, Blk(l, "indexer.proj.weight"), {kHidden, kIdxHeads});

    if (l < kLeadingDense) {
      AddF32(b, Blk(l, "ffn_gate.weight"), {kHidden, kInter});
      AddF32(b, Blk(l, "ffn_up.weight"), {kHidden, kInter});
      AddF32(b, Blk(l, "ffn_down.weight"), {kInter, kHidden});
    } else {
      AddF32(b, Blk(l, "ffn_gate_inp.weight"), {kHidden, kExperts});
      AddF32(b, Blk(l, "exp_probs_b.bias"), {kExperts});
      const bool expand_this = expand_one_tower && l == kLeadingDense;
      if (expand_this) {
        // F32 has no keep-quant residency, so this tower expands and the
        // loader must refuse it by name.
        AddF32(b, Blk(l, "ffn_gate_exps.weight"),
               {kHidden, kMoeInter, kExperts});
      } else {
        AddQ8(b, Blk(l, "ffn_gate_exps.weight"),
              {kHidden, kMoeInter, kExperts}, 11u + static_cast<uint32_t>(l));
      }
      AddQ8(b, Blk(l, "ffn_up_exps.weight"), {kHidden, kMoeInter, kExperts},
            41u + static_cast<uint32_t>(l));
      AddQ8(b, Blk(l, "ffn_down_exps.weight"), {kMoeInter, kHidden, kExperts},
            71u + static_cast<uint32_t>(l));
      AddF32(b, Blk(l, "ffn_gate_shexp.weight"), {kHidden, kMoeInter});
      AddF32(b, Blk(l, "ffn_up_shexp.weight"), {kHidden, kMoeInter});
      AddF32(b, Blk(l, "ffn_down_shexp.weight"), {kMoeInter, kHidden});
    }
  }
  // The MTP tail's own four tensors, on top of the full block above — the real
  // file's `blk.78` carries both.
  AddF32(b, Blk(kBackbone, "nextn.eh_proj.weight"), {kHidden, 2 * kHidden});
  AddF32(b, Blk(kBackbone, "nextn.enorm.weight"), {kHidden});
  AddF32(b, Blk(kBackbone, "nextn.hnorm.weight"), {kHidden});
  AddF32(b, Blk(kBackbone, "nextn.shared_head_norm.weight"), {kHidden});

  if (!extra_tensor.empty()) AddF32(b, extra_tensor, {kHidden});
  return b.Build();
}

std::string RefusalOf(const std::function<void()>& fn, const char* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  FAIL_CHECK("expected a refusal from " << what << ", got none");
  return {};
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// The load, through the production entry point.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: a complete `glm-dsa` GGUF loads through LoadedEngine::FromModelDir") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  vllm::entrypoints::EngineParams params;
  // THE REACHABILITY ASSERTION. This is the entry point a user arrives through;
  // deleting the `kGgufArchArms` row, the `REGISTER_VLLM_MODEL` line or the GGUF
  // branch of `LoadGlmMoeDsaForCausalLM` all red this.
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine;
  REQUIRE_NOTHROW(engine = vllm::entrypoints::LoadedEngine::FromModelDir(
                      f.path(), params));
  REQUIRE(engine != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// What the loader produced. Driven through the same file, read directly so the
// weights can be inspected.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: the loaded weights carry the schedule, the towers and the accounting") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const vllm::GlmMoeDsaWeights w =
      vllm::LoadGlmMoeDsaFromGguf(g, c, /*policy=*/nullptr);

  // The backbone, not the block count.
  REQUIRE(static_cast<int64_t>(w.layers.size()) == kBackbone);
  CHECK(w.params.num_hidden_layers == kBackbone);
  CHECK(w.params.num_nextn_predict_layers == 1);

  // ── the accounting closes, and the MTP tail is counted rather than ignored ──
  CHECK(w.file_tensors == static_cast<int64_t>(g.Tensors().size()));
  CHECK(w.accounted_tensors + w.mtp_block_tensors_dropped == w.file_tensors);
  CHECK(w.mtp_block_tensors_dropped > 0);
  // Spec D3: the broadcast indexer surplus is the `shared` layers x 5.
  CHECK(w.broadcast_indexer_tensors_dropped == kSharedLayers * 5);

  // ── the heterogeneous schedule reached the WEIGHTS, not only the config ──
  // A `full` layer carries its indexer; a `shared` layer carries none, because
  // it runs no indexer at all and attends through the preceding full layer's
  // selection. This is the assertion that would fail if the loader believed the
  // file (which ships indexer weights on every block) instead of the schedule.
  int64_t with_indexer = 0;
  for (int64_t l = 0; l < kBackbone; ++l) {
    const bool full = w.params.indexer_types[static_cast<size_t>(l)] ==
                      vllm::GlmMoeDsaIndexerKind::kFull;
    const vllm::GlmMoeDsaIndexerWeights& ix = w.layers[static_cast<size_t>(l)].attn.indexer;
    CAPTURE(l);
    CHECK(ix.Empty() == !full);
    if (full) {
      ++with_indexer;
      // The bias is what makes this a LayerNorm rather than an RMSNorm.
      CHECK(!ix.k_norm_bias.Empty());
      CHECK(ix.wq_b.shape[0] == kIdxHeads * kIdxHead);
      CHECK(ix.wq_b.shape[1] == kQLora);
      CHECK(ix.weights_proj.shape[0] == kIdxHeads);
      // `weights_proj` decides the selection outright and stays f32.
      CHECK(ix.weights_proj.dtype == vt::DType::kF32);
    }
  }
  CHECK(with_indexer == kFullLayers);

  // ── the dense/MoE layout ──
  for (int64_t l = 0; l < kBackbone; ++l) {
    CAPTURE(l);
    const vllm::GlmMoeDsaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    CHECK(lw.is_moe == (l >= kLeadingDense));
    CHECK(lw.dense.Empty() == lw.is_moe);
    CHECK(lw.moe.Empty() == !lw.is_moe);
    if (!lw.is_moe) {
      // The leading dense block uses `intermediate_size`, not
      // `moe_intermediate_size`.
      CHECK(lw.dense.gate_proj.shape[0] == kInter);
    } else {
      // ── THE TOWERS STAYED IN BLOCKS ──
      // This is the assertion the whole residency plan rests on. A tower that
      // dequantized to bf16 here would be 4x the bytes and, worse, would take
      // the whole model out of the streaming lane, because
      // `GgufExpertTowersReachSlotLane` is all-or-nothing.
      CHECK(lw.moe.gate_exps.dtype == vt::DType::kQ8_0);
      CHECK(lw.moe.up_exps.dtype == vt::DType::kQ8_0);
      CHECK(lw.moe.down_exps.dtype == vt::DType::kQ8_0);
      // Held flat as [E*out, K], which is the shape the streaming lane's only
      // gated client uses.
      CHECK(lw.moe.gate_exps.shape[0] == kExperts * kMoeInter);
      CHECK(lw.moe.gate_exps.shape[1] == kHidden);
      CHECK(lw.moe.down_exps.shape[0] == kExperts * kHidden);
      CHECK(lw.moe.down_exps.shape[1] == kMoeInter);
      // The router and its noaux_tc bias are f32: the bias is ADDED to the
      // sigmoid scores before a top-k, and a top-k is a discrete outcome that no
      // tolerance bounds.
      CHECK(lw.moe.router.dtype == vt::DType::kF32);
      CHECK(lw.moe.e_score_correction_bias.dtype == vt::DType::kF32);
      // The shared expert is `moe_intermediate_size * n_shared_experts`, NOT
      // `intermediate_size`.
      CHECK(lw.moe.shared.gate_proj.shape[0] == kMoeInter);
    }
  }

  // ── model level ──
  CHECK(w.embed_tokens.shape[0] == kVocab);
  CHECK(w.embed_tokens.shape[1] == kHidden);
  // The tie is read off the FILE. This fixture ships `output.weight`, so the
  // model is untied and `lm_head` is populated.
  CHECK(!w.lm_head.Empty());
  CHECK(!w.rope_cos_sin_cache.Empty());
  CHECK(w.rope_cos_sin_cache.shape[1] == kQkRope);
}

// ─────────────────────────────────────────────────────────────────────────────
// The two refusals, each proven by making the file violate it.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: a tensor this port does not claim refuses the load by name") {
  // A tensor that is neither claimed nor in the MTP tail. Without the
  // accounting this loads clean and the extra weight is silently absent from
  // the model, which no token gate on this row could ever see (spec O1).
  gguf_test::TempFile f(BuildCompleteGlmDsa("blk.2.attn_sinks.weight"));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const std::string msg = RefusalOf(
      [&] { (void)vllm::LoadGlmMoeDsaFromGguf(g, c, nullptr); },
      "LoadGlmMoeDsaFromGguf");
  CHECK(msg.find("blk.2.attn_sinks.weight") != std::string::npos);
  CHECK(msg.find("silently absent") != std::string::npos);
}

TEST_CASE("glm-dsa W7: an expert tower that would EXPAND refuses the load by name") {
  gguf_test::TempFile f(BuildCompleteGlmDsa("", /*expand_one_tower=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const std::string msg = RefusalOf(
      [&] { (void)vllm::LoadGlmMoeDsaFromGguf(g, c, nullptr); },
      "LoadGlmMoeDsaFromGguf");
  CHECK(msg.find("ffn_gate_exps.weight") != std::string::npos);
  // The message must say WHY it matters, because the consequence is not local:
  // one expanded tower takes all of them out of the streaming lane.
  CHECK(msg.find("streaming lane") != std::string::npos);
  CHECK(msg.find("all-or-nothing") != std::string::npos);
}
