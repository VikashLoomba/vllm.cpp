// GLM-5.3 (`GlmMoeDsaForCausalLM`, HF `model_type: glm_moe_dsa`) — the CONFIG
// surface, the `glm-dsa` GGUF arm and the refuse-by-name forward. W2 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHY THIS MODEL HAS ITS OWN PARAMS STRUCT ────────────────────────────────
// Upstream at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98` is a
// literal zero-override subclass: `deepseek_v2.py:1930` is
// `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`, and the only
// behavioural special case is the fp32 router dtype forced on
// `model_type == "glm_moe_dsa"` at `deepseek_v2.py:127`. Sharing
// `DeepseekV2Params` would therefore be the obvious move, and it is the wrong
// one: `ParseDeepseekV2Params` REFUSES any checkpoint carrying `index_topk`
// (`deepseek_v2_weights.cpp:358-364`) and any `quantization_config` (`:365-369`),
// and GLM-5.3 carries both. Relaxing either refusal to let this model through
// turns a wall into a choice for DeepSeek-V2 as well — that tripwire is what
// stops a V3.2 checkpoint silently loading onto a dense-attention forward.
// So GLM-5.3 gets its own struct and DeepSeek-V2's refusals are not touched.
// Recorded in the spec §3.7 W2 Exclusions.
//
// ─── WHAT THIS TRANSLATION UNIT DOES NOT DO ──────────────────────────────────
// No forward math and no weight materialization. `GlmMoeDsaModel::Forward`
// refuses by name and lists every primitive that is missing, the wave that owes
// it and the issue that tracks it, so a reader meets the gap at the call rather
// than discovering it. The GGUF loader is W7's; the safetensors arms are refused
// permanently (spec D1: the published bf16/fp8 checkpoint is 703.74 GiB with no
// streaming loader and no MoE block-fp8 rung, so the quantized arm is the only
// one that can be fed on this fleet).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

// llama.cpp's `general.architecture` for this family, `b10451`
// `src/llama-arch.cpp:85` (`{ LLM_ARCH_GLM_DSA, "glm-dsa" }`). It is the FAMILY
// key, not the HF class name, which is why the config builder below sets
// `architectures` explicitly instead of deriving it from this string.
inline constexpr const char* kGlmMoeDsaGgufArch = "glm-dsa";

// Whether a decoder layer runs its own lightning indexer, or reuses the
// selection of the preceding `kFull` layer. Upstream never stores this as a
// list: it evaluates the rule per layer inside `DeepseekV2MLAAttention.__init__`
// (`deepseek_v2.py:1092-1103`) and builds an indexer at `:1115` when
// `not _skip_topk or is_mtp_layer`. `kShared` is upstream's `_skip_topk == True`.
enum class GlmMoeDsaIndexerKind { kFull, kShared };

// Whether a decoder layer's MLP is the dense one or the 256-expert MoE block.
// Upstream derives this from `first_k_dense_replace` / `moe_layer_freq`
// (`deepseek_v2.py:1214-1218`) and never reads the checkpoint's own
// `mlp_layer_types` — `grep -c mlp_layer_types` over `deepseek_v2.py` at the pin
// is 0. See `ParseGlmMoeDsaParams` for why we read it anyway.
enum class GlmMoeDsaMlpKind { kDense, kSparse };

// The resolved GLM-5.3 configuration. Field-for-field an HF read; no derived
// geometry that a forward would own.
struct GlmMoeDsaParams {
  // --- backbone ---
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;  // BACKBONE depth, excluding the MTP block
  int64_t vocab_size = 0;
  int64_t num_attention_heads = 0;
  int64_t intermediate_size = 0;  // the dense MLP width of the leading layers
  double rms_norm_eps = 0.0;
  int64_t max_position_embeddings = 0;
  double rope_theta = 10000.0;

  // --- MLA geometry (deepseek_v2.py:1040-1074) ---
  int64_t q_lora_rank = 0;
  int64_t kv_lora_rank = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;
  // INTERLEAVED (GPT-J) rope, ALWAYS. Upstream passes `is_neox_style=False`
  // unconditionally at `deepseek_v2.py:1073` and reads no top-level
  // `rope_interleave` for it, so no checkpoint key can move this. It is a field
  // rather than an implicit default because `mla::MlaBlockDims::is_neox_style`
  // ALSO defaults false (`mla_attention.h:136`) — the value is right today by
  // coincidence of two defaults agreeing, and a default that is never read is a
  // value nothing can gate. `ParseGlmMoeDsaParams` writes it; a test asserts it.
  bool is_neox_style = false;
  // The INDEXER's rope, which is a separate decision and IS config-driven:
  // `is_neox_style=not getattr(config, "indexer_rope_interleave", False)`
  // (`deepseek_v2.py:1120`). GLM-5.3 ships `indexer_rope_interleave: true`, so
  // this is false on the published checkpoint and true on a checkpoint that
  // omits the key. The two rope styles are NOT the same field.
  bool indexer_rope_is_neox_style = true;

  // --- MoE (deepseek_v2.py:286-393) ---
  int64_t n_routed_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t n_shared_experts = 0;
  int64_t first_k_dense_replace = 0;
  int64_t moe_layer_freq = 1;
  int64_t n_group = 1;
  int64_t topk_group = 1;
  bool norm_topk_prob = false;
  bool has_e_score_correction_bias = false;  // `topk_method == "noaux_tc"`
  double routed_scaling_factor = 1.0;
  // The router GEMM runs in f32. THIS IS THE ANNOTATED EXCEPTION the dtype
  // polarity requires: `_get_moe_router_dtype` (`deepseek_v2.py:123-133`) forces
  // `torch.float32` on `model_type == "glm_moe_dsa"` at `:127`, BEFORE the
  // generic `moe_router_dtype == "float32"` branch at `:131`. GLM-5.3 also
  // declares `moe_router_dtype: float32`, so the special case is redundant on
  // this checkpoint and still fires first. Everything else on the model path
  // inherits the checkpoint's bf16.
  bool router_dtype_is_f32 = true;

  // --- the DSA lightning indexer (deepseek_v2.py:803-842) ---
  int64_t index_topk = 0;
  int64_t index_n_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk_freq = 1;
  int64_t index_skip_topk_offset = 2;

  // --- MTP, skipped rather than implemented (spec O5) ---
  int64_t num_nextn_predict_layers = 0;
  bool index_share_for_mtp_iteration = false;

  // --- the per-layer schedules, both `num_hidden_layers` long ---
  std::vector<GlmMoeDsaIndexerKind> indexer_types;
  std::vector<GlmMoeDsaMlpKind> mlp_layer_types;

  // Upstream's own rule, `deepseek_v2.py:1214-1218`, mirrored exactly as
  // `DeepseekV2Params::is_moe_layer` (`deepseek_v2.h:125-128`) mirrors it.
  bool is_moe_layer(int64_t layer) const {
    return n_routed_experts > 0 && layer >= first_k_dense_replace &&
           (moe_layer_freq <= 1 || layer % moe_layer_freq == 0);
  }
  // The one latent row the paged cache stores per token.
  int64_t mla_kv_head_size() const { return kv_lora_rank + qk_rope_head_dim; }
};

// Upstream's DERIVED indexer schedule, `deepseek_v2.py:1097-1101`, evaluated for
// every backbone layer instead of once per constructed layer:
//
//     _skip_topk = max(layer_id - offset + 1, 0) % freq != 0
//
// This is the ONLY source of the schedule this port synthesizes. llama.cpp
// survives a file that states nothing by falling back to a hardcoded 78-entry
// table (`b10451:src/models/glm-dsa.cpp:6-27`, `GLM_5_2_DEFAULT_INDEXER_TYPES`)
// that is bit-identical to GLM-5.3's own list; we deliberately do not copy it,
// because a constant that happens to be right is the shape that silently becomes
// wrong on the next checkpoint (spec D3).
std::vector<GlmMoeDsaIndexerKind> DeriveGlmMoeDsaIndexerSchedule(
    int64_t num_hidden_layers, int64_t index_topk_freq,
    int64_t index_skip_topk_offset);

// Upstream's dense/MoE layout for the same layers, `deepseek_v2.py:1214-1218`.
std::vector<GlmMoeDsaMlpKind> DeriveGlmMoeDsaMlpSchedule(
    int64_t num_hidden_layers, int64_t first_k_dense_replace,
    int64_t moe_layer_freq, int64_t n_routed_experts);

// Resolve `GlmMoeDsaParams` from an `HfConfig`. Throws `std::runtime_error` with
// a precise message on every field this port cannot serve. Pure/host — testable
// without a checkpoint.
GlmMoeDsaParams ParseGlmMoeDsaParams(const HfConfig& config);

// ─── W4: the heterogeneous per-layer MLA schedule ────────────────────────────
// The `mla::MlaBlockDims` for ONE backbone layer, with the indexer geometry
// present on a `kFull` layer and `skip_topk` set on a `kShared` one. This is the
// only place the two-way split becomes block geometry, and it reads
// `p.indexer_types` — the schedule `ParseGlmMoeDsaParams` resolved from the
// checkpoint — rather than re-deriving the rule. On GLM-5.3 it puts an indexer
// on 21 of the 78 backbone layers; the 22nd is the MTP block, which upstream
// forces full at `deepseek_v2.py:1110-1115` and which this row skips (spec O5).
//
// Throws `std::out_of_range` when `layer` is outside `[0, num_hidden_layers)`,
// and whatever `mla::MlaBlockDims::Validate` throws when the resolved geometry
// is one the MLA block refuses.
mla::MlaBlockDims GlmMoeDsaMlaBlockDims(const GlmMoeDsaParams& p, int64_t layer);

// Every backbone layer's dims, in layer order, each already `Validate()`d. The
// caller allocates ONE `mla::MlaSharedSelection` for the model and hands it to
// every layer in this order, which is what makes a `kShared` layer read the
// selection its owning `kFull` layer wrote (`mla.py:180`).
std::vector<mla::MlaBlockDims> GlmMoeDsaMlaSchedule(const GlmMoeDsaParams& p);

// How many of `p.indexer_types` are `kFull`. Named because the split is the
// wave's headline number and a reader should not have to count a vector.
int64_t GlmMoeDsaFullIndexerLayerCount(const GlmMoeDsaParams& p);

// The registry's config hook. The resolve IS the validation.
void ParseGlmMoeDsaConfig(const HfConfig& config);

// True when `gguf` carries `general.architecture == kGlmMoeDsaGgufArch`.
bool IsGlmMoeDsaGguf(const GgufFile& gguf);

// Synthesize an HF-shaped config from a `glm-dsa` GGUF header and hand it to the
// SAME `ParseGlmMoeDsaParams` a `config.json` descends through, so both sources
// meet one validator (the `Glm5NextHfConfigFromGguf` arrangement).
HfConfig GlmMoeDsaHfConfigFromGguf(const GgufFile& gguf);

// One MLA group, `kv_lora_rank + qk_rope_head_dim` wide. The indexer's own
// 132 B/token side cache is a SECOND group upstream
// (`DeepseekV32IndexerCache`, `deepseek_v2.py:696-701`) and does not exist here
// yet — spec O4, owed by W5 / `KV-DSV4-MULTICACHE`
// ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)).
v1::KVCacheConfig MakeGlmMoeDsaKVCache(const HfConfig& config, int block_size,
                                       int num_blocks);

// The weights this model would carry. W2 loads none: the struct exists so the
// registry's `LoadedModel` subclass has something to hold and so the shape of
// what W7 fills is visible.
struct GlmMoeDsaWeights {
  GlmMoeDsaParams params;
};

// The forward, REFUSE-by-name. Both entry points list every missing primitive.
class GlmMoeDsaModel {
 public:
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const GlmMoeDsaWeights& weights,
                                    vt::Queue& queue,
                                    const std::vector<int32_t>& logits_indices);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GlmMoeDsaWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices);
};

// The refusal text both entry points raise, exposed so a test can assert the
// message NAMES each missing primitive rather than re-spelling it.
const char* GlmMoeDsaForwardRefusal();

// The refusal the safetensors arm raises (spec D1).
const char* GlmMoeDsaSafetensorsRefusal();

}  // namespace vllm
