// dots3-note VISION tower — the DENSE arm (W6a, #2512). See the header for the
// complete port map and for the ONE deliberate RMSNorm rounding difference.
//
// Every anchor in this file was read in `~/_git/vllm` at `9035151d6`.
#include "vllm/model_executor/models/dots3_note_vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"  // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

using vt::DType;
using vt::Tensor;
using namespace dense_attn;  // Dev / DBuf / MakeTensor / ResidentWeight

namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadMergedBf16RawNK;

// ── config resolution ───────────────────────────────────────────────────────
//
// The reader mirrors `dots3_note.cpp`'s: a MISSING key that upstream's
// `DotsMoEVitConfig.__init__` defaults gets that default; a key that is present
// and of the wrong TYPE refuses by name, because a silently-ignored value is
// the shape of every §4 trap on this row.
int64_t ReadIntOr(const nlohmann::json& j, const char* key, int64_t fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  VT_CHECK(it->is_number_integer() || it->is_number_unsigned(),
           std::string("dots3-note vision_config: '") + key +
               "' must be an integer (DotsMoEVitConfig, vision.py:27-105 @ "
               "9035151d6), got " + it->dump());
  return it->get<int64_t>();
}

double ReadNumOr(const nlohmann::json& j, const char* key, double fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  VT_CHECK(it->is_number(),
           std::string("dots3-note vision_config: '") + key +
               "' must be a number, got " + it->dump());
  return it->get<double>();
}

bool ReadBoolOr(const nlohmann::json& j, const char* key, bool fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  VT_CHECK(it->is_boolean(),
           std::string("dots3-note vision_config: '") + key +
               "' must be a boolean, got " + it->dump());
  return it->get<bool>();
}

std::string ReadStrOr(const nlohmann::json& j, const char* key,
                      const std::string& fallback) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  VT_CHECK(it->is_string(),
           std::string("dots3-note vision_config: '") + key +
               "' must be a string, got " + it->dump());
  return it->get<std::string>();
}

std::string ShapeOf(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

void RequireVisionShape(const OwnedTensor& t, const std::string& name,
                        const std::vector<int64_t>& want) {
  std::vector<int64_t> got(t.shape, t.shape + t.rank);
  VT_CHECK(got == want,
           "dots3-note vision tower: '" + name + "' ships " + ShapeOf(got) +
               ", the config implies " + ShapeOf(want) +
               ". Refusing rather than reading a differently-shaped weight. "
               "See .agents/specs/dots3-note.md §4.11 and issue #2512.");
  // THE MEMORY FORMAT, asserted rather than assumed (porting.md). Every dense
  // vision tensor is BF16 in the released index; a widened one still passes
  // every shape check and every token gate while moving twice the bytes, and
  // this row's W2 F1 fixture row already proves a re-typed tensor fires.
  VT_CHECK(t.dtype == DType::kBF16,
           "dots3-note vision tower: '" + name +
               "' is not BF16 after load. The released checkpoint carries the "
               "whole dense vision arm in BF16; a wider store is a memory-format "
               "defect a token gate cannot see (porting.md).");
}

}  // namespace

Dots3NoteVisionParams ParseDots3NoteVisionParams(const HfConfig& config) {
  Dots3NoteVisionParams v;
  const auto it = config.raw.find("vision_config");
  if (it == config.raw.end() || it->is_null()) return v;  // present == false
  VT_CHECK(it->is_object(),
           "dots3-note: `vision_config` must be an object, got " + it->dump());
  const nlohmann::json& j = *it;
  v.present = true;

  v.embed_dim = ReadIntOr(j, "embed_dim", 1536);
  // From the LANGUAGE config, not from `j`. See the field's comment: this is
  // the width `EncodeMmDots3NoteForCausalLM` compares `adapter_out_dim`
  // against, and reading `vision_config`'s own copy of it instead would leave
  // the refusal answering a different question from the route.
  v.text_hidden_size = config.hidden_size;
  v.intermediate_size = ReadIntOr(j, "intermediate_size", 4224);
  v.moe_intermediate_size = ReadIntOr(j, "moe_intermediate_size", 2112);
  v.num_hidden_layers = ReadIntOr(j, "num_hidden_layers", 42);
  v.num_attention_heads = ReadIntOr(j, "num_attention_heads", 24);
  v.num_channels = ReadIntOr(j, "num_channels", 3);
  v.patch_size = ReadIntOr(j, "patch_size", 14);
  v.spatial_merge_size = ReadIntOr(j, "spatial_merge_size", 2);
  v.temporal_patch_size = ReadIntOr(j, "temporal_patch_size", 1);
  v.rms_norm_eps = ReadNumOr(j, "rms_norm_eps", 1e-5);
  v.use_bias = ReadBoolOr(j, "use_bias", false);
  v.use_qk_norm = ReadBoolOr(j, "use_qk_norm", true);
  v.is_causal = ReadBoolOr(j, "is_causal", false);
  v.post_norm = ReadBoolOr(j, "post_norm", true);
  v.pre_pixel_shuffle = ReadBoolOr(j, "pre_pixel_shuffle", false);
  v.capacity_factor = ReadNumOr(j, "capacity_factor", 2.0);
  v.router_scoring_func = ReadStrOr(j, "router_scoring_func", "sigmoid");
  v.router_scale = ReadNumOr(j, "router_scale", 1.0);
  v.adapter_type = ReadStrOr(j, "adapter_type", "pixel_shuffle_mlp");
  v.adapter_in_dim = ReadIntOr(j, "adapter_in_dim", 1536);
  v.adapter_out_dim = ReadIntOr(j, "adapter_out_dim", 2048);
  v.adapter_merge_size = ReadIntOr(j, "adapter_merge_size", 2);

  const auto pyr = j.find("pyramid_num_routed");
  if (pyr != j.end() && !pyr->is_null()) {
    VT_CHECK(pyr->is_array(),
             "dots3-note vision_config: `pyramid_num_routed` must be a list "
             "(vision.py:90 @ 9035151d6), got " + pyr->dump());
    for (const nlohmann::json& e : *pyr) {
      VT_CHECK(e.is_number_integer(),
               "dots3-note vision_config: `pyramid_num_routed` entries must be "
               "integers, got " + e.dump());
      v.pyramid_num_routed.push_back(e.get<int64_t>());
    }
  }

  // Upstream's own validation, mirrored: `adapter_type` is checked in the
  // constructor and raises there (`vision.py:98-102`). Anything else is a
  // config this port cannot represent AT ALL, so it refuses at PARSE rather
  // than at load — an unknown adapter is not owed to a later brick.
  VT_CHECK(v.adapter_type == "pixel_shuffle_mlp" ||
               v.adapter_type == "patch_merger",
           "dots3-note vision_config: adapter_type must be 'pixel_shuffle_mlp' "
           "or 'patch_merger' (vision.py:98-102 @ 9035151d6), got '" +
               v.adapter_type + "'");

  // Geometry that cannot be true of any dots3-note tower, checked where the key
  // name is still in hand.
  VT_CHECK(v.num_attention_heads > 0 && v.embed_dim > 0 &&
               v.embed_dim % v.num_attention_heads == 0,
           "dots3-note vision_config: embed_dim " +
               std::to_string(v.embed_dim) +
               " is not a whole multiple of num_attention_heads " +
               std::to_string(v.num_attention_heads));
  VT_CHECK(v.head_dim() % 2 == 0,
           "dots3-note vision_config: head_dim " +
               std::to_string(v.head_dim()) +
               " is odd; the 2-D vision RoPE splits it into a height half and a "
               "width half (vision.py:503-504 @ 9035151d6)");
  VT_CHECK(v.spatial_merge_size > 0 && v.adapter_merge_size > 0,
           "dots3-note vision_config: spatial_merge_size and adapter_merge_size "
           "must be positive");
  VT_CHECK(static_cast<int64_t>(v.pyramid_num_routed.size()) == 0 ||
               static_cast<int64_t>(v.pyramid_num_routed.size()) >=
                   v.num_hidden_layers,
           "dots3-note vision_config: `pyramid_num_routed` has " +
               std::to_string(v.pyramid_num_routed.size()) + " entries for " +
               std::to_string(v.num_hidden_layers) +
               " blocks. Upstream indexes it by layer number "
               "(vision.py:346-350 @ 9035151d6), so a short list would make the "
               "tail silently dense.");
  return v;
}

std::string Dots3NoteVisionRefusal(
    const Dots3NoteVisionParams& v, const std::string& quant_method,
    const std::vector<int64_t>& weight_block_size) {
  if (!v.present) {
    return "the checkpoint's `config.json` carries no `vision_config`, so this "
           "load has no vision tower to build (multimodal.py:113-118 @ "
           "9035151d6)";
  }
  // ORDER IS BRICK ORDER, and the message names ONE thing: a reader is told
  // what to build, not that something is missing.
  if (!weight_block_size.empty() || quant_method == "fp8") {
    return "the checkpoint is BLOCKWISE-QUANTIZED (`quantization_config"
           ".weight_block_size`), and the vision tower's FP32-scale FP8 arm "
           "(`MoESwiGLUFFNFP8`, vision.py:222-297 @ 9035151d6, and "
           "nvidia/vision_moe.py's `note_vision_fused_moe_fp8`) is W9";
  }
  const int64_t moe = v.num_moe_blocks();
  if (moe > 0) {
    int64_t first = -1;
    for (int64_t i = 0; i < v.num_hidden_layers; ++i) {
      if (v.is_moe_block(i)) { first = i; break; }
    }
    return "vision block " + std::to_string(first) + " is a PYRAMID MoE block (" +
           std::to_string(v.pyramid_num_routed[static_cast<size_t>(first)]) +
           " routed experts), and " + std::to_string(moe) + " of the tower's " +
           std::to_string(v.num_hidden_layers) +
           " blocks are. The MoE ViT — `mlp.gate_weight` + `mlp.router_bias`, "
           "sigmoid scoring and the capacity-factor top-k (`MoESwiGLUFFN`, "
           "vision.py:139-219 @ 9035151d6) — is W6b. W6a ships the DENSE arm "
           "only, and refusing is what stops this port from serving a tower "
           "whose pyramid it silently skipped on a row that has no oracle "
           "(spec §6.4)";
  }
  if (v.adapter_type != "patch_merger") {
    return "`adapter_type` is '" + v.adapter_type +
           "'. W6a implements `patch_merger` (`PatchMergerAdapter`, "
           "vision.py:441-472 @ 9035151d6), which is what the released "
           "`dots-studio/dots3-note-prev` selects. `pixel_shuffle_mlp` is a "
           "DIFFERENT token order from the same pixels and a DIFFERENT state "
           "dict (`proj.0`/`proj.1`/`proj.3` against `ln_q`/`mlp.0`/`mlp.2`), "
           "so it is refused rather than mapped onto this one. W6b";
  }
  if (!v.post_norm) {
    return "`post_norm` is false, so upstream builds no `post_trunk_norm` "
           "(vision.py:513-514 @ 9035151d6). Every published dots3-note tower "
           "sets it true and W6a's arm assumes it. W6b";
  }
  if (v.use_bias) {
    return "`use_bias` is true, so `attn.qkv`, `attn.proj` and every "
           "`mlp.fc*` would carry a bias the released checkpoint does not ship "
           "(vision.py:143-144 of vision_attention.py @ 9035151d6). W6b";
  }
  if (!v.use_qk_norm) {
    return "`use_qk_norm` is false, so attention would run with no per-head "
           "`q_norm`/`k_norm` (vision_attention.py:145-147 @ 9035151d6) while "
           "the checkpoint ships both. W6b";
  }
  if (v.is_causal) {
    return "`is_causal` is true. The dots3-note ViT attends bidirectionally "
           "(vision_attention.py:118 @ 9035151d6) and W6a's attention call "
           "passes `causal=false`. W6b";
  }
  if (v.temporal_patch_size != 1) {
    return "`temporal_patch_size` is " + std::to_string(v.temporal_patch_size) +
           ", which is the VIDEO arm: `DotsPatchEmbed.forward` takes "
           "`[:, :, 0]` of the temporal axis (vision.py:317-325 @ 9035151d6) "
           "and the multi-frame `cu_seqlens` builder is a different one "
           "(vision.py:613-624). Video is W7";
  }
  if (v.adapter_in_dim != v.embed_dim) {
    return "`adapter_in_dim` " + std::to_string(v.adapter_in_dim) +
           " is not the tower's `embed_dim` " + std::to_string(v.embed_dim) +
           ", so `ln_q` would normalize a width the trunk does not produce "
           "(vision.py:466 @ 9035151d6). W6b";
  }
  // ── THE TWO THE ENCODER ASSERTS ON ─────────────────────────────────────────
  //
  // These name NO brick, because nothing is owed: they are configs no
  // dots3-note tower can be served under. They are here because a refusal
  // predicate that is a strict SUBSET of the request-time asserts is not a
  // refusal. `EncodeMmDots3NoteForCausalLM` makes both comparisons again inside
  // the ENGINE's busy loop, where a throw sets `AsyncLLM::errored_`
  // permanently (`async_llm.cpp:584-601`) — the server then starts, serves
  // text, 500s the first image, and answers every LATER request, text ones
  // included, with "request submitted to a stopped AsyncLLM". Asking here turns
  // the same answer into a REFUSING seam: HTTP 400, text path untouched.
  //
  // The refusal and the route must be the SAME predicate. This is the row's
  // second recurrence of that finding (the first is the sparse-routing entry
  // under `## Owed`), and it is what retired the tautology that used to sit
  // where the second check now is: `adapter_merge_size**2 * adapter_in_dim !=
  // merged_dim()` compared `merged_dim()` against its own definition.
  if (v.adapter_out_dim != v.text_hidden_size) {
    return "`adapter_out_dim` " + std::to_string(v.adapter_out_dim) +
           " is not the TEXT tower's `hidden_size` " +
           std::to_string(v.text_hidden_size) +
           ", so `adapter.mlp.2` emits rows that cannot be scattered into the "
           "prompt at all (vision.py:461 @ 9035151d6 against "
           "`config.hidden_size`). This is the comparison "
           "`EncodeMmDots3NoteForCausalLM` makes on a served request";
  }
  if (v.adapter_merge_size != v.spatial_merge_size) {
    return "`adapter_merge_size` " + std::to_string(v.adapter_merge_size) +
           " is not `spatial_merge_size` " +
           std::to_string(v.spatial_merge_size) +
           ". The PROMPT side expands one image marker into "
           "`prod(grid) // spatial_merge_size**2` placeholders "
           "(multimodal.py:151-155 @ 9035151d6, and this port's "
           "`Dots3NoteProcessorConfig::merge_size`, which is read from that "
           "key) while the ADAPTER folds `adapter_merge_size**2` trunk tokens "
           "into each emitted row (vision.py:441-449). Upstream keeps the two "
           "as independent keys with independent defaults, so a checkpoint can "
           "carry them disagreeing; serving it would either leave the trunk "
           "length not grouping into whole merger rows, or emit a row count "
           "the placeholder span cannot hold";
  }
  return "";
}

std::string Dots3NoteVisionRefusalFor(const HfConfig& config) {
  // The `quantization_config` block is read HERE rather than through
  // `ParseDots3NoteParams`, because this overload must answer for a checkpoint
  // whose LANGUAGE config the caller has not validated — the chat seam runs at
  // server start and holds only a path.
  std::string quant_method;
  std::vector<int64_t> weight_block_size;
  const auto qc = config.raw.find("quantization_config");
  if (qc != config.raw.end() && qc->is_object()) {
    const auto qm = qc->find("quant_method");
    if (qm != qc->end() && qm->is_string()) quant_method = qm->get<std::string>();
    const auto wb = qc->find("weight_block_size");
    if (wb != qc->end() && wb->is_array()) {
      for (const nlohmann::json& e : *wb)
        if (e.is_number_integer()) weight_block_size.push_back(e.get<int64_t>());
    }
  }
  return Dots3NoteVisionRefusal(ParseDots3NoteVisionParams(config), quant_method,
                                weight_block_size);
}

std::vector<Dots3NoteTensor> EnumerateDots3NoteVisionTensors(
    const Dots3NoteVisionParams& v) {
  std::vector<Dots3NoteTensor> out;
  if (!v.present) return out;
  const std::string p = "vision_encoder.";
  out.push_back({p + "patch_embed.proj.weight", "vision.patch_embed"});
  out.push_back({p + "patch_embed.proj.bias", "vision.patch_embed"});
  out.push_back({p + "patch_embed.norm.weight", "vision.patch_embed"});
  for (int64_t b = 0; b < v.num_hidden_layers; ++b) {
    if (v.is_moe_block(b)) continue;  // W6b's
    const std::string pre = p + "blocks." + std::to_string(b) + ".";
    out.push_back({pre + "norm_1.weight", "vision.block.norm_1"});
    out.push_back({pre + "norm_2.weight", "vision.block.norm_2"});
    out.push_back({pre + "attn.qkv.weight", "vision.block.attn.qkv"});
    out.push_back({pre + "attn.proj.weight", "vision.block.attn.proj"});
    out.push_back({pre + "attn.q_norm.weight", "vision.block.attn.q_norm"});
    out.push_back({pre + "attn.k_norm.weight", "vision.block.attn.k_norm"});
    out.push_back({pre + "mlp.fc1.weight", "vision.block.mlp.gate"});
    out.push_back({pre + "mlp.fc2.weight", "vision.block.mlp.down"});
    out.push_back({pre + "mlp.fc3.weight", "vision.block.mlp.up"});
  }
  if (v.post_norm) {
    out.push_back({p + "post_trunk_norm.weight", "vision.post_trunk_norm"});
  }
  out.push_back({p + "adapter.ln_q.weight", "vision.adapter.ln_q"});
  out.push_back({p + "adapter.ln_q.bias", "vision.adapter.ln_q"});
  out.push_back({p + "adapter.mlp.0.weight", "vision.adapter.mlp.0"});
  out.push_back({p + "adapter.mlp.0.bias", "vision.adapter.mlp.0"});
  out.push_back({p + "adapter.mlp.2.weight", "vision.adapter.mlp.2"});
  out.push_back({p + "adapter.mlp.2.bias", "vision.adapter.mlp.2"});
  return out;
}

Dots3NoteVisionWeights MaterializeDots3NoteVision(
    const std::vector<SafetensorsFile>& shards,
    const Dots3NoteVisionParams& v) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& f : shards) {
    for (const std::string& n : f.Names()) where.emplace(n, &f);
  }
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(),
             "dots3-note vision tower: tensor not found: " + name);
    return it->second->Get(name);
  };

  const int64_t E = v.embed_dim, I = v.intermediate_size, D = v.head_dim();
  const std::string p = "vision_encoder.";
  Dots3NoteVisionWeights w;

  // `DotsPatchEmbed.proj` is an `nn.Conv2d(C, E, kernel=stride=patch)`
  // (vision.py:308-314), so its weight ships [E, C, p, p]. The forward takes
  // `[:, :, 0]` of a temporal axis of size 1 and then applies a kernel that
  // covers exactly one patch with no overlap, which is a Linear over the
  // flattened patch row — so it is read as [E, C*tp*p*p] here. The shape
  // OVERRIDE is what makes that reinterpretation explicit rather than implied,
  // and the check below is against the ON-DISK rank-4 shape.
  {
    const StTensor& t = get(p + "patch_embed.proj.weight");
    const std::vector<int64_t> want{E, v.num_channels, v.patch_size,
                                    v.patch_size};
    VT_CHECK(t.shape == want,
             "dots3-note vision tower: '" + p +
                 "patch_embed.proj.weight' ships " + ShapeOf(t.shape) +
                 ", the config implies " + ShapeOf(want));
  }
  w.patch_proj_w = LoadBf16Direct(get, p + "patch_embed.proj.weight",
                                  {E, v.patch_row()});
  RequireVisionShape(w.patch_proj_w, p + "patch_embed.proj.weight",
                     {E, v.patch_row()});
  w.patch_proj_b = LoadBf16Direct(get, p + "patch_embed.proj.bias");
  RequireVisionShape(w.patch_proj_b, p + "patch_embed.proj.bias", {E});
  w.patch_norm = LoadBf16Direct(get, p + "patch_embed.norm.weight");
  RequireVisionShape(w.patch_norm, p + "patch_embed.norm.weight", {E});

  for (int64_t b = 0; b < v.num_hidden_layers; ++b) {
    VT_CHECK(!v.is_moe_block(b),
             "dots3-note vision tower: block " + std::to_string(b) +
                 " is a pyramid MoE block and W6a does not load it. "
                 "MaterializeDots3NoteVision was reached with a config "
                 "Dots3NoteVisionRefusal should have refused; that is a caller "
                 "defect, not a checkpoint one.");
    const std::string pre = p + "blocks." + std::to_string(b) + ".";
    Dots3NoteVisionBlockWeights bw;
    bw.norm_1 = LoadBf16Direct(get, pre + "norm_1.weight");
    RequireVisionShape(bw.norm_1, pre + "norm_1.weight", {E});
    bw.norm_2 = LoadBf16Direct(get, pre + "norm_2.weight");
    RequireVisionShape(bw.norm_2, pre + "norm_2.weight", {E});
    bw.qkv = LoadBf16Direct(get, pre + "attn.qkv.weight");
    RequireVisionShape(bw.qkv, pre + "attn.qkv.weight", {3 * E, E});
    bw.proj = LoadBf16Direct(get, pre + "attn.proj.weight");
    RequireVisionShape(bw.proj, pre + "attn.proj.weight", {E, E});
    bw.q_norm = LoadBf16Direct(get, pre + "attn.q_norm.weight");
    RequireVisionShape(bw.q_norm, pre + "attn.q_norm.weight", {D});
    bw.k_norm = LoadBf16Direct(get, pre + "attn.k_norm.weight");
    RequireVisionShape(bw.k_norm, pre + "attn.k_norm.weight", {D});
    // fc1 = the SwiGLU gate, fc3 = the up projection
    // (`fc2(F.silu(fc1(x)) * fc3(x))`, vision.py:132-133). The merge is what
    // routes this pair through `layers::MlpGateUpMethodBase` instead of a
    // hand-written parallel path.
    bw.gate_up = LoadMergedBf16RawNK(
        get, {pre + "mlp.fc1.weight", pre + "mlp.fc3.weight"});
    RequireVisionShape(bw.gate_up, pre + "mlp.{fc1,fc3}.weight", {2 * I, E});
    bw.down = LoadBf16Direct(get, pre + "mlp.fc2.weight");
    RequireVisionShape(bw.down, pre + "mlp.fc2.weight", {E, I});
    w.blocks.push_back(std::move(bw));
  }

  if (v.post_norm) {
    w.post_trunk_norm = LoadBf16Direct(get, p + "post_trunk_norm.weight");
    RequireVisionShape(w.post_trunk_norm, p + "post_trunk_norm.weight", {E});
  }
  const int64_t M = v.merged_dim(), O = v.adapter_out_dim;
  w.adapter_ln_w = LoadBf16Direct(get, p + "adapter.ln_q.weight");
  RequireVisionShape(w.adapter_ln_w, p + "adapter.ln_q.weight",
                     {v.adapter_in_dim});
  w.adapter_ln_b = LoadBf16Direct(get, p + "adapter.ln_q.bias");
  RequireVisionShape(w.adapter_ln_b, p + "adapter.ln_q.bias",
                     {v.adapter_in_dim});
  w.adapter_mlp0_w = LoadBf16Direct(get, p + "adapter.mlp.0.weight");
  RequireVisionShape(w.adapter_mlp0_w, p + "adapter.mlp.0.weight", {M, M});
  w.adapter_mlp0_b = LoadBf16Direct(get, p + "adapter.mlp.0.bias");
  RequireVisionShape(w.adapter_mlp0_b, p + "adapter.mlp.0.bias", {M});
  w.adapter_mlp2_w = LoadBf16Direct(get, p + "adapter.mlp.2.weight");
  RequireVisionShape(w.adapter_mlp2_w, p + "adapter.mlp.2.weight", {O, M});
  w.adapter_mlp2_b = LoadBf16Direct(get, p + "adapter.mlp.2.bias");
  RequireVisionShape(w.adapter_mlp2_b, p + "adapter.mlp.2.bias", {O});
  w.present = true;
  return w;
}

std::vector<std::array<int64_t, 2>> Dots3NoteVisionPosIds(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v) {
  // `get_pos_ids_by_grid` (vision.py:566-603 @ 9035151d6). When
  // `pre_pixel_shuffle` is set the positions follow the qwen `merge_size`
  // GROUPED layout, because the preprocessor already emitted the patch rows in
  // that order; otherwise they are flat row-major regardless of
  // `spatial_merge_size` (upstream's own comment at :567-570).
  const int64_t rope_merge =
      v.pre_pixel_shuffle ? (v.spatial_merge_size > 1 ? v.spatial_merge_size : 2)
                          : 1;
  const int64_t t = grid_thw[0], h = grid_thw[1], wgrid = grid_thw[2];
  VT_CHECK(h % rope_merge == 0 && wgrid % rope_merge == 0,
           "dots3-note vision tower: grid " + std::to_string(h) + "x" +
               std::to_string(wgrid) +
               " is not divisible by the RoPE merge size " +
               std::to_string(rope_merge) +
               ", so the grouped position reshape (vision.py:576-590 @ "
               "9035151d6) has no answer");
  std::vector<std::array<int64_t, 2>> one;
  one.reserve(static_cast<size_t>(h * wgrid));
  // `reshape(h/m, m, w/m, m).permute(0, 2, 1, 3).flatten()` over an [h, w]
  // array whose value is the ROW index (h_pos) or the COLUMN index (w_pos).
  for (int64_t bh = 0; bh < h / rope_merge; ++bh) {
    for (int64_t bw = 0; bw < wgrid / rope_merge; ++bw) {
      for (int64_t sh = 0; sh < rope_merge; ++sh) {
        for (int64_t sw = 0; sw < rope_merge; ++sw) {
          one.push_back({bh * rope_merge + sh, bw * rope_merge + sw});
        }
      }
    }
  }
  std::vector<std::array<int64_t, 2>> out;
  out.reserve(one.size() * static_cast<size_t>(t));
  for (int64_t f = 0; f < t; ++f)
    out.insert(out.end(), one.begin(), one.end());
  return out;
}

std::vector<float> Dots3NoteVisionRopeCache(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v) {
  // `VisionRotaryEmbedding(head_dim // 2)` (vision.py:503-504) builds
  // `inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)` with `dim = head_dim/2`
  // (vision_attention.py:67), i.e. head_dim/4 frequencies per spatial axis.
  // `rot_pos_emb` gathers [L, 2, nf] and flattens to [L, 2*nf] = [L, head_dim/2]
  // (vision.py:604-611). `apply_rotary_pos_emb_vision` then repeats that to
  // head_dim as [f | f] and applies NeoX rotate_half
  // (vision_attention.py:39-49) — which is exactly the [cos(hd/2) | sin(hd/2)]
  // cache `vt::RopeFromCache` consumes at `rotary_dim == head_dim`.
  const int64_t hd = v.head_dim();
  const int64_t dim = hd / 2;
  const int64_t nf = dim / 2;  // frequencies per axis
  VT_CHECK(nf * 2 == dim,
           "dots3-note vision tower: head_dim/2 is odd, so the rope frequency "
           "table has no whole per-axis half");
  std::vector<double> inv_freq(static_cast<size_t>(nf));
  for (int64_t i = 0; i < nf; ++i) {
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(10000.0, static_cast<double>(2 * i) /
                                    static_cast<double>(dim));
  }
  const std::vector<std::array<int64_t, 2>> pos =
      Dots3NoteVisionPosIds(grid_thw, v);
  const int64_t L = static_cast<int64_t>(pos.size());
  std::vector<float> cache(static_cast<size_t>(L * hd));
  for (int64_t r = 0; r < L; ++r) {
    for (int64_t axis = 0; axis < 2; ++axis) {
      const double p = static_cast<double>(pos[static_cast<size_t>(r)][
          static_cast<size_t>(axis)]);
      for (int64_t i = 0; i < nf; ++i) {
        const double ang = p * inv_freq[static_cast<size_t>(i)];
        const int64_t c = axis * nf + i;
        cache[static_cast<size_t>(r * hd + c)] =
            static_cast<float>(std::cos(ang));
        cache[static_cast<size_t>(r * hd + dim + c)] =
            static_cast<float>(std::sin(ang));
      }
    }
  }
  return cache;
}

namespace {

// out[M,N] = x[M,K] @ W[N,K]^T (+ bias[N]). bf16 throughout, exactly the
// projection shape the rest of this tree spells.
void LinearBias(Dev d, DBuf& out, const Tensor& x, const Tensor& w,
                const Tensor* bias) {
  vt::MatmulBT(d.q, out.t(), x, w);
  if (bias != nullptr) vt::Add(d.q, out.t(), out.t(), *bias);
}

std::vector<float> DownloadF32(Dev d, DBuf& buf, int64_t n) {
  std::vector<uint16_t> bits(static_cast<size_t>(n));
  buf.Download(d, bits.data());
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    out[static_cast<size_t>(i)] = vt::BF16ToF32(bits[static_cast<size_t>(i)]);
  return out;
}

}  // namespace

std::vector<float> Dots3NoteVisionForward(
    const std::vector<uint16_t>& pixel_values_bf16,
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionWeights& w,
    const Dots3NoteVisionParams& v, vt::Backend& backend,
    Dots3NoteVisionCapture* cap) {
  VT_CHECK(w.present,
           "dots3-note vision tower: the weights were never materialized. The "
           "loader only materializes a tower Dots3NoteVisionRefusal accepted.");
  VT_CHECK(grid_thw[0] == 1,
           "dots3-note vision tower: grid_t is " + std::to_string(grid_thw[0]) +
               ". W6a serves single-frame IMAGE items; the multi-frame "
               "`cu_seqlens` builder (vision.py:613-624 @ 9035151d6) is the "
               "VIDEO arm and belongs to W7.");
  const int64_t E = v.embed_dim, I = v.intermediate_size;
  const int64_t nh = v.num_attention_heads, hd = v.head_dim();
  const int64_t L = grid_thw[0] * grid_thw[1] * grid_thw[2];
  const int64_t P = v.patch_row();
  VT_CHECK(L > 0, "dots3-note vision tower: an empty grid");
  VT_CHECK(static_cast<int64_t>(pixel_values_bf16.size()) == L * P,
           "dots3-note vision tower: the processor produced " +
               std::to_string(pixel_values_bf16.size()) +
               " patch values for a grid implying " + std::to_string(L) + " x " +
               std::to_string(P) + " = " + std::to_string(L * P));

  vt::Queue q = backend.CreateQueue();
  Dev d{backend, q};
  const vt::RmsNormArgs rms{static_cast<float>(v.rms_norm_eps), /*gemma=*/false};

  // ── patch_embed (vision.py:302-332) ────────────────────────────────────────
  DBuf hidden(d, DType::kBF16, {L, E});
  {
    DBuf px(d, DType::kBF16, {L, P}, pixel_values_bf16.data());
    DBuf proj(d, DType::kBF16, {L, E});
    Tensor pw = ResidentWeight(d, w.patch_proj_w, {E, P});
    Tensor pb = ResidentWeight(d, w.patch_proj_b, {E});
    LinearBias(d, proj, px.t(), pw, &pb);
    Tensor nw = ResidentWeight(d, w.patch_norm, {E});
    vt::RmsNorm(d.q, hidden.t(), proj.t(), nw, rms);
  }
  if (cap != nullptr) cap->patch_embed_out = DownloadF32(d, hidden, L * E);

  // ── the 2-D vision rope cache, host f32 -> bf16 device (vision.py:604-611) ──
  const std::vector<float> cache_f = Dots3NoteVisionRopeCache(grid_thw, v);
  if (cap != nullptr) cap->rope_cache = cache_f;
  std::vector<uint16_t> cache_bits(cache_f.size());
  for (size_t i = 0; i < cache_f.size(); ++i)
    cache_bits[i] = vt::F32ToBF16(cache_f[i]);
  DBuf cache(d, DType::kBF16, {L, hd}, cache_bits.data());
  std::vector<int32_t> pos_idx(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) pos_idx[static_cast<size_t>(i)] =
      static_cast<int32_t>(i);
  DBuf posb(d, DType::kI32, {L}, pos_idx.data());
  vt::RopeArgs ra;
  ra.rotary_dim = static_cast<int>(hd);
  ra.is_neox_style = true;

  // Softmax scale: `1 / sqrt(head_dim)` (vision_attention.py:199, :233).
  const float scale =
      1.0f / std::sqrt(static_cast<float>(hd));
  const vt::AttentionArgs aargs{scale, /*causal=*/false};

  DBuf n1(d, DType::kBF16, {L, E});
  DBuf qkv(d, DType::kBF16, {L, 3 * E});
  DBuf qb(d, DType::kBF16, {L, E});
  DBuf kb(d, DType::kBF16, {L, E});
  DBuf vb(d, DType::kBF16, {L, E});
  DBuf ao(d, DType::kBF16, {L, nh, hd});
  DBuf attn(d, DType::kBF16, {L, E});
  DBuf n2(d, DType::kBF16, {L, E});
  DBuf mlp_out(d, DType::kBF16, {L, E});

  for (size_t b = 0; b < w.blocks.size(); ++b) {
    const Dots3NoteVisionBlockWeights& bw = w.blocks[b];
    // `apply_vision_attention_residual` (vision_attention.py:436-456):
    // `hidden + attn(norm_1(hidden))`. PRE-norm, and the residual is the
    // UN-normalized stream.
    vt::RmsNorm(d.q, n1.t(), hidden.t(), ResidentWeight(d, bw.norm_1, {E}), rms);
    {
      Tensor wq = ResidentWeight(d, bw.qkv, {3 * E, E});
      vt::MatmulBT(d.q, qkv.t(), n1.t(), wq);
      vt::QkvSplit(d.q, qb.t(), kb.t(), vb.t(), qkv.t());
    }
    // `_qkv_with_rope` (vision_attention.py:149-166): the per-head q/k RMSNorm
    // runs BEFORE the rope. Swapping the two is silent — same shapes, same
    // magnitudes, different numbers — and this row has no oracle downstream to
    // catch it, which is why the order has its own gate case.
    {
      Tensor qn = qb.t();
      qn.rank = 2; qn.shape[0] = L * nh; qn.shape[1] = hd;
      qn.stride[0] = hd; qn.stride[1] = 1;
      Tensor kn = kb.t();
      kn.rank = 2; kn.shape[0] = L * nh; kn.shape[1] = hd;
      kn.stride[0] = hd; kn.stride[1] = 1;
      vt::RmsNorm(d.q, qn, qn, ResidentWeight(d, bw.q_norm, {hd}), rms);
      vt::RmsNorm(d.q, kn, kn, ResidentWeight(d, bw.k_norm, {hd}), rms);
    }
    Tensor q3 = qb.t();
    q3.rank = 3; q3.shape[0] = L; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.t();
    k3.rank = 3; k3.shape[0] = L; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    Tensor v3 = vb.t();
    v3.rank = 3; v3.shape[0] = L; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    vt::RopeFromCache(d.q, q3, &k3, posb.t(), cache.t(), ra);
    // ONE window: `grid_t == 1` is asserted above, so `cu_seqlens` is
    // `[0, h*w]` under either of upstream's two builders and the whole item is
    // a single bidirectional block.
    vt::AttentionDenseFlash(d.q, ao.t(), q3, k3, v3, aargs);
    {
      Tensor ao2 = ao.t();
      ao2.rank = 2; ao2.shape[0] = L; ao2.shape[1] = E;
      ao2.stride[0] = E; ao2.stride[1] = 1;
      Tensor wp = ResidentWeight(d, bw.proj, {E, E});
      LinearBias(d, attn, ao2, wp, nullptr);
    }
    vt::Add(d.q, hidden.t(), hidden.t(), attn.t());

    // `hidden + mlp(norm_2(hidden))` (vision.py:369).
    vt::RmsNorm(d.q, n2.t(), hidden.t(), ResidentWeight(d, bw.norm_2, {E}), rms);
    {
      // THE SHARED SEAM. `fc2(silu(fc1(x)) * fc3(x))` is a mergeable gate/up
      // pair, so it rides `layers::MlpGateUpMethodBase` rather than two
      // hand-written GEMMs (AGENTS.md, "Shared seams").
      DBuf act = layers::UnquantizedMlpGateUpMethod(&bw.gate_up, I).Apply(d, n2.t());
      Tensor wd = ResidentWeight(d, bw.down, {E, I});
      LinearBias(d, mlp_out, act.t(), wd, nullptr);
    }
    vt::Add(d.q, hidden.t(), hidden.t(), mlp_out.t());
    if (cap != nullptr && b == 0) cap->block0_out = DownloadF32(d, hidden, L * E);
  }

  // ── post_trunk_norm (vision.py:513-514, 676-677) ───────────────────────────
  DBuf trunk(d, DType::kBF16, {L, E});
  if (v.post_norm) {
    vt::RmsNorm(d.q, trunk.t(), hidden.t(),
                ResidentWeight(d, w.post_trunk_norm, {E}), rms);
  } else {
    backend.Copy(q, trunk.ptr(), hidden.ptr(), hidden.bytes());
  }
  if (cap != nullptr) cap->trunk_out = DownloadF32(d, trunk, L * E);

  // ── the patch_merger adapter (vision.py:441-490) ───────────────────────────
  //
  // `ln_q` normalizes over the PER-TOKEN dim with a HARD-CODED eps of 1e-6
  // (vision.py:466) — NOT `rms_norm_eps`, and it is a LayerNorm with a bias,
  // not an RMSNorm. Then `reshape(-1, merged_dim)` views every 4 consecutive
  // 2x2-grouped tokens as one row: no permutation, because `pre_pixel_shuffle`
  // put the tokens in that order already.
  const int64_t merge_unit = v.adapter_merge_size * v.adapter_merge_size;
  VT_CHECK(L % merge_unit == 0,
           "dots3-note vision tower: " + std::to_string(L) +
               " trunk tokens do not group into whole " +
               std::to_string(merge_unit) +
               "-token merger rows. The processor's grid and the adapter's "
               "merge size disagree.");
  const int64_t Nm = L / merge_unit;
  const int64_t M = v.merged_dim(), O = v.adapter_out_dim;

  DBuf lnq(d, DType::kBF16, {L, E});
  {
    Tensor lw = ResidentWeight(d, w.adapter_ln_w, {v.adapter_in_dim});
    Tensor lb = ResidentWeight(d, w.adapter_ln_b, {v.adapter_in_dim});
    vt::LayerNorm(d.q, lnq.t(), trunk.t(), &lw, &lb, vt::LayerNormArgs{1e-6f});
  }
  DBuf fc1(d, DType::kBF16, {Nm, M});
  {
    Tensor xv = lnq.t();  // [L, E] contiguous IS [Nm, M]
    xv.rank = 2; xv.shape[0] = Nm; xv.shape[1] = M;
    xv.stride[0] = M; xv.stride[1] = 1;
    Tensor w0 = ResidentWeight(d, w.adapter_mlp0_w, {M, M});
    Tensor b0 = ResidentWeight(d, w.adapter_mlp0_b, {M});
    LinearBias(d, fc1, xv, w0, &b0);
  }
  // `nn.GELU()` with no `approximate` argument is the EXACT erf gelu
  // (vision.py:469). The tanh approximation is a different function and a
  // silent one at this magnitude.
  vt::GeluErf(d.q, fc1.t(), fc1.t());
  DBuf out(d, DType::kBF16, {Nm, O});
  {
    Tensor w2 = ResidentWeight(d, w.adapter_mlp2_w, {O, M});
    Tensor b2 = ResidentWeight(d, w.adapter_mlp2_b, {O});
    LinearBias(d, out, fc1.t(), w2, &b2);
  }
  return DownloadF32(d, out, Nm * O);
}

}  // namespace vllm
