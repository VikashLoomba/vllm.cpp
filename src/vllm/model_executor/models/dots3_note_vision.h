// dots3-note VISION tower — the DENSE arm (W6a, #2512).
//
// Ported from vLLM read in the local clone `~/_git/vllm` at
// **`9035151d6`**, the merge of vllm#51255 that added the architecture. That
// SHA is written beside every anchor in this file on purpose: `dots3_note` does
// not exist at our parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, and
// upstream has ALREADY moved — `nvidia/vision_attention.py` is 477 lines at
// `9035151d6` and 494 lines at vLLM `main` `7a100bb61`. An anchor with no
// revision beside it is a line number read in the wrong tree.
//
//   vllm/models/dots3_note/nvidia/vision.py @ 9035151d6 (677 lines)
//     DotsMoEVitConfig            :27    -> Dots3NoteVisionParams
//     RMSNorm                     :107   -> vt::RmsNorm (one deliberate
//                                          rounding difference, see below)
//     DotsSwiGLUFFN               :126   -> layers::UnquantizedMlpGateUpMethod
//     DotsPatchEmbed              :302   -> the patch GEMM + RMSNorm
//     MoEVisionBlock              :334   -> Dots3NoteVisionBlockWeights
//     PatchMergerAdapter          :441   -> the adapter
//     DotsMoEVitModel             :492   -> Dots3NoteVisionForward
//       get_pos_ids_by_grid       :566   -> Dots3NoteVisionPosIds
//       rot_pos_emb               :604   -> Dots3NoteVisionRopeCache
//       forward                   :634
//   vllm/models/dots3_note/nvidia/vision_attention.py @ 9035151d6 (477 lines)
//     rotate_half                 :33
//     apply_rotary_pos_emb_vision :39    -> vt::RopeFromCache (NeoX, rotary_dim
//                                          == head_dim, [L, head_dim] cache)
//     VisionRotaryEmbedding       :52    -> Dots3NoteVisionRopeCache
//     _RMSNorm                    :97    -> the per-head q_norm/k_norm
//     _VisionAttentionBase        :134   -> the qkv/proj pair
//       _qkv_with_rope            :149   -> qk-norm BEFORE rope; the order is
//                                          load-bearing and silent when wrong
//     VisionAttentionV2           :207   -> vt::AttentionDenseFlash, causal=false
//     apply_vision_attention_residual :436
//
// WHAT THIS FILE IS NOT. `nvidia/vision_moe.py` @ `9035151d6` (149 lines) and
// `MoESwiGLUFFN` / `MoESwiGLUFFNFP8` (`vision.py:139`, `:222`) are the PYRAMID
// arm. Nothing here reads them, and `Dots3NoteVisionRefusal` names W6b for any
// block the config marks routed. The RELEASED `dots-studio/dots3-note-prev` has
// 17 such blocks out of 42, so its vision tower still refuses BY NAME — which
// is exactly what W3 did to the language tower for four bricks before W5 lifted
// it, not a new exception. See `.agents/specs/dots3-note.md` §4.11.
//
// WHY IT SHARES NO CODE WITH `qwen3_vl_vision`. The two towers agree on the
// block OUTLINE and on almost nothing below it: RMSNorm vs LayerNorm, no bias
// anywhere vs bias everywhere, a per-head qk-norm applied BEFORE rope, a
// three-tensor SwiGLU vs a two-tensor GELU MLP, a patch-merger adapter vs a
// pixel-shuffle merger, no DeepStack, no interpolated position-embedding table.
// Extending `Qwen3VLVisionConfig` to carry all of that would be a parallel path
// wearing one struct's name. What IS shared is every seam underneath: the
// `vt::` ops, `dense_attn`'s device glue, `dense_loaders`' weight readers and
// `layers::MlpGateUpMethodBase` for the mergeable projections.
//
// THE ONE DELIBERATE FORMULA DIFFERENCE, named rather than left to be found.
// Upstream's `RMSNorm.forward` (`vision.py:112-114`) is
// `self._norm(x.float()).type_as(x) * self.weight` — it rounds BACK to the
// activation dtype before multiplying by the weight. `vt::RmsNorm` keeps f32
// through the weight multiply and rounds once on the store (`vt/ops.h`'s own
// note on it). Using the shared op is the seam rule; the difference is one bf16
// rounding step, it is what the gate's tolerance carries, and the in-test
// double-precision reference mirrors UPSTREAM rather than this file so that the
// difference is measured rather than defined away.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_
#define VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vt/backend.h"

namespace vllm {

class SafetensorsFile;
struct HfConfig;
struct Dots3NoteTensor;

// `DotsMoEVitConfig.__init__` (`vision.py:27-105` @ `9035151d6`), reduced to
// the fields this arm reads plus the ones it REFUSES on. Every default here is
// upstream's own default, so a config that omits a key gets what upstream's
// constructor would have given it — never a value chosen locally.
struct Dots3NoteVisionParams {
  // False when `config.json` carries no `vision_config` at all. Upstream builds
  // no `DotsMoEVitModel` in that case (`multimodal.py:113-118` @ `9035151d6`,
  // the `vision_config_dict is not None` guard), and neither does this port.
  bool present = false;

  int64_t embed_dim = 1536;
  // The TEXT tower's `hidden_size`, copied from the LANGUAGE config at parse.
  // It is NOT a `vision_config` key, and it replaces one that was: upstream's
  // `vision_config.hidden_size` is a SECOND copy of the same number, nothing
  // read it, and a refusal that read it would still not be the predicate the
  // encoder applies. `EncodeMmDots3NoteForCausalLM` compares `adapter_out_dim`
  // against `config.hidden_size`, so that is the value that has to be here for
  // `Dots3NoteVisionRefusal` to answer the same question the route asks.
  int64_t text_hidden_size = 0;
  int64_t intermediate_size = 4224;  // the DENSE SwiGLU width
  int64_t moe_intermediate_size = 2112;  // W6b's; read only to report it
  int64_t num_hidden_layers = 42;
  int64_t num_attention_heads = 24;
  int64_t num_channels = 3;
  int64_t patch_size = 14;
  int64_t spatial_merge_size = 2;
  int64_t temporal_patch_size = 1;
  double rms_norm_eps = 1e-5;
  bool use_bias = false;
  bool use_qk_norm = true;
  bool is_causal = false;
  bool post_norm = true;
  // `pre_pixel_shuffle` (`vision.py:60-63`): when TRUE the PREPROCESSOR already
  // emits patch rows in 2x2-grouped order (`common/processor.py:185-197`), so
  // the RoPE positions are regrouped to match (`get_pos_ids_by_grid:566-575`).
  // It is NOT the same switch as `adapter_type`, and conflating the two is the
  // reading #2512's prose invites — see `.agents/specs/dots3-note.md` §4.11.1.
  //
  // FALSE is `DotsMoEVitConfig`'s own default (`vision.py:64`), which is why it
  // is the default here; the RELEASED checkpoint sets it TRUE in its
  // `vision_config` and the parser reads that. The struct default matters
  // because this field selects between two INCOMPATIBLE token orders, so a
  // default-constructed params must not silently claim the regrouped one.
  bool pre_pixel_shuffle = false;

  // The per-block routed-expert counts (`vision.py:90`). `is_moe` is
  // `pyramid_num_routed[i] > 0` (`vision.py:346-350`), so the released
  // checkpoint's leading 25 entries of `-1` are DENSE and the trailing 17
  // (4, 8, ... 60, 64, 64) are W6b's.
  std::vector<int64_t> pyramid_num_routed;
  double capacity_factor = 2.0;         // W6b's top-k; read only to report it
  std::string router_scoring_func = "sigmoid";
  double router_scale = 1.0;

  // The adapter (`vision.py:441-472`). `patch_merger` is the arm the released
  // checkpoint selects; `pixel_shuffle_mlp` is a DIFFERENT token order from the
  // same pixels and is refused by name.
  std::string adapter_type = "pixel_shuffle_mlp";
  int64_t adapter_in_dim = 1536;
  int64_t adapter_out_dim = 2048;
  int64_t adapter_merge_size = 2;

  int64_t head_dim() const { return embed_dim / num_attention_heads; }
  // `merged_dim = in_dim * merge_size**2` (`vision.py:462`).
  int64_t merged_dim() const {
    return adapter_in_dim * adapter_merge_size * adapter_merge_size;
  }
  // One patch row as the processor ships it:
  // `channel * temporal_patch_size * patch_size * patch_size`
  // (`common/processor.py:216-218` @ `9035151d6`).
  int64_t patch_row() const {
    return num_channels * temporal_patch_size * patch_size * patch_size;
  }
  bool is_moe_block(int64_t layer) const {
    return layer >= 0 &&
           layer < static_cast<int64_t>(pyramid_num_routed.size()) &&
           pyramid_num_routed[static_cast<size_t>(layer)] > 0;
  }
  // How many DENSE blocks this config has, counted over ALL blocks rather than
  // as a leading run: a config that interleaved them would report the truth
  // instead of the length of its first run.
  int64_t num_dense_blocks() const {
    int64_t n = 0;
    for (int64_t i = 0; i < num_hidden_layers; ++i)
      if (!is_moe_block(i)) ++n;
    return n;
  }
  int64_t num_moe_blocks() const { return num_hidden_layers - num_dense_blocks(); }
};

// Resolve + validate `config.json`'s `vision_config`. Returns `present=false`
// when the key is absent. Throws (VT_CHECK) naming the key on a value this arm
// cannot represent AND that no later brick owns — a shape that IS owed to a
// later brick is reported by `Dots3NoteVisionRefusal` instead, because a
// checkpoint whose tower is owed must still LOAD its language half.
Dots3NoteVisionParams ParseDots3NoteVisionParams(const HfConfig& config);

// Why the vision tower cannot be materialized, or "" when it can. Names ONE
// thing — the first unrepresentable feature in brick order — and the brick that
// owes it. A non-empty answer leaves the 2195 `vision_encoder.*` tensors in the
// accounting's existing `vision` deferral bucket, exactly as before W6a, so
// every W2 count assertion is unchanged.
std::string Dots3NoteVisionRefusal(const Dots3NoteVisionParams& v,
                                   const std::string& quant_method,
                                   const std::vector<int64_t>& weight_block_size);

// The same answer from a CONFIG alone, for a caller that holds a checkpoint
// directory and no loaded model. The multimodal CHAT seam is that caller, and
// it is the reason this overload exists rather than a convenience.
//
// A refusal raised from `encode_mm` is FATAL: it is thrown inside the engine's
// busy loop, which stops `AsyncLLM` and turns every later request — including
// TEXT ones — into a 500. Measured on this row's served-request gate before
// this function existed. The entrypoint is where a "this server cannot serve
// images for this checkpoint" answer belongs, and `InstallMultiModalChatSeam`
// already has the shape for it: a factory that throws installs a REFUSING seam,
// which answers an image request with HTTP 400 naming the architecture and the
// reason while the text path keeps working. The `encode_mm` check stays as
// defence in depth, on the same polarity as Qwen3-VL's ("reaching this point is
// a defect").
std::string Dots3NoteVisionRefusalFor(const HfConfig& config);

// One DENSE block's weights, by the names the checkpoint ships
// (`vision_encoder.blocks.{B}.*`). Every tensor is BF16 on disk and BF16 here:
// the released index carries 37944 BF16 + 62 F32 and every F32 of those is a
// `router_bias` or an `e_score_correction_bias`, neither of which this arm
// reads. Widening any of these would be invisible to a token gate and is what
// `porting.md`'s memory-format rule is about.
struct Dots3NoteVisionBlockWeights {
  OwnedTensor norm_1;   // [E]
  OwnedTensor norm_2;   // [E]
  OwnedTensor qkv;      // [3E, E]   (NO bias: use_bias == false)
  OwnedTensor proj;     // [E, E]
  OwnedTensor q_norm;   // [head_dim]
  OwnedTensor k_norm;   // [head_dim]
  // fc1 (the SwiGLU gate) and fc3 (the up projection) MERGED into one [2I, E]
  // raw-NK operand, so the pair rides `layers::MlpGateUpMethodBase` rather than
  // a hand-written parallel path (AGENTS.md, "Shared seams"). The order is
  // gate-then-up because `vt::SiluAndMul` reads `silu(x[:, :I]) * x[:, I:]` and
  // upstream is `fc2(F.silu(fc1(x)) * fc3(x))` (`vision.py:132-133`).
  OwnedTensor gate_up;  // [2I, E] = concat(fc1, fc3)
  OwnedTensor down;     // fc2 [E, I]
};

struct Dots3NoteVisionWeights {
  bool present = false;
  OwnedTensor patch_proj_w;  // [E, C*tp*p*p]  (on disk [E, C, p, p])
  OwnedTensor patch_proj_b;  // [E]
  OwnedTensor patch_norm;    // [E]
  std::vector<Dots3NoteVisionBlockWeights> blocks;  // the DENSE blocks, in order
  OwnedTensor post_trunk_norm;  // [E]
  OwnedTensor adapter_ln_w;     // [in_dim]
  OwnedTensor adapter_ln_b;     // [in_dim]
  OwnedTensor adapter_mlp0_w;   // [merged_dim, merged_dim]
  OwnedTensor adapter_mlp0_b;   // [merged_dim]
  OwnedTensor adapter_mlp2_w;   // [out_dim, merged_dim]
  OwnedTensor adapter_mlp2_b;   // [out_dim]
};

// Every `vision_encoder.*` name the DENSE arm claims, with its named consumer.
// Over the released checkpoint this is exactly 235 of the 2195 vision tensors;
// the other 1960 belong to W6b and stay deferred.
std::vector<Dots3NoteTensor> EnumerateDots3NoteVisionTensors(
    const Dots3NoteVisionParams& v);

// Read the DENSE tower out of `shards`. REFUSES BY NAME on the first tensor
// whose shape disagrees with the config. Only called when
// `Dots3NoteVisionRefusal` is empty.
Dots3NoteVisionWeights MaterializeDots3NoteVision(
    const std::vector<SafetensorsFile>& shards, const Dots3NoteVisionParams& v);

// `get_pos_ids_by_grid` (`vision.py:566-603` @ `9035151d6`) for ONE grid.
// Returns L = t*h*w pairs {h_pos, w_pos} in the tower's token order.
std::vector<std::array<int64_t, 2>> Dots3NoteVisionPosIds(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v);

// `VisionRotaryEmbedding` + `rot_pos_emb` (`vision_attention.py:52-89`,
// `vision.py:604-611`) folded into the [L, head_dim] = [cos(hd/2) | sin(hd/2)]
// cache `vt::RopeFromCache` consumes. f32 host precompute, deterministic.
std::vector<float> Dots3NoteVisionRopeCache(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v);

// Optional intermediate capture, for the unit gate only. Production passes
// nullptr and pays nothing.
struct Dots3NoteVisionCapture {
  std::vector<float> rope_cache;      // [L, head_dim]
  std::vector<float> patch_embed_out; // [L, E]
  std::vector<float> block0_out;      // [L, E]
  std::vector<float> trunk_out;       // [L, E], after post_trunk_norm
};

// THE TOWER. `pixel_values_bf16` is [L, patch_row()] raw bf16 bits as the
// processor ships them; `grid_thw` is {t, h, w}. Returns
// [L / merge^2, adapter_out_dim] host f32 — the rows the placeholder span is
// expanded to (`multimodal.py:151-155` @ `9035151d6`:
// `grid.prod(-1) // merge_size**2`).
std::vector<float> Dots3NoteVisionForward(
    const std::vector<uint16_t>& pixel_values_bf16,
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionWeights& w,
    const Dots3NoteVisionParams& v, vt::Backend& backend,
    Dots3NoteVisionCapture* capture = nullptr);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_
