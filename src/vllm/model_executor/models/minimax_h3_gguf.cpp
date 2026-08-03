// MiniMax-H3 GGUF arm — the quantized checkpoints that actually FIT one GB10.
//
// ─── WHY THIS EXISTS ─────────────────────────────────────────────────────────
// The bf16 MiniMax-H3 release is ~354 GB and needs 4x B300, which is why the
// safetensors arm is hardware-blocked here. The community ComfyUI-format GGUFs
// change that verdict: the DiT at Q3_K_M is ~15.6 GB and the Qwen3-VL encoder at
// Q4_K_M is ~14.6 GB, so a whole working set (plus the two fp16/fp32 VAEs) lands
// around ~41 GB — comfortably inside the 119 GiB unified pool. See
// .agents/specs/minimax-h3.md section 0.
//
// ─── THE NAME MAP IS THE IDENTITY (verified against a real checkpoint) ───────
// ComfyUI's H3 GGUF keeps the checkpoint's own parameter names, so every one of
// the 535 tensors in `MiniMax-H3-FL2VA-Q3_K_M.gguf` matches the contract
// `EnumerateMiniMaxH3DitTensors` derives from upstream source — no rename table.
// That equivalence is GATED on the real manifest
// (tests/vllm/models/minimax_h3_gguf_manifest.inc, produced by
// scripts/gen-minimax-h3-gguf-manifest.py from the file's header alone).
//
// Two shape rules, and they are the whole subtlety:
//   1. GGUF `ne` is REVERSED relative to torch: a `[out, in]` weight is stored
//      `[in, out]` (qkv_proj is ne=[5376, 21504] for logical [21504, 5376]).
//   2. When ComfyUI had to RESHAPE a tensor so its fastest axis is a whole
//      number of quant blocks, it records the true torch shape in the metadata
//      key `comfy.gguf.orig_shape.<name>`, which is already in `[out, in]`
//      order and therefore must NOT be reversed. The 50 AdaLN projections are
//      exactly this case: logical [96768, 2688], but 2688 is not a multiple of
//      the 256-element Q3_K block, so they ship as ne=[256, 1016064].
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

// Parse the trailing integer of a `prefix<N>.` path segment; -1 when absent.
int64_t IndexAfter(const std::string& name, const std::string& prefix) {
  if (name.compare(0, prefix.size(), prefix) != 0) return -1;
  size_t pos = prefix.size();
  if (pos >= name.size() || name[pos] < '0' || name[pos] > '9') return -1;
  int64_t value = 0;
  while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
    value = value * 10 + (name[pos] - '0');
    ++pos;
  }
  return value;
}

}  // namespace

// The logical (torch) shape of a tensor from its RAW GGUF `ne` dims: reverse the
// ne order, unless ComfyUI recorded `orig_shape`, which is already torch order.
// NOTE the asymmetry with `EnumerateMiniMaxH3GgufTensors` below: `GgufTensorInfo::shape`
// has ALREADY been reversed by the reader, so that path must not reverse again.
// This overload is the one that speaks the on-disk order, and it is what the
// real-manifest gate exercises.
std::vector<int64_t> MiniMaxH3GgufLogicalShape(const std::vector<int64_t>& gguf_dims,
                                               const std::vector<int64_t>& orig_shape) {
  if (!orig_shape.empty()) return orig_shape;
  std::vector<int64_t> shape(gguf_dims.rbegin(), gguf_dims.rend());
  // GGUF pads trailing dims with 1; drop the leading 1s the reversal produces so
  // a 1-D tensor stays 1-D.
  while (shape.size() > 1 && shape.front() == 1) shape.erase(shape.begin());
  return shape;
}

// Derive the H3 geometry from the tensor manifest alone. A ComfyUI GGUF carries
// no transformer config.json, so the shapes ARE the config: this is what lets a
// GGUF checkpoint load without the original repo.
MiniMaxH3DitParams ParseMiniMaxH3DitParamsFromGgufManifest(
    const std::vector<MiniMaxH3TensorSpec>& manifest) {
  MiniMaxH3DitParams p;
  int64_t max_block = -1, max_refiner = -1;
  bool saw_hidden = false, saw_head_dim = false;

  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const std::string& name = spec.name;
    max_block = std::max(max_block, IndexAfter(name, "blocks."));
    max_refiner = std::max(max_refiner, IndexAfter(name, "token_refiner.blocks."));

    if (name == "final_layer.norm.weight" && spec.shape.size() == 1) {
      p.hidden_size = spec.shape[0];
      saw_hidden = true;
    } else if (name == "blocks.0.attn.q_norm.weight" && spec.shape.size() == 1) {
      p.attention_head_dim = spec.shape[0];
      saw_head_dim = true;
    } else if (name == "condition_proj.weight" && spec.shape.size() == 2) {
      p.text_dim = spec.shape[1];
    } else if (name == "video_patch_proj.weight" && spec.shape.size() == 2) {
      // [hidden, latents_dim * patch volume]; the patch is fixed at (1,2,2).
      p.latents_dim = spec.shape[1] / (p.patch_size_t * p.patch_size_h * p.patch_size_w);
    } else if (name == "audio_patch_proj.weight" && spec.shape.size() == 2) {
      p.audio_latents_dim = spec.shape[1];
    } else if (name == "time_embedder.proj_in.weight" && spec.shape.size() == 2) {
      p.time_embed_hidden_size = spec.shape[0];
      p.timestep_input_dim = spec.shape[1];
    } else if (name == "time_embedder.proj_out.weight" && spec.shape.size() == 2) {
      p.time_embed_dim = spec.shape[0];
    } else if (name == "blocks.0.mlp.fc2.weight" && spec.shape.size() == 2) {
      p.ffn_hidden_size = spec.shape[1];
    } else if (name == "blocks.0.adaln_proj.linear.weight" && spec.shape.size() == 2) {
      p.adaln_out_features = spec.shape[0];
    } else if (name == "final_layer.adaln_proj.linear.weight" && spec.shape.size() == 2) {
      p.final_adaln_out_features = spec.shape[0];
    } else if (name == "rope.inv_freq" && spec.shape.size() == 1) {
      p.rope_inv_freq_len = spec.shape[0];
    }
  }

  VT_CHECK(saw_hidden, "minimax_h3 gguf: final_layer.norm.weight is required to size the model");
  VT_CHECK(saw_head_dim, "minimax_h3 gguf: blocks.0.attn.q_norm.weight is required");
  VT_CHECK(max_block >= 0, "minimax_h3 gguf: no blocks.<N> tensors found");
  p.num_layers = max_block + 1;
  p.token_refiner_num_layers = max_refiner + 1;

  // MHA: qkv is 3 * heads * head_dim wide, so the head count follows.
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    if (spec.name == "blocks.0.attn.qkv_proj.weight" && spec.shape.size() == 2) {
      VT_CHECK(spec.shape[0] % (3 * p.attention_head_dim) == 0,
               "minimax_h3 gguf: qkv width is not 3 * heads * head_dim");
      p.num_attention_heads = spec.shape[0] / (3 * p.attention_head_dim);
    }
  }
  VT_CHECK(p.num_attention_heads > 0, "minimax_h3 gguf: could not derive the head count");
  VT_CHECK(p.adaln_out_features == 6 * p.hidden_size * kMiniMaxH3AdalnModalityNum,
           "minimax_h3 gguf: adaln width does not match 6 * hidden * 3");
  VT_CHECK(p.rope_rot_dim() <= p.attention_head_dim,
           "minimax_h3 gguf: 6 * rope_inv_freq_len exceeds attention_head_dim");
  return p;
}

// Read the manifest (names + logical shapes + ggml types) out of a GGUF.
std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3GgufTensors(const GgufFile& file) {
  std::vector<MiniMaxH3TensorSpec> out;
  out.reserve(file.Tensors().size());
  for (const GgufTensorInfo& info : file.Tensors()) {
    MiniMaxH3TensorSpec spec;
    spec.name = info.name;
    // `info.shape` is already torch row-major (the reader reverses ne), so it is
    // used as-is; `orig_shape` overrides it when ComfyUI reshaped the tensor for
    // quant-block alignment.
    spec.shape = info.shape;
    if (const GgufValue* kv = file.FindKv("comfy.gguf.orig_shape." + info.name)) {
      if (const GgufArray* array = std::get_if<GgufArray>(&kv->v)) {
        std::vector<int64_t> orig;
        orig.reserve(array->elems.size());
        for (const GgufValue& elem : array->elems) {
          switch (elem.TypeId()) {
            case kGgufU32: orig.push_back(std::get<uint32_t>(elem.v)); break;
            case kGgufI32: orig.push_back(std::get<int32_t>(elem.v)); break;
            case kGgufU64: orig.push_back(static_cast<int64_t>(std::get<uint64_t>(elem.v))); break;
            case kGgufI64: orig.push_back(std::get<int64_t>(elem.v)); break;
            default:
              VT_CHECK(false, "minimax_h3 gguf: comfy.gguf.orig_shape must hold integers");
          }
        }
        if (!orig.empty()) spec.shape = orig;
      }
    }
    // The fp32 islands stay unquantized in the GGUF too (the ComfyUI quantizer
    // leaves norms, biases, and the small projections alone).
    spec.fp32 = info.ggml_type == 0;
    out.push_back(std::move(spec));
  }
  return out;
}

}  // namespace vllm
