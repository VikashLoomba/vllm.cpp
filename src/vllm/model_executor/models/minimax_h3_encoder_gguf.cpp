// H3-Encoder KEEP-QUANT loader — the Qwen3-VL tower from a ComfyUI-format GGUF.
//
// WHY THIS EXISTS: the encoder is 32B. The safetensors loader materializes f32,
// which is ~128 GB and does not fit the box we test on. Held in its ggml blocks the
// tower is ~14.6 GB, and the block-quant GEMM has no arch gate, so it runs natively
// on hardware that cannot do FP4.
//
// THE FUSIONS ARE DONE ON QUANTIZED BYTES. The forward consumes `qkv_proj` and
// `gate_up_proj` fused (what vLLM consumes); the checkpoint ships q/k/v and gate/up
// separate. Concatenating the raw ggml bytes is sound because ggml rows are
// INDEPENDENT — a row is a whole number of blocks, and every K here is a multiple of
// the 256-element Q4_K block — so the result is a valid block-quant tensor whose
// rows are [q_all | k_all | v_all]. No dequantize/requantize round trip, and no
// precision lost to one.
//
// GGUF stores `ne` REVERSED vs torch, so a logical [out, in] weight appears as
// {in, out}; the reader already reverses it, which is why `shape[0]` below is `out`.
#include "vllm/model_executor/models/minimax_h3.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {

const vt::Tensor& MiniMaxH3EncoderQuantWeights::Get(const std::string& name) const {
  const auto it = views.find(name);
  VT_CHECK(it != views.end(), "minimax_h3 encoder gguf: missing tensor (by name)");
  return it->second;
}

namespace {

// A rank-2 ggml tensor's row stride in BYTES.
size_t RowBytes(vt::DType dtype, int64_t k) { return vt::RowSizeBytes(dtype, k); }

}  // namespace

MiniMaxH3EncoderQuantWeights LoadMiniMaxH3EncoderFromGguf(const GgufFile& file,
                                                          int64_t max_layers) {
  MiniMaxH3EncoderQuantWeights out;

  // Index by name. GGUF drops the safetensors `language_model.` level: the text
  // tower is `model.layers.N.*` and the vision tower is `visual.*` (no `model.`).
  // GgufFile has no membership query, so build one from the tensor list.
  std::set<std::string> present;
  for (const GgufTensorInfo& info : file.Tensors()) present.insert(info.name);
  auto has = [&](const std::string& n) { return present.count(n) != 0; };

  auto keep = [&](const std::string& src, const std::string& dst) {
    const GgufTensorInfo& info = file.Get(src);
    VT_CHECK(info.shape.size() == 2, "minimax_h3 encoder gguf: expected a rank-2 projection");
    vt::DType block = vt::DType::kF32;
    VT_CHECK(KeepQuantDType(info.ggml_type, &block),
             "minimax_h3 encoder gguf: unsupported quant encoding for a projection");
    const int64_t rows = info.shape[0], k = info.shape[1];
    VT_CHECK(k % vt::BlockElems(block) == 0,
             "minimax_h3 encoder gguf: K is not a whole number of blocks");
    const size_t bytes = static_cast<size_t>(rows) * RowBytes(block, k);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(info.data);
    out.quant_storage[dst].assign(src_bytes, src_bytes + bytes);
    out.views[dst] = vt::Tensor::Contiguous(out.quant_storage[dst].data(), block, vt::Device{},
                                            {rows, k});
  };

  auto plain = [&](const std::string& src, const std::string& dst) {
    const GgufTensorInfo& info = file.Get(src);
    int64_t numel = 1;
    for (int64_t d : info.shape) numel *= d;
    out.storage[dst] = DequantGgufRowToF32(info.ggml_type, info.data, numel);
    std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
    vt::Tensor t;
    t.data = out.storage[dst].data();
    t.dtype = vt::DType::kF32;
    t.device = vt::Device{};
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    out.views[dst] = t;
  };

  // Fuse a group of same-K quantized projections by concatenating whole ROWS of
  // ggml bytes. Sound because rows are independent block sequences.
  auto fuse = [&](const std::vector<std::string>& srcs, const std::string& dst) {
    vt::DType block = vt::DType::kF32;
    int64_t total_rows = 0, k = -1;
    for (const std::string& s : srcs) {
      const GgufTensorInfo& info = file.Get(s);
      VT_CHECK(info.shape.size() == 2, "minimax_h3 encoder gguf: fuse expects rank-2");
      vt::DType b = vt::DType::kF32;
      VT_CHECK(KeepQuantDType(info.ggml_type, &b),
               "minimax_h3 encoder gguf: unsupported quant encoding in a fused group");
      if (k < 0) {
        k = info.shape[1];
        block = b;
      }
      VT_CHECK(info.shape[1] == k, "minimax_h3 encoder gguf: fused group disagrees on K");
      VT_CHECK(b == block, "minimax_h3 encoder gguf: fused group mixes quant encodings");
      total_rows += info.shape[0];
    }
    VT_CHECK(k % vt::BlockElems(block) == 0,
             "minimax_h3 encoder gguf: fused K is not a whole number of blocks");
    const size_t row_bytes = RowBytes(block, k);
    std::vector<uint8_t>& dstbuf = out.quant_storage[dst];
    dstbuf.clear();
    dstbuf.reserve(static_cast<size_t>(total_rows) * row_bytes);
    for (const std::string& s : srcs) {
      const GgufTensorInfo& info = file.Get(s);
      const uint8_t* p = static_cast<const uint8_t*>(info.data);
      dstbuf.insert(dstbuf.end(), p, p + static_cast<size_t>(info.shape[0]) * row_bytes);
    }
    out.views[dst] =
        vt::Tensor::Contiguous(dstbuf.data(), block, vt::Device{}, {total_rows, k});
  };

  for (int64_t layer = 0;; ++layer) {
    const std::string src = "model.layers." + std::to_string(layer) + ".";
    if (!has(src + "input_layernorm.weight")) break;
    if (max_layers > 0 && layer >= max_layers) break;
    const std::string dst = "layers." + std::to_string(layer) + ".";

    plain(src + "input_layernorm.weight", dst + "input_layernorm.weight");
    plain(src + "post_attention_layernorm.weight", dst + "post_attention_layernorm.weight");
    plain(src + "self_attn.q_norm.weight", dst + "self_attn.q_norm.weight");
    plain(src + "self_attn.k_norm.weight", dst + "self_attn.k_norm.weight");
    keep(src + "self_attn.o_proj.weight", dst + "self_attn.o_proj.weight");
    keep(src + "mlp.down_proj.weight", dst + "mlp.down_proj.weight");
    fuse({src + "self_attn.q_proj.weight", src + "self_attn.k_proj.weight",
          src + "self_attn.v_proj.weight"},
         dst + "self_attn.qkv_proj.weight");
    fuse({src + "mlp.gate_proj.weight", src + "mlp.up_proj.weight"},
         dst + "mlp.gate_up_proj.weight");
  }
  VT_CHECK(out.views.count("layers.0.self_attn.qkv_proj.weight") != 0,
           "minimax_h3 encoder gguf: no text-tower layers were loaded");

  // The embedding table stays QUANTIZED too — at [151936, 5120] it is the single
  // largest tensor, and a caller only ever gathers a handful of its rows.
  if (has("model.embed_tokens.weight")) keep("model.embed_tokens.weight", "embed_tokens.weight");

  // `model.norm.weight` is deliberately NOT bound: H3 reads the UNNORMALIZED
  // truncated output, so carrying it would imply it is applied.

  // Recover the geometry from the fused shapes. qkv rows are q + 2*kv, and the
  // per-head dim comes from q_norm, which is [head_dim].
  const vt::Tensor& qkv = out.Get("layers.0.self_attn.qkv_proj.weight");
  const vt::Tensor& qn = out.Get("layers.0.self_attn.q_norm.weight");
  const vt::Tensor& gate_up = out.Get("layers.0.mlp.gate_up_proj.weight");
  out.config.hidden_size = qkv.shape[1];
  out.config.head_dim = qn.shape[0];
  const int64_t o_rows = out.Get("layers.0.self_attn.o_proj.weight").shape[1];
  out.config.num_attention_heads = o_rows / out.config.head_dim;
  out.config.num_key_value_heads =
      (qkv.shape[0] - o_rows) / (2 * out.config.head_dim);
  out.config.intermediate_size = gate_up.shape[0] / 2;
  int64_t layers = 0;
  while (out.views.count("layers." + std::to_string(layers) + ".input_layernorm.weight") != 0) {
    ++layers;
  }
  out.config.num_hidden_layers = layers;
  return out;
}

}  // namespace vllm
