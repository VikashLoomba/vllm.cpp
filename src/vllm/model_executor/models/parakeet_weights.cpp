// Checkpoint loader for `ParakeetForCTC` — spike row
// `MODEL-AUDIO-PARAKEET-ENCODER` (work item P4 of
// .agents/specs/parakeet-conformer-encoder.md).
//
// Ported from transformers 5.3.0:
//   configuration_parakeet.py ParakeetEncoderConfig:23 (fields :96-149),
//                             ParakeetCTCConfig:152 (fields :196-218)
//   modeling_parakeet.py      the module tree whose `state_dict()` keys this
//                             maps: ParakeetForCTC:678-682,
//                             ParakeetEncoder:563-568,
//                             ParakeetEncoderSubsamplingConv2D:368-391,
//                             ParakeetEncoderBlock:431-440,
//                             ParakeetEncoderAttention:272-289,
//                             ParakeetEncoderConvolutionModule:134-149
// vLLM does not implement the encoder (parakeet.py:14,61 delegate to HF), so the
// HF state-dict layout IS the checkpoint layout — see the recorded deviation in
// include/vllm/model_executor/models/parakeet_encoder.h.
//
// Key map (`encoder.` prefix as `ParakeetForCTC` stores it; an encoder-only
// checkpoint with no prefix is accepted too):
//   encoder.subsampling.layers.0.{weight,bias}       dense 1->C conv   (:369-371)
//   encoder.subsampling.layers.{2+3i}.{weight,bias}  stage i depthwise (:375-384)
//   encoder.subsampling.layers.{3+3i}.{weight,bias}  stage i pointwise (:386)
//     (layers.1 and layers.{4+3i} are the parameterless ReLUs, :372/:388)
//   encoder.subsampling.linear.{weight,bias}                           (:391)
//   encoder.layers.N.feed_forward{1,2}.linear{1,2}.{weight,bias}       (:104-106)
//   encoder.layers.N.self_attn.{q,k,v,o}_proj.{weight,bias}            (:272-283)
//   encoder.layers.N.self_attn.relative_k_proj.weight  (never biased,   :285)
//   encoder.layers.N.self_attn.bias_{u,v}                              (:287-289)
//   encoder.layers.N.conv.pointwise_conv{1,2}.{weight,bias}            (:134-149)
//   encoder.layers.N.conv.depthwise_conv.{weight,bias}                 (:137-145)
//   encoder.layers.N.conv.norm.{weight,bias,running_mean,running_var}  (:146)
//   encoder.layers.N.norm_{feed_forward1,self_att,conv,feed_forward2,out}.*
//                                                                      (:436-440)
//   ctc_head.{weight,bias}                                             (:682)
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/parakeet_encoder.h"
#include "vt/dtype.h"

namespace vllm::multimodal {
namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) throw std::runtime_error("parakeet: cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

int64_t JsonInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<int64_t>();
}

bool JsonBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

std::string JsonStr(const nlohmann::json& doc, const char* key,
                    const std::string& fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_string()) return fallback;
  return it->get<std::string>();
}

// Convert one safetensors tensor to host f32, whatever float storage it uses.
std::vector<float> ToF32(const std::string& name, const StTensor& t) {
  int64_t numel = 1;
  for (int64_t d : t.shape) numel *= d;
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.dtype == "F32") {
    if (t.nbytes != out.size() * sizeof(float)) {
      throw std::runtime_error("parakeet: byte count mismatch for " + name);
    }
    std::memcpy(out.data(), t.data, t.nbytes);
  } else if (t.dtype == "F16" || t.dtype == "BF16") {
    if (t.nbytes != out.size() * sizeof(uint16_t)) {
      throw std::runtime_error("parakeet: byte count mismatch for " + name);
    }
    const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = (t.dtype == "F16") ? vt::F16ToF32(src[i]) : vt::BF16ToF32(src[i]);
    }
  } else {
    throw std::runtime_error("parakeet: unsupported dtype " + t.dtype + " for " + name);
  }
  return out;
}

// Resolves a state-dict key across one or many safetensors shards, accepting an
// optional `encoder.`-style prefix drop so an encoder-only checkpoint loads too.
class Resolver {
 public:
  Resolver(const std::string& dir) {
    const std::string index = dir + "/model.safetensors.index.json";
    std::ifstream probe(index, std::ios::binary);
    if (probe.good()) {
      probe.close();
      for (const auto& [tensor, shard] : LoadSafetensorsIndex(index)) {
        shard_of_[tensor] = shard;
        if (files_.find(shard) == files_.end()) {
          files_.emplace(shard, std::make_unique<SafetensorsFile>(
                                    SafetensorsFile::Open(dir + "/" + shard)));
        }
      }
    } else {
      const std::string single = "model.safetensors";
      files_.emplace(single, std::make_unique<SafetensorsFile>(
                                 SafetensorsFile::Open(dir + "/" + single)));
      for (const std::string& n : files_.at(single)->Names()) shard_of_[n] = single;
    }
  }

  bool Has(const std::string& name) const {
    return shard_of_.find(name) != shard_of_.end();
  }

  std::vector<float> Get(const std::string& name) const {
    const auto it = shard_of_.find(name);
    if (it == shard_of_.end()) {
      throw std::runtime_error("parakeet: checkpoint is missing tensor " + name);
    }
    return ToF32(name, files_.at(it->second)->Get(name));
  }

  // Optional tensor: empty when absent. Used for the biases a checkpoint may
  // legitimately drop (attention_bias / convolution_bias false — the case vLLM
  // parakeet.py:112-131 documents for transformers v5).
  std::vector<float> GetOptional(const std::string& name) const {
    return Has(name) ? Get(name) : std::vector<float>();
  }

 private:
  std::map<std::string, std::string> shard_of_;
  std::map<std::string, std::unique_ptr<SafetensorsFile>> files_;
};

void RequireSize(const std::string& name, const std::vector<float>& v, int64_t expect) {
  if (static_cast<int64_t>(v.size()) != expect) {
    throw std::runtime_error("parakeet: tensor " + name + " has " +
                             std::to_string(v.size()) + " elements, expected " +
                             std::to_string(expect));
  }
}

ParakeetEncoderConfig ConfigFromJson(const nlohmann::json& doc) {
  // ParakeetCTCConfig nests the encoder under "encoder_config" (:201, :209-213);
  // a bare ParakeetEncoderConfig checkpoint has the fields at top level.
  const nlohmann::json* enc = &doc;
  const auto it = doc.find("encoder_config");
  if (it != doc.end() && it->is_object()) enc = &*it;

  ParakeetEncoderConfig cfg;
  cfg.hidden_size = JsonInt(*enc, "hidden_size", cfg.hidden_size);
  cfg.num_hidden_layers = JsonInt(*enc, "num_hidden_layers", cfg.num_hidden_layers);
  cfg.num_attention_heads =
      JsonInt(*enc, "num_attention_heads", cfg.num_attention_heads);
  // :124 — upstream forces num_key_value_heads == num_attention_heads.
  cfg.num_key_value_heads =
      JsonInt(*enc, "num_key_value_heads", cfg.num_attention_heads);
  cfg.intermediate_size = JsonInt(*enc, "intermediate_size", cfg.intermediate_size);
  cfg.hidden_act = JsonStr(*enc, "hidden_act", cfg.hidden_act);
  cfg.attention_bias = JsonBool(*enc, "attention_bias", cfg.attention_bias);
  cfg.convolution_bias = JsonBool(*enc, "convolution_bias", cfg.convolution_bias);
  cfg.conv_kernel_size = JsonInt(*enc, "conv_kernel_size", cfg.conv_kernel_size);
  cfg.subsampling_factor = JsonInt(*enc, "subsampling_factor", cfg.subsampling_factor);
  cfg.subsampling_conv_channels =
      JsonInt(*enc, "subsampling_conv_channels", cfg.subsampling_conv_channels);
  cfg.num_mel_bins = JsonInt(*enc, "num_mel_bins", cfg.num_mel_bins);
  cfg.subsampling_conv_kernel_size =
      JsonInt(*enc, "subsampling_conv_kernel_size", cfg.subsampling_conv_kernel_size);
  cfg.subsampling_conv_stride =
      JsonInt(*enc, "subsampling_conv_stride", cfg.subsampling_conv_stride);
  cfg.max_position_embeddings =
      JsonInt(*enc, "max_position_embeddings", cfg.max_position_embeddings);
  cfg.scale_input = JsonBool(*enc, "scale_input", cfg.scale_input);
  // The CTC head's two fields live at the TOP level (:205, :216).
  cfg.vocab_size = JsonInt(doc, "vocab_size", cfg.vocab_size);
  cfg.pad_token_id = static_cast<int32_t>(JsonInt(doc, "pad_token_id", cfg.pad_token_id));
  return cfg;
}

ParakeetSubsamplingWeights LoadSubsampling(const Resolver& r, const std::string& prefix,
                                           const ParakeetEncoderConfig& cfg) {
  const std::string base = prefix + "subsampling.";
  const int64_t k = cfg.subsampling_conv_kernel_size;
  const int64_t channels = cfg.subsampling_conv_channels;

  ParakeetSubsamplingWeights w;
  w.conv0_w = r.Get(base + "layers.0.weight");
  RequireSize(base + "layers.0.weight", w.conv0_w, channels * 1 * k * k);
  w.conv0_b = r.Get(base + "layers.0.bias");
  RequireSize(base + "layers.0.bias", w.conv0_b, channels);

  for (int64_t i = 0; i + 1 < cfg.num_subsampling_layers(); ++i) {
    // layers[0]=conv, [1]=ReLU, then (depthwise, pointwise, ReLU) per stage.
    const std::string dw = base + "layers." + std::to_string(2 + 3 * i) + ".";
    const std::string pw = base + "layers." + std::to_string(3 + 3 * i) + ".";
    ParakeetSubsamplingWeights::Stage stage;
    stage.depthwise_w = r.Get(dw + "weight");
    RequireSize(dw + "weight", stage.depthwise_w, channels * 1 * k * k);
    stage.depthwise_b = r.Get(dw + "bias");
    RequireSize(dw + "bias", stage.depthwise_b, channels);
    stage.pointwise_w = r.Get(pw + "weight");
    RequireSize(pw + "weight", stage.pointwise_w, channels * channels);
    stage.pointwise_b = r.Get(pw + "bias");
    RequireSize(pw + "bias", stage.pointwise_b, channels);
    w.stages.push_back(std::move(stage));
  }

  w.linear_w = r.Get(base + "linear.weight");
  RequireSize(base + "linear.weight", w.linear_w,
              cfg.hidden_size * channels * cfg.subsampling_out_freq());
  w.linear_b = r.Get(base + "linear.bias");
  RequireSize(base + "linear.bias", w.linear_b, cfg.hidden_size);
  return w;
}

ParakeetEncoderLayerWeights LoadLayer(const Resolver& r, const std::string& prefix,
                                      int64_t index,
                                      const ParakeetEncoderConfig& cfg) {
  const std::string base = prefix + "layers." + std::to_string(index) + ".";
  const int64_t hidden = cfg.hidden_size;
  const int64_t inter = cfg.intermediate_size;
  const int64_t heads = cfg.num_attention_heads;
  const int64_t kv = cfg.num_key_value_heads;
  const int64_t hd = cfg.head_dim();

  ParakeetEncoderLayerWeights w;

  auto load_ff = [&](const std::string& p) {
    ParakeetFeedForwardWeights ff;
    ff.linear1_w = r.Get(p + "linear1.weight");
    RequireSize(p + "linear1.weight", ff.linear1_w, inter * hidden);
    ff.linear1_b = r.GetOptional(p + "linear1.bias");
    ff.linear2_w = r.Get(p + "linear2.weight");
    RequireSize(p + "linear2.weight", ff.linear2_w, hidden * inter);
    ff.linear2_b = r.GetOptional(p + "linear2.bias");
    return ff;
  };
  w.feed_forward1 = load_ff(base + "feed_forward1.");
  w.feed_forward2 = load_ff(base + "feed_forward2.");

  const std::string attn = base + "self_attn.";
  w.self_attn.q_w = r.Get(attn + "q_proj.weight");
  RequireSize(attn + "q_proj.weight", w.self_attn.q_w, heads * hd * hidden);
  w.self_attn.q_b = r.GetOptional(attn + "q_proj.bias");
  w.self_attn.k_w = r.Get(attn + "k_proj.weight");
  RequireSize(attn + "k_proj.weight", w.self_attn.k_w, kv * hd * hidden);
  w.self_attn.k_b = r.GetOptional(attn + "k_proj.bias");
  w.self_attn.v_w = r.Get(attn + "v_proj.weight");
  RequireSize(attn + "v_proj.weight", w.self_attn.v_w, kv * hd * hidden);
  w.self_attn.v_b = r.GetOptional(attn + "v_proj.bias");
  w.self_attn.o_w = r.Get(attn + "o_proj.weight");
  RequireSize(attn + "o_proj.weight", w.self_attn.o_w, hidden * heads * hd);
  w.self_attn.o_b = r.GetOptional(attn + "o_proj.bias");
  w.self_attn.relative_k_w = r.Get(attn + "relative_k_proj.weight");
  RequireSize(attn + "relative_k_proj.weight", w.self_attn.relative_k_w,
              heads * hd * hidden);
  w.self_attn.bias_u = r.Get(attn + "bias_u");
  RequireSize(attn + "bias_u", w.self_attn.bias_u, heads * hd);
  w.self_attn.bias_v = r.Get(attn + "bias_v");
  RequireSize(attn + "bias_v", w.self_attn.bias_v, heads * hd);

  const std::string conv = base + "conv.";
  w.conv.pointwise1_w = r.Get(conv + "pointwise_conv1.weight");
  RequireSize(conv + "pointwise_conv1.weight", w.conv.pointwise1_w, 2 * hidden * hidden);
  w.conv.pointwise1_b = r.GetOptional(conv + "pointwise_conv1.bias");
  w.conv.depthwise_w = r.Get(conv + "depthwise_conv.weight");
  RequireSize(conv + "depthwise_conv.weight", w.conv.depthwise_w,
              hidden * cfg.conv_kernel_size);
  w.conv.depthwise_b = r.GetOptional(conv + "depthwise_conv.bias");
  w.conv.norm_w = r.Get(conv + "norm.weight");
  RequireSize(conv + "norm.weight", w.conv.norm_w, hidden);
  w.conv.norm_b = r.Get(conv + "norm.bias");
  RequireSize(conv + "norm.bias", w.conv.norm_b, hidden);
  w.conv.norm_running_mean = r.Get(conv + "norm.running_mean");
  RequireSize(conv + "norm.running_mean", w.conv.norm_running_mean, hidden);
  w.conv.norm_running_var = r.Get(conv + "norm.running_var");
  RequireSize(conv + "norm.running_var", w.conv.norm_running_var, hidden);
  w.conv.pointwise2_w = r.Get(conv + "pointwise_conv2.weight");
  RequireSize(conv + "pointwise_conv2.weight", w.conv.pointwise2_w, hidden * hidden);
  w.conv.pointwise2_b = r.GetOptional(conv + "pointwise_conv2.bias");

  auto load_norm = [&](const std::string& name, std::vector<float>& weight,
                       std::vector<float>& bias) {
    weight = r.Get(base + name + ".weight");
    RequireSize(base + name + ".weight", weight, hidden);
    bias = r.Get(base + name + ".bias");
    RequireSize(base + name + ".bias", bias, hidden);
  };
  load_norm("norm_feed_forward1", w.norm_feed_forward1_w, w.norm_feed_forward1_b);
  load_norm("norm_self_att", w.norm_self_att_w, w.norm_self_att_b);
  load_norm("norm_conv", w.norm_conv_w, w.norm_conv_b);
  load_norm("norm_feed_forward2", w.norm_feed_forward2_w, w.norm_feed_forward2_b);
  load_norm("norm_out", w.norm_out_w, w.norm_out_b);
  return w;
}

}  // namespace

ParakeetEncoderConfig LoadParakeetConfig(const std::string& dir) {
  return ConfigFromJson(nlohmann::json::parse(ReadFile(dir + "/config.json")));
}

ParakeetForCTCWeights LoadParakeetForCTC(const std::string& dir,
                                         ParakeetEncoderConfig* out_cfg) {
  const ParakeetEncoderConfig cfg = LoadParakeetConfig(dir);
  if (out_cfg != nullptr) *out_cfg = cfg;

  Resolver r(dir);
  // `ParakeetForCTC` nests the encoder under `encoder.` (:680); a standalone
  // `ParakeetEncoder` checkpoint has no prefix.
  const std::string prefix = r.Has("encoder.subsampling.layers.0.weight") ? "encoder." : "";

  ParakeetForCTCWeights w;
  w.encoder.subsampling = LoadSubsampling(r, prefix, cfg);
  w.encoder.layers.reserve(static_cast<size_t>(cfg.num_hidden_layers));
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    w.encoder.layers.push_back(LoadLayer(r, prefix, i, cfg));
  }
  w.ctc_head_w = r.Get("ctc_head.weight");
  RequireSize("ctc_head.weight", w.ctc_head_w, cfg.vocab_size * cfg.hidden_size);
  w.ctc_head_b = r.Get("ctc_head.bias");
  RequireSize("ctc_head.bias", w.ctc_head_b, cfg.vocab_size);
  return w;
}

}  // namespace vllm::multimodal
