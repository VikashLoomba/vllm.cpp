// `CLAIM-DEEPSEEK-V4-MTP` R1 (#1314) — the MTP tail is CLASSIFIED, not just counted.
//
// The shapes below are not invented. They are read from the safetensors headers of
// `/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3` (2026-08-31), so this gate
// states the real artifact as data and needs no 100 GB file to run.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"

using vllm::ClassifyDeepseekV4MtpTail;
using vllm::DeepseekV4MtpTensorDesc;

namespace {

// One head as the artifact stores it: fp8-block attention at exactly 128x128,
// MXFP4 routed experts at group 32, and plain norms.
std::vector<DeepseekV4MtpTensorDesc> RealHead(int64_t l, int64_t experts = 2) {
  const std::string p = "mtp." + std::to_string(l) + ".";
  std::vector<DeepseekV4MtpTensorDesc> v{
      {p + "attn.attn_sink", "F32", {64}},
      {p + "attn.kv_norm.weight", "BF16", {512}},
      {p + "attn_norm.weight", "BF16", {4096}},
      {p + "attn.wkv.weight", "F8_E4M3", {512, 4096}},
      {p + "attn.wkv.weight.scale", "F8_E8M0", {4, 32}},
      {p + "attn.wq_b.weight", "F8_E4M3", {32768, 1024}},
      {p + "attn.wq_b.weight.scale", "F8_E8M0", {256, 8}},
      {p + "ffn.shared_experts.w1.weight", "F8_E4M3", {2048, 4096}},
      {p + "ffn.shared_experts.w1.weight.scale", "F8_E8M0", {16, 32}},
      {p + "ffn.gate.weight", "BF16", {216, 4096}},
  };
  for (int64_t e = 0; e < experts; ++e) {
    const std::string b = p + "ffn.experts." + std::to_string(e) + ".w1.weight";
    v.push_back({b, "I8", {2048, 2048}});          // [N, K/2], so K = 4096
    v.push_back({b + ".scale", "F8_E8M0", {2048, 128}});  // K/32 == 128
  }
  return v;
}

}  // namespace

TEST_CASE("R1: the real MTP tail classifies with no refusal") {
  const auto inv = ClassifyDeepseekV4MtpTail(RealHead(0));
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 1);
  CHECK(inv.plain == 4);      // attn_sink, kv_norm, attn_norm, ffn.gate
  CHECK(inv.fp8_block == 3);  // wkv, wq_b, shared_experts.w1
  CHECK(inv.mxfp4 == 2);      // the two routed experts
}

TEST_CASE("R1: THREE heads are counted, which is the artifact's own shape") {
  // The config says `num_nextn_predict_layers = 1`; this artifact carries three,
  // and three is what makes a K5 draft possible. The count must come from the
  // TENSORS, never from the config, or the extra heads are silently invisible.
  std::vector<DeepseekV4MtpTensorDesc> all;
  for (int64_t l = 0; l < 3; ++l) {
    const auto h = RealHead(l);
    all.insert(all.end(), h.begin(), h.end());
  }
  const auto inv = ClassifyDeepseekV4MtpTail(all);
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 3);
  CHECK(inv.mxfp4 == 6);
  CHECK(inv.fp8_block == 9);
}

TEST_CASE("R1: an absent tail is EMPTY, not a refusal") {
  // The two shipped GGUFs have no tail at all. That is a legitimate checkpoint,
  // and it must read as "no head here" rather than as a broken one.
  const auto inv = ClassifyDeepseekV4MtpTail({});
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 0);
  CHECK(inv.plain + inv.fp8_block + inv.mxfp4 == 0);
}

TEST_CASE("R1: NVFP4's group of 16 is REFUSED, not read as MXFP4") {
  // The 156.7 GiB checkpoint's tail uses NVFP4: group 16, a double scale. Reading
  // it with the MXFP4 group-32 path would dequantize every weight against the
  // wrong exponent and produce a plausible, entirely wrong draft head.
  auto v = RealHead(0, /*experts=*/1);
  for (auto& d : v) {
    if (d.name.find("experts.0.w1.weight.scale") != std::string::npos) {
      d.shape = {2048, 256};  // K/16, the NVFP4 grouping
    }
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("experts.0.w1") != std::string::npos);
  CHECK(inv.mxfp4 == 0);
}

TEST_CASE("R1: a quantized weight with NO scale is refused by name") {
  auto v = RealHead(0, /*experts=*/1);
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i].name.find("attn.wkv.weight.scale") != std::string::npos) {
      v.erase(v.begin() + static_cast<long>(i));
      break;
    }
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("attn.wkv.weight") != std::string::npos);
  // wq_b AND shared_experts.w1 still classify; only wkv is refused.
  CHECK(inv.fp8_block == 2);
}

TEST_CASE("R1: a scale that does not tile its weight at 128x128 is refused") {
  // A block size that is not 128 is a DIFFERENT fp8 layout. Accepting it would
  // walk the scale array with the wrong stride.
  auto v = RealHead(0, /*experts=*/1);
  for (auto& d : v) {
    if (d.name == "mtp.0.attn.wkv.weight.scale") d.shape = {8, 32};  // 64x128
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("128x128") != std::string::npos);
}
