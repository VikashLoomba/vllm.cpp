// Qwen4-Exp W5f — `Qwen4ExpTextModel::Forward` against the ORACLE, end to end.
//
// Row MODEL-MM-QWEN4-EXP, issues #2031 and #2336, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS SUITE CAN SEE THAT NO EARLIER ONE COULD ───────────────────────
// Eleven waves of this row gated ONE thing each: `RunGdnBlockPaged`,
// `RunQwen4ExpQsaBlockPaged`, `RunQwen4ExpMoeBlock`, `RunQwen4ExpPleBlock`, the
// two hyper-connection ops. Every one of them is green and NONE of them can see:
//
//   * the LAYER ORDER — which index runs Gated DeltaNet and which runs Qwen
//     Sparse Attention;
//   * the hyper-connection ARITHMETIC BETWEEN the blocks — that `hyper_input` is
//     the RAW stream, that the write-back is rank-1 onto it, and that there are
//     TWO sites per layer and not one;
//   * where the PLE layer SITS — first in its own decoder layer, on the hc-wide
//     stream, and added to it;
//   * the `repeat(1, 1, hc_count)` WIDEN at the top and the `use_combine=False`
//     mixer at the bottom, with NO final RMSNorm between the mixer and the head.
//
// A composition of correct parts in the wrong order is exactly the defect class
// a per-block gate is blind to, and it produces fluent wrong text rather than a
// crash. The golden below is an observation of `Qwen4ExpTextModel.forward`
// itself, so it sees all four.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// transformers **5.16.0**, this row's accepted lane pin
// (`.agents/oracles/transformers.md`), sha256
// `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`, generated
// by `scripts/gen-qwen4-exp-forward-goldens.py`, which ASSERTS that sha256
// against the file it imported before it observes anything. vLLM registers no
// `qwen4_exp` at the pinned revision, so there is no primary oracle to mirror.
//
// ─── THE TOLERANCE, AND WHY IT IS NOT 1e-5 ───────────────────────────────────
// The oracle runs the whole tower in f32; this tree runs the model path at the
// dtype the model resolves, which is bf16, exactly as AGENTS.md "Inherit vLLM
// defaults" requires. Four layers of bf16 rounding is the residual this gate
// measures, and it is NOT a tolerance chosen to make the case pass: the WEIGHTS
// are bf16-exact by construction (the generator asserts a bf16 round trip per
// tensor), so the only rounding is in the activations. The bound is stated as a
// constant below, the MEASURED residual is printed, and every mutation in the
// spec's record separates by more than an order of magnitude above it — which is
// the property that makes the bound a gate rather than a mute switch.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen4_exp_forward.h"
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "support/qwen4_exp_gguf_fixture.h"

#include "../../support/max_abs_diff.h"

namespace {

#include "qwen4_exp_forward_goldens.inc"

using vllm::OwnedTensor;
using vllm::Qwen4ExpLayerKind;
using vllm::Qwen4ExpParams;
using vllm::Qwen4ExpWeights;
using vt::DType;

constexpr int64_t kStream = kFwdHcCount * kFwdHiddenSize;
constexpr int64_t kNgramHeads = (kFwdNgramSize - 1) * kFwdHeadsPerNgram;

// THE BOUND. See the header note: bf16 activations against an f32 oracle over
// four layers. Reported as a measured value by the case itself.
constexpr float kTol = 3.0e-2F;

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

OwnedTensor Owned(DType dt, const std::vector<int64_t>& shape, bool nk = false) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  t.nk = nk;
  t.bytes.assign(static_cast<size_t>(Numel(shape)) * vt::SizeOf(dt), 0);
  return t;
}

// Every weight below arrives from the golden in the ORACLE's own orientation and
// in the RAW HuggingFace gamma parameterization, which is precisely what the
// loader leaves in `Qwen4ExpWeights` (#2218). Nothing is transposed or folded
// here: a fixture that pre-chewed either would be gating itself.
//
// `nk` DEFAULTS TO TRUE BECAUSE `LoadMatmul` PRODUCES TRUE, unconditionally
// (`qwen4_exp_weights.cpp`: every arm — keep-quant blocks, f16 and the bf16
// expansion — passes `nk = true`). The flag is not decoration: consumers that
// branch on it read the SAME BYTES in two orientations. `BorrowWhole` in the MoE
// adapter preserves the source's flag and hands it to `MatmulF32D`, so a fixture
// that built the shared expert's projections `nk = false` fed the seam a [K, N]
// weight where a [N, K] one was meant — the same element count, no shape error,
// and a wrong answer. That was the first defect this gate found, and it was in
// the fixture rather than in the loop, which is why the default is stated here
// once instead of at thirty call sites.
OwnedTensor Bf16(const float* src, const std::vector<int64_t>& shape,
                 bool nk = true) {
  OwnedTensor t = Owned(DType::kBF16, shape, nk);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  const int64_t n = Numel(shape);
  for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(src[i]);
  return t;
}

OwnedTensor F32(const float* src, const std::vector<int64_t>& shape) {
  OwnedTensor t = Owned(DType::kF32, shape);
  std::memcpy(t.bytes.data(), src, t.bytes.size());
  return t;
}

Qwen4ExpParams MakeParams() {
  Qwen4ExpParams p;
  p.hidden_size = kFwdHiddenSize;
  p.num_hidden_layers = kFwdLayers;
  p.vocab_size = kFwdVocabSize;
  p.rms_norm_eps = kFwdRmsNormEps;
  p.hc_count = kFwdHcCount;
  p.hc_lowrank = kFwdHcLowrank;
  p.num_experts = kFwdNumExperts;
  p.num_experts_per_tok = kFwdNumExpertsPerTok;
  p.moe_intermediate_size = kFwdMoeIntermediateSize;
  p.shared_expert_intermediate_size = kFwdSharedExpertIntermediateSize;
  p.linear_num_key_heads = kFwdLinearNumKeyHeads;
  p.linear_num_value_heads = kFwdLinearNumValueHeads;
  p.linear_key_head_dim = kFwdLinearKeyHeadDim;
  p.linear_value_head_dim = kFwdLinearValueHeadDim;
  p.linear_conv_kernel_dim = kFwdLinearConvKernelDim;
  p.eos_token_id = kFwdEosTokenId;
  p.num_attention_heads = kFwdNumAttentionHeads;
  p.num_key_value_heads = kFwdNumKeyValueHeads;
  p.head_dim = kFwdHeadDim;
  p.rotary_dim = kFwdRotaryDim;
  p.qsa.n_heads = kFwdIndexerNHeads;
  p.qsa.kv_heads = kFwdIndexerKvHeads;
  p.qsa.head_dim = kFwdIndexerHeadDim;
  p.qsa.budget = kFwdIndexerBudget;
  p.qsa.compress_ratio = kFwdIndexerCompressRatio;
  p.ple.embed_dim = kFwdPleEmbedDim;
  p.ple.conv_kernel_size = kFwdPleConvKernelSize;
  p.ple.ngram_size = kFwdNgramSize;
  p.ple.heads_per_ngram = kFwdHeadsPerNgram;
  p.ple.ngram_vocab_size_base = kFwdNgramVocabBase;
  p.ple.make_ngram_vocab_size_divisible_by = kFwdNgramVocabDivisor;
  p.ple.seed = kFwdSeed;
  p.ple.layer_ids_zero_based = {kFwdPleLayerZeroBased};
  // THE LAYER SCHEDULE, spelled out rather than synthesized from an interval,
  // because it is the thing under test. `full_attention_interval` 4 over 4
  // layers gives 3 linear then 1 sparse, which is the released pattern at its
  // shortest, and the golden's own config declares exactly this list.
  p.layer_types = {Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kQwenSparseAttention};
  return p;
}

vllm::HfConfig MakeHfConfig(const Qwen4ExpParams& p) {
  vllm::HfConfig c;
  c.model_type = "qwen4_exp_text";
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.rms_norm_eps = p.rms_norm_eps;
  c.num_experts = p.num_experts;
  c.linear_num_key_heads = p.linear_num_key_heads;
  c.linear_num_value_heads = p.linear_num_value_heads;
  c.linear_key_head_dim = p.linear_key_head_dim;
  c.linear_value_head_dim = p.linear_value_head_dim;
  c.linear_conv_kernel_dim = p.linear_conv_kernel_dim;
  c.rotary_dim = p.rotary_dim;
  c.rope_theta = kFwdRopeTheta;
  c.rope_parameters.rope_theta = kFwdRopeTheta;
  // THE VALUE THE CHECKPOINT ASKED FOR, and the reason `Qwen4ExpGdnHfConfig`
  // carries it rather than deriving it: `Qwen4ExpParams` has no field for it.
  c.output_gate_type = kFwdOutputGateType;
  c.mamba_ssm_dtype = "float32";
  return c;
}

struct HcArrays {
  const float* norm;
  const float* down;
  const float* up;
  const float* inject;  // nullptr on the terminal mixer
};

void FillHc(vllm::Qwen4ExpGatedResidualWeights& g, const HcArrays& a,
            const Qwen4ExpParams& p) {
  g.hc_norm = Bf16(a.norm, {kStream});
  g.down = Bf16(a.down, {p.hc_lowrank, kStream});
  g.up = Bf16(a.up, {kStream, p.hc_lowrank});
  g.has_inject = a.inject != nullptr;
  if (g.has_inject) g.inject = Bf16(a.inject, {p.hc_count, kStream});
}

struct GdnArrays {
  const float *qkv, *z, *b, *a, *conv, *alog, *dt, *norm, *out;
};

void FillGdn(vllm::Qwen4ExpGdnWeights& g, const GdnArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t num_v = p.linear_num_value_heads;
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = num_v * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  // `nk = true` — the `[N, K]` orientation both GGUF loaders produce
  // (`MakeGdnProj` under `GgufLoadPolicy::gdn_expand_nk`). Passing these as
  // `nk = false` would be read transposed with no shape error.
  g.in_proj_qkv = Bf16(s.qkv, {conv_dim, H}, /*nk=*/true);
  g.in_proj_z = Bf16(s.z, {value_dim, H}, /*nk=*/true);
  g.in_proj_b = Bf16(s.b, {num_v, H}, /*nk=*/true);
  g.in_proj_a = Bf16(s.a, {num_v, H}, /*nk=*/true);
  g.conv1d = Bf16(s.conv, {conv_dim, p.linear_conv_kernel_dim});
  g.a_log = F32(s.alog, {num_v});
  g.dt_bias = F32(s.dt, {num_v});
  // THE ONE GAMMA WITH NO FOLD ANYWHERE: `Qwen4ExpTextRMSNormGated.weight` is
  // ones-init and its forward multiplies by `weight` directly, which is why
  // ggml-org/llama.cpp#27742 excludes exactly this tensor and the loader does
  // not unshift it.
  g.norm_weight = Bf16(s.norm, {p.linear_value_head_dim});
  g.out_proj = Bf16(s.out, {H, value_dim}, /*nk=*/true);
}

struct MoeArrays {
  const float *router, *gate_exps, *up_exps, *down_exps, *shared_gate,
      *shared_gate_proj, *shared_up_proj, *shared_down_proj;
};

void FillMoe(vllm::Qwen4ExpMoeWeights& m, const MoeArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t I = p.moe_intermediate_size;
  const int64_t S = p.shared_expert_intermediate_size;
  // f32 BY THE LOADER'S OWN CHOICE (`qwen4_exp_weights.cpp:437-447`), and
  // `Qwen4ExpMoeBlockWeights` refuses either of them at any other dtype.
  m.router = F32(s.router, {E, H});
  m.shared_gate = F32(s.shared_gate, {H});
  m.gate_exps = Bf16(s.gate_exps, {E, I, H});
  m.up_exps = Bf16(s.up_exps, {E, I, H});
  m.down_exps = Bf16(s.down_exps, {E, H, I});
  m.shared_gate_proj = Bf16(s.shared_gate_proj, {S, H});
  m.shared_up_proj = Bf16(s.shared_up_proj, {S, H});
  m.shared_down_proj = Bf16(s.shared_down_proj, {H, S});
}

Qwen4ExpWeights MakeWeights(const Qwen4ExpParams& p) {
  Qwen4ExpWeights w;
  w.params = p;
  w.embed_tokens = Bf16(kFwdEmbedTokens, {p.vocab_size, p.hidden_size});
  w.tied_word_embeddings = true;
  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));

  const HcArrays hc[kFwdLayers][2] = {
      {{kFwdL0AttnHcNorm, kFwdL0AttnHcDown, kFwdL0AttnHcUp, kFwdL0AttnHcInject},
       {kFwdL0MlpHcNorm, kFwdL0MlpHcDown, kFwdL0MlpHcUp, kFwdL0MlpHcInject}},
      {{kFwdL1AttnHcNorm, kFwdL1AttnHcDown, kFwdL1AttnHcUp, kFwdL1AttnHcInject},
       {kFwdL1MlpHcNorm, kFwdL1MlpHcDown, kFwdL1MlpHcUp, kFwdL1MlpHcInject}},
      {{kFwdL2AttnHcNorm, kFwdL2AttnHcDown, kFwdL2AttnHcUp, kFwdL2AttnHcInject},
       {kFwdL2MlpHcNorm, kFwdL2MlpHcDown, kFwdL2MlpHcUp, kFwdL2MlpHcInject}},
      {{kFwdL3AttnHcNorm, kFwdL3AttnHcDown, kFwdL3AttnHcUp, kFwdL3AttnHcInject},
       {kFwdL3MlpHcNorm, kFwdL3MlpHcDown, kFwdL3MlpHcUp, kFwdL3MlpHcInject}},
  };
  const GdnArrays gdn[3] = {
      {kFwdL0GdnInProjQkv, kFwdL0GdnInProjZ, kFwdL0GdnInProjB, kFwdL0GdnInProjA,
       kFwdL0GdnConv1d, kFwdL0GdnALog, kFwdL0GdnDtBias, kFwdL0GdnNorm,
       kFwdL0GdnOutProj},
      {kFwdL1GdnInProjQkv, kFwdL1GdnInProjZ, kFwdL1GdnInProjB, kFwdL1GdnInProjA,
       kFwdL1GdnConv1d, kFwdL1GdnALog, kFwdL1GdnDtBias, kFwdL1GdnNorm,
       kFwdL1GdnOutProj},
      {kFwdL2GdnInProjQkv, kFwdL2GdnInProjZ, kFwdL2GdnInProjB, kFwdL2GdnInProjA,
       kFwdL2GdnConv1d, kFwdL2GdnALog, kFwdL2GdnDtBias, kFwdL2GdnNorm,
       kFwdL2GdnOutProj},
  };
  const MoeArrays moe[kFwdLayers] = {
      {kFwdL0MoeRouter, kFwdL0MoeGateExps, kFwdL0MoeUpExps, kFwdL0MoeDownExps,
       kFwdL0MoeSharedGate, kFwdL0MoeSharedGateProj, kFwdL0MoeSharedUpProj,
       kFwdL0MoeSharedDownProj},
      {kFwdL1MoeRouter, kFwdL1MoeGateExps, kFwdL1MoeUpExps, kFwdL1MoeDownExps,
       kFwdL1MoeSharedGate, kFwdL1MoeSharedGateProj, kFwdL1MoeSharedUpProj,
       kFwdL1MoeSharedDownProj},
      {kFwdL2MoeRouter, kFwdL2MoeGateExps, kFwdL2MoeUpExps, kFwdL2MoeDownExps,
       kFwdL2MoeSharedGate, kFwdL2MoeSharedGateProj, kFwdL2MoeSharedUpProj,
       kFwdL2MoeSharedDownProj},
      {kFwdL3MoeRouter, kFwdL3MoeGateExps, kFwdL3MoeUpExps, kFwdL3MoeDownExps,
       kFwdL3MoeSharedGate, kFwdL3MoeSharedGateProj, kFwdL3MoeSharedUpProj,
       kFwdL3MoeSharedDownProj},
  };

  int64_t gi = 0;
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    vllm::Qwen4ExpLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    lw.is_linear_attention =
        p.layer_types[static_cast<size_t>(il)] == Qwen4ExpLayerKind::kLinearAttention;
    FillHc(lw.attn_hc, hc[il][0], p);
    FillHc(lw.mlp_hc, hc[il][1], p);
    FillMoe(lw.moe, moe[il], p);
    if (lw.is_linear_attention) {
      FillGdn(lw.gdn, gdn[gi++], p);
    } else {
      const int64_t H = p.hidden_size;
      lw.qsa.q_proj = Bf16(kFwdL3QsaQProj, {p.num_attention_heads * p.head_dim * 2, H});
      lw.qsa.k_proj = Bf16(kFwdL3QsaKProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.v_proj = Bf16(kFwdL3QsaVProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.o_proj = Bf16(kFwdL3QsaOProj, {H, p.num_attention_heads * p.head_dim});
      lw.qsa.q_norm = Bf16(kFwdL3QsaQNorm, {p.head_dim});
      lw.qsa.k_norm = Bf16(kFwdL3QsaKNorm, {p.head_dim});
      lw.qsa.idx_q_proj = Bf16(kFwdL3QsaIdxQProj, {p.qsa.n_heads * p.qsa.head_dim, H});
      lw.qsa.idx_k_proj = Bf16(kFwdL3QsaIdxKProj, {p.qsa.kv_heads * p.qsa.head_dim, H});
      lw.qsa.idx_q_norm = Bf16(kFwdL3QsaIdxQNorm, {p.qsa.head_dim});
      lw.qsa.idx_k_norm = Bf16(kFwdL3QsaIdxKNorm, {p.qsa.head_dim});
    }
    if (il == kFwdPleLayerZeroBased) {
      lw.has_ple = true;
      lw.ple.key_proj = Bf16(kFwdL1PleKeyProj, {kStream, p.ple.embed_dim});
      lw.ple.value_proj = Bf16(kFwdL1PleValueProj, {p.hidden_size, p.ple.embed_dim});
      lw.ple.norm_key = Bf16(kFwdL1PleNormKey, {kStream});
      lw.ple.norm_query = Bf16(kFwdL1PleNormQuery, {kStream});
      lw.ple.norm_conv = Bf16(kFwdL1PleNormConv, {kStream});
      lw.ple.conv1d = Bf16(kFwdL1PleConv1d, {kStream, p.ple.conv_kernel_size});
      w.ngram_table =
          Bf16(kFwdL1PleNgramTable, {kFwdNgramPaddedVocab, kFwdNgramHeadDim});
    }
  }
  FillHc(w.mixer, {kFwdMixerNorm, kFwdMixerDown, kFwdMixerUp, nullptr}, p);
  return w;
}

std::vector<float> Download(vllm::dense_attn::Dev d, const vt::Tensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == DType::kBF16) {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    d.b.Copy(d.q, raw.data(), t.data, raw.size() * sizeof(uint16_t));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] = vt::BF16ToF32(raw[static_cast<size_t>(i)]);
  } else {
    REQUIRE(t.dtype == DType::kF32);
    d.b.Copy(d.q, out.data(), t.data, out.size() * sizeof(float));
  }
  return out;
}


}  // namespace

TEST_CASE(
    "qwen4_exp layer loop: Qwen4ExpTextModel::Forward matches the transformers "
    "5.16.0 oracle end to end") {
  // ─── THE FIXTURE'S OWN CONDITIONING, ASSERTED BEFORE ANYTHING RUNS ───────
  // A top-k router is a DISCRETE SELECTION and its error is BIMODAL: either the
  // two sides pick the same experts and the residual is bf16-sized, or they pick
  // different ones and it is O(1). No tolerance can straddle that. This suite is
  // gateable only because the fixture's worst router margin is far above the
  // hidden-state residual the bf16 model path introduces, and the generator
  // measures that margin on the same run that produced the golden.
  //
  // THE FLOOR IS NOT DECORATION. The first draft of this fixture had a margin of
  // 0.0164 against a residual of 0.0152, layer 0 flipped an expert, and the
  // end-to-end residual read 0.466 — a fixture artefact that reads exactly like a
  // broken layer loop. Regenerating the golden onto another draw can put it back
  // there, and this REQUIRE is what makes that loud instead of silent.
  REQUIRE_MESSAGE(kFwdMinRouterMargin > 0.25,
                  "the fixture's worst router margin is "
                      << kFwdMinRouterMargin
                      << ", which is too close to the bf16 hidden-state residual "
                         "for a top-k selection to be stable; regenerate with "
                         "scripts/gen-qwen4-exp-forward-goldens.py and pick a "
                         "better-conditioned SALT / ROUTER_SCALE rather than "
                         "widening the tolerance");

  const Qwen4ExpParams p = MakeParams();
  const vllm::HfConfig config = MakeHfConfig(p);
  Qwen4ExpWeights w = MakeWeights(p);

  // THE GDN OUTPUT GATE IS SIGMOID HERE AND THE SHARED READER DEFAULTS TO SILU.
  // Asserted rather than assumed, because it is the one field `Qwen4ExpParams`
  // cannot carry and the whole reason `Qwen4ExpGdnHfConfig` takes a source
  // config: the two activations differ on 36 of 48 layers of the released model
  // and no shape check can tell them apart (#489).
  REQUIRE(std::string(kFwdOutputGateType) == "sigmoid");
  REQUIRE(vllm::Qwen4ExpGdnHfConfig(p, config).output_gate_type == "sigmoid");

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t T = kFwdSeqLen;
  std::vector<int32_t> token_ids(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    token_ids[static_cast<size_t>(t)] = static_cast<int32_t>(kFwdInputIds[t]);
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  // ── the step metadata, exactly as the runner builds it for a prefill ──
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  // ── the caches ──
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = p.linear_num_value_heads * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = p.linear_conv_kernel_dim - 1;
  const int64_t ssm_row =
      p.linear_num_value_heads * p.linear_value_head_dim * p.linear_key_head_dim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  vllm::Qwen4ExpForwardCaches caches;
  caches.gdn.resize(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, p.linear_num_value_heads,
                                            p.linear_value_head_dim,
                                            p.linear_key_head_dim},
                       ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    caches.gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    caches.gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(T * p.num_key_value_heads *
                                               p.head_dim * 2),
                           0);
  std::vector<uint16_t> index_key(
      static_cast<size_t>(T * p.qsa.head_dim), 0);
  std::vector<int32_t> bt{0};
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) slots[static_cast<size_t>(t)] = t;
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, 1, T, p.num_key_value_heads, p.head_dim},
                              kv.data());
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {T, p.qsa.head_dim},
                               index_key.data());
  vllm::dense_attn::DBuf bt_b(d, DType::kI32, {1, 1}, bt.data());
  vllm::dense_attn::DBuf slot_b(d, DType::kI64, {T}, slots.data());
  caches.qsa.resize(1);
  caches.qsa[0].kv.data = kv_b.t().data;
  caches.qsa[0].kv.dtype = DType::kBF16;
  caches.qsa[0].kv.num_blocks = 1;
  caches.qsa[0].kv.block_size = T;
  caches.qsa[0].kv.num_kv_heads = p.num_key_value_heads;
  caches.qsa[0].kv.head_size = p.head_dim;
  caches.qsa[0].block_table = bt_b.t();
  caches.qsa[0].slot_mapping = slot_b.t();
  caches.qsa[0].index_key = idx_b.t();

  const int64_t state_len = p.ple.short_conv_state_len();
  std::vector<float> ple_conv(static_cast<size_t>(kStream * state_len), 0.0F);
  std::vector<int64_t> ple_tok(static_cast<size_t>(p.ple.ngram_size - 1), 0);
  vllm::dense_attn::DBuf ple_conv_b(d, DType::kF32, {1, kStream, state_len},
                                    ple_conv.data());
  caches.ple.resize(1);
  caches.ple[0].conv_state = ple_conv_b.t();
  caches.ple[0].tokens = vllm::dense_attn::MakeTensor(
      ple_tok.data(), DType::kI64, vt::Device{vt::DeviceType::kCPU, 0},
      {1, p.ple.ngram_size - 1});
  caches.ple[0].state_row = 0;

  // ── the n-gram LAYOUT, checked against the oracle's own buffers ──
  // The C++ side DERIVES the prime chain and the offsets; the golden carries
  // what upstream derived. Comparing them is what stops this fixture from
  // agreeing with itself about a 320-million-row table's addressing.
  {
    const vllm::qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
    REQUIRE(static_cast<int64_t>(layout.head_vocab_sizes.size()) == kNgramHeads);
    for (int64_t i = 0; i < kNgramHeads; ++i) {
      CHECK(layout.head_vocab_sizes[static_cast<size_t>(i)] ==
            kFwdNgramHeadVocabSizes[i]);
      CHECK(layout.head_offsets[static_cast<size_t>(i)] == kFwdNgramHeadOffsets[i]);
    }
    CHECK(layout.padded_vocab_size == kFwdNgramPaddedVocab);
    REQUIRE(layout.layer_multipliers.size() == 3);
    for (int i = 0; i < 3; ++i)
      CHECK(layout.layer_multipliers[static_cast<size_t>(i)] ==
            kFwdNgramLayerMultipliers[i]);
  }

  const vllm::Qwen4ExpTextModelOutput out = vllm::Qwen4ExpTextModelForward(
      d, w, config, token_ids, positions, am, gm, caches, /*past_len=*/0);

  REQUIRE(out.storage != nullptr);
  REQUIRE(out.tensor.rank == 2);
  REQUIRE(out.tensor.shape[0] == T);
  REQUIRE(out.tensor.shape[1] == p.hidden_size);

  const std::vector<float> got = Download(d, out.tensor);
  REQUIRE(got.size() == static_cast<size_t>(T * p.hidden_size));

  // FINITENESS BEFORE ANY TOLERANCE (#2272, #449). `vllm_test::MaxAbsDiff`
  // returns +infinity on a non-finite operand rather than the NaN-blind zero the
  // two obvious `std::max` folds return, and it raises on the offending index —
  // but an explicit scan first says WHICH failure a red is.
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE_MESSAGE(std::isfinite(got[i]),
                    "the layer loop emitted a non-finite value at index " << i);
  }

  const double worst = vllm_test::MaxAbsDiff(got, kFwdExpectedHidden,
                                             got.size());
  MESSAGE("layer loop vs transformers 5.16.0: max|diff| = " << worst
          << " against a bound of " << kTol);
  CHECK(worst < kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// REACHABILITY (AGENTS.md "Nothing lands dead", `.agents/reachability.md`).
//
// The case above drives `Qwen4ExpTextModelForward` DIRECTLY, which proves the
// arithmetic and proves nothing about whether anything reaches it. This one
// enters through a PRODUCTION entry point — `ModelRegistry::Forward` — on a
// model produced by `ModelRegistry::Load` from a real `qwen4exp` GGUF, and it is
// the first time this architecture has run a forward from one.
//
// WHAT IT CAN AND CANNOT BE. It cannot be a token gate: the synthetic fixture's
// weights are a deterministic ramp, not a checkpoint, so there is no reference
// token stream. It cannot run through `GPUModelRunner` either — that path sets
// `multi_kv`, which `ModelRegistry::Forward` refuses by name for every model, a
// refusal #2353 established must NOT be lifted yet. What it IS: proof that the
// registered hook is entered, opens its handle, assembles the caches, runs all
// four block seams over 4 layers and returns finite logits of the right shape.
// The mutation that reds it is deleting the `Qwen4ExpTextModelForward` call
// site, which is exactly the mutation `.agents/reachability.md` step 5 asks for
// and which the three preceding waves could only record as VACUOUS.
TEST_CASE(
    "qwen4_exp layer loop: ModelRegistry::Forward reaches it on a loaded "
    "qwen4exp GGUF") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // Every id in range and the LAST one EOS, so the n-gram hash sees a segment
    // boundary rather than a uniform ramp.
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // The two POSITIONAL channels, sized from the fixture's own geometry: three
  // Gated DeltaNet layers and one Qwen Sparse Attention layer.
  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(
      static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, 1, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids,     pos,      am, gm, attn_kv,
                             gdn,     config,   q,  logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // ─── WHAT THIS CASE ASSERTS, AND WHY IT IS NOT A COMPLETED FORWARD ────────
  //
  // THE SHARED GGUF FIXTURE IS INTERNALLY INCONSISTENT AND NOTHING COULD SEE IT
  // UNTIL A FORWARD RAN THE PLE LAYER ON IT. `tests/support/qwen4_exp_gguf_fixture.h`
  // STATES `qwen4exp.ple.head_vocab_sizes = {23, 29}`, and its own comment says
  // those are "what the HF derivation would produce from
  // `ngram_vocab_size_base = 20`". But the GGUF CONTAINER HAS NO
  // `ngram_vocab_size_base` KEY — `Qwen4ExpHfConfigFromGguf` never reads one, and
  // ggml-org/llama.cpp#27742 writes the RESOLVED sizes instead of the base — so
  // the parsed config carries upstream's DEFAULT of 20,000,000 and the prime
  // chain derives 20,000,003. W5e-2's `Qwen4ExpPleLayout` cross-checks the two
  // and refuses BY NAME, which is correct behaviour on a genuine disagreement
  // and is what a tiny fixture stating sizes from a tiny base will always hit.
  //
  // ON A REAL `qwen4exp` FILE THE TWO AGREE, which is why this is a fixture
  // defect and not a port one: the released config's base IS 20,000,000, so the
  // chain derives exactly the sizes the file states. A fixture cannot have both
  // — a table addressed from base 20,000,000 needs 40 million rows.
  //
  // SO THE REACH THIS CASE PROVES IS EXACT AND IT IS STATED AS SUCH:
  // `ModelRegistry::Forward` enters the registered hook, the hook opens its
  // handle, assembles the caches and CALLS `Qwen4ExpTextModelForward`, and the
  // loop runs layer 0 (Gated DeltaNet, MoE, both hyper-connection sites) and
  // reaches layer 1's PLE block, where the fixture's own inconsistency stops it.
  // The ARITHMETIC of the whole tower — PLE included — is gated by the golden
  // case above, which drives the same function directly.
  //
  // TWO-SIDED ON PURPOSE. Asserting only that the PLE message appears would be a
  // spelling gate; asserting ALSO that the old unconditional refusal is GONE is
  // what makes it a reach claim. Deleting the `Qwen4ExpTextModelForward` call
  // site reds both halves, which is the mutation `.agents/reachability.md`
  // step 5 asks for and which W5b-5, W5d-3, W5d-4, W5e-1 and W5e-2 could each
  // only record as VACUOUS.
  std::string message;
  try {
    (void)vllm::ModelRegistry::Forward(*model, in);
    FAIL("ModelRegistry::Forward returned; the fixture cannot get past the PLE "
         "layout cross-check, so this case's premise is stale");
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("ModelRegistry::Forward said: ", message);
  // It got INTO the loop and INTO the PLE block.
  CHECK(message.find("qwen4_exp ple layout") != std::string::npos);
  CHECK(message.find("vocabulary size disagrees") != std::string::npos);
  // And it is NOT the pre-W5f unconditional refusal, which is what deleting the
  // call site would put back.
  CHECK(message.find("the forward is not ported") == std::string::npos);
  CHECK(message.find("LAYER LOOP") == std::string::npos);

  // ─── THE TWO REFUSALS THIS HOOK ADVERTISES, GATED BY THEIR MESSAGE ────────
  //
  // A BARE `CHECK_THROWS` HERE WOULD BE A MUTE SWITCH, and it was one until this
  // repair. Every input below reaches the loop if its refusal is deleted, and
  // the loop then throws anyway at the SAME PLE layout cross-check the golden
  // reach above lands on. An unrelated exception satisfied the assertion, so
  // deleting either `VT_CHECK` in `qwen4_exp_registry.cpp` left this suite at
  // 2 cases / 76 assertions SUCCESS and the scaffold suite at 12 / 294.
  //
  // So each refusal is asserted TWO-SIDED on its own MESSAGE: the bytes that
  // identify it are present, AND the PLE message that means the input got past
  // it is absent. Deleting either `VT_CHECK` flips both halves of its pair.
  auto refusal_message = [&](const vllm::ModelForwardInput& bad,
                             const char* what) {
    std::string msg;
    try {
      (void)vllm::ModelRegistry::Forward(*model, bad);
      FAIL(what);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // `past_len != 0` means the QSA indexer side cache and the PLE conv ring and
  // n-gram history would have had to persist across a step, and no channel
  // carries them. The refusal must fire BEFORE the loop runs.
  vllm::v1::CommonAttentionMetadata am2 = am;
  am2.seq_lens.assign(1, static_cast<int32_t>(T + 1));
  vllm::ModelForwardInput in2{ids,     pos,      am2, gm, attn_kv,
                              gdn,     config,   q,   logits_indices};
  in2.num_reqs = 1;
  in2.gdn_state_slots = 1;
  const std::string past_len_msg = refusal_message(
      in2,
      "ModelRegistry::Forward returned on a step at past_len 1; the "
      "single-shot-prefill refusal is gone");
  INFO("the past_len refusal said: ", past_len_msg);
  CHECK(past_len_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  CHECK(past_len_msg.find("SINGLE-SHOT") != std::string::npos);
  // The VALUE, not only the word: the hook reports the past_len it was handed.
  CHECK(past_len_msg.find("at past_len 1") != std::string::npos);
  // It stopped at the boundary and never entered the loop, so the PLE layout
  // cross-check the golden reach above lands on cannot be what threw.
  CHECK(past_len_msg.find("qwen4_exp ple layout") == std::string::npos);

  // `num_reqs != 1` is the OTHER refusal, and nothing drove it at all before
  // this repair. `RunQwen4ExpQsaBlockPaged` takes a block_table of i32
  // [1, max_pages], so a ragged multi-request batch has no plumbing here.
  vllm::ModelForwardInput in3{ids,     pos,      am, gm, attn_kv,
                              gdn,     config,   q,  logits_indices};
  in3.num_reqs = 2;
  in3.gdn_state_slots = 1;
  const std::string num_reqs_msg = refusal_message(
      in3,
      "ModelRegistry::Forward returned on a step carrying num_reqs 2; the "
      "one-sequence-per-call refusal is gone");
  INFO("the num_reqs refusal said: ", num_reqs_msg);
  CHECK(num_reqs_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  CHECK(num_reqs_msg.find("ONE sequence per call") != std::string::npos);
  // The VALUE again: the count the step actually carried.
  CHECK(num_reqs_msg.find("the step carries 2") != std::string::npos);
  CHECK(num_reqs_msg.find("qwen4_exp ple layout") == std::string::npos);
}
