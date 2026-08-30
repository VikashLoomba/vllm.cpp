// Qwen4-Exp W5e-2 — `Qwen4ExpTextPLELayer` as one production block. See
// `qwen4_exp_ple_block.h` for why this file exists, which two silent contracts
// it carries, and what it deliberately does not cover.
//
// ALGORITHM ORACLE: transformers 5.16.0 (this row's accepted lane pin, sha256
// `77fec77d…c459`), `models/qwen4_exp/modeling_qwen4_exp.py`. Every line below
// cites the upstream line it mirrors, re-derived by reading the pinned file.
// OP ORACLE: vLLM, through the `vt::` primitives — this block introduces no
// arithmetic of its own, which is the whole point of it being a composition
// rather than a kernel.
#include "vllm/model_executor/models/qwen4_exp_ple_block.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // ResidentWeight
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using vt::DType;
using vt::Tensor;

// A contiguous ROW-RANGE view of a contiguous tensor, reshaped. Same helper the
// QSA block carries, and for the same reason: the element count must agree,
// which is the check that stops a reshape from quietly renaming a stride.
Tensor RowsView(const Tensor& t, int64_t start, int64_t count,
                const std::vector<int64_t>& shape) {
  VT_CHECK(t.rank >= 1 && t.IsContiguous(),
           "qwen4_exp ple block: RowsView needs a contiguous tensor");
  VT_CHECK(start >= 0 && count >= 0 && start + count <= t.shape[0],
           "qwen4_exp ple block: RowsView range outside the tensor");
  int64_t row_elems = 1;
  for (int i = 1; i < t.rank; ++i) row_elems *= t.shape[i];
  int64_t want = 1;
  for (int64_t s : shape) want *= s;
  VT_CHECK(want == count * row_elems,
           "qwen4_exp ple block: RowsView shape does not cover the rows it names");
  return dense_attn::MakeTensor(
      static_cast<char*>(t.data) +
          static_cast<size_t>(start * row_elems) * vt::SizeOf(t.dtype),
      t.dtype, t.device, shape);
}

// The whole tensor under a different shape, same bytes.
Tensor Reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  return RowsView(t, 0, t.rank == 0 ? 0 : t.shape[0], shape);
}

// A VIEW with caller-chosen strides. `vt::BatchedMatmul` constrains only the
// innermost stride, which is what lets the two [T, hc*H] buffers be read as
// [T*hc, 1, H] and [T*hc, H, 1] with no copy — #2336's claim, and the reason
// the `:1180` dot needs no new op.
Tensor StridedView(const Tensor& t, const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& stride) {
  Tensor v;
  v.data = t.data;
  v.dtype = t.dtype;
  v.device = t.device;
  v.rank = static_cast<int>(shape.size());
  for (int i = 0; i < v.rank; ++i) {
    v.shape[i] = shape[static_cast<size_t>(i)];
    v.stride[i] = stride[static_cast<size_t>(i)];
  }
  return v;
}

}  // namespace

qwen4_exp::PleGeometry Qwen4ExpPleGeometry(const Qwen4ExpParams& p) {
  qwen4_exp::PleGeometry g;
  g.hidden_size = p.hidden_size;
  g.hc_count = p.hc_count;
  g.ple_embed_dim = p.ple.embed_dim;
  g.ple_conv_kernel_size = p.ple.conv_kernel_size;
  g.ngram_size = p.ple.ngram_size;
  g.heads_per_ngram = p.ple.heads_per_ngram;
  g.ngram_vocab_size_base = p.ple.ngram_vocab_size_base;
  g.make_ngram_vocab_size_divisible_by = p.ple.make_ngram_vocab_size_divisible_by;
  g.vocab_size = p.vocab_size;
  // `-1` is `PleGeometry`'s UNSET SENTINEL and W2 refuses it rather than
  // defaulting it, because eos is the second id in the hash mix and there is no
  // safe value to pick. Copied through unchanged so that refusal fires with the
  // name of the config key that is missing, instead of being masked here.
  g.eos_token_id = p.eos_token_id;
  g.seed = p.ple.seed;
  g.rms_norm_eps = p.rms_norm_eps;
  return g;
}

qwen4_exp::NGramTableLayout Qwen4ExpPleLayout(const Qwen4ExpParams& p,
                                              int64_t ple_layer_index) {
  VT_CHECK(ple_layer_index >= 0,
           "qwen4_exp ple layout: ple_layer_index must be >= 0; it is the index INTO "
           "ple_layer_ids (upstream's `config.ple_layer_ids.index(layer_idx + 1)`), "
           "not the decoder layer index");
  const qwen4_exp::PleGeometry geom = Qwen4ExpPleGeometry(p);
  qwen4_exp::NGramTableLayout layout =
      qwen4_exp::BuildNGramTableLayout(geom, ple_layer_index);

  // THE TWO SOURCES ARE CROSS-CHECKED WHERE BOTH EXIST, AND NOWHERE ELSE DOES
  // THAT. `NgramTableRows` (`qwen4_exp_weights.cpp`) takes the STATED sizes when
  // the file has them and derives the prime chain otherwise, so it never sees
  // both at once and cannot compare them. Here the derivation has just run and
  // the stated set is in hand.
  //
  // A disagreement is not a shape error and would not throw anywhere
  // downstream: `head_offsets` is an exclusive prefix sum, so ONE wrong size
  // shifts every later head's rows inside a table whose row count both sides
  // agree on. Every gathered vector would be somebody else's, with no crash.
  const std::vector<int64_t>& stated = p.ple.head_vocab_sizes;
  if (!stated.empty()) {
    VT_CHECK(static_cast<int64_t>(stated.size()) == p.ple.ngram_heads(),
             "qwen4_exp ple layout: the config states " + std::to_string(stated.size()) +
                 " per-head vocabulary sizes but this geometry has " +
                 std::to_string(p.ple.ngram_heads()) +
                 " n-gram heads ((ngram_size - 1) * heads_per_ngram)");
    for (size_t h = 0; h < stated.size(); ++h) {
      VT_CHECK(stated[h] == layout.head_vocab_sizes[h],
               "qwen4_exp ple layout: head " + std::to_string(h) +
                   " vocabulary size disagrees — the source STATES " +
                   std::to_string(stated[h]) + " and the prime chain from "
                   "ngram_vocab_size_base " +
                   std::to_string(p.ple.ngram_vocab_size_base) + " derives " +
                   std::to_string(layout.head_vocab_sizes[h]) +
                   ". The stated set is what the shipped table was built against, so "
                   "this is a real disagreement and not a rounding one: the head "
                   "offsets are an exclusive prefix sum, so one wrong size silently "
                   "re-points every later head at another head's rows");
    }
  }
  return layout;
}

Qwen4ExpPleBlockOutput RunQwen4ExpPleBlock(Dev d, const Qwen4ExpPleWeights& w,
                                           const OwnedTensor& ngram_table,
                                           const Qwen4ExpParams& p,
                                           const qwen4_exp::NGramTableLayout& layout,
                                           const Tensor& hidden, const int64_t* input_ids,
                                           const unsigned char* conv_mask,
                                           const Qwen4ExpPleCaches& caches,
                                           int64_t past_len) {
  const int64_t T = hidden.rank == 2 ? hidden.shape[0] : 0;
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_count;
  const int64_t W = p.stream_width();
  const int64_t E = p.ple.embed_dim;
  const int64_t heads = p.ple.ngram_heads();
  const int64_t hd = p.ple.head_dim_per_ngram();
  const int64_t K = p.ple.conv_kernel_size;
  const int64_t dilation = p.ple.ngram_size;
  const int64_t L = p.ple.short_conv_state_len();
  const int64_t ctx = p.ple.ngram_size - 1;
  const auto eps = static_cast<float>(p.rms_norm_eps);

  VT_CHECK(T > 0, "qwen4_exp ple block: T must be positive");
  VT_CHECK(hidden.rank == 2 && hidden.shape[1] == W,
           "qwen4_exp ple block: hidden must be [T, hc_count*hidden_size] = [T," +
               std::to_string(W) +
               "] — the hc-WIDE hyper-connection stream, never the collapsed "
               "hidden state (upstream's own docstring: the returned tensor has "
               "shape (batch, seq, hc_count * hidden_size))");
  VT_CHECK(hidden.IsContiguous(), "qwen4_exp ple block: hidden must be contiguous");
  // THE STREAM DTYPE IS INHERITED, NOT CHOSEN (AGENTS.md, "Inherit vLLM
  // defaults"): every buffer below carries `hidden`'s dtype, so a bf16 model
  // moves bf16 bytes through all of it. The ONE f32 buffer is the [T, hc] score,
  // and it is f32 because `vt::Qwen4ExpPleGate` refuses a bf16 score — the
  // argument of a sigmoid must not be rounded, which is the contract that op and
  // `vt::SigmoidGateBf16` both state.
  const DType dt = hidden.dtype;
  VT_CHECK(dt == DType::kF32 || dt == DType::kBF16,
           "qwen4_exp ple block: hidden must be f32 or bf16");
  VT_CHECK(input_ids != nullptr, "qwen4_exp ple block: input_ids must not be null");
  VT_CHECK(past_len >= 0, "qwen4_exp ple block: past_len must not be negative");
  VT_CHECK(heads > 0 && hd > 0 && E == heads * hd,
           "qwen4_exp ple block: ple_embed_dim must be ngram_heads * head_dim_per_ngram");

  // The n-gram table is MODEL-level, and it may be block-quantized: W6a made
  // `GgufTensorRole::kEmbeddingTable` keep-quant eligible and taught
  // `vt::Embedding` to decode one row per gathered id, which is what turns
  // 102.4 GB of expanded bf16 into 28.8 GB of resident blocks. Nothing here
  // expands it.
  VT_CHECK(ngram_table.rank == 2 && ngram_table.shape[1] == hd,
           "qwen4_exp ple block: the n-gram table must be [padded_vocab, "
           "ple_embed_dim/ngram_heads] = [V," + std::to_string(hd) + "]");
  VT_CHECK(ngram_table.shape[0] == layout.padded_vocab_size,
           "qwen4_exp ple block: the n-gram table has " +
               std::to_string(ngram_table.shape[0]) + " rows and the layout addresses " +
               std::to_string(layout.padded_vocab_size) +
               "; the head offsets are an exclusive prefix sum over the LAYOUT, so a "
               "table of another height is a different table");

  const Tensor& conv_state = caches.conv_state;
  const Tensor& tokens = caches.tokens;
  VT_CHECK(conv_state.rank == 3 && conv_state.shape[1] == W && conv_state.shape[2] == L,
           "qwen4_exp ple block: conv_state must be [N," + std::to_string(W) + "," +
               std::to_string(L) +
               "] — (kernel-1)*ngram_size deep, NOT kernel-1: the conv is DILATED, so "
               "the state is nine columns at the released config and not three");
  VT_CHECK(conv_state.dtype == DType::kF32 && conv_state.IsContiguous(),
           "qwen4_exp ple block: conv_state must be a contiguous f32 tensor");
  VT_CHECK(tokens.rank == 2 && tokens.shape[1] == ctx && tokens.dtype == DType::kI64 &&
               tokens.IsContiguous(),
           "qwen4_exp ple block: the n-gram history must be a contiguous i64 [N," +
               std::to_string(ctx) +
               "] — int64 because it holds TOKEN IDS, and a float store would ROUND them");
  VT_CHECK(tokens.device.type == vt::DeviceType::kCPU,
           "qwen4_exp ple block: the n-gram history is read and written on the HOST, "
           "because the splitmix64 hash that consumes it is a host int64 computation and "
           "no vt:: op computes it; a device-resident history needs the hash on the "
           "device, which the spec's `## Owed` records");
  VT_CHECK(caches.state_row >= 0 && caches.state_row < conv_state.shape[0] &&
               caches.state_row < tokens.shape[0],
           "qwen4_exp ple block: state_row is outside one of the two caches");

  // ─── THE SEEDING PREDICATE (upstream :1073-1076, :1080-1088) ───────────────
  // `past_len == 0` IS `has_previous_state(layer_idx, state_idx=2) == False`.
  //
  // ZERO IS A VALID TOKEN ID, which is the whole reason this branch exists.
  // `CacheBuffer` zero-fills, exactly as upstream's `update_conv_state` pads
  // with zeros, and upstream works around its own pad with an explicit EOS
  // left-pad at :1086-1087. A port that trusted the zero-filled cache would hash
  // token 0 into the first `ngram_size - 1` positions of every sequence and get
  // a fluent wrong answer.
  //
  // The conv ring is zeroed in the same branch, which is NOT the same statement:
  // a zeroed conv state is bit-identical to upstream's first call
  // (cache_utils.py:1053-1060 left-zero-pads), so this only matters when the
  // runner hands back a slot a previous sequence used. Together the two are
  // exactly `qwen4_exp::PleSequenceState::Reset`.
  auto* hist = tokens.Ptr<int64_t>() + caches.state_row * ctx;
  if (past_len == 0) {
    for (int64_t i = 0; i < ctx; ++i) hist[i] = p.eos_token_id;
    float* ring = conv_state.Ptr<float>() + caches.state_row * W * L;
    for (int64_t i = 0; i < W * L; ++i) ring[i] = 0.0F;
  }

  // ─── THE MASK'S PAIRED HALF, ENFORCED (spec `## Owed`, since W2) ───────────
  // The activations are masked at :1186-1187 AND `input_ids` must already carry
  // EOS at every masked position, because the hash at :1101-1106 reads token ids
  // and not activations. Masking only the activations leaks padding into the
  // hash, and nothing downstream can see it: the ids are in range, the gather
  // succeeds, and the answer is wrong. This is the first enforcer that half has
  // ever had.
  if (conv_mask != nullptr) {
    for (int64_t t = 0; t < T; ++t) {
      if (conv_mask[t] != 0) continue;
      VT_CHECK(input_ids[t] == p.eos_token_id,
               "qwen4_exp ple block: conv_mask masks token " + std::to_string(t) +
                   " but input_ids[" + std::to_string(t) + "] is " +
                   std::to_string(input_ids[t]) + " rather than eos_token_id " +
                   std::to_string(p.eos_token_id) +
                   ". Masking is a PAIRED obligation: the n-gram hash reads token ids, "
                   "not activations, so a masked position whose id is not EOS leaks "
                   "padding into the hash and the answer is wrong with no error");
    }
  }

  // ─── :1176  the n-gram ids, then the gather ────────────────────────────────
  // The hash is host int64 bit-mixing (`_splitmix64`, :979-983) over TOKEN IDS,
  // and `qwen4_exp::BuildNGramIds` is W2's port of it. This is its first caller
  // outside its own translation unit. It advances the history in place, which is
  // upstream's `update_conv_state(..., state_idx=2)`.
  const qwen4_exp::PleGeometry geom = Qwen4ExpPleGeometry(p);
  qwen4_exp::PleSequenceState hash_state;
  hash_state.tokens.assign(hist, hist + ctx);
  std::vector<int64_t> ids(static_cast<size_t>(T * heads));
  qwen4_exp::BuildNGramIds(geom, layout, input_ids, T, &hash_state, ids.data());
  for (int64_t i = 0; i < ctx; ++i) hist[i] = hash_state.tokens[static_cast<size_t>(i)];

  DBuf d_ids(d, DType::kI64, {T * heads}, ids.data());
  // :1114 — `self.ngram_embedding(ngram_ids).flatten(-2)`. The gather emits
  // [T*heads, hd] and the flatten is the same bytes viewed [T, heads*hd].
  DBuf emb(d, dt, {T * heads, hd});
  {
    Tensor table = dense_attn::ResidentWeight(d, ngram_table,
                                              {ngram_table.shape[0], ngram_table.shape[1]});
    Tensor out = emb.t();
    vt::Embedding(d.q, out, table, d_ids.t());
  }
  Tensor embeddings = Reshape(emb.t(), {T, E});

  // ─── :1177  key_normed = norm_key(key_proj(embeddings)) ────────────────────
  DBuf key(d, dt, {T, W});
  vt::MatmulBT(d.q, key.t(), embeddings, dense_attn::ResidentWeight(d, w.key_proj, {W, E}));
  DBuf key_normed(d, dt, {T, W});
  // `gemma = true` and `group_size = hidden_size` are the two halves of
  // `Qwen4ExpTextRMSNorm(hc*H, group_size=H)` (:158-181, :1138). The gamma is
  // stored RAW and the `+1` belongs here (#2218); the group extent is what makes
  // each of the hc streams normalize independently, and `vt::RmsNormGroup`
  // refuses `group_size == 0` rather than degenerating to a whole-row norm.
  const vt::RmsNormGroupArgs norm_args{eps, /*gemma=*/true, H};
  vt::RmsNormGroup(d.q, key_normed.t(), key.t(),
                   dense_attn::ResidentWeight(d, w.norm_key, {W}), norm_args);

  // ─── :1178  value = value_proj(embeddings) ─────────────────────────────────
  DBuf value(d, dt, {T, H});
  vt::MatmulBT(d.q, value.t(), embeddings,
               dense_attn::ResidentWeight(d, w.value_proj, {H, E}));

  // ─── :1179  query_normed = norm_query(hidden_states) ───────────────────────
  DBuf query_normed(d, dt, {T, W});
  vt::RmsNormGroup(d.q, query_normed.t(), hidden,
                   dense_attn::ResidentWeight(d, w.norm_query, {W}), norm_args);

  // ─── :1180  the per-(t, j) dot, WITHOUT a new op ───────────────────────────
  // `(key_normed * query_normed).sum(-1, keepdim=True)` over the [T, hc, H]
  // unflatten is `vt::BatchedMatmul` over VIEWS: [T*hc, 1, H] x [T*hc, H, 1] ->
  // [T*hc, 1, 1]. Only the innermost dim must be unit-stride, so both views are
  // free and neither buffer is copied or tiled. A private scoring loop here
  // would be the parallel path AGENTS.md "Shared seams" forbids.
  //
  // THE SCORE IS f32 AND THE OPERANDS ARE NOT. `BatchedMatmul` accumulates in
  // f32 whatever the operands' width, and the gate needs an unrounded sigmoid
  // argument; the `/ sqrt(hidden_size)` divide is NOT applied here — it rides
  // into the gate as `gate_divisor`, so the op performs upstream's own division
  // rather than a multiply by a pre-computed reciprocal.
  DBuf score(d, DType::kF32, {T, hc});
  {
    const int64_t g = T * hc;
    Tensor a = StridedView(key_normed.t(), {g, 1, H}, {H, H, 1});
    Tensor b = StridedView(query_normed.t(), {g, H, 1}, {H, 1, 1});
    Tensor o = StridedView(score.t(), {g, 1, 1}, {1, 1, 1});
    vt::BatchedMatmul(d.q, o, a, b);
  }

  // ─── :1181-1182, flattened at :1184 ────────────────────────────────────────
  // The signed square root and the broadcast sigmoid scale, as ONE op (W5e-1).
  // The clamp is applied BEFORE the square root, so the floor on |gate| is 1e-3
  // and not 1e-6; exactly zero maps to zero because `sign(0) == 0`, and a fully
  // masked row reaches that origin.
  vt::Qwen4ExpPleGateArgs gate_args;
  gate_args.gate_divisor = static_cast<float>(std::sqrt(static_cast<double>(H)));
  DBuf gated(d, dt, {T, W});
  vt::Qwen4ExpPleGate(d.q, gated.t(), score.t(), value.t(), gate_args);

  // ─── :1183  gated_value_normed = norm_conv(gated_value.flatten(-2)) ────────
  // THE FORK the spec warns about: the conv sees this NORMED copy and the skip
  // term added back at :1188 is the UN-NORMED one, and it is the normed copy the
  // nine-column state keeps.
  DBuf gated_normed(d, dt, {T, W});
  vt::RmsNormGroup(d.q, gated_normed.t(), gated.t(),
                   dense_attn::ResidentWeight(d, w.norm_conv, {W}), norm_args);

  // ─── :1185-1187  apply_mask_to_padding_states, to BOTH ─────────────────────
  // `hidden_states * attention_mask[:, :, None]` (:211) is a per-TOKEN row
  // scale, and the shared surface has no per-row broadcast multiply:
  // `vt::MulColVecF32` scales per output COLUMN and `vt::SigmoidGateBf16`
  // refuses by element count. A masked position is padding and therefore rare,
  // so this is `vt::MulScalar` on the row view — upstream's own two lines at
  // upstream's own two sites, rather than a new general op nothing else calls.
  //
  // BOTH tensors, and masking only one is a real port defect the goldens see:
  // `gated_value` is the skip term and `gated_value_normed` is what enters the
  // conv AND what the ring keeps, so an interior masked position moves output
  // rows t, t+3, t+6 and t+9 as well as its own.
  if (conv_mask != nullptr) {
    for (int64_t t = 0; t < T; ++t) {
      if (conv_mask[t] != 0) continue;
      Tensor g_row = RowsView(gated.t(), t, 1, {1, W});
      Tensor n_row = RowsView(gated_normed.t(), t, 1, {1, W});
      vt::MulScalar(d.q, g_row, g_row, 0.0);
      vt::MulScalar(d.q, n_row, n_row, 0.0);
    }
  }

  // ─── :1188  output = gated_value + self._short_conv(gated_value_normed) ────
  // `vt::Qwen4ExpPleConv` is `_short_conv` (:1150-1167) INCLUDING the state read
  // and the write-back `update_conv_state(..., state_idx=1)` performs around it,
  // so the ring is advanced by the op and not by this file. `state_idx` is
  // resolved by which tensor is passed, and `conv_state_indices` names the cache
  // ROW this sequence owns — a different axis with a deliberately different name
  // (see the op's contract).
  const int32_t qsl_host[2] = {0, static_cast<int32_t>(T)};
  const auto row_host = static_cast<int32_t>(caches.state_row);
  DBuf qsl(d, DType::kI32, {2}, qsl_host);
  DBuf row(d, DType::kI32, {1}, &row_host);
  DBuf conv_out(d, dt, {T, W});
  {
    vt::Qwen4ExpPleConvArgs cargs;
    cargs.dilation = dilation;
    Tensor state = conv_state;
    Tensor idx = row.t();
    vt::Qwen4ExpPleConv(d.q, conv_out.t(), gated_normed.t(),
                        dense_attn::ResidentWeight(d, w.conv1d, {W, K}), state, qsl.t(), &idx,
                        cargs);
  }

  DBuf out(d, dt, {T, W});
  vt::Add(d.q, out.t(), gated.t(), conv_out.t());

  Qwen4ExpPleBlockOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

}  // namespace vllm
