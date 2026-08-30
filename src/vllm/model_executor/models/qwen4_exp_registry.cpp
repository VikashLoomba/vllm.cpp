// Qwen4-Exp registry TU — the ADDITIVE self-registration seam (W1 of
// MODEL-MM-QWEN4-EXP, #1981). Follows the dots3_note_registry.cpp /
// gemma4_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array.
//
// UPSTREAM. `Qwen4ExpForConditionalGeneration` is registered by NO vLLM
// revision. Read live 2026-08-26 at vLLM `origin/main` = `6a5e8f5979`: no
// `qwen4*` path, no `registry.py` entry, and a repository-wide search for
// `qwen4` returns zero results; `vllm-omni` likewise. That is absence from
// vLLM `main` rather than staleness in our parity pin `555967922`, so this TU
// deliberately carries no pinned upstream module/class anchor, the convention
// `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` follows for a beyond-pin arm.
// The ALGORITHM source is transformers **5.16.0**, the accepted lane pin; see
// `.agents/oracles/transformers.md` and `.agents/specs/qwen4-exp-flash-next.md`.
//
// The MTP head is deliberately NOT registered as a second architecture, and
// unlike dots3-note that is not a scheduling choice: upstream carries it as an
// `mtp` block INSIDE the same text config rather than as a separate registry
// entry, so there is no second architecture string to register. That is why
// this row moves the MODEL row ratchet by ONE and not by two.
//
// SCOPE HONESTY, RESTATED AT W5c-1 (#2031). Registering this arch makes it
// RESOLVE, parse and validate its config, LOAD a `qwen4exp` GGUF on a CPU
// device (W5a) and PUBLISH its three KV-cache groups (W5c-1). `make_kv_cache`
// no longer refuses; the sentence that said it did was true at W5a's parent and
// stopped being true one wave later, which is the drift this paragraph keeps
// being rewritten to remove.
//
// THREE THINGS STILL REFUSE, and the one that matters is the FORWARD: no token
// has been decoded by this architecture. The other two are load-time and are
// listed here so that the count is checkable rather than rhetorical -- the
// SAFETENSORS arm refuses unconditionally at the end of
// `LoadQwen4ExpForConditionalGeneration` (every published artifact is larger
// than every device this project owns), and the GGUF arm refuses a source that
// names the kind without carrying a file. Both are stated at their own sites.
//
// The forward's polarity matters more here than usual, because no oracle for
// this model runs on any hardware this project owns yet (`gateable = no`,
// blocked on memory rather than software), so there is no downstream token gate
// that would catch a forward returning plausible garbage. Refusing is the only
// safe default.
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform — the load-time device gate

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen3_5_internal.h"  // ResolveMambaSsmCacheDType
#include "vllm/model_executor/models/dense_attn_block.h"   // ResidentWeight
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_forward.h"  // W5f: the layer loop
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vt/ops.h"
#include "vllm/v1/kv_cache_dtype.h"     // ResolveKvCacheDType
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// Text generation, multimodal (image AND video: the published config carries
// `image_token_id`, `video_token_id` and a `vision_config`), and HYBRID —
// 36 of 48 layers are Gated DeltaNet carrying recurrent state, so this belongs
// with the hybrids and not with the pure-attention arms.
inline constexpr ModelInfo kQwen4ExpInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    // FALSE by the house convention the blanket assertion in
    // test_model_registry.cpp enforces: our ModelInfo is a consumed subset
    // whose only reader short-circuits on is_hybrid, so the GDN-hybrid
    // wrappers (kQwen3_5Info, kKimiLinearInfo) all leave this false even
    // though upstream's class carries HasInnerState.
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

// `Qwen4ExpLoadedModel` — the concrete model this hook produces — is declared in
// `qwen4_exp_weights.h` rather than here. That header says why: an anonymous
// type is unreachable by `dynamic_cast` from another translation unit, and a
// reachability case that cannot open the handle cannot tell a real load from a
// hook that returns `Qwen4ExpWeights{}`.

std::unique_ptr<LoadedModel> LoadQwen4ExpForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W5a (#2031) LOADS it. The GGUF k-quant arm is OWED, not optional
    // (AGENTS.md, porting-a-model.md §2), and for this row it is the ONLY arm
    // that fits a host we own: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S is
    // 67.56 GiB of weights against ~119.6 GiB usable on GB10, where every
    // safetensors artifact (bf16 ~360 GB, FP8 ~180 GB, NVFP4 ~128 GB) does not.
    //
    // Both blockers W1 named have since landed. W6a (#1989) added the IQ4_NL
    // and Q5_0 reader arms, so the file opens at all, and made
    // `GgufTensorRole::kEmbeddingTable` keep-quant eligible with a dequantizing
    // gather behind it, so the 51.2 G-parameter n-gram table stays resident as
    // blocks instead of expanding to 102.4 GB of bf16.
    //
    // A null `gguf` reaches here from a caller that set the KIND without the
    // FILE. Refused by name rather than dereferenced: the alternative is a
    // segmentation fault inside a loader the reader is entitled to read as
    // "GGUF is not supported here".
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "Qwen4ExpForConditionalGeneration: the model source says GGUF but "
          "carries no file. See .agents/specs/qwen4-exp-flash-next.md and "
          "issue #2031.");
    }
    return std::make_unique<Qwen4ExpLoadedModel>(
        registration,
        LoadQwen4ExpFromGguf(*source.gguf, config,
                             platforms::CurrentPlatform().device_type()));
  }
  (void)registration;
  (void)config;
  // The safetensors arm stays refused, and NOT because it is the harder one.
  // Every published safetensors artifact of this model is larger than every
  // device this project owns, so an arm that read them would be code nothing
  // could ever run. The spec's `## Owed` records it with that reason rather
  // than as an unqualified to-do.
  throw std::runtime_error(
      "Qwen4ExpForConditionalGeneration: the safetensors weight loader is not "
      "ported (every published safetensors artifact — bf16 ~360 GB, FP8 ~180 "
      "GB, NVFP4 ~128 GB — exceeds every device this project owns, so the GGUF "
      "arm is the supported one). See .agents/specs/qwen4-exp-flash-next.md and "
      "issue #1978.");
}

void PrepareQwen4ExpForConditionalGeneration(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

// The single-shot prefill this hook serves is assembled below from the two
// POSITIONAL cache channels plus the three states no channel carries.
//
// WHY THE SCRATCH IS BUILT HERE AND NOT IN THE LOOP. `Qwen4ExpTextModelForward`
// takes every cache as an operand and has no opinion about where they came
// from, which is what lets its gate drive it at any `past_len`. The RESTRICTION
// is the engine's: `ModelForwardInput` carries `attn_kv` and `gdn_state` and
// nothing else, so the QSA indexer side cache and the PLE layer's conv ring and
// n-gram history have no home across steps. Putting the scratch — and the
// refusal that makes it sound — at this boundary keeps the limit where it is
// true instead of baking it into the loop.

ForwardLogits ForwardQwen4ExpForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  // THE DOWNCAST COMES FIRST NOW, and the inversion this comment used to argue
  // for is retired with the refusal it protected. The house shape opens the
  // type-erased handle with `ModelAs<Qwen4ExpLoadedModel>` before doing anything
  // else, because a bare `static_cast` down the hierarchy is undefined behaviour
  // on an object that is not really that type (#775, #730). W5a made
  // `load_weights` produce a genuine `Qwen4ExpLoadedModel`, and this wave gives
  // the opened handle something to be used FOR, which is exactly the condition
  // the previous comment named for restoring it: "W5b restores `ModelAs` in the
  // same change that gives it something to read."
  Qwen4ExpLoadedModel& m =
      ModelAs<Qwen4ExpLoadedModel>(model, "Qwen4ExpForConditionalGeneration");
  Qwen4ExpWeights& w = m.weights();
  const Qwen4ExpParams& p = w.params;

  // ─── WHAT THIS HOOK STILL REFUSES, AND WHY IT IS NOT THE LOOP ──────────────
  //
  // THE REFUSAL NO LONGER NAMES THE LAYER LOOP, because W5f wrote it and a
  // refusal that enumerates finished work sends the next reader to rebuild it —
  // #2288, which this string has now produced eight times. What refuses is the
  // ENGINE's cache plumbing, and the boundary is exact:
  //
  //   * `ModelForwardInput` carries exactly two POSITIONAL cache channels,
  //     `attn_kv` and `gdn_state` (`model_registry.h:439-440`). The QSA INDEXER
  //     side cache and the PLE layer's conv ring and n-gram history are NEITHER.
  //     At `past_len == 0` all three are per-call scratch and this hook
  //     allocates them; at any other `past_len` they would have to persist, and
  //     there is nowhere to persist them.
  //   * The only channel that could carry them is `multi_kv`, which
  //     `ModelRegistry::Forward` refuses by name before this hook is entered
  //     (`model_registry.cpp`, the `if (input.multi_kv != nullptr)` block —
  //     named rather than numbered because the line anchor this carried,
  //     `:461-478`, had drifted onto `MultiKvCacheIndex::PayloadAt`), and this
  //     model publishes three groups so
  //     the runner sets it. #2353 established that this refusal must NOT be
  //     lifted yet: none of the three arriving architectures has a consuming
  //     forward, and the by-name channel cannot address recurrent (`MambaSpec`)
  //     members at all — their states go to `gdn_state` POSITIONALLY and their
  //     layer names do not resolve. That is engine work, owned by
  //     KV-DSV4-MULTICACHE W5 (#1925, #2068) and #2353, and not by this row.
  //   * `RunQwen4ExpQsaBlockPaged` takes a `block_table` of i32 `[1, max_pages]`
  //     — ONE sequence per call — so `num_reqs > 1` is out of reach for the same
  //     seam work.
  //
  // So this hook SERVES a single-shot prefill of one sequence and refuses
  // everything else BY NAME. It does not refuse unconditionally any more, which
  // is the difference this wave makes: `Qwen4ExpTextModel::Forward` exists, it
  // is called from here, and a reachable production path runs all four block
  // seams. Nothing about that is a decode: a decode needs the second step, and
  // the second step is what the engine cannot address.
  VT_CHECK(input.num_reqs == 1,
           "Qwen4ExpForConditionalGeneration: this forward serves ONE sequence "
           "per call and the step carries " +
               std::to_string(input.num_reqs) +
               ". RunQwen4ExpQsaBlockPaged's block_table is i32 [1, max_pages], "
               "so a ragged multi-request batch needs query_start_loc plumbing "
               "no block on this row carries. Owned by W5f under #2031; see "
               ".agents/specs/qwen4-exp-flash-next.md.");
  const auto T = static_cast<int64_t>(input.token_ids.size());
  VT_CHECK(T > 0,
           "Qwen4ExpForConditionalGeneration: the step carries no tokens");
  VT_CHECK(input.attn_meta.seq_lens.size() == 1,
           "Qwen4ExpForConditionalGeneration: one sequence per call, so "
           "attn_meta.seq_lens must hold exactly one entry");
  const int64_t past_len =
      static_cast<int64_t>(input.attn_meta.seq_lens[0]) - T;
  VT_CHECK(
      past_len == 0,
      "Qwen4ExpForConditionalGeneration: this forward serves a SINGLE-SHOT "
      "PREFILL (past_len == 0) and this step continues a sequence at past_len " +
          std::to_string(past_len) +
          ". The layer loop itself has no such limit — it takes every cache as "
          "an operand — but ModelForwardInput carries only the POSITIONAL "
          "attn_kv and gdn_state channels, and the QSA indexer side cache and "
          "the PLE conv ring and n-gram history are neither, so at past_len > 0 "
          "there is nowhere for them to have persisted. The channel that would "
          "carry them is multi_kv, which ModelRegistry::Forward refuses by name "
          "and which #2353 established must not be lifted until a consuming "
          "forward and a recurrent-member channel exist. Owned by "
          "KV-DSV4-MULTICACHE W5 (#1925, #2068) and #2353; see "
          ".agents/specs/qwen4-exp-flash-next.md and issues #2031 and #2336.");

  vt::Backend& backend = vt::GetBackend(input.queue.device.type);
  dense_attn::Dev d{backend, input.queue};

  // ─── the caches, in the loop's own three-vector shape ─────────────────────
  int64_t n_gdn = 0, n_qsa = 0;
  for (Qwen4ExpLayerKind k : p.layer_types) {
    if (k == Qwen4ExpLayerKind::kLinearAttention) {
      ++n_gdn;
    } else {
      ++n_qsa;
    }
  }
  VT_CHECK(static_cast<int64_t>(input.gdn_state.size()) == n_gdn,
           "Qwen4ExpForConditionalGeneration: the runner handed " +
               std::to_string(input.gdn_state.size()) +
               " recurrent state caches for " + std::to_string(n_gdn) +
               " linear_attention layers");
  VT_CHECK(static_cast<int64_t>(input.attn_kv.size()) == n_qsa,
           "Qwen4ExpForConditionalGeneration: the runner handed " +
               std::to_string(input.attn_kv.size()) +
               " paged K/V caches for " + std::to_string(n_qsa) +
               " qwen_sparse_attention layers");

  Qwen4ExpForwardCaches caches;
  caches.gdn = input.gdn_state;

  // ONE block table and ONE slot mapping for the whole step, taken from the
  // runner's own metadata rather than rebuilt: `dense_attn::AttnBlock` does not
  // build them either (`StepInputs` carries the runner's own), and a locally
  // invented mapping would store this step's K/V at pages nothing else reads.
  const int64_t cols = input.attn_meta.block_table_num_cols;
  VT_CHECK(cols > 0 && static_cast<int64_t>(
                           input.attn_meta.block_table_tensor.size()) >= cols,
           "Qwen4ExpForConditionalGeneration: the step carries no block table");
  VT_CHECK(static_cast<int64_t>(input.attn_meta.slot_mapping.size()) == T,
           "Qwen4ExpForConditionalGeneration: the slot mapping has " +
               std::to_string(input.attn_meta.slot_mapping.size()) +
               " entries for " + std::to_string(T) + " tokens");
  std::vector<int32_t> bt(input.attn_meta.block_table_tensor.begin(),
                          input.attn_meta.block_table_tensor.begin() + cols);
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    slots[static_cast<size_t>(t)] = input.attn_meta.slot_mapping[static_cast<size_t>(t)];

  // The scratch buffers live for exactly this call, which is what `past_len ==
  // 0` buys and what the refusal above protects.
  std::vector<std::vector<uint16_t>> index_keys(static_cast<size_t>(n_qsa));
  std::vector<std::vector<float>> ple_convs(p.ple.layer_ids_zero_based.size());
  std::vector<std::vector<int64_t>> ple_tokens(p.ple.layer_ids_zero_based.size());

  dense_attn::DBuf d_bt(d, vt::DType::kI32, {1, cols}, bt.data());
  dense_attn::DBuf d_slots(d, vt::DType::kI64, {T}, slots.data());

  // ─── THE QSA INDEXER SIDE CACHE, IN THE ENGINE'S OWN PAGED SHAPE (W5i) ────
  //
  // The block reads and writes group 2 through a block table now, so the scratch
  // this hook allocates is shaped the way the engine allocates group 2 —
  // `[num_pages, block_size, indexer_head_dim]`, the FUSED 3-dim MLA page — and
  // is addressed through a table. Nothing here is a second geometry invented for
  // the scratch: `page` is the group-0 page size the runner handed us, and
  // `MakeQwen4ExpKVCache` publishes group 2 at that same `block_size` with
  // `compress_ratio = 1`, which is W5h's capacity law (one indexer row per token
  // slot) written as a shape.
  //
  // THE TABLE IS THE IDENTITY, AND THAT IS WHAT A PRIVATE PER-CALL BUFFER MEANS,
  // NOT A SHORTCUT. There is no allocator behind this scratch, so there are no
  // physical pages to permute; logical page `i` IS physical page `i`. It is NOT
  // group 0's table: those are group 0's physical page ids in group 0's pool, and
  // adopting them here would name pages this buffer does not have. The
  // permutation the translation exists for is exercised by the block's own gate,
  // which runs a non-identity `{2, 6, 1}` against a logical `{0, 1, 2}` with a
  // partial final page (`test_qwen4_exp_qsa_block.cpp`, the W5i case).
  //
  // WHAT REPLACES IT. When `ModelRegistry::Forward` stops refusing `multi_kv`
  // (#2353, W5j), these two lines become group 2's own cache and
  // `group_block_tables[2]` — the vector W5c-2 already gathers — and NOTHING in
  // the block changes. That substitution is the whole point of paging it here.
  const int64_t idx_page =
      input.attn_kv[0].block_size > 0 ? input.attn_kv[0].block_size : T;
  const int64_t idx_pages = (T + idx_page - 1) / idx_page;
  std::vector<int32_t> idx_bt(static_cast<size_t>(idx_pages));
  for (int64_t k = 0; k < idx_pages; ++k) idx_bt[static_cast<size_t>(k)] = static_cast<int32_t>(k);
  dense_attn::DBuf d_idx_bt(d, vt::DType::kI32, {1, idx_pages}, idx_bt.data());

  std::vector<dense_attn::DBuf> idx_bufs;
  idx_bufs.reserve(static_cast<size_t>(n_qsa));
  caches.qsa.resize(static_cast<size_t>(n_qsa));
  for (int64_t i = 0; i < n_qsa; ++i) {
    const int64_t id = p.qsa.head_dim;
    index_keys[static_cast<size_t>(i)].assign(
        static_cast<size_t>(idx_pages * idx_page * id), 0);
    idx_bufs.emplace_back(d, vt::DType::kBF16,
                          std::vector<int64_t>{idx_pages, idx_page, id},
                          index_keys[static_cast<size_t>(i)].data());
    caches.qsa[static_cast<size_t>(i)].kv = input.attn_kv[static_cast<size_t>(i)];
    caches.qsa[static_cast<size_t>(i)].block_table = d_bt.t();
    caches.qsa[static_cast<size_t>(i)].slot_mapping = d_slots.t();
    caches.qsa[static_cast<size_t>(i)].index_key = idx_bufs.back().t();
    caches.qsa[static_cast<size_t>(i)].index_block_table = d_idx_bt.t();
  }

  std::vector<dense_attn::DBuf> ple_conv_bufs;
  ple_conv_bufs.reserve(p.ple.layer_ids_zero_based.size());
  caches.ple.resize(p.ple.layer_ids_zero_based.size());
  for (size_t i = 0; i < p.ple.layer_ids_zero_based.size(); ++i) {
    const int64_t stream = p.stream_width();
    const int64_t state_len = p.ple.short_conv_state_len();
    const int64_t ctx = p.ple.ngram_size - 1;
    ple_convs[i].assign(static_cast<size_t>(stream * state_len), 0.0F);
    ple_tokens[i].assign(static_cast<size_t>(ctx), 0);
    ple_conv_bufs.emplace_back(d, vt::DType::kF32,
                               std::vector<int64_t>{1, stream, state_len},
                               ple_convs[i].data());
    caches.ple[i].conv_state = ple_conv_bufs.back().t();
    // HOST-RESIDENT AND REFUSED OTHERWISE by the PLE block itself: the n-gram
    // hash is a host int64 computation, so the history is a host tensor over
    // this vector's bytes rather than a device buffer.
    caches.ple[i].tokens = dense_attn::MakeTensor(
        ple_tokens[i].data(), vt::DType::kI64,
        vt::Device{vt::DeviceType::kCPU, 0}, {1, ctx});
    caches.ple[i].state_row = 0;
  }

  const Qwen4ExpTextModelOutput hidden = Qwen4ExpTextModelForward(
      d, w, input.config, input.token_ids, input.positions, input.attn_meta,
      input.gdn_meta, caches, past_len);

  // ─── the lm_head, which is `Qwen4ExpForCausalLM` and not the text model ────
  // `Qwen4ExpTextModel` has NO final RMSNorm (the mixer's `hc_norm` is the last
  // normalization in the model), so the tail is the loop's output straight into
  // one `kMatmulBT`. The head is TIED to the embedding table when the file
  // carries no `output.weight`, which is read off the FILE by the loader.
  const OwnedTensor& head =
      w.tied_word_embeddings ? w.embed_tokens : w.lm_head;
  VT_CHECK(head.rank == 2 && head.shape[0] == p.vocab_size &&
               head.shape[1] == p.hidden_size,
           "Qwen4ExpForConditionalGeneration: the lm_head must be [vocab, "
           "hidden_size]");

  // ONLY THE ROWS THE SAMPLER ASKED FOR, which on a prefill is the last token
  // of each sequence. Computing the full [T, vocab] and discarding it is the
  // shape every other forward in this tree avoids.
  std::vector<int32_t> rows = input.logits_indices;
  if (rows.empty()) rows.push_back(static_cast<int32_t>(T - 1));
  const auto R = static_cast<int64_t>(rows.size());
  for (int32_t r : rows) {
    VT_CHECK(r >= 0 && static_cast<int64_t>(r) < T,
             "Qwen4ExpForConditionalGeneration: logits index " +
                 std::to_string(r) + " is outside this step's " +
                 std::to_string(T) + " tokens");
  }
  dense_attn::DBuf gathered(d, hidden.tensor.dtype, {R, p.hidden_size});
  {
    dense_attn::DBuf d_rows(d, vt::DType::kI32, {R}, rows.data());
    vt::Tensor g = gathered.t();
    vt::IndexSelect(input.queue, g, hidden.tensor, d_rows.t());
  }
  dense_attn::DBuf logits(d, vt::DType::kF32, {R, p.vocab_size});
  {
    vt::Tensor o = logits.t();
    vt::MatmulBT(input.queue, o, gathered.t(),
                 dense_attn::ResidentWeight(d, head, {p.vocab_size, p.hidden_size}));
  }

  ForwardLogits r;
  r.rows = R;
  r.vocab = p.vocab_size;
  r.device_tensor = logits.t();
  r.device_storage = logits.ReleaseShared();
  return r;
}

// ─── The KV-cache spec (W5c, #2031) ──────────────────────────────────────────
//
// THREE published groups, and the shape of them is the decision this function
// exists to record:
//
//   0. the QSA layers' paged K+V              `FullAttentionSpec`
//   1. EVERY linear-attention layer's state   `MambaSpec`, N states
//   2. the QSA layers' indexer side cache     `MLAAttentionSpec`, ONE ROW PER TOKEN
//
// ONE UNIFORM RECURRENT GROUP, NOT ONE PER LAYER, AND THE COST IS DELIBERATE.
// Only ONE linear-attention layer carries the PLE conv and the n-gram token
// history (`ple_layer_ids` selects 0-based layer 1 on the published
// checkpoint), so a per-layer spec would give 35 of the 36 recurrent layers a
// smaller state set. Upstream cannot express that and does not try:
// `get_mamba_state_shape_from_config` is a CLASSMETHOD taking only the config
// (`vllm/model_executor/models/interfaces.py:809-812` at the pin
// `5559679229`), all 18 implementations of it declare ONE shape model-wide and
// not one of them takes a `layer_idx`, and `get_mamba_groups`
// (`vllm/v1/worker/mamba_utils.py:441`) asserts
// `all(mamba_specs[0] == spec for spec in mamba_specs)` — every recurrent spec
// in the model equal, field for field. Upstream pays the uniform cost by
// PADDING rather than by splitting (`vllm/v1/core/kv_cache_utils.py:1101-1109`
// sets `page_size_padded=max_page_size` on the smaller `MambaSpec`).
//
// The cost here, derived from the published shapes rather than measured: the
// PLE conv is `10240 x 9` at bf16 = 184320 B and the n-gram history is 2 int64
// = 16 B, so 184336 B per sequence on each of the 35 linear layers that do not
// use them. At the default `max_num_seqs` of 8 that is 49.2 MiB — 0.09% of the
// GB10 headroom the row's `## Hardware` section accounts. Splitting the group
// to recover it would need a SECOND recurrent group, which
// `.agents/specs/recurrent-multistate.md` records as owed generic engine debt
// and which this topology does not need.
//
// STATE ORDER IS A DELIBERATE DIVERGENCE FROM UPSTREAM'S LIST ORDER, and it is
// the same bytes either way. Upstream keeps the three CONV states adjacent
// (`number_of_conv_states = 3`: GDN conv, PLE conv, n-gram history) with the
// temporal state after them. This tree publishes
// `[gdn_conv, temporal, ple_conv, ngram]` because `GdnStateCache` exposes
// `conv_state = states[0]` and `ssm_state = states[1]` as NAMED fields that
// THREE model families already read (`qwen3_5.cpp`, `kimi_linear_device.cpp`
// and the `nemotron_h` pair `nemotron_h_device.cpp` / `nemotron_h_forward.h`),
// and moving the temporal state off slot 1 would silently re-point every one
// of them. Recorded in the row spec.
//
// THE COUNT IS THREE, NOT FOUR (#2203). `gemma4_mm.cpp` was named here and in
// `.agents/specs/recurrent-multistate.md` as a fourth reader, and it reads
// NEITHER field: its only two mentions of the type are an include comment and
// `std::vector<GdnStateCache> no_gdn_state;` (`gemma4_mm.cpp:221`), passed
// EMPTY, which is the file proving Gemma-4 has no recurrent arm.
// `muse_glimmer_mm.cpp:340` and `qwen3_vl.cpp:621` carry that same empty-vector
// shape. Grepping the FIELD name over-counts in the other direction:
// `glm5_next_kda.cpp` matches `conv_state` 13 times on
// `Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
// (`glm5_next_kda.h:314`), where this one is a `vt::Tensor` (`qwen3_5.h:111`),
// and that file has zero occurrences of `GdnStateCache`. Grep the TYPE.
//
// REAL PER-LAYER NAMES, NEVER PLACEHOLDERS. `ResolveKVCacheGroupLayerNames`
// (`src/vllm/v1/kv_cache_interface.cpp`) rewrites a placeholder group set into
// per-layer names, but its fallback classification can name only a TARGET
// attention group and one `fa_draft` slot: a third attention group gets
// `layer_names.clear()` and an unnamed group is then refused by the runner's
// multi-cache admission check, because its names "do not all resolve to
// distinct in-range layer indices". Publishing the real names also makes the
// rewrite a no-op by its own idempotence guard, so what the runner allocates is
// what this function said.
//
// GROUP 2 IS AN `MLAAttentionSpec` AND THAT IS LOAD-BEARING. `MLAAttentionSpec`
// is not an MLA claim — it is the KEY-ONLY page budget, one vector per stored
// row instead of a K+V pair. A `FullAttentionSpec` here would be absorbed by the
// runner as the single `fa_draft` draft-KV slot instead (`gpu/runner.cpp`, the
// `draft_slot_taken` arm of the leftover scan), `multi_cache_topology` would
// stay false, and the side cache would be published and never allocated — in
// silence.
//
// ITS `compress_ratio` IS 1 AND NOT `indexer_compress_ratio` (W5h). This
// paragraph used to say the ratio "is what makes a state cover four tokens",
// which is what `compress_ratio` MEANS
// (`vllm/v1/kv_cache_interface.py:386`, `:393-395`) and is not what this cache
// STORES. The oracle is unambiguous — transformers 5.16.0
// `models/qwen4_exp/modeling_qwen4_exp.py`, the lane pin recorded in
// `.agents/oracles/transformers.md`:
//
//   :650      `q, raw_keys = q.reshape(*hidden_shape),
//              token_k.reshape(*hidden_shape).squeeze(2)` — one UN-normed,
//              UN-roped key per TOKEN. (W5i re-derived this at the pin: the
//              anchor read `:645-646`, which is inside the `torch.split(` call
//              that FEEDS the line, not the line quoted here.)
//   :654-655  `raw_keys = past_key_values.update_indexer(raw_keys,
//              self.layer_idx)` — and `cache_utils.py:340` calls that "update
//              the indexer key cache by concatenation", returning
//              `[batch_size, total_len, index_head_dim]` (`:346`). Those two are
//              the DOCSTRING; the executing line is
//              `self.indexer_keys = torch.cat([self.indexer_keys,
//              indexer_key_states], dim=1)` (`:350-351`). The static arm returns
//              `[batch_size, max_cache_len, index_head_dim]` (`:672`)
//   :679-682  the POOLED block keys are rebuilt from those raw keys on EVERY
//              step (`raw_keys[batch_idx].index_select(0, ...)` at `:679` then
//              `.float().mean(dim=1)` at `:681` and `k_layernorm` at `:682`).
//              Nothing caches a pooled key. (W5i re-derived this too: the anchor
//              read `:678-681` and `:678` is blank.)
//
// So `indexer_compress_ratio` belongs to the indexer's ALGORITHM — `block_topk =
// token_budget // compress_ratio` (`:622`) and `complete_keys = (kv_len / CR) *
// CR` in `Qwen4ExpQsaIndex` — and never to this cache's page geometry. Our own
// consumer already said so, and since W5i it says it in the ENGINE's own shape:
// `Qwen4ExpQsaPagedCaches::index_key` is the fused MLA page
// `[num_pages, block_size, indexer_head_dim]`, ONE ROW PER TOKEN SLOT, written
// at the physical rows of logical positions `[past_len, past_len + T)` and read
// over the physical rows of `[0, kv_len)`. Both go through `IndexerRows` in
// `qwen4_exp_qsa_block.cpp` — see that helper for the translation, and see
// `Qwen4ExpQsaPagedCaches` for why group 2 carries its own block table.
//
// WHAT THE FOUR COST, AND WHY IT WAS WORSE THAN A SHORT CACHE. The allocation is
// `num_blocks * page_size_bytes()` and `MLAAttentionSpec::real_page_size_bytes`
// takes `storage_block_size()` (`src/vllm/v1/kv_cache_interface.cpp:151-152`),
// while the `PagedKvCache` VIEW the runner hands the forward carries
// `kv.block_size = fa_dims[i].block_size` — the spec's own `block_size`
// (`src/vllm/v1/worker/gpu/runner.cpp:1532` filled at `:1333`). At ratio 4 the
// view claimed 16 rows per page over an allocation of 4, so a consumer that
// trusted the view read four times past the buffer. The two agree exactly when
// `storage_block_size() == block_size`, which is what ratio 1 makes true. It was
// unreachable only because `ModelRegistry::Forward` refuses a multi-cache
// topology before any of it runs.
v1::KVCacheConfig MakeQwen4ExpKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // The row's own resolve-and-validate, not a second reading of the raw config.
  // It is what rewrites `full_attention` into `qwen_sparse_attention`, so the
  // classification below is upstream's post-`__post_init__` one.
  const Qwen4ExpParams p = ParseQwen4ExpParams(config);

  VT_CHECK(block_size > 0,
           "qwen4_exp KV spec: block_size must be positive, got " +
               std::to_string(block_size));

  std::vector<std::string> qsa_layers;
  std::vector<std::string> qsa_indexer_layers;
  std::vector<std::string> linear_layers;
  for (size_t l = 0; l < p.layer_types.size(); ++l) {
    const std::string idx = std::to_string(l);
    if (p.layer_types[l] == Qwen4ExpLayerKind::kLinearAttention) {
      // The name `ResolveKVCacheGroupLayerNames` builds for a recurrent layer,
      // so the runner's by-name membership sees the same string either way.
      linear_layers.push_back("model.layers." + idx + ".linear_attn");
    } else {
      qsa_layers.push_back("model.layers." + idx + ".self_attn.attn");
      // Upstream addresses a side cache by its own module prefix
      // (`vllm/models/deepseek_v4/attention.py:761-767` registers the indexer
      // key cache under `...indexer.k_cache`); the runner parses the
      // `.layers.<N>.` segment out of it, so the suffix is free to say which
      // cache it is.
      qsa_indexer_layers.push_back("model.layers." + idx +
                                   ".self_attn.indexer.k_cache");
    }
  }

  VT_CHECK(!qsa_layers.empty(),
           "qwen4_exp KV spec: the config declares no qwen_sparse_attention "
           "layer, so there is no attention KV to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");
  VT_CHECK(!linear_layers.empty(),
           "qwen4_exp KV spec: the config declares no linear_attention layer, "
           "so there is no recurrent state to publish. See "
           ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // QSA is optional as a WHOLE in the config layer (all five `indexer_*` fields
  // or none), while the `full_attention` -> `qwen_sparse_attention` rewrite is
  // unconditional. So a config CAN declare sparse layers and no indexer, and
  // that combination has no side cache to size. Refuse rather than publish two
  // groups where the model needs three.
  VT_CHECK(p.qsa.compress_ratio > 0 && p.qsa.head_dim > 0 &&
               p.qsa.kv_heads > 0,
           "qwen4_exp KV spec: the config declares " +
               std::to_string(qsa_layers.size()) +
               " qwen_sparse_attention layer(s) but no `indexer_*` group, so "
               "the QSA indexer side cache cannot be sized. See "
               ".agents/specs/qwen4-exp-flash-next.md and issue #2031.");

  // W5h: THE BLOCK-SIZE DIVISIBILITY REFUSAL IS GONE, BECAUSE THE COMPRESSION IT
  // GUARDED WAS NEVER THIS CACHE'S. It read `block_size % indexer_compress_ratio
  // == 0` and explained that `storage_block_size()` truncates under integer
  // division (`vllm/v1/kv_cache_interface.py:393-395`). Every word of that was
  // true of a COMPRESSED page and this page is not one — see the group-2
  // construction below for the oracle that decides it — so at `compress_ratio`
  // 1 there is no division and nothing to truncate. Deleting a guard needs an
  // argument rather than a green run, and the argument is that its replacement
  // is STRICTLY STRONGER: `test_qwen4_exp_kv_cache.cpp`'s W5h case asserts the
  // side cache's row capacity EQUALS the paged K/V group's token capacity, at
  // both a dividing and a non-dividing block size. The old refusal could only
  // see one arithmetic accident; the capacity law sees any spec that cannot hold
  // what the model stores.

  // The recurrent state set, in the order stated above.
  //
  // The two dtypes come from the SAME resolver every other hybrid in this tree
  // uses, rather than a second reading of `mamba_ssm_dtype`; its refusal
  // message is spelled `qwen3_5:` because that is where the one copy lives.
  const vt::DType conv_dtype = vt::DType::kBF16;
  const vt::DType ssm_dtype =
      detail::ResolveMambaSsmCacheDType(config, conv_dtype);

  std::vector<std::vector<int64_t>> state_shapes{
      // GDN conv: the concatenated q|k|v stream, `conv_kernel - 1` taps.
      {p.linear_conv_dim(), p.linear_conv_kernel_dim - 1},
      // GDN temporal.
      {p.linear_num_value_heads, p.linear_value_head_dim,
       p.linear_key_head_dim},
  };
  std::vector<vt::DType> state_dtypes{conv_dtype, ssm_dtype};

  // `number_of_conv_states` is 3 exactly when the model has a PLE layer, and 1
  // otherwise (`Qwen4ExpParams::number_of_conv_states`, mirroring upstream).
  // The two extra conv states are the PLE conv and the n-gram token history,
  // which upstream keeps in the linear-attention cache beside the GDN conv
  // because the state manipulations are identical (`modular_qwen4_exp.py`
  // :178-180).
  if (p.number_of_conv_states() == 3) {
    // The PLE conv is DILATED by `ngram_size`, so its state is
    // `(kernel - 1) * ngram_size` = 9 columns deep, not `kernel - 1`, and it
    // runs over the FULL hyper-connection stream width.
    state_shapes.push_back(
        {p.stream_width(), p.ple.short_conv_state_len()});
    state_dtypes.push_back(conv_dtype);
    // TOKEN IDS, and `kI64` is not a widening. The history holds
    // `input_ids.long()` and feeds a `uint64_t` hash multiply; storing it in a
    // float dtype rounds a token id, which the row spec records as one of the
    // three silent divergence sites. `ENG-RECURRENT-MULTISTATE` (#2131) is what
    // made an integer recurrent state expressible at all.
    state_shapes.push_back({p.ple.ngram_size - 1});
    state_dtypes.push_back(vt::DType::kI64);
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_layers),
      std::make_shared<v1::FullAttentionSpec>(
          block_size, static_cast<int>(p.num_key_value_heads),
          static_cast<int>(p.head_dim), v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::move(linear_layers),
      std::make_shared<v1::MambaSpec>(block_size, std::move(state_shapes),
                                      std::move(state_dtypes)));
  kv.kv_cache_groups.emplace_back(
      std::move(qsa_indexer_layers),
      std::make_shared<v1::MLAAttentionSpec>(
          block_size, static_cast<int>(p.qsa.head_dim),
          v1::ResolveKvCacheDType(), static_cast<int>(p.qsa.kv_heads),
          v1::KVQuantMode::kNone, /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/std::nullopt,
          // ONE ROW PER TOKEN — `compress_ratio` 1, NOT
          // `indexer_compress_ratio`. See the "GROUP 2" paragraph above this
          // function for the oracle lines that decide it.
          /*compress_ratio=*/1,
          /*model_version=*/std::nullopt));
  return kv;
}

const ModelFactory kQwen4ExpFactory{
    .parse_config = &ParseQwen4ExpConfig,
    .load_weights = &LoadQwen4ExpForConditionalGeneration,
    .prepare = &PrepareQwen4ExpForConditionalGeneration,
    .forward = &ForwardQwen4ExpForConditionalGeneration,
    .make_kv_cache = &MakeQwen4ExpKVCache,
    .is_dense_model = false,
};

}  // namespace

REGISTER_VLLM_MODEL(qwen4_exp, "Qwen4ExpForConditionalGeneration",
                    kQwen4ExpFactory, kQwen4ExpInfo)

}  // namespace vllm
