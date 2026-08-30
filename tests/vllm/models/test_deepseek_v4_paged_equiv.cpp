// `KV-DSV4-MULTICACHE` W5 (#2323) — DeepSeek-V4's attention IS `vt::MlaDecodeAttention`.
//
// THE CLAIM THIS FILE EXISTS TO TEST. W5 routes V4's decode onto the shared
// paged MLA op instead of its bespoke loop over a contiguous `deck`. That is only
// sound if the op computes the SAME function, and the spec's argument for it is a
// reading of two source files. This checks it by running both.
//
// V4's attention (`deepseek_v4.cpp`, "5. attention with per-head sink softmax"):
//
//     sc[j]   = dot(q[t,h], kv[j]) * scale          over the FULL head_dim
//     prob    = softmax_with_sink(sc, sink[h])      sink in the DENOMINATOR only
//     o[t,h]  = sum_j prob[j] * kv[j]               the FULL row is the value
//
// `vt::MlaDecodeAttention` dots the query over `head_size` columns of the cache
// row and accumulates the value over the LEADING `v_head_dim` columns of that
// same row. So V4 is the case `head_size == v_head_dim == head_dim`: the whole
// latent is both key and value. That degenerate-looking choice is the entire
// reason the op fits a model it was not written for.
//
// NOT BIT-IDENTITY, and the reason is stated rather than hidden: V4's host loop
// is a two-pass softmax that sums in key order; the op's CPU kernel is an ONLINE
// softmax with running rescales. Same arithmetic, different association, so the
// f32 results differ in the last bits. The bound below is relative to the output
// scale and tight enough that a wrong VALUE SOURCE, a missing sink or a wrong
// scale could not pass it.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

std::vector<float> Rand(size_t n, uint32_t seed, float scale) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * scale;
  }
  return v;
}

vt::Tensor Contig(void* p, vt::DType dt, vt::Device dev,
                  std::initializer_list<int64_t> shape) {
  return vt::Tensor::Contiguous(p, dt, dev, shape);
}

}  // namespace

TEST_CASE("W5: vt::MlaDecodeAttention reproduces DeepSeek-V4's sink attention") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // V4-Flash's real widths, so the case cannot pass by accident on a toy shape.
  const int64_t hd = 512;   // head_dim — key AND value width
  const int64_t nh = 4;     // a few heads, each with its own sink
  const int64_t n_keys = 37;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 7u, 0.35f);
  const std::vector<float> qv = Rand(static_cast<size_t>(nh * hd), 11u, 0.30f);
  std::vector<float> sink(static_cast<size_t>(nh));
  for (int64_t h = 0; h < nh; ++h) sink[static_cast<size_t>(h)] = -0.3f + 0.2f * static_cast<float>(h);

  // ── (A) V4's own arithmetic, transcribed from the forward ────────────────
  std::vector<float> want(static_cast<size_t>(nh * hd), 0.0f);
  for (int64_t h = 0; h < nh; ++h) {
    std::vector<float> sc(static_cast<size_t>(n_keys));
    for (int64_t j = 0; j < n_keys; ++j) {
      float dot = 0.0f;
      for (int64_t d = 0; d < hd; ++d)
        dot += qv[static_cast<size_t>(h * hd + d)] * kv[static_cast<size_t>(j * hd + d)];
      sc[static_cast<size_t>(j)] = dot * scale;
    }
    // The SAME host reference the forward calls.
    const std::vector<float> prob =
        vllm::deepseek_v4::SoftmaxWithSink(sc, sink[static_cast<size_t>(h)]);
    for (int64_t j = 0; j < n_keys; ++j) {
      const float w = prob[static_cast<size_t>(j)];
      for (int64_t d = 0; d < hd; ++d)
        want[static_cast<size_t>(h * hd + d)] += w * kv[static_cast<size_t>(j * hd + d)];
    }
  }

  // ── (B) the shared op, over a PAGED cache holding the same keys ──────────
  const int64_t block_size = 16;
  const int64_t num_blocks = (n_keys + block_size - 1) / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
  for (int64_t j = 0; j < n_keys; ++j) {
    for (int64_t d = 0; d < hd; ++d)
      cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
  }
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(n_keys)};

  std::vector<float> got(static_cast<size_t>(nh * hd), 0.0f);
  vt::Tensor t_out = Contig(got.data(), vt::DType::kF32, q.device, {1, nh, hd});
  vt::Tensor t_q = Contig(const_cast<float*>(qv.data()), vt::DType::kF32, q.device, {1, nh, hd});
  vt::Tensor t_c = Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});
  vt::Tensor t_bt = Contig(block_table.data(), vt::DType::kI32, q.device, {1, num_blocks});
  vt::Tensor t_sl = Contig(seq_lens.data(), vt::DType::kI32, q.device, {1});
  vt::Tensor t_sink = Contig(sink.data(), vt::DType::kF32, q.device, {nh});

  vt::MlaDecodeAttentionArgs args;
  args.scale = scale;
  args.attn_sink = &t_sink;
  vt::MlaDecodeAttention(q, t_out, nullptr, t_q, t_c, t_bt, t_sl, args);

  // ── the comparison ───────────────────────────────────────────────────────
  double worst = 0.0, mag = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(got[i]));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
    worst = std::max(worst, std::abs(static_cast<double>(want[i] - got[i])));
  }
  // NON-TRIVIAL first: an all-zero pair satisfies any bound and proves nothing.
  REQUIRE(mag > 1e-3);
  CHECK(worst <= 1e-5 * mag);

  // AND THE SINK IS LOAD-BEARING. Without it the op computes a different answer,
  // so the agreement above is evidence about the sink and not only about the dot
  // product and the value source.
  std::vector<float> no_sink(static_cast<size_t>(nh * hd), 0.0f);
  vt::Tensor t_out2 = Contig(no_sink.data(), vt::DType::kF32, q.device, {1, nh, hd});
  vt::MlaDecodeAttentionArgs plain = args;
  plain.attn_sink = nullptr;
  vt::MlaDecodeAttention(q, t_out2, nullptr, t_q, t_c, t_bt, t_sl, plain);
  double diff = 0.0;
  for (size_t i = 0; i < want.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(want[i] - no_sink[i])));
  CHECK(diff > 1e-4 * mag);
}
