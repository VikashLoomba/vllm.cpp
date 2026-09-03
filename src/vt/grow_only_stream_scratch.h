// vllm.cpp original — portable bookkeeping for a grow-only, per-stream device
// scratch pool.
//
// Several device backends keep one scratch block per stream and grow it on
// demand: the ROCm activation-quant scratch (rocm_grouped_gemm.hip
// EnsureQuantScratch) and the hipBLASLt workspace (rocm_matmul_hipblaslt.hip
// LtWorkspace). Two rules govern such a pool, and the shipped code obeyed
// neither completely.
//
// RULE 1 — ONE LOCK ACROSS THE WHOLE OPERATION. Locking only the map lookup and
// then mutating the entry unlocked is not synchronisation. Two callers on one
// stream both write `buf` and both write `bytes`, and the pair that survives can
// be MISMATCHED: `bytes` from the larger request, `buf` from the smaller
// allocation. Every later caller then passes the capacity check and is handed a
// block smaller than it asked for. That is the corruption; the torn read is only
// how it starts (#2712).
//
// RULE 2 — RETIRE, NEVER FREE. Freeing the old block on growth is unsafe once
// graph capture is live: a captured graph bakes the pointer its nodes were built
// with, and a later, larger forward would free memory that graph still replays
// against. Per-call malloc/free/synchronize is illegal under capture outright.
// So the old block stays resident for the process. Growth is O(log(max/min))
// events over a process, so the retained bytes are bounded and negligible; this
// mirrors vt::cuda::RetireGraphScratch (src/vt/cuda/graph_safe_scratch.h) and the
// DevicePool's own "never returned to the driver" discipline.
//
// Like graph_safe_scratch.h, this header holds ONLY the portable bookkeeping.
// The device allocation stays at the call site and arrives as a callable, so the
// header carries no HIP or CUDA include and is compiled — and unit-tested — in
// every build, including a CPU-only CI runner with no accelerator. That is the
// same reason include/vt/rocm/rocm_arch.h is deliberately HIP-free.
#ifndef VT_GROW_ONLY_STREAM_SCRATCH_H_
#define VT_GROW_ONLY_STREAM_SCRATCH_H_

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vt {

// StreamT is the backend's stream handle (hipStream_t, cudaStream_t, ...). It is
// used only as a map key, so this header needs no backend type.
template <typename StreamT>
class GrowOnlyStreamScratch {
 public:
  // Return a block for `stream` of at least `need` bytes, allocating a larger one
  // when the current block is too small.
  //
  // `alloc` is called as `alloc(need)` and returns the new block, and it is
  // called with the pool's lock HELD, which is what makes the capacity check and
  // the publish one operation. It may throw; the entry is then unchanged.
  //
  // Returns nullptr when `need` is 0 and nothing was ever allocated, and when
  // `alloc` could not allocate.
  template <typename Alloc>
  void* Ensure(StreamT stream, std::size_t need, Alloc&& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry& entry = pools_[stream];
    if (need > entry.bytes) {
      void* block = alloc(need);
      // A callable that cannot allocate must throw or return nullptr, and
      // neither publishes. Publishing a nullptr with `need` bytes would poison
      // the entry: every later caller would pass the capacity check and get
      // nothing back.
      if (block == nullptr) return nullptr;
      if (entry.buf != nullptr) retired_.push_back(entry.buf);
      entry.buf = block;
      entry.bytes = need;
    }
    return entry.buf;
  }

  // Blocks kept resident because a captured graph may have baked them. Monotonic
  // for the pool's lifetime. Diagnostics and unit tests.
  std::size_t RetiredCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return retired_.size();
  }

  // The capacity currently published for `stream`, or 0 when it has none.
  std::size_t CapacityFor(StreamT stream) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pools_.find(stream);
    return it == pools_.end() ? 0 : it->second.bytes;
  }

  // The block currently published for `stream`, or nullptr when it has none.
  void* BlockFor(StreamT stream) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pools_.find(stream);
    return it == pools_.end() ? nullptr : it->second.buf;
  }

 private:
  struct Entry {
    void* buf = nullptr;
    std::size_t bytes = 0;
  };

  mutable std::mutex mutex_;
  std::unordered_map<StreamT, Entry> pools_;
  std::vector<void*> retired_;
};

}  // namespace vt

#endif  // VT_GROW_ONLY_STREAM_SCRATCH_H_
