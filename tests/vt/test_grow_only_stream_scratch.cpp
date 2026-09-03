// Gates the synchronisation of the grow-only per-stream device scratch pool
// (src/vt/grow_only_stream_scratch.h), the bookkeeping the ROCm activation-quant
// scratch and the hipBLASLt workspace both route through (#2712).
//
// COMPILED AND RUN IN EVERY BUILD, including a CPU-only runner with no AMD GPU
// and no ROCm toolchain. That is why the pool's device allocation arrives as a
// callable: the part carrying the decision is portable, so the part carrying the
// defect can be gated where the hardware is not.
//
// The defect these cases exist for is NOT a torn read. `ScratchFor` returned a
// reference under a lock_guard that died with the call, so two callers on ONE
// stream both wrote `buf` and both wrote `bytes`. The pair that survives can be
// MISMATCHED — `bytes` from the larger request, `buf` from the smaller
// allocation — and every later caller then passes the capacity check and gets a
// block smaller than it asked for.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <doctest/doctest.h>

#include "vt/grow_only_stream_scratch.h"

namespace {

// Stands in for hipStream_t / cudaStream_t: the pool uses it only as a map key.
using FakeStream = int;

// Hands out a distinct, non-null block per call and remembers each block's real
// capacity, so a caller can be asked the one question that matters: is the block
// I was given at least as large as the size I asked for?
class RecordingAllocator {
 public:
  explicit RecordingAllocator(int widen_us) : widen_us_(widen_us) {}

  void* Allocate(std::size_t need) {
    // Widen the publish window so the interleaving is REACHED rather than hoped
    // for. With the lock held across the whole operation this only serialises;
    // it cannot change the answer.
    if (widen_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds(widen_us_));
    const std::uintptr_t id = next_.fetch_add(1, std::memory_order_relaxed) + 1;
    void* block = reinterpret_cast<void*>(id * 4096);
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_[block] = need;
    return block;
  }

  // 0 for a block this allocator never produced, which is itself a failure.
  std::size_t CapacityOf(void* block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = capacity_.find(block);
    return it == capacity_.end() ? 0 : it->second;
  }

  std::size_t Allocations() const { return next_.load(std::memory_order_relaxed); }

 private:
  int widen_us_;
  std::atomic<std::uintptr_t> next_{0};
  mutable std::mutex mutex_;
  std::unordered_map<void*, std::size_t> capacity_;
};

}  // namespace

TEST_CASE("grow-only scratch grows once and reuses on one thread") {
  vt::GrowOnlyStreamScratch<FakeStream> pool;
  RecordingAllocator alloc(0);
  auto call = [&](std::size_t need) {
    return pool.Ensure(7, need, [&](std::size_t n) { return alloc.Allocate(n); });
  };

  void* small = call(64);
  REQUIRE(small != nullptr);
  CHECK(alloc.CapacityOf(small) >= 64);
  CHECK(alloc.Allocations() == 1);

  // Inside the current capacity: no allocation, same block.
  CHECK(call(32) == small);
  CHECK(alloc.Allocations() == 1);

  // Past it: a new, larger block, and the old one is RETIRED rather than freed,
  // because a captured graph may have baked it.
  void* big = call(4096);
  CHECK(big != small);
  CHECK(alloc.CapacityOf(big) >= 4096);
  CHECK(alloc.Allocations() == 2);
  CHECK(pool.RetiredCount() == 1);
}

TEST_CASE("grow-only scratch keeps two streams independent") {
  vt::GrowOnlyStreamScratch<FakeStream> pool;
  RecordingAllocator alloc(0);
  auto call = [&](FakeStream s, std::size_t need) {
    return pool.Ensure(s, need, [&](std::size_t n) { return alloc.Allocate(n); });
  };

  void* a = call(1, 128);
  void* b = call(2, 256);
  CHECK(a != b);
  CHECK(pool.CapacityFor(1) == 128);
  CHECK(pool.CapacityFor(2) == 256);
  // Growing one stream must not republish the other's block.
  call(1, 8192);
  CHECK(pool.BlockFor(2) == b);
  CHECK(pool.CapacityFor(2) == 256);
}

TEST_CASE("every concurrent caller on one stream gets a block at least its own size") {
  // THE #2712 CASE. All threads contend on ONE stream key with different sizes.
  // Under the shipped discipline -- lock scoped to the map lookup, entry mutated
  // unlocked -- a caller receives a block another thread published for a smaller
  // request.
  constexpr int kThreads = 8;
  constexpr int kRounds = 24;

  for (int round = 0; round < kRounds; ++round) {
    vt::GrowOnlyStreamScratch<FakeStream> pool;
    RecordingAllocator alloc(50);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> undersized{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        // Every thread asks for a different size, all on stream key 0.
        const std::size_t need = static_cast<std::size_t>(t + 1) * 1024;
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
        }
        void* block = pool.Ensure(0, need, [&](std::size_t n) { return alloc.Allocate(n); });
        if (block == nullptr || alloc.CapacityOf(block) < need) {
          undersized.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) {
    }
    go.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    // 1. No caller may be handed a block smaller than it asked for.
    CHECK(undersized.load(std::memory_order_relaxed) == 0);
    // 2. The surviving pair must describe each other: the published capacity is
    //    the real capacity of the published block. A mismatch here is what makes
    //    every LATER caller unsafe, long after the racing pair have returned.
    void* published = pool.BlockFor(0);
    REQUIRE(published != nullptr);
    CHECK(alloc.CapacityOf(published) == pool.CapacityFor(0));
    // 3. Every block the pool replaced is retained, never dropped on the floor.
    CHECK(pool.RetiredCount() + 1 == alloc.Allocations());
  }
}
