/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/connectors/hive/CacheHostRegistration.h"

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
using Clock = std::chrono::steady_clock;
using Traits = memory::AllocationTraits;
constexpr uint64_t kAll = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kInitialBlockBytes = 64ULL << 20;
constexpr uint64_t kMaxBlockBytes = 256ULL << 20;

uint64_t nanos(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now() - start)
      .count();
}

void count(
    const std::shared_ptr<IoStats>& stats,
    const char* name,
    uint64_t value = 1,
    RuntimeCounter::Unit unit = RuntimeCounter::Unit::kNone) {
  if (stats) {
    stats->addCounter(name, RuntimeCounter(saturateCast(value), unit));
  }
}

// Keep reuse local to the allocating CPU's NUMA node. Allocation/first touch
// also inherit the calling worker's memory policy; no background thread with
// an unrelated policy prepares these blocks. Production workers bind CPU and
// memory together. This does not migrate already resident malloc backing.
int currentNumaNode() {
#ifdef SYS_getcpu
  unsigned node = 0;
  if (::syscall(SYS_getcpu, nullptr, &node, nullptr) == 0) {
    return node;
  }
#endif
  return -1;
}
} // namespace

struct CacheHostRegistration::Pool : cache::CacheAllocator {
  struct Block {
    Block(Pool* owner, uint64_t size, int numa, int cudaDevice)
        : pool(owner),
          bytes(size),
          node(numa),
          device(cudaDevice),
          freeBits((Traits::numPages(size) + 63) / 64, ~uint64_t{0}) {}

    // Registry mutex protects metadata. Allocator/CUDA calls never run under
    // this mutex. The pool owns every block until explicit retirement.
    Pool* const pool;
    const uint64_t bytes;
    const int node;
    const int device;
    memory::ContiguousAllocation allocation;
    std::vector<uint64_t> freeBits;
    uint64_t usedPages{0};
    uint64_t users{0};
    bool registered{false};
    bool retiring{false};
    bool metricsReported{false};
    uint64_t prepareNanos{0};
    uint64_t registerNanos{0};

    int32_t findSpace(uint64_t pages) const {
      const auto end = Traits::numPages(bytes);
      if (pages > end - usedPages) {
        return -1;
      }
      int32_t cursor = 0;
      while (cursor < end) {
        const auto begin = bits::findFirstBit(freeBits.data(), cursor, end);
        if (begin < 0 || begin + pages > end) {
          return -1;
        }
        cursor = begin;
        while (cursor < begin + pages &&
               bits::isBitSet(freeBits.data(), cursor)) {
          ++cursor;
        }
        if (cursor == begin + pages) {
          return begin;
        }
        ++cursor;
      }
      return -1;
    }
  };

  struct Registry {
    std::mutex mutex;
    std::condition_variable ready;
    std::atomic<bool> enabled{false};
    uint64_t maxBytes{0};
    uint64_t initialBytes{kInitialBlockBytes};
    uint64_t maxBlockBytes{kMaxBlockBytes};
    uint64_t reservedBytes{0};
    uint64_t retainedBytes{0};
    uint64_t pendingBlocks{0};
    uint64_t registerCalls{0};
    uint64_t unregisterCalls{0};
    uint64_t budgetFallbacks{0};
    uint64_t allocationFallbacks{0};
    uint64_t registrationFailures{0};
    uint32_t failureAt{0};
    std::vector<std::shared_ptr<Block>> blocks;
  };
  static Registry& registry() {
    // Cache destruction explicitly tears down CUDA registrations before the
    // root allocator and driver disappear. No static-destructor CUDA calls.
    static auto* value = new Registry;
    return *value;
  }

  explicit Pool(cache::AsyncDataCache* cache) : allocator(cache->allocator()) {}

  static std::shared_ptr<cache::CacheAllocator> create(
      cache::AsyncDataCache* cache) {
    return std::make_shared<Pool>(cache);
  }

  std::shared_ptr<cache::CacheAllocation> allocate(uint64_t bytes) override;

  // Mark under the registry mutex, then unregister/free outside it. A cache
  // slice can be returned under a shard lock without making any CUDA call.
  static uint64_t retireEmpty(Pool* pool, uint64_t target) {
    auto& registry = Pool::registry();
    std::vector<std::shared_ptr<Block>> victims;
    uint64_t freed = 0;
    {
      std::lock_guard lock(registry.mutex);
      for (const auto& block : registry.blocks) {
        if ((!pool || block->pool == pool) && !block->retiring &&
            block->usedPages == 0 && block->users == 0 && freed < target) {
          victims.push_back(block);
          freed += block->bytes;
        }
      }
      for (const auto& block : victims) {
        block->retiring = true;
      }
    }
    for (const auto& block : victims) {
      unregister(block);
      block->pool->allocator->freeContiguous(block->allocation);
      std::lock_guard lock(registry.mutex);
      registry.retainedBytes -= block->bytes;
      std::erase(registry.blocks, block);
    }
    return freed;
  }

  static void unregister(const std::shared_ptr<Block>& block) noexcept {
    if (!block->registered) {
      return;
    }
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{block->device}};
    const auto status = cudaHostUnregister(block->allocation.data());
    if (status != cudaSuccess) {
      // Never recycle or unmap memory whose DMA registration cannot be retired.
      LOG(FATAL) << "Cannot unregister cache slab: "
                 << cudaGetErrorString(status);
    }
    auto& registry = Pool::registry();
    std::lock_guard lock(registry.mutex);
    block->registered = false;
    registry.reservedBytes -= block->bytes;
    ++registry.unregisterCalls;
  }

  uint64_t reclaim(uint64_t bytes) override {
    return retireEmpty(this, bytes);
  }

  void shutdown() noexcept override {
    try {
      auto& registry = Pool::registry();
      {
        std::lock_guard lock(registry.mutex);
        VELOX_CHECK(!building, "Cache shutdown requires quiescent allocations");
        closed = true;
        for (const auto& block : registry.blocks) {
          if (block->pool == this) {
            VELOX_CHECK_EQ(
                block->usedPages, 0, "Cache slices survived shutdown");
            VELOX_CHECK_EQ(block->users, 0, "DMA survived cache shutdown");
            VELOX_CHECK(!block->retiring);
          }
        }
      }
      retireEmpty(this, kAll);
    } catch (const std::exception& error) {
      LOG(FATAL) << "Cannot destroy registered cache backing: " << error.what();
    }
  }

  memory::MemoryAllocator* const allocator;
  // Protected by Registry::mutex. Only slab growth is serialized, not reads,
  // slice allocation/release, root allocation arbitration, or H2D submission.
  bool building{false};
  bool closed{false};
  uint64_t nextBlockBytes{0};
};

struct CacheHostRegistration::Slice : cache::CacheAllocation {
  Slice(std::shared_ptr<Pool::Block> owner, uint64_t firstPage, uint64_t pages)
      : block(std::move(owner)), offset(firstPage), pages(pages) {}

  ~Slice() override {
    if (!allocated) {
      return;
    }
    std::lock_guard lock(Pool::registry().mutex);
    for (auto page = offset; page < offset + pages; ++page) {
      bits::setBit(block->freeBits.data(), page);
    }
    block->usedPages -= pages;
    // The pool still owns the block. No unregister, allocator free, cache
    // callback, allocation, or device wait is permitted in this destructor.
  }

  char* data() const override {
    return block->allocation.data<char>() + Traits::pageBytes(offset);
  }
  uint64_t capacity() const override {
    return Traits::pageBytes(pages);
  }

  const std::shared_ptr<Pool::Block> block;
  const uint64_t offset;
  const uint64_t pages;
  bool allocated{false}; // Set before publication, immutable afterwards.
};

std::shared_ptr<cache::CacheAllocation> CacheHostRegistration::Pool::allocate(
    uint64_t bytes) {
  auto& registry = Pool::registry();
  const auto pages = Traits::numPages(bytes);
  const uint64_t sliceBytes = Traits::pageBytes(pages);
  const auto node = currentNumaNode();
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  std::shared_ptr<Block> block;
  uint64_t blockBytes = 0;
  {
    std::unique_lock lock(registry.mutex);
    for (;;) {
      if (!registry.enabled || closed) {
        return nullptr;
      }
      if (sliceBytes > registry.maxBytes) {
        ++registry.budgetFallbacks;
        return nullptr;
      }
      for (const auto& candidate : registry.blocks) {
        if (candidate->pool != this || candidate->node != node ||
            !candidate->registered || candidate->retiring) {
          continue;
        }
        const auto offset = candidate->findSpace(pages);
        if (offset >= 0) {
          // Allocate Slice metadata before changing the free bitmap.
          auto slice = std::make_shared<Slice>(candidate, offset, pages);
          for (auto page = offset; page < offset + pages; ++page) {
            bits::clearBit(candidate->freeBits.data(), page);
          }
          candidate->usedPages += pages;
          slice->allocated = true;
          return slice;
        }
      }
      if (building) {
        // The synchronous grower makes progress without submitting any jobs to
        // this executor. No nested-executor dependency or mutex is held.
        registry.ready.wait(lock, [&] { return !building; });
        continue;
      }
      const auto available = registry.maxBytes - registry.reservedBytes;
      if (sliceBytes > available) {
        // An empty slab on another NUMA node can return its budget. Only this
        // cache's slabs are retired here: another cache may be shutting down.
        // Live slices are never invalidated or copied to make an allocation fit.
        lock.unlock();
        const auto released = retireEmpty(this, sliceBytes - available);
        lock.lock();
        if (released != 0) {
          continue;
        }
        ++registry.budgetFallbacks;
        return nullptr;
      }
      const auto regularBytes =
          nextBlockBytes ? nextBlockBytes : registry.initialBytes;
      // Oversized individual requests get dedicated page-rounded blocks. Small
      // requests share geometrically growing slabs. Neither moves old data.
      blockBytes = std::min<uint64_t>(
          std::max(sliceBytes, regularBytes),
          std::min<uint64_t>(available, allocator->capacity()));
      blockBytes -= blockBytes % Traits::kPageSize;
      if (sliceBytes > blockBytes) {
        return nullptr;
      }
      block = std::make_shared<Block>(this, blockBytes, node, device);
      building = true;
      ++registry.pendingBlocks;
      registry.reservedBytes += blockBytes;
      break;
    }
  }

  bool published = false;
  bool registered = false;
  auto cleanup = folly::makeGuard([&] {
    if (!published) {
      if (registered) {
        const auto status = cudaHostUnregister(block->allocation.data());
        if (status != cudaSuccess) {
          LOG(FATAL) << "Cannot roll back cache slab: "
                     << cudaGetErrorString(status);
        }
      }
      allocator->freeContiguous(block->allocation);
    }
    {
      std::lock_guard lock(registry.mutex);
      if (!published) {
        registry.reservedBytes -= blockBytes;
        registry.unregisterCalls += registered ? 1 : 0;
      }
      --registry.pendingBlocks;
      building = false;
    }
    registry.ready.notify_all();
  });

  const auto prepareStart = Clock::now();
  if (!allocator->allocateContiguous(
          Traits::numPages(blockBytes), nullptr, block->allocation)) {
    std::lock_guard lock(registry.mutex);
    ++registry.allocationFallbacks;
    return nullptr;
  }
  // Fault only owned/charged pages on the allocating NUMA worker. Never
  // register an entire size-class virtual arena or an adjacent allocation.
  for (uint64_t offset = 0; offset < blockBytes; offset += Traits::kPageSize) {
    block->allocation.data<volatile char>()[offset] = 0;
  }
  block->prepareNanos = nanos(prepareStart);
  bool injectFailure = false;
  {
    std::lock_guard lock(registry.mutex);
    injectFailure = registry.failureAt != 0 && --registry.failureAt == 0;
    ++registry.registerCalls;
  }
  const auto start = Clock::now();
  const auto status = injectFailure
      ? cudaErrorMemoryAllocation
      : cudaHostRegister(
            block->allocation.data(), blockBytes, cudaHostRegisterPortable);
  block->registerNanos = nanos(start);
  if (status != cudaSuccess) {
    {
      std::lock_guard lock(registry.mutex);
      ++registry.registrationFailures;
    }
    if (status != cudaErrorMemoryAllocation &&
        status != cudaErrorNotSupported && status != cudaErrorInvalidValue &&
        status != cudaErrorHostMemoryAlreadyRegistered) {
      CUDF_CUDA_TRY(status);
    }
    if (!injectFailure) {
      cudaGetLastError();
    }
    return nullptr;
  }
  registered = true;
  auto slice = std::make_shared<Slice>(block, 0, pages);
  {
    std::lock_guard lock(registry.mutex);
    registry.blocks.push_back(block);
    for (uint64_t page = 0; page < pages; ++page) {
      bits::clearBit(block->freeBits.data(), page);
    }
    block->usedPages = pages;
    slice->allocated = true;
    block->registered = true;
    registry.retainedBytes += blockBytes;
    nextBlockBytes = std::min<uint64_t>(
        registry.maxBlockBytes,
        std::min(blockBytes, registry.maxBlockBytes) * 2);
    published = true;
  }
  return slice;
}

CacheHostRegistration::Lease::~Lease() {
  release();
}

CacheHostRegistration::Lease::Lease(Lease&& other) noexcept
    : slices_(std::move(other.slices_)),
      pins_(std::move(other.pins_)),
      acquired_(std::exchange(other.acquired_, false)) {}

void CacheHostRegistration::Lease::release() noexcept {
  if (acquired_) {
    std::lock_guard lock(Pool::registry().mutex);
    for (const auto& slice : slices_) {
      --slice->block->users;
    }
    acquired_ = false;
  }
  // Pin/Slice release can take cache/registry locks. Keep both outside the
  // accounting critical section and release only after the reader's DMA fence.
  pins_.clear();
  slices_.clear();
}

void CacheHostRegistration::configure(bool enabled, uint64_t maxBytes) {
  auto& registry = Pool::registry();
  std::vector<std::shared_ptr<Pool::Block>> blocks;
  {
    std::lock_guard lock(registry.mutex);
    VELOX_CHECK_EQ(
        registry.pendingBlocks,
        0,
        "Reconfiguration requires quiescent allocations");
    for (const auto& block : registry.blocks) {
      VELOX_CHECK_EQ(block->users, 0, "Reconfiguration requires completed DMA");
      VELOX_CHECK(!block->retiring);
    }
    blocks = registry.blocks;
    registry.enabled = false;
    for (const auto& block : blocks) {
      block->retiring = true;
    }
  }
  for (const auto& block : blocks) {
    Pool::unregister(block);
  }
  {
    std::lock_guard lock(registry.mutex);
    for (const auto& block : blocks) {
      block->retiring = false;
      block->pool->nextBlockBytes = 0;
    }
  }
  Pool::retireEmpty(nullptr, kAll);
  {
    std::lock_guard lock(registry.mutex);
    VELOX_CHECK_EQ(registry.reservedBytes, 0);
    registry.maxBytes = maxBytes;
    registry.registerCalls = 0;
    registry.unregisterCalls = 0;
    registry.budgetFallbacks = 0;
    registry.allocationFallbacks = 0;
    registry.registrationFailures = 0;
    registry.enabled = enabled && maxBytes != 0;
  }
  cache::AsyncDataCache::setAllocatorFactory(enabled ? Pool::create : nullptr);
}

bool CacheHostRegistration::enabled() {
  return Pool::registry().enabled.load();
}

std::optional<CacheHostRegistration::Lease> CacheHostRegistration::tryAcquire(
    std::span<const cache::CachePin> pins,
    std::shared_ptr<IoStats> stats) {
  if (!enabled() || pins.empty()) {
    return std::nullopt;
  }
  count(stats, "cudfCacheHostRegistrationAttempts");
  const auto pool = poolStats();
  count(stats, "cudfCacheHostPoolSlabsSamples", pool.slabs);
  count(stats, "cudfCacheHostPoolRegisterCallsSamples", pool.registerCalls);
  count(stats, "cudfCacheHostPoolUnregisterCallsSamples", pool.unregisterCalls);
  count(stats, "cudfCacheHostPoolBudgetFallbacksSamples", pool.budgetFallbacks);
  count(
      stats,
      "cudfCacheHostPoolAllocationFallbacksSamples",
      pool.allocationFallbacks);
  count(
      stats,
      "cudfCacheHostPoolRegistrationFailuresSamples",
      pool.registrationFailures);
  count(
      stats,
      "cudfCacheHostPoolUsedBytesSamples",
      pool.usedBytes,
      RuntimeCounter::Unit::kBytes);
  count(
      stats,
      "cudfCacheHostPoolFreeBytesSamples",
      pool.retainedBytes - pool.usedBytes,
      RuntimeCounter::Unit::kBytes);
  Lease lease;
  std::unordered_set<const cache::AsyncDataCacheEntry*> seen;
  for (const auto& pin : pins) {
    if (pin.empty() || !pin.entry()->isShared()) {
      return std::nullopt;
    }
    if (!seen.insert(pin.entry()).second) {
      continue;
    }
    auto slice =
        std::dynamic_pointer_cast<Slice>(pin.entry()->allocationOwner());
    if (!slice) {
      return std::nullopt;
    }
    lease.pins_.push_back(pin);
    lease.slices_.push_back(std::move(slice));
  }
  auto& registry = Pool::registry();
  uint64_t calls = 0, registrationBytes = 0, registerNanos = 0,
           prepareNanos = 0;
  uint64_t sharedBytes = 0, reusedBytes = 0, reservedBytes = 0,
           retainedBytes = 0;
  {
    std::lock_guard lock(registry.mutex);
    if (!registry.enabled ||
        std::any_of(
            lease.slices_.begin(), lease.slices_.end(), [](const auto& slice) {
              return !slice->block->registered || slice->block->retiring;
            })) {
      return std::nullopt;
    }
    for (const auto& slice : lease.slices_) {
      auto& block = *slice->block;
      (block.users != 0 ? sharedBytes : reusedBytes) += slice->capacity();
      ++block.users;
      if (!block.metricsReported && stats) {
        ++calls;
        registrationBytes += block.bytes;
        prepareNanos += block.prepareNanos;
        registerNanos += block.registerNanos;
        block.metricsReported = true;
      }
    }
    lease.acquired_ = true;
    reservedBytes = registry.reservedBytes;
    retainedBytes = registry.retainedBytes;
  }
  count(stats, "cudfCacheHostRegisterCalls", calls);
  count(
      stats,
      "cudfCacheHostPoolPrepareNanos",
      prepareNanos,
      RuntimeCounter::Unit::kNanos);
  count(
      stats,
      "cudfCacheHostRegisterNanos",
      registerNanos,
      RuntimeCounter::Unit::kNanos);
  count(
      stats,
      "cudfCacheHostRegisteredBytes",
      registrationBytes,
      RuntimeCounter::Unit::kBytes);
  count(
      stats,
      "cudfCacheHostRegistrationReusedBytes",
      reusedBytes,
      RuntimeCounter::Unit::kBytes);
  count(
      stats,
      "cudfCacheHostRegistrationSharedBytes",
      sharedBytes,
      RuntimeCounter::Unit::kBytes);
  count(
      stats,
      "cudfCacheHostRegistrationReservedBytesSamples",
      reservedBytes,
      RuntimeCounter::Unit::kBytes);
  count(
      stats,
      "cudfCacheHostRegistrationRetainedBytesSamples",
      retainedBytes,
      RuntimeCounter::Unit::kBytes);
  return lease;
}

CacheHostRegistration::PoolStats CacheHostRegistration::poolStats() {
  auto& registry = Pool::registry();
  std::lock_guard lock(registry.mutex);
  PoolStats stats;
  stats.reservedBytes = registry.reservedBytes;
  stats.retainedBytes = registry.retainedBytes;
  stats.slabs = registry.blocks.size();
  stats.registerCalls = registry.registerCalls;
  stats.unregisterCalls = registry.unregisterCalls;
  stats.budgetFallbacks = registry.budgetFallbacks;
  stats.allocationFallbacks = registry.allocationFallbacks;
  stats.registrationFailures = registry.registrationFailures;
  for (const auto& block : registry.blocks) {
    stats.usedBytes += Traits::pageBytes(block->usedPages);
  }
  return stats;
}

uint64_t CacheHostRegistration::testingReservedBytes() {
  return poolStats().reservedBytes;
}

uint64_t CacheHostRegistration::testingRetainedBytes() {
  return poolStats().retainedBytes;
}

void CacheHostRegistration::setBlockSizesForTesting(
    uint64_t initial,
    uint64_t max) {
  VELOX_CHECK(initial > 0 && initial <= max);
  VELOX_CHECK_EQ(initial % Traits::kPageSize, 0);
  VELOX_CHECK_EQ(max % Traits::kPageSize, 0);
  auto& registry = Pool::registry();
  std::lock_guard lock(registry.mutex);
  registry.initialBytes = initial;
  registry.maxBlockBytes = max;
}

void CacheHostRegistration::setFailureAtForTesting(uint32_t ordinal) {
  auto& registry = Pool::registry();
  std::lock_guard lock(registry.mutex);
  registry.failureAt = ordinal;
}

} // namespace facebook::velox::cudf_velox::connector::hive
