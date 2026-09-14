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

#pragma once

#include "velox/common/caching/AsyncDataCache.h"

#include <optional>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Opt-in registered backing allocator for AsyncDataCache, shared by the GPU
/// BufferedInput and KvikIO readers. Cache entries suballocate stable slices of
/// root-allocator-owned slabs. Registration is paid when a slab is created,
/// not on each entry, query, or H2D. Entry eviction returns its slice without
/// unregistering the slab. No idle CachePins are retained.
class CacheHostRegistration {
  struct Pool;
  struct Slice;

 public:
  class Lease {
   public:
    ~Lease();
    Lease(Lease&&) noexcept;
    Lease& operator=(Lease&&) = delete;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    /// Fence submitted H2D work before release. Pins keep entries alive; the
    /// separate slab lease excludes unregister until device reads complete.
    void release() noexcept;

   private:
    friend class CacheHostRegistration;
    std::vector<std::shared_ptr<Slice>> slices_;
    std::vector<cache::CachePin> pins_;
    bool acquired_{false};
    Lease() = default;
  };

  /// Startup/test configuration, with quiescent readers and allocators only.
  /// Disabling unregisters slabs but preserves any still-live cached bytes.
  /// Backing pages are freed only when all slices have been returned.
  static void configure(bool enabled, uint64_t maxBytes);
  static bool enabled();

  /// Acquires already registered slab storage, never registers an arbitrary
  /// entry. Pins must be shared and fully filled. Ordinary/pageable entries,
  /// tiny entries, and budget/allocation/registration failures use the existing
  /// staging path without refetching. Registration costs are reported once, to
  /// the first acquisition using a slab (which may differ from its preloader).
  static std::optional<Lease> tryAcquire(
      std::span<const cache::CachePin> pins,
      std::shared_ptr<IoStats> stats = nullptr);

  struct PoolStats {
    uint64_t reservedBytes{0}; // Full registered + pending slab capacity.
    uint64_t retainedBytes{0}; // All backing, including pageable live slabs.
    uint64_t usedBytes{0}; // Page-rounded live slices.
    uint64_t slabs{0};
    uint64_t registerCalls{0};
    uint64_t unregisterCalls{0};
    uint64_t budgetFallbacks{0};
    uint64_t allocationFallbacks{0};
    uint64_t registrationFailures{0};
  };
  static PoolStats poolStats();
  static uint64_t testingReservedBytes();
  static uint64_t testingRetainedBytes();
  /// Test-only; configure with quiescent allocators before changing these.
  static void setBlockSizesForTesting(uint64_t initialBytes, uint64_t maxBytes);
  /// Fail the Nth subsequent new slab registration (zero disables).
  static void setFailureAtForTesting(uint32_t ordinal);
};

} // namespace facebook::velox::cudf_velox::connector::hive
