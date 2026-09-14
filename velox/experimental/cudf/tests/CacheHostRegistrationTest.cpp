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

#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/MmapAllocator.h"

#include <cudf/utilities/error.hpp>

#include <gtest/gtest.h>
#include <sched.h>

#include <array>
#include <atomic>
#include <cstring>
#include <future>
#include <limits>
#include <thread>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
using Registration = CacheHostRegistration;
using Traits = memory::AllocationTraits;

class CacheHostRegistrationTest : public testing::Test {
 protected:
  static constexpr uint64_t kSize = 64 << 10;
  static constexpr uint64_t kSlab = 1 << 20;
  void SetUp() override {
    CUDF_CUDA_TRY(cudaSetDevice(0));
    // Stable locality for exact slab-count assertions. Child test threads
    // inherit affinity; registration is portable across CUDA devices.
    ASSERT_EQ(::sched_getaffinity(0, sizeof(oldAffinity_), &oldAffinity_), 0);
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(::sched_getcpu(), &affinity);
    ASSERT_EQ(::sched_setaffinity(0, sizeof(affinity), &affinity), 0);
    Registration::configure(true, 4 * kSlab);
    Registration::setBlockSizesForTesting(kSlab, 2 * kSlab);
  }
  void TearDown() override {
    cache_->shutdown();
    EXPECT_EQ(Registration::testingReservedBytes(), 0);
    EXPECT_EQ(Registration::testingRetainedBytes(), 0);
    Registration::configure(false, 0);
    Registration::setBlockSizesForTesting(64ULL << 20, 256ULL << 20);
    Registration::setFailureAtForTesting(0);
    EXPECT_EQ(::sched_setaffinity(0, sizeof(oldAffinity_), &oldAffinity_), 0);
  }
  cache::CachePin makePin(uint64_t offset, uint64_t size = kSize) {
    auto pin = cache_->findOrCreate({file_.id(), offset}, size);
    if (pin.checkedEntry()->isExclusive()) {
      for (const auto& range : pin.checkedEntry()->dataRanges(size)) {
        std::memset(range.data(), 'x', range.size());
      }
      pin.checkedEntry()->setExclusiveToShared(false);
    }
    return pin;
  }
  std::optional<Registration::Lease> acquire(const cache::CachePin& pin) {
    return Registration::tryAcquire(
        std::span<const cache::CachePin>(&pin, 1), stats_);
  }
  std::shared_ptr<memory::MallocAllocator> allocator_ =
      std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 8 << 20,
              .reservationByteLimit = 0});
  std::shared_ptr<cache::AsyncDataCache> cache_ =
      cache::AsyncDataCache::create(allocator_.get());
  StringIdLease file_{fileIds(), "registered-cache-test"};
  std::shared_ptr<IoStats> stats_ = std::make_shared<IoStats>();
  cpu_set_t oldAffinity_;
};

TEST_F(CacheHostRegistrationTest, disabledUsesOrdinaryBacking) {
  Registration::configure(false, 0);
  auto pin = makePin(0);
  EXPECT_EQ(pin.entry()->allocationOwner(), nullptr);
  EXPECT_FALSE(acquire(pin));
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  EXPECT_TRUE(stats_->stats().empty());
}

TEST_F(CacheHostRegistrationTest, manySmallEntriesShareOneChargedRegistration) {
  std::vector<cache::CachePin> pins;
  for (int i = 0; i < 16; ++i) {
    pins.push_back(makePin(i * kSize));
    ASSERT_NE(pins.back().entry()->allocationOwner(), nullptr);
    EXPECT_EQ(pins.back().entry()->dataCapacity(), kSize);
    auto lease = acquire(pins.back());
    ASSERT_TRUE(lease);
  }
  const auto pool = Registration::poolStats();
  EXPECT_EQ(pool.slabs, 1);
  EXPECT_EQ(pool.registerCalls, 1);
  EXPECT_EQ(pool.reservedBytes, kSlab);
  EXPECT_EQ(pool.usedBytes, kSlab);
  EXPECT_EQ(allocator_->numAllocated() * Traits::kPageSize, kSlab);
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisterCalls").sum, 1);
}

TEST_F(
    CacheHostRegistrationTest,
    idleSpaceIsChargedAndEntryEvictionReusesSlab) {
  auto pin = makePin(0);
  auto* address = pin.entry()->contiguousData();
  ASSERT_TRUE(acquire(pin));
  pin.clear();
  EXPECT_EQ(cache_->refreshStats().numShared, 0); // No registry-owned CachePin.
  cache_->clear();
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
  EXPECT_EQ(Registration::poolStats().usedBytes, 0);
  EXPECT_EQ(Registration::testingReservedBytes(), kSlab);
  EXPECT_EQ(allocator_->numAllocated() * Traits::kPageSize, kSlab);
  Registration::setFailureAtForTesting(1);
  auto next = makePin(123 * kSize);
  EXPECT_EQ(next.entry()->contiguousData(), address);
  ASSERT_TRUE(acquire(next));
  EXPECT_EQ(Registration::poolStats().registerCalls, 1);
  EXPECT_EQ(Registration::poolStats().unregisterCalls, 0);
}

TEST_F(CacheHostRegistrationTest, growthIsGeometricAndNeverMovesLiveSlices) {
  auto first = makePin(0, kSlab);
  auto* address = first.entry()->contiguousData();
  auto second = makePin(kSlab, kSize);
  EXPECT_EQ(Registration::testingReservedBytes(), 3 * kSlab);
  EXPECT_EQ(first.entry()->contiguousData(), address);
  EXPECT_EQ(std::string_view(address, kSize), std::string(kSize, 'x'));
  auto third = makePin(2 * kSlab, 2 * kSlab - kSize);
  EXPECT_EQ(Registration::poolStats().slabs, 2);
  auto fourth = makePin(4 * kSlab, kSize);
  // Remaining budget caps this regular slab at 1MiB, including its free space.
  EXPECT_EQ(Registration::testingReservedBytes(), 4 * kSlab);
  EXPECT_EQ(Registration::poolStats().slabs, 3);
}

TEST_F(
    CacheHostRegistrationTest,
    oversizedEntryUsesDedicatedBlockWithinBudget) {
  auto pin = makePin(0, 3 * kSlab + 17);
  EXPECT_EQ(pin.entry()->dataCapacity(), 3 * kSlab + Traits::kPageSize);
  EXPECT_EQ(Registration::testingReservedBytes(), pin.entry()->dataCapacity());
  EXPECT_TRUE(acquire(pin));
}

TEST_F(
    CacheHostRegistrationTest,
    budgetExhaustionFallsBackWithoutInvalidatingLiveData) {
  Registration::configure(true, kSlab);
  auto first = makePin(0, kSlab);
  auto lease = acquire(first);
  ASSERT_TRUE(lease);
  auto fallback = makePin(kSlab);
  EXPECT_EQ(fallback.entry()->allocationOwner(), nullptr);
  EXPECT_FALSE(acquire(fallback));
  EXPECT_EQ(Registration::testingReservedBytes(), kSlab);
  EXPECT_TRUE(acquire(first));
}

TEST_F(
    CacheHostRegistrationTest,
    registrationFailureReleasesReservationAndBacking) {
  Registration::setFailureAtForTesting(1);
  auto fallback = makePin(0);
  EXPECT_EQ(fallback.entry()->allocationOwner(), nullptr);
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  EXPECT_EQ(Registration::testingRetainedBytes(), 0);
  EXPECT_EQ(allocator_->numAllocated() * Traits::kPageSize, kSize);
  auto next = makePin(kSize);
  EXPECT_TRUE(acquire(next));
}

TEST_F(
    CacheHostRegistrationTest,
    failedFillReturnsSliceWithoutRegistrationChurn) {
  {
    auto pin = cache_->findOrCreate({file_.id(), 0}, kSize);
    ASSERT_TRUE(pin.entry()->isExclusive());
    EXPECT_FALSE(acquire(pin));
  }
  EXPECT_EQ(Registration::poolStats().usedBytes, 0);
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
  auto next = makePin(0);
  EXPECT_TRUE(acquire(next));
  EXPECT_EQ(Registration::poolStats().registerCalls, 1);
}

TEST_F(CacheHostRegistrationTest, tinyAndPreexistingPageableEntriesUseStaging) {
  auto tiny = makePin(0, 127);
  EXPECT_FALSE(acquire(tiny));
  Registration::configure(false, 0);
  auto ordinary = makePin(kSize);
  auto* address = ordinary.entry()->dataRanges(kSize).front().data();
  CUDF_CUDA_TRY(cudaHostRegister(address, kSize, cudaHostRegisterPortable));
  Registration::configure(true, 4 * kSlab);
  // Never adopt somebody else's registration or register a partial malloc page.
  EXPECT_FALSE(acquire(ordinary));
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  CUDF_CUDA_TRY(cudaHostUnregister(address));
}

TEST_F(
    CacheHostRegistrationTest,
    duplicateAndPartialAcquisitionsDoNotLeakUsers) {
  auto pin = makePin(0);
  const std::array repeated{pin, pin, pin};
  auto lease = Registration::tryAcquire(repeated, stats_);
  ASSERT_TRUE(lease);
  auto tiny = makePin(kSize, 100);
  const std::array partial{pin, tiny};
  EXPECT_FALSE(Registration::tryAcquire(partial, stats_));
  lease.reset();
  // A failed partially collected lease must not decrement unacquired users.
  EXPECT_NO_THROW(Registration::configure(false, 0));
}

TEST_F(
    CacheHostRegistrationTest,
    activeLeasePreservesSlicesAcrossClearAndShrink) {
  auto pin = makePin(0);
  auto* address = pin.entry()->contiguousData();
  auto lease = acquire(pin);
  pin.clear();
  cache_->clear();
  EXPECT_EQ(cache_->shrink(kSlab), 0);
  EXPECT_GT(cache_->refreshStats().numEntries, 0);
  EXPECT_EQ(std::string_view(address, kSize), std::string(kSize, 'x'));
  EXPECT_THROW(Registration::configure(false, 0), VeloxRuntimeError);
  lease.reset();
  EXPECT_EQ(cache_->shrink(kSlab), kSlab);
  EXPECT_EQ(allocator_->numAllocated(), 0);
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
}

TEST_F(
    CacheHostRegistrationTest,
    partialSlabEvictionDoesNotClaimFreedCapacity) {
  auto first = makePin(0);
  auto second = makePin(kSize);
  first.clear();
  EXPECT_EQ(cache_->shrink(kSize), 0);
  EXPECT_EQ(Registration::poolStats().usedBytes, kSize);
  EXPECT_EQ(allocator_->numAllocated() * Traits::kPageSize, kSlab);
  second.clear();
  EXPECT_EQ(cache_->shrink(kSize), kSlab);
  EXPECT_EQ(allocator_->numAllocated(), 0);
}

TEST_F(
    CacheHostRegistrationTest,
    freeAdjacentSlicesCoalesceWithoutCopyingLiveData) {
  auto first = makePin(0, kSlab / 4);
  auto second = makePin(kSlab / 4, kSlab / 4);
  auto third = makePin(kSlab / 2, kSlab / 2);
  auto* firstAddress = first.entry()->contiguousData();
  auto* thirdAddress = third.entry()->contiguousData();
  first.clear();
  second.clear();
  cache_->clear();
  auto combined = makePin(2 * kSlab, kSlab / 2);
  EXPECT_EQ(combined.entry()->contiguousData(), firstAddress);
  EXPECT_EQ(third.entry()->contiguousData(), thirdAddress);
  EXPECT_EQ(std::string_view(thirdAddress, kSize), std::string(kSize, 'x'));
  EXPECT_EQ(Registration::poolStats().registerCalls, 1);
}

TEST_F(CacheHostRegistrationTest, rootAllocatorPressureReclaimsEmptySlabs) {
  auto pin = makePin(0);
  pin.clear();
  cache_->clear();
  memory::ContiguousAllocation query;
  ASSERT_TRUE(allocator_->allocateContiguous(
      Traits::numPages(8 * kSlab), nullptr, query));
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  EXPECT_EQ(Registration::testingRetainedBytes(), 0);
  allocator_->freeContiguous(query);
}

TEST_F(CacheHostRegistrationTest, rootPressureCanEvictDataAndThenFreeItsSlab) {
  auto pin = makePin(0);
  pin.clear();
  EXPECT_GT(cache_->refreshStats().numEntries, 0);
  memory::ContiguousAllocation query;
  ASSERT_TRUE(allocator_->allocateContiguous(
      Traits::numPages(8 * kSlab), nullptr, query));
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  allocator_->freeContiguous(query);
}

TEST_F(CacheHostRegistrationTest, queryMetricsChargeEachSlabOnlyOnce) {
  auto pin = makePin(0);
  ASSERT_TRUE(acquire(pin));
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisterCalls").sum, 1);
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisteredBytes").sum, kSlab);
  pin.clear();
  cache_->clear();
  stats_ = std::make_shared<IoStats>();
  auto next = makePin(kSize);
  ASSERT_TRUE(acquire(next));
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisterCalls").sum, 0);
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisterNanos").sum, 0);
  EXPECT_EQ(stats_->stats().at("cudfCacheHostPoolPrepareNanos").sum, 0);
  EXPECT_EQ(stats_->stats().at("cudfCacheHostPoolRegisterCallsSamples").max, 1);
}

TEST_F(CacheHostRegistrationTest, backingExceptionDoesNotStrandExclusiveEntry) {
  struct ThrowOnce : cache::CacheAllocator {
    std::shared_ptr<cache::CacheAllocation> allocate(uint64_t) override {
      if (std::exchange(fail, false)) {
        VELOX_FAIL("Injected backing failure");
      }
      return nullptr;
    }
    uint64_t reclaim(uint64_t) override {
      return 0;
    }
    void shutdown() noexcept override {}
    bool fail{true};
  };
  cache::AsyncDataCache::setAllocatorFactory(
      +[](cache::AsyncDataCache*) -> std::shared_ptr<cache::CacheAllocator> {
        return std::make_shared<ThrowOnce>();
      });
  auto isolated = std::make_shared<cache::AsyncDataCache>(allocator_.get());
  EXPECT_THROW(
      isolated->findOrCreate({file_.id(), 0}, kSize), VeloxRuntimeError);
  EXPECT_FALSE(isolated->find({file_.id(), 0}).has_value());
  auto retry = isolated->findOrCreate({file_.id(), 0}, kSize);
  ASSERT_FALSE(retry.empty());
  EXPECT_TRUE(retry.entry()->isExclusive());
  retry.clear();
  isolated->shutdown();
}

TEST_F(CacheHostRegistrationTest, mmapAllocatorAccountingAndReclamation) {
  auto allocator = std::make_shared<memory::MmapAllocator>(
      memory::MemoryAllocator::Options{.capacity = 64 << 20});
  auto cache = cache::AsyncDataCache::create(allocator.get());
  auto pin = cache->findOrCreate({file_.id(), 0}, kSize);
  ASSERT_NE(pin.entry()->allocationOwner(), nullptr);
  pin.entry()->setExclusiveToShared(false);
  EXPECT_EQ(allocator->numAllocated() * Traits::kPageSize, kSlab);
  EXPECT_EQ(allocator->numExternalMapped() * Traits::kPageSize, kSlab);
  pin.clear();
  cache->clear();
  EXPECT_EQ(cache->shrink(kSize), kSlab);
  EXPECT_EQ(allocator->numAllocated(), 0);
  EXPECT_TRUE(allocator->checkConsistency());
  cache->shutdown();
}

TEST_F(
    CacheHostRegistrationTest,
    fileInvalidationReturnsSlicesAndPreservesOtherFiles) {
  auto first = makePin(0);
  StringIdLease other(fileIds(), "registered-cache-other-file");
  auto second = cache_->findOrCreate({other.id(), 0}, kSize);
  second.entry()->setExclusiveToShared(false);
  first.clear();
  folly::F14FastSet<uint64_t> retained;
  EXPECT_TRUE(cache_->removeFileEntries({file_.id()}, retained));
  EXPECT_TRUE(retained.empty());
  EXPECT_FALSE(cache_->exists({file_.id(), 0}));
  EXPECT_TRUE(cache_->exists({other.id(), 0}));
  EXPECT_EQ(Registration::poolStats().usedBytes, kSize);
  EXPECT_EQ(Registration::poolStats().unregisterCalls, 0);
}

TEST_F(
    CacheHostRegistrationTest,
    reconfigureUnpinsButPreservesLiveCachedBytes) {
  auto pin = makePin(0);
  auto* data = pin.entry()->contiguousData();
  Registration::configure(false, 0);
  EXPECT_EQ(Registration::testingReservedBytes(), 0);
  EXPECT_EQ(Registration::testingRetainedBytes(), kSlab);
  EXPECT_EQ(std::string_view(data, kSize), std::string(kSize, 'x'));
  EXPECT_FALSE(acquire(pin));
  pin.clear();
  cache_->clear();
  EXPECT_EQ(cache_->shrink(kSize), kSlab);
}

TEST_F(
    CacheHostRegistrationTest,
    shutdownAndDestructionReleaseBackingBeforeAllocator) {
  auto pin = makePin(0);
  ASSERT_TRUE(acquire(pin));
  pin.clear();
  cache_->shutdown();
  EXPECT_EQ(Registration::testingRetainedBytes(), 0);
  EXPECT_EQ(allocator_->numAllocated(), 0);
  // Direct ownership exercises the destructor. create() instead installs a
  // strong reference in the allocator and requires explicit cache shutdown.
  auto other = std::make_shared<cache::AsyncDataCache>(allocator_.get());
  auto otherPin = other->findOrCreate({file_.id(), 0}, kSize);
  otherPin.entry()->setExclusiveToShared(false);
  otherPin.clear();
  other.reset();
  EXPECT_EQ(Registration::testingRetainedBytes(), 0);
  EXPECT_EQ(allocator_->numAllocated(), 0);
}

TEST_F(
    CacheHostRegistrationTest,
    multipleCachesShareBudgetNotBackingLifetimes) {
  auto pin = makePin(0);
  auto otherAllocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 8 << 20, .reservationByteLimit = 0});
  auto other = cache::AsyncDataCache::create(otherAllocator.get());
  auto otherPin = other->findOrCreate({file_.id(), 0}, kSize);
  otherPin.entry()->setExclusiveToShared(false);
  EXPECT_EQ(Registration::testingReservedBytes(), 2 * kSlab);
  otherPin.clear();
  other->shutdown();
  EXPECT_EQ(Registration::testingReservedBytes(), kSlab);
  EXPECT_TRUE(acquire(pin));
}

TEST_F(CacheHostRegistrationTest, idleSlabsDoNotRetainQueryStats) {
  auto pin = makePin(0);
  auto stats = std::make_shared<IoStats>();
  std::weak_ptr<IoStats> weak = stats;
  auto lease = Registration::tryAcquire(
      std::span<const cache::CachePin>(&pin, 1), stats);
  ASSERT_TRUE(lease);
  stats.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(CacheHostRegistrationTest, concurrentAllocationClearAndLeaseReuse) {
  std::atomic<bool> start{false};
  std::vector<std::future<void>> workers;
  for (int thread = 0; thread < 8; ++thread) {
    workers.push_back(std::async(std::launch::async, [&, thread] {
      CUDF_CUDA_TRY(cudaSetDevice(0));
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 50; ++i) {
        auto pin = makePin((thread * 100 + i) * kSize);
        auto lease =
            Registration::tryAcquire(std::span<const cache::CachePin>(&pin, 1));
        ASSERT_TRUE(lease);
        EXPECT_EQ(pin.entry()->contiguousData()[0], 'x');
        if (i % 4 == 0) {
          cache_->clear();
        }
      }
    }));
  }
  start = true;
  for (auto& worker : workers) {
    worker.get();
  }
  cache_->clear();
  EXPECT_EQ(Registration::poolStats().usedBytes, 0);
  EXPECT_LE(Registration::testingReservedBytes(), 4 * kSlab);
  EXPECT_LT(Registration::poolStats().registerCalls, 10);
}

TEST_F(CacheHostRegistrationTest, portableSlabCanBeLeasedOnAnotherDevice) {
  int devices = 0;
  CUDF_CUDA_TRY(cudaGetDeviceCount(&devices));
  if (devices < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  auto pin = makePin(0);
  CUDF_CUDA_TRY(cudaSetDevice(1));
  auto lease = acquire(pin);
  ASSERT_TRUE(lease);
  char* device = nullptr;
  CUDF_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device), kSize));
  CUDF_CUDA_TRY(cudaMemcpy(
      device, pin.entry()->contiguousData(), kSize, cudaMemcpyHostToDevice));
  std::string result(kSize, '\0');
  CUDF_CUDA_TRY(
      cudaMemcpy(result.data(), device, kSize, cudaMemcpyDeviceToHost));
  EXPECT_EQ(result, std::string(kSize, 'x'));
  CUDF_CUDA_TRY(cudaFree(device));
  lease.reset();
  CUDF_CUDA_TRY(cudaSetDevice(0));
}
} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
