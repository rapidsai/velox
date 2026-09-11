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

#include "velox/experimental/cudf/connectors/hive/CachingDataSource.h"

#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/MallocAllocator.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
using namespace std::chrono_literals;

struct ReadState {
  std::string data = std::string(8192, 'x');
  std::atomic<size_t> reads{0};
  std::atomic<bool> shortRead{false};
  std::atomic<bool> failRead{false};
  std::atomic<bool> destroyed{false};
  std::shared_future<void> gate;
};

class MemorySource : public cudf::io::datasource {
 public:
  explicit MemorySource(std::shared_ptr<ReadState> state)
      : state_(std::move(state)) {}
  ~MemorySource() override {
    state_->destroyed = true;
  }
  size_t size() const override {
    return state_->data.size();
  }
  bool supports_device_read() const override {
    return true;
  }
  bool is_device_read_preferred(size_t) const override {
    return true;
  }
  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t bytes)
      override {
    std::vector<uint8_t> result(
        offset < size() ? std::min(bytes, size() - offset) : 0);
    result.resize(host_read(offset, result.size(), result.data()));
    return datasource::buffer::create(std::move(result));
  }
  size_t host_read(size_t offset, size_t bytes, uint8_t* dst) override {
    ++state_->reads;
    if (state_->gate.valid()) {
      state_->gate.wait();
    }
    VELOX_CHECK(!state_->failRead.exchange(false), "Injected remote failure");
    bytes = offset < size() ? std::min(bytes, size() - offset) : 0;
    if (state_->shortRead.exchange(false) && bytes > 0) {
      --bytes;
    }
    if (bytes > 0) {
      std::memcpy(dst, state_->data.data() + offset, bytes);
    }
    return bytes;
  }

 private:
  std::shared_ptr<ReadState> state_;
};

struct StreamGate {
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
};

void CUDART_CB waitForStreamGate(void* argument) {
  auto* gate = static_cast<StreamGate*>(argument);
  gate->entered.set_value();
  gate->released.wait();
}

class CachingDataSourceTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This test executable owns its remote executor.
    setenv("KVIKIO_NTHREADS", "2", 1);
  }

  std::unique_ptr<cudf::io::datasource> source(bool cacheable = true) {
    return maybeCacheKvikioDataSource(
        std::make_unique<MemorySource>(read_),
        path_,
        cache_.get(),
        cacheable,
        stats_);
  }
  std::string fromDevice(const uint8_t* device, size_t bytes) {
    std::string result(bytes, '\0');
    CUDF_CUDA_TRY(
        cudaMemcpy(result.data(), device, bytes, cudaMemcpyDeviceToHost));
    return result;
  }

  std::shared_ptr<memory::MallocAllocator> allocator_ =
      std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 16 << 20,
              .reservationByteLimit = 0});
  std::shared_ptr<cache::AsyncDataCache> cache_ =
      cache::AsyncDataCache::create(allocator_.get());
  std::shared_ptr<ReadState> read_ = std::make_shared<ReadState>();
  std::shared_ptr<IoStats> stats_ = std::make_shared<IoStats>();
  const std::string path_ = "s3://test/caching-datasource";
};

TEST_F(CachingDataSourceTest, cacheOffAndNonCacheablePreserveDelegateIdentity) {
  for (const bool cacheable : {false, true}) {
    auto delegate = std::make_unique<MemorySource>(read_);
    auto* original = delegate.get();
    auto actual = maybeCacheKvikioDataSource(
        std::move(delegate),
        path_,
        cacheable ? nullptr : cache_.get(),
        cacheable);
    EXPECT_EQ(actual.get(), original);
  }
}

TEST_F(CachingDataSourceTest, hostMissHitAndFileIdentity) {
  auto input = source();
  auto first = input->host_read(17, 512);
  auto second = input->host_read(17, 512);
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(std::memcmp(first->data(), second->data(), 512), 0);
  const auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfKvikioCacheMissBytes").sum, 512);
  EXPECT_EQ(metrics.at("cudfKvikioCacheHitBytes").sum, 512);
  auto anotherFile = maybeCacheKvikioDataSource(
      std::make_unique<MemorySource>(read_),
      "s3://test/another-file",
      cache_.get(),
      true);
  anotherFile->host_read(17, 512);
  EXPECT_EQ(read_->reads, 2);
}

TEST_F(CachingDataSourceTest, failedAndShortReadsAreNotPublished) {
  auto input = source();
  read_->shortRead = true;
  EXPECT_THROW(input->host_read(31, 100), VeloxRuntimeError);
  read_->failRead = true;
  EXPECT_THROW(input->host_read(31, 100), VeloxRuntimeError);
  EXPECT_EQ(input->host_read(31, 100)->size(), 100);
  EXPECT_EQ(read_->reads, 3);
  input->host_read(31, 100);
  EXPECT_EQ(read_->reads, 3);
}

TEST_F(CachingDataSourceTest, nonContiguousCacheHit) {
  StringIdLease file(fileIds(), path_);
  auto pin = cache_->findOrCreate({file.id(), 0}, read_->data.size(), false);
  auto* entry = pin.checkedEntry();
  ASSERT_FALSE(entry->hasContiguousData());
  for (const auto& range : entry->dataRanges(read_->data.size())) {
    std::memset(range.data(), 'p', range.size());
  }
  entry->setExclusiveToShared();
  auto input = source();
  auto bytes = input->host_read(0, read_->data.size());
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size()),
      std::string(read_->data.size(), 'p'));
  EXPECT_EQ(read_->reads, 0);
}

TEST_F(CachingDataSourceTest, emptyEofAndOverflowingEnd) {
  auto input = source();
  EXPECT_EQ(input->host_read(0, 0)->size(), 0);
  EXPECT_EQ(input->host_read(input->size(), 1)->size(), 0);
  EXPECT_EQ(
      input->host_read(std::numeric_limits<size_t>::max(), 1, nullptr), 0);
  EXPECT_EQ(read_->reads, 0);
  EXPECT_EQ(
      input->host_read(input->size() - 3, std::numeric_limits<size_t>::max())
          ->size(),
      3);
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, longerRangeReplacesShortEntry) {
  auto input = source();
  input->host_read(1, 100);
  EXPECT_EQ(input->host_read(1, 200)->size(), 200);
  EXPECT_EQ(read_->reads, 2);
  input->host_read(1, 50);
  EXPECT_EQ(read_->reads, 2);
}

TEST_F(CachingDataSourceTest, cacheCapacityFailureFallsBackWithoutPublishing) {
  read_->data.assign(32 << 20, 'b');
  auto input = source();
  auto result = input->host_read(0, read_->data.size());
  EXPECT_EQ(result->size(), read_->data.size());
  EXPECT_EQ(result->data()[result->size() - 1], 'b');
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(
      stats_->stats().at("cudfKvikioCacheBypassBytes").sum, result->size());
}

TEST_F(CachingDataSourceTest, concurrentReadersShareOneFill) {
  auto input = source();
  std::promise<void> start;
  auto started = start.get_future().share();
  std::vector<std::future<void>> readers;
  for (int i = 0; i < 8; ++i) {
    readers.push_back(std::async(std::launch::async, [&input, started] {
      started.wait();
      auto result = input->host_read(0, 4096);
      EXPECT_EQ(result->size(), 4096);
      EXPECT_EQ(result->data()[4095], 'x');
    }));
  }
  start.set_value();
  for (auto& reader : readers) {
    reader.get();
  }
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, hostFutureRetainsDelegate) {
  auto input = source();
  auto future = input->host_read_async(0, 100);
  input.reset();
  EXPECT_FALSE(read_->destroyed);
  EXPECT_EQ(future.get()->size(), 100);
  EXPECT_TRUE(read_->destroyed);
}

TEST_F(CachingDataSourceTest, deviceReadServesSecondReadFromCache) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      1024, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  EXPECT_EQ(input->device_read_async(0, 1024, dst, stream.view()).get(), 1024);
  EXPECT_EQ(fromDevice(dst, 1024), read_->data.substr(0, 1024));
  EXPECT_EQ(input->device_read_async(0, 1024, dst, stream.view()).get(), 1024);
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(fromDevice(dst, 1024), read_->data.substr(0, 1024));
}

TEST_F(CachingDataSourceTest, owningDeviceReadUsesCache) {
  auto input = source();
  rmm::cuda_stream stream;
  auto first = input->device_read(9, 200, stream.view());
  auto second = input->device_read(9, 200, stream.view());
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(
      fromDevice(first->data(), first->size()), read_->data.substr(9, 200));
  EXPECT_EQ(
      fromDevice(second->data(), second->size()), read_->data.substr(9, 200));
}

TEST_F(CachingDataSourceTest, synchronousReadsUseCacheAndClampRanges) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      200, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  // Exercise a miss and both synchronous cache-hit overloads.
  EXPECT_EQ(input->device_read(9, 200, dst, stream.view()), 200);
  EXPECT_EQ(input->device_read(9, 200, dst, stream.view()), 200);
  auto owned = input->device_read(9, 200, stream.view());
  EXPECT_EQ(read_->reads, 1);
  EXPECT_EQ(fromDevice(dst, 200), read_->data.substr(9, 200));
  EXPECT_EQ(
      fromDevice(owned->data(), owned->size()), read_->data.substr(9, 200));

  EXPECT_EQ(input->device_read(0, 0, nullptr, stream.view()), 0);
  EXPECT_EQ(input->device_read(input->size(), 1, nullptr, stream.view()), 0);
  EXPECT_EQ(
      input->device_read(
          std::numeric_limits<size_t>::max(), 1, nullptr, stream.view()),
      0);
  EXPECT_EQ(
      input->device_read(
          input->size() - 3,
          std::numeric_limits<size_t>::max(),
          dst,
          stream.view()),
      3);
  EXPECT_EQ(fromDevice(dst, 3), read_->data.substr(input->size() - 3));
  EXPECT_THROW(
      input->device_read(0, 1, nullptr, stream.view()), VeloxRuntimeError);
}

TEST_F(CachingDataSourceTest, synchronousReadsFromSingleThreadCaller) {
  for (const bool owning : {false, true}) {
    SCOPED_TRACE(owning);
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    folly::CPUThreadPoolExecutor pool(1);
    auto input = maybeCacheKvikioDataSource(
        std::make_unique<MemorySource>(read_),
        path_,
        cache_.get(),
        true,
        stats_);
    // Exercise each overload on a cache hit from a single-threaded caller.
    ASSERT_EQ(input->host_read(0, 100)->size(), 100);
    std::unique_ptr<cudf::io::datasource::buffer> owned;
    std::promise<size_t> completed;
    auto result = completed.get_future();
    pool.add([&] {
      try {
        if (owning) {
          owned = input->device_read(0, 100, stream.view());
          completed.set_value(owned->size());
        } else {
          completed.set_value(input->device_read(0, 100, dst, stream.view()));
        }
      } catch (...) {
        completed.set_exception(std::current_exception());
      }
    });
    const auto initial = result.wait_for(5s);
    EXPECT_EQ(initial, std::future_status::ready)
        << "Synchronous read did not complete from the caller executor";
    EXPECT_EQ(result.get(), 100);
    EXPECT_EQ(
        fromDevice(owning ? owned->data() : dst, 100), std::string(100, 'x'));
    pool.join();
  }
}

TEST_F(CachingDataSourceTest, synchronousReadWaitsForH2D) {
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      100, stream.view(), cudf::get_current_device_resource_ref());
  stream.synchronize();
  auto* dst = static_cast<uint8_t*>(destination.data());
  StreamGate gate;
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.value(), waitForStreamGate, &gate));
  gate.entered.get_future().wait();
  auto waiter = std::async(std::launch::async, [&] {
    return input->device_read(10, 100, dst, stream.view());
  });
  EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
  gate.release.set_value();
  EXPECT_EQ(waiter.get(), 100);
  EXPECT_EQ(fromDevice(dst, 100), read_->data.substr(10, 100));
}

TEST_F(CachingDataSourceTest, deviceFutureRetainsDelegate) {
  std::promise<void> release;
  read_->gate = release.get_future().share();
  auto input = source();
  rmm::cuda_stream stream;
  rmm::device_buffer destination(
      100, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  auto future = input->device_read_async(10, 100, dst, stream.view());
  input.reset();
  EXPECT_FALSE(read_->destroyed);
  release.set_value();
  EXPECT_EQ(future.get(), 100);
  EXPECT_EQ(fromDevice(dst, 100), read_->data.substr(10, 100));
}

TEST_F(CachingDataSourceTest, usesTheDestinationStreamsDevice) {
  int count = 0;
  CUDF_CUDA_TRY(cudaGetDeviceCount(&count));
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  auto input = source();
  for (int device : {0, 1, 0}) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    EXPECT_EQ(input->device_read_async(0, 100, dst, stream.view()).get(), 100);
    EXPECT_EQ(fromDevice(dst, 100), std::string(100, 'x'));
  }
  EXPECT_EQ(read_->reads, 1);
}

TEST_F(CachingDataSourceTest, completionAndDiscardWaitForH2D) {
  for (bool discard : {false, true}) {
    auto input = source();
    rmm::cuda_stream stream;
    rmm::device_buffer destination(
        100, stream.view(), cudf::get_current_device_resource_ref());
    stream.synchronize();
    auto* dst = static_cast<uint8_t*>(destination.data());
    StreamGate gate;
    CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.value(), waitForStreamGate, &gate));
    gate.entered.get_future().wait();
    // Use distinct ranges so both passes exercise asynchronous cache fills.
    auto future =
        input->device_read_async(discard ? 101 : 1, 100, dst, stream.view());
    auto waiter = std::async(
        std::launch::async, [f = std::move(future), discard]() mutable {
          if (discard) {
            f = {};
          } else {
            EXPECT_EQ(f.get(), 100);
          }
        });
    EXPECT_EQ(waiter.wait_for(50ms), std::future_status::timeout);
    gate.release.set_value();
    waiter.get();
    EXPECT_EQ(fromDevice(dst, 100), std::string(100, 'x'));
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
