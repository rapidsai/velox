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

#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CacheHostRegistration.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderIOHelpers.h"
#include "velox/experimental/cudf/connectors/hive/PinnedStagingArena.h"

#include "velox/common/caching/FileIds.h"
#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;
using facebook::velox::StringIdLease;
using facebook::velox::cache::AsyncDataCache;
using facebook::velox::cache::ScanTracker;
using facebook::velox::dwio::common::BufferedInput;
using facebook::velox::dwio::common::CachedBufferedInput;
using facebook::velox::dwio::common::DirectBufferedInput;

class TestCudaStream {
 public:
  TestCudaStream() {
    CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  rmm::cuda_stream_view view() const {
    return rmm::cuda_stream_view{stream_};
  }

 private:
  cudaStream_t stream_{nullptr};
};

struct StreamGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool open{false};
  std::promise<void> entered;
};

void CUDART_CB waitForGate(void* opaque) {
  auto* gate = static_cast<StreamGate*>(opaque);
  gate->entered.set_value();
  std::unique_lock<std::mutex> lock(gate->mutex);
  gate->cv.wait(lock, [gate] { return gate->open; });
}

void releaseGate(StreamGate& gate) {
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.open = true;
  }
  gate.cv.notify_all();
}

class ExecutorBufferedInput final : public BufferedInput {
 public:
  ExecutorBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(readFile), pool), executor_(executor) {}

  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* const executor_;
};

// A BufferedInput implementation whose streams own/refill their own buffers,
// not the base class's loaded allocation. It must keep the copying fallback.
class StreamOnlyBufferedInput final : public BufferedInput {
 public:
  using BufferedInput::BufferedInput;

  std::unique_ptr<dwio::common::SeekableInputStream> enqueue(
      common::Region region,
      const dwio::common::StreamIdentifier* = nullptr) override {
    return read(region.offset, region.length, dwio::common::LogType::FILE);
  }

  void load(dwio::common::LogType) override {}
};

// Hold storage reads open so the test can distinguish parallel read-ahead
// from demand reads issued only after the preceding quantum completes.
class GatedReadFile final : public InMemoryReadFile {
 public:
  using InMemoryReadFile::InMemoryReadFile;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    {
      std::unique_lock lock(mutex_);
      ++startedReads_;
      cv_.notify_all();
      cv_.wait(lock, [&] { return released_; });
    }
    return InMemoryReadFile::preadv(offset, buffers, context);
  }

  bool waitForReads(size_t count) const {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return startedReads_ >= count; });
  }

  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  mutable size_t startedReads_{0};
  bool released_{false};
};

// The first region can finish while a later, independent storage read stays
// in flight. Also supports failure after an earlier H2D was submitted.
class GatedTailReadFile final : public InMemoryReadFile {
 public:
  GatedTailReadFile(const std::string& data, uint64_t tailOffset)
      : InMemoryReadFile(data), tailOffset_(tailOffset) {}

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    if (offset >= tailOffset_) {
      std::unique_lock lock(mutex_);
      entered_ = true;
      cv_.notify_all();
      cv_.wait(lock, [&] { return released_; });
      if (fail_) {
        ++failures_;
        cv_.notify_all();
        VELOX_FAIL("Injected tail read failure");
      }
    }
    return InMemoryReadFile::preadv(offset, buffers, context);
  }

  bool waitForTail() const {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return entered_; });
  }

  bool waitForFailures(size_t count) const {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return failures_ >= count; });
  }

  void release(bool fail = false) {
    std::lock_guard lock(mutex_);
    fail_ = fail;
    released_ = true;
    cv_.notify_all();
  }

 private:
  const uint64_t tailOffset_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  mutable bool entered_{false};
  mutable size_t failures_{0};
  bool released_{false};
  bool fail_{false};
};

struct PendingReadState {
  PendingReadState() : releaseFuture(release.get_future().share()) {}

  std::atomic<bool> destroyed{false};
  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> releaseFuture;
};

class PendingDeviceDataSource final : public cudf::io::datasource {
 public:
  explicit PendingDeviceDataSource(std::shared_ptr<PendingReadState> state)
      : state_(std::move(state)) {}

  ~PendingDeviceDataSource() override {
    state_->destroyed = true;
  }

  std::unique_ptr<datasource::buffer> host_read(
      size_t /*offset*/,
      size_t /*size*/) override {
    return datasource::buffer::create(std::vector<uint8_t>{});
  }

  size_t host_read(size_t /*offset*/, size_t /*size*/, uint8_t* /*dst*/)
      override {
    return 0;
  }

  bool supports_device_read() const override {
    return true;
  }

  std::future<size_t> device_read_async(
      size_t /*offset*/,
      size_t size,
      uint8_t* /*dst*/,
      rmm::cuda_stream_view /*stream*/) override {
    return std::async(std::launch::async, [state = state_, size] {
      state->entered.set_value();
      state->releaseFuture.wait();
      return size;
    });
  }

  size_t size() const override {
    return 16;
  }

 private:
  std::shared_ptr<PendingReadState> state_;
};

class CudfSplitReaderHelpersTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::Options options;
    options.trackDefaultUsage = true;
    memory::MemoryManager::testingSetInstance(options);
  }

  void TearDown() override {
    PinnedStagingArena::setAllocationFailureForTesting(false);
    CacheHostRegistration::configure(false, 0);
    EXPECT_EQ(CacheHostRegistration::testingReservedBytes(), 0);
    CacheHostRegistration::setFailureAtForTesting(0);
  }

  std::shared_ptr<ExecutorBufferedInput> makeInput(
      const std::string& data,
      folly::Executor* executor) {
    return std::make_shared<ExecutorBufferedInput>(
        std::make_shared<InMemoryReadFile>(data), *pool_, executor);
  }

  std::shared_ptr<DirectBufferedInput> makeDirectInput(
      const std::string& data,
      folly::Executor* executor = nullptr) {
    io::ReaderOptions options(pool_.get());
    options.setLoadQuantum(64 << 10);
    return std::make_shared<DirectBufferedInput>(
        std::make_shared<InMemoryReadFile>(data),
        dwio::common::MetricsLog::voidLog(),
        StringIdLease(fileIds(), "direct-buffered-file"),
        nullptr,
        StringIdLease(fileIds(), "direct-buffered-group"),
        std::make_shared<dwio::common::IoStatistics>(),
        nullptr,
        executor,
        options);
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

class RegisteredBufferedInputTest : public CudfSplitReaderHelpersTest {
 protected:
  void SetUp() override {
    CacheHostRegistration::configure(true, 32ULL << 20);
    CacheHostRegistration::setBlockSizesForTesting(1 << 20, 2 << 20);
    PinnedStagingArena::configure(true, 64 << 10, 1, 1);
    data_.resize(2 << 20);
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<char>(i % 251);
    }
  }
  void TearDown() override {
    cache_->shutdown();
    CacheHostRegistration::setBlockSizesForTesting(64ULL << 20, 256ULL << 20);
    CudfSplitReaderHelpersTest::TearDown();
  }
  std::shared_ptr<CachedBufferedInput> makeCachedInput(
      std::shared_ptr<ReadFile> readFile = nullptr,
      folly::Executor* executor = nullptr) {
    io::ReaderOptions options(pool_.get());
    options.setLoadQuantum(64 << 10);
    options.setCacheable(true);
    if (readFile) {
      // Keep the gated test's two regions in separate coalesced loads.
      options.setMaxCoalesceDistance(0);
      options.setMaxCoalesceBytes(64 << 10);
    } else {
      readFile = std::make_shared<InMemoryReadFile>(data_);
    }
    return std::make_shared<CachedBufferedInput>(
        std::move(readFile),
        dwio::common::MetricsLog::voidLog(),
        StringIdLease(fileIds(), "registered-buffered-file"),
        cache_.get(),
        nullptr,
        StringIdLease(fileIds(), "registered-buffered-group"),
        std::make_shared<io::IoStatistics>(),
        stats_,
        executor,
        options);
  }

  int64_t readyBatches() const {
    const auto metrics = stats_->stats();
    const auto it = metrics.find("cudfBufferedReadyH2DBatches");
    return it == metrics.end() ? 0 : it->second.sum;
  }

  bool waitForReadyBatch(int64_t previous) const {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (readyBatches() <= previous &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return readyBatches() > previous;
  }

  void checkCopyBeforeTailCompletes(
      bool cachedPrefix,
      bool singleRequest = false) {
    constexpr size_t kSize = 64 << 10;
    const size_t tailOffset = singleRequest ? kSize : 1 << 20;
    auto file = std::make_shared<GatedTailReadFile>(data_, tailOffset);
    folly::CPUThreadPoolExecutor executor(2);
    BufferedInputDataSource source(makeCachedInput(file, &executor), stats_);
    TestCudaStream stream;
    if (cachedPrefix) {
      std::ignore = source.device_read(0, kSize, stream.view());
    }
    const auto previous = readyBatches();
    rmm::device_buffer destination(
        2 * kSize, stream.view(), cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    CUDF_CUDA_TRY(cudaMemsetAsync(dst, 0xff, 2 * kSize, stream.view().value()));
    const std::vector<cudf::io::datasource::device_read_request> requests =
        singleRequest
        ? std::vector<
              cudf::io::datasource::device_read_request>{{0, 2 * kSize, dst}}
        : std::vector<cudf::io::datasource::device_read_request>{
              {0, kSize, dst}, {tailOffset, kSize, dst + kSize}};
    auto completion = source.device_read_batch_async(requests, stream.view());
    auto releaseTail = folly::makeGuard([&] { file->release(); });
    const bool tailStarted = file->waitForTail();
    const bool prefixSubmitted = waitForReadyBatch(previous);
    if (tailStarted && prefixSubmitted) {
      // The tail cannot submit a copy while its read is gated. Synchronizing
      // the stream here proves the prefix really reached GPU memory, not
      // merely that the final completion future was constructed early.
      stream.view().synchronize();
      std::string actual(kSize, '\0');
      CUDF_CUDA_TRY(
          cudaMemcpy(actual.data(), dst, kSize, cudaMemcpyDeviceToHost));
      EXPECT_EQ(actual, data_.substr(0, kSize));
    }
    file->release();
    EXPECT_EQ(
        completion.get(),
        singleRequest ? std::vector<size_t>{2 * kSize}
                      : (std::vector<size_t>{kSize, kSize}));
    EXPECT_TRUE(tailStarted);
    EXPECT_TRUE(prefixSubmitted);
    std::string actual(2 * kSize, '\0');
    CUDF_CUDA_TRY(
        cudaMemcpy(actual.data(), dst, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, data_.substr(0, kSize) + data_.substr(tailOffset, kSize));
    const auto metrics = stats_->stats();
    EXPECT_EQ(
        metrics.at("cudfBufferedH2DBytesBeforeLastSourceReady").sum, kSize);
    EXPECT_GT(metrics.at("cudfBufferedReadyH2DLeadNanos").sum, 0);
    EXPECT_EQ(metrics.count("cudfPinnedStagingBytes"), 0);
  }
  std::string data_;
  std::shared_ptr<memory::MallocAllocator> allocator_ =
      std::make_shared<memory::MallocAllocator>(
          memory::MemoryAllocator::Options{
              .capacity = 16 << 20,
              .reservationByteLimit = 0});
  std::shared_ptr<AsyncDataCache> cache_ =
      AsyncDataCache::create(allocator_.get());
  std::shared_ptr<IoStats> stats_ = std::make_shared<IoStats>();
};

TEST_F(RegisteredBufferedInputTest, coldPrefixCopiesBeforeTailCompletes) {
  checkCopyBeforeTailCompletes(false);
}

TEST_F(RegisteredBufferedInputTest, cachedPrefixCopiesBeforeTailCompletes) {
  checkCopyBeforeTailCompletes(true);
}

TEST_F(RegisteredBufferedInputTest, copiesBeforeNextEntryInSameRequest) {
  checkCopyBeforeTailCompletes(false, true);
}

TEST_F(RegisteredBufferedInputTest, laterReadFailureDrainsEarlierH2D) {
  constexpr size_t kSize = 64 << 10;
  constexpr size_t kTailOffset = 1 << 20;
  auto file = std::make_shared<GatedTailReadFile>(data_, kTailOffset);
  folly::CPUThreadPoolExecutor executor(2);
  auto source = std::make_unique<BufferedInputDataSource>(
      makeCachedInput(file, &executor), stats_);
  TestCudaStream stream;
  rmm::device_buffer destination(
      2 * kSize, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));
  const std::vector<cudf::io::datasource::device_read_request> requests{
      {0, kSize, dst}, {kTailOffset, kSize, dst + kSize}};
  auto completion = source->device_read_batch_async(requests, stream.view());
  source.reset();
  auto result = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        return completion.get();
      });
  auto releaseReads = folly::makeGuard([&] {
    file->release();
    releaseGate(gate);
  });
  const bool streamBlocked = entered.wait_for(5s) == std::future_status::ready;
  const bool tailStarted = file->waitForTail();
  const bool prefixSubmitted = waitForReadyBatch(0);
  file->release(true);
  // Coalesced failure is retried once by CacheInputStream's demand read.
  const bool tailFailed = file->waitForFailures(2);
  if (streamBlocked && prefixSubmitted && tailFailed) {
    EXPECT_EQ(result.wait_for(100ms), std::future_status::timeout);
    cache_->clear();
    EXPECT_GT(cache_->refreshStats().numEntries, 0);
  }
  releaseGate(gate);
  EXPECT_THROW(result.get(), VeloxRuntimeError);
  EXPECT_TRUE(streamBlocked);
  EXPECT_TRUE(tailStarted);
  EXPECT_TRUE(prefixSubmitted);
  EXPECT_TRUE(tailFailed);
  std::string actual(kSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(actual.data(), dst, kSize, cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, data_.substr(0, kSize));
  cache_->clear();
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
  EXPECT_EQ(CacheHostRegistration::poolStats().usedBytes, 0);
}

TEST_F(RegisteredBufferedInputTest, coldAndHotBatchesAvoidStaging) {
  BufferedInputDataSource source(makeCachedInput(), stats_);
  TestCudaStream stream;
  rmm::device_buffer destination(
      data_.size(), stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  const std::vector<cudf::io::datasource::device_read_request> requests{
      {19, (1 << 20) - 19, dst}, {1 << 20, 1 << 20, dst + (1 << 20) - 19}};
  for (int iteration = 0; iteration < 2; ++iteration) {
    const auto sizes =
        source.device_read_batch_async(requests, stream.view()).get();
    EXPECT_EQ(sizes, (std::vector<size_t>{(1 << 20) - 19, 1 << 20}));
    EXPECT_GT(CacheHostRegistration::testingReservedBytes(), 0);
    std::string actual(data_.size() - 19, '\0');
    CUDF_CUDA_TRY(
        cudaMemcpy(actual.data(), dst, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, data_.substr(19));
  }
  const auto metrics = stats_->stats();
  EXPECT_EQ(
      metrics.at("cudfCacheRegisteredH2DBytes").sum, 2 * (data_.size() - 19));
  EXPECT_EQ(metrics.count("cudfPinnedStagingBytes"), 0);
  EXPECT_EQ(metrics.count("cudfCacheHostRegistrationFallbackBytes"), 0);
  EXPECT_GT(metrics.at("cudfCacheHostRegistrationReusedBytes").sum, 0);
  EXPECT_EQ(metrics.count("cudfCacheHostUnregisteredBytes"), 0);
  EXPECT_GT(cache_->refreshStats().numEntries, 0);
}

TEST_F(RegisteredBufferedInputTest, budgetFailureKeepsStagingFallback) {
  CacheHostRegistration::configure(true, 1);
  BufferedInputDataSource source(makeCachedInput(), stats_);
  TestCudaStream stream;
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto result = source.device_read(0, data_.size(), stream.view());
    std::string actual(result->size(), '\0');
    CUDF_CUDA_TRY(cudaMemcpy(
        actual.data(), result->data(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, data_);
    EXPECT_EQ(CacheHostRegistration::testingReservedBytes(), 0);
  }
  const auto metrics = stats_->stats();
  EXPECT_EQ(metrics.at("cudfPinnedStagingBytes").sum, 2 * data_.size());
  EXPECT_EQ(
      metrics.at("cudfCacheHostRegistrationFallbackBytes").sum,
      2 * data_.size());
}

TEST_F(
    RegisteredBufferedInputTest,
    oneUnregisteredEntryDoesNotStageEntireBatch) {
  CacheHostRegistration::setFailureAtForTesting(1);
  BufferedInputDataSource source(makeCachedInput(), stats_);
  TestCudaStream stream;
  // The first 64-KiB entry uses pageable memory after the injected failure;
  // subsequent allocations use the registered pool. Test both cold and hot.
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto result = source.device_read(0, data_.size(), stream.view());
    std::string actual(result->size(), '\0');
    CUDF_CUDA_TRY(cudaMemcpy(
        actual.data(), result->data(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, data_);
  }
  constexpr size_t kPageableBytes = 64 << 10;
  const auto metrics = stats_->stats();
  EXPECT_EQ(
      metrics.at("cudfCacheRegisteredH2DBytes").sum,
      2 * (data_.size() - kPageableBytes));
  EXPECT_EQ(
      metrics.at("cudfCacheHostRegistrationFallbackBytes").sum,
      2 * kPageableBytes);
  EXPECT_EQ(metrics.at("cudfDirectHostToDeviceBytes").sum, 2 * kPageableBytes);
  EXPECT_EQ(metrics.at("cudfCacheBackedSourceBytes").sum, 2 * data_.size());
  EXPECT_EQ(metrics.count("cudfPinnedStagingBytes"), 0);
}

TEST_F(RegisteredBufferedInputTest, smallerReadsReuseCompleteOwner) {
  BufferedInputDataSource source(makeCachedInput(), stats_);
  TestCudaStream stream;
  auto first = source.device_read(0, data_.size(), stream.view());
  const auto calls = stats_->stats().at("cudfCacheHostRegisterCalls").sum;
  CacheHostRegistration::setFailureAtForTesting(1);
  auto slice = source.device_read(0, 4096, stream.view());
  std::string actual(4096, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(), slice->data(), actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, data_.substr(0, 4096));
  EXPECT_EQ(stats_->stats().at("cudfCacheHostRegisterCalls").sum, calls);
  EXPECT_GT(stats_->stats().at("cudfCacheHostRegistrationReusedBytes").sum, 0);
  EXPECT_EQ(stats_->stats().count("cudfCacheHostRegistrationFallbackBytes"), 0);
}

TEST_F(RegisteredBufferedInputTest, uncachedInputDoesNotAttemptRegistration) {
  BufferedInputDataSource source(makeDirectInput(data_), stats_);
  TestCudaStream stream;
  auto result = source.device_read(0, data_.size(), stream.view());
  EXPECT_EQ(result->size(), data_.size());
  EXPECT_EQ(stats_->stats().count("cudfCacheHostRegistrationAttempts"), 0);
  EXPECT_EQ(stats_->stats().at("cudfPinnedStagingBytes").sum, data_.size());
}

TEST_F(
    RegisteredBufferedInputTest,
    discardRetainsRegistrationAndCachePinsUntilH2D) {
  constexpr size_t kSize = 64 << 10;
  auto input = makeCachedInput();
  auto source = std::make_unique<BufferedInputDataSource>(input, stats_);
  TestCudaStream stream;
  auto initial = source->device_read(0, kSize, stream.view());
  StringIdLease file(fileIds(), "registered-buffered-file");
  auto pin = cache_->findOrCreate({file.id(), 0}, kSize, false);
  auto registration = CacheHostRegistration::tryAcquire(
      std::span<const cache::CachePin>(&pin, 1));
  ASSERT_TRUE(registration);
  rmm::device_buffer destination(
      kSize, stream.view(), cudf::get_current_device_resource_ref());
  StreamGate gate;
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));
  auto future = source->device_read_async(
      0, kSize, static_cast<uint8_t*>(destination.data()), stream.view());
  source.reset();
  input.reset();
  auto completion = std::async(
      std::launch::async,
      [future = std::move(future)]() mutable { future = {}; });
  const auto shared = [&] {
    const auto metrics = stats_->stats();
    const auto found = metrics.find("cudfCacheHostRegistrationSharedBytes");
    return found != metrics.end() && found->second.sum > 0;
  };
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!shared() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool hasShared = shared();
  if (hasShared) {
    registration.reset();
    pin.clear();
    cache_->clear();
    EXPECT_GT(cache_->refreshStats().numEntries, 0);
    EXPECT_GT(CacheHostRegistration::testingReservedBytes(), 0);
    EXPECT_EQ(completion.wait_for(0s), std::future_status::timeout);
  }
  releaseGate(gate);
  completion.get();
  EXPECT_TRUE(hasShared);
  registration.reset();
  pin.clear();
  EXPECT_GT(CacheHostRegistration::testingReservedBytes(), 0);
  std::string actual(kSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(), destination.data(), kSize, cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, data_.substr(0, kSize));
  cache_->clear();
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
  EXPECT_EQ(CacheHostRegistration::poolStats().usedBytes, 0);
}

TEST_F(CudfSplitReaderHelpersTest, knownKvikioFileSizeUsesObjectMetadata) {
  auto split = CudfHiveConnectorSplitBuilder("s3://bucket/file.parquet")
                   .start(40)
                   .length(50)
                   .infoColumn("$file_size", "200")
                   .build();
  EXPECT_EQ(knownKvikioFileSize(*split), 200);

  // Large objects and the unspecified-length sentinel are supported.
  split = CudfHiveConnectorSplitBuilder("s3://bucket/file.parquet")
              .infoColumn("$file_size", "5368709120")
              .build();
  EXPECT_EQ(knownKvikioFileSize(*split), 5368709120ULL);

  split = CudfHiveConnectorSplitBuilder("s3://bucket/empty.parquet")
              .length(0)
              .infoColumn("$file_size", "0")
              .build();
  EXPECT_EQ(knownKvikioFileSize(*split), 0);
}

TEST_F(CudfSplitReaderHelpersTest, knownKvikioFileSizeNeverInfersFromRange) {
  auto split = CudfHiveConnectorSplitBuilder("s3://bucket/file.parquet")
                   .length(200)
                   .build();
  EXPECT_FALSE(knownKvikioFileSize(*split).has_value());

  for (const auto* value :
       {"",
        "-1",
        "+200",
        " 200",
        "200 ",
        "200x",
        "9223372036854775808",
        "18446744073709551616"}) {
    SCOPED_TRACE(value);
    split->infoColumns["$file_size"] = value;
    EXPECT_FALSE(knownKvikioFileSize(*split).has_value());
  }
}

TEST_F(
    CudfSplitReaderHelpersTest,
    knownKvikioFileSizeRejectsInconsistentRange) {
  const auto check = [](uint64_t start, uint64_t length) {
    auto split = CudfHiveConnectorSplitBuilder("s3://bucket/file.parquet")
                     .start(start)
                     .length(length)
                     .infoColumn("$file_size", "200")
                     .build();
    return knownKvikioFileSize(*split);
  };
  EXPECT_FALSE(check(201, 0).has_value());
  EXPECT_FALSE(check(150, 51).has_value());
  // Validate without overflowing start + length.
  EXPECT_FALSE(
      check(150, std::numeric_limits<uint64_t>::max() - 1).has_value());
  EXPECT_EQ(check(150, 50), 200);
  EXPECT_EQ(check(150, std::numeric_limits<uint64_t>::max()), 200);
}

TEST_F(CudfSplitReaderHelpersTest, normalizeKvikioS3Uri) {
  EXPECT_EQ(normalizeKvikioUri("s3://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("s3a://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("s3n://bucket/key"), "s3://bucket/key");
  EXPECT_EQ(normalizeKvikioUri("file:/tmp/input"), "file:/tmp/input");
  EXPECT_EQ(
      normalizeKvikioUri("https://example.test/a"), "https://example.test/a");
}

TEST_F(CudfSplitReaderHelpersTest, bufferedInputPrefersDeviceReads) {
  folly::CPUThreadPoolExecutor executor(1);
  BufferedInputDataSource dataSource(makeInput("buffered-input", &executor));

  EXPECT_TRUE(dataSource.supports_device_read());
  EXPECT_TRUE(dataSource.is_device_read_preferred(1));
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadBatchPreservesRequestsAndOrder) {
  std::string inputData;
  for (int index = 0; index < 64; ++index) {
    inputData.push_back(static_cast<char>('A' + index % 26));
  }
  folly::CPUThreadPoolExecutor executor(1);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(makeInput(inputData, &executor), ioStats);

  TestCudaStream stream;
  constexpr size_t kSingleByteReads = 34;
  rmm::device_buffer destination(
      kSingleByteReads + 2,
      stream.view(),
      cudf::get_current_device_resource_ref());
  auto* destinationData = static_cast<uint8_t*>(destination.data());

  std::vector<cudf::io::datasource::device_read_request> requests;
  requests.reserve(kSingleByteReads + 2);
  for (size_t index = 0; index < kSingleByteReads; ++index) {
    requests.push_back({index, 1, destinationData + index});
  }
  requests.push_back({9, 0, nullptr});
  requests.push_back(
      {inputData.size() - 2, 17, destinationData + kSingleByteReads});

  auto completion = dataSource.device_read_batch_async(
      cudf::host_span<cudf::io::datasource::device_read_request const>{
          requests.data(), requests.size()},
      stream.view());
  // The datasource contract requires an immediate descriptor copy.
  std::fill(
      requests.begin(),
      requests.end(),
      cudf::io::datasource::device_read_request{0, 0, nullptr});

  std::vector<size_t> expectedResults(kSingleByteReads, 1);
  expectedResults.push_back(0);
  expectedResults.push_back(2);
  EXPECT_EQ(completion.get(), expectedResults);

  std::string actual(kSingleByteReads + 2, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual.substr(0, kSingleByteReads), inputData.substr(0, 34));
  EXPECT_EQ(
      actual.substr(kSingleByteReads), inputData.substr(inputData.size() - 2));
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadBatches").sum, 1);
  EXPECT_EQ(
      metrics.at("cudfBufferedDeviceReadRequests").sum, kSingleByteReads + 2);
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadBytes").sum, actual.size());
  EXPECT_EQ(
      metrics.at("cudfBufferedDeviceReadFragments").sum, kSingleByteReads + 1);
  EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, actual.size());
  EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
  EXPECT_EQ(metrics.at("cudfPinnedStagingSmallReadBypasses").sum, 1);
  EXPECT_EQ(metrics.at("cudfDirectHostToDeviceBytes").sum, actual.size());
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadBatchRetainsCachedRuns) {
  PinnedStagingArena::configure(false, 0, 0, 0);
  constexpr size_t kLoadQuantum = 64 << 10;
  std::string inputData(kLoadQuantum * 2 + 113, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }

  auto allocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 16 << 20, .reservationByteLimit = 0});
  auto cache = AsyncDataCache::create(allocator.get());
  auto tracker = std::make_shared<ScanTracker>(
      "cudfBatchRetainedRuns", nullptr, kLoadQuantum);
  auto ioStatistics = std::make_shared<facebook::velox::io::IoStatistics>();
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  folly::CPUThreadPoolExecutor executor(2);
  facebook::velox::io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(kLoadQuantum);
  readerOptions.setCacheable(true);

  auto& ids = facebook::velox::fileIds();
  auto input = std::make_shared<CachedBufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData),
      facebook::velox::dwio::common::MetricsLog::voidLog(),
      StringIdLease(ids, "cudfBatchRetainedRunsFile"),
      cache.get(),
      tracker,
      StringIdLease(ids, "cudfBatchRetainedRunsGroup"),
      ioStatistics,
      ioStats,
      &executor,
      readerOptions);
  auto dataSource = std::make_shared<BufferedInputDataSource>(input, ioStats);

  constexpr size_t kFirstOffset = kLoadQuantum - 29;
  constexpr size_t kFirstSize = 83;
  constexpr size_t kSecondOffset = kLoadQuantum * 2 - 17;
  constexpr size_t kSecondSize = 71;
  TestCudaStream stream;
  rmm::device_buffer destination(
      kFirstSize + kSecondSize,
      stream.view(),
      cudf::get_current_device_resource_ref());
  auto* destinationData = static_cast<uint8_t*>(destination.data());
  const std::vector<cudf::io::datasource::device_read_request> requests{
      {kFirstOffset, kFirstSize, destinationData},
      {kSecondOffset, kSecondSize, destinationData + kFirstSize}};

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));
  auto completion = dataSource->device_read_batch_async(
      cudf::host_span<cudf::io::datasource::device_read_request const>{
          requests.data(), requests.size()},
      stream.view());
  // The asynchronous operation owns the BufferedInput and all pins needed by
  // the transfer; the datasource need not remain alive after scheduling.
  dataSource.reset();
  input.reset();
  auto getResult = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        return completion.get();
      });

  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    std::ignore = getResult.get();
    FAIL() << "CUDA stream callback did not start";
  }
  const auto cacheDeadline = std::chrono::steady_clock::now() + 10s;
  while (cache->refreshStats().numEntries == 0 &&
         std::chrono::steady_clock::now() < cacheDeadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (cache->refreshStats().numEntries == 0) {
    releaseGate(gate);
    std::ignore = getResult.get();
    FAIL() << "Cached device read did not populate the cache";
  }

  // executeDeviceReadBatch releases its CacheInputStreams before submitting
  // H2D. Clearing the cache while the stream is gated therefore verifies that
  // the transfer plan's independent pins retain the source fragments.
  cache->clear();
  EXPECT_GT(cache->refreshStats().numEntries, 0);
  EXPECT_EQ(getResult.wait_for(0s), std::future_status::timeout);
  releaseGate(gate);
  EXPECT_EQ(getResult.get(), (std::vector<size_t>{kFirstSize, kSecondSize}));

  std::string actual(kFirstSize + kSecondSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      actual.substr(0, kFirstSize), inputData.substr(kFirstOffset, kFirstSize));
  EXPECT_EQ(
      actual.substr(kFirstSize), inputData.substr(kSecondOffset, kSecondSize));
  cache->clear();
  EXPECT_EQ(cache->refreshStats().numEntries, 0);
  cache->shutdown();
}

TEST_F(
    CudfSplitReaderHelpersTest,
    retainedUncachedReadSurvivesReloadBeforePacking) {
  for (const bool direct : {false, true}) {
    SCOPED_TRACE(direct);
    constexpr size_t kReadSize = (1 << 20) + 31;
    PinnedStagingArena::configure(true, 256 << 10, 2, 1);
    auto arenaBlocker = PinnedStagingArena::acquirePair();
    ASSERT_TRUE(arenaBlocker.has_value());
    const std::string inputData =
        std::string(kReadSize, 'a') + std::string(kReadSize, 'b');
    std::shared_ptr<BufferedInput> input = direct
        ? makeDirectInput(inputData)
        : std::make_shared<BufferedInput>(
              std::make_shared<InMemoryReadFile>(inputData), *pool_);
    auto ioStats = std::make_shared<facebook::velox::IoStats>();
    BufferedInputDataSource dataSource(input, ioStats);
    TestCudaStream stream;
    rmm::device_buffer destination(
        kReadSize, stream.view(), cudf::get_current_device_resource_ref());
    auto completion = dataSource.device_read_async(
        0, kReadSize, static_cast<uint8_t*>(destination.data()), stream.view());
    auto result = std::async(
        std::launch::async,
        [f = std::move(completion)]() mutable { return f.get(); });
    bool waitingForArena = false;
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline) {
      const auto metrics = ioStats->stats();
      if (metrics.count("cudfPinnedStagingAttempts") != 0) {
        waitingForArena = true;
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    if (!waitingForArena) {
      arenaBlocker->release();
      std::ignore = result.get();
      FAIL() << "Uncached read did not reach the occupied staging arena";
    }

    // Source acquisition is complete and its input lock is released. Reset and
    // reuse the original input while the retained bytes still await packing.
    input->reset();
    input->enqueue({kReadSize, kReadSize});
    input->load(dwio::common::LogType::FILE);
    input->reset();
    EXPECT_GT(pool_->usedBytes(), 0);
    arenaBlocker->release();
    EXPECT_EQ(result.get(), kReadSize);
    std::string actual(kReadSize, '\0');
    CUDF_CUDA_TRY(cudaMemcpy(
        actual.data(),
        destination.data(),
        actual.size(),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, inputData.substr(0, kReadSize));
    EXPECT_EQ(pool_->usedBytes(), 0);
    const auto metrics = ioStats->stats();
    EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, kReadSize);
    EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
  }
}

TEST_F(CudfSplitReaderHelpersTest, directBufferedReadsRetainIoBuffers) {
  // Hive selects DirectBufferedInput for cache-off Parquet. Exercise that
  // path, not only BufferedInput's base implementation, including fallback
  // H2D without staging, preloaded files, duplicate requests and load rollover.
  constexpr size_t kReadSize = (1 << 20) + 37;
  std::string inputData(kReadSize + 128, '\0');
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<char>(i % 251);
  }
  folly::CPUThreadPoolExecutor executor(1);
  for (const bool preload : {false, true}) {
    for (const bool staging : {false, true}) {
      SCOPED_TRACE(fmt::format("preload={} staging={}", preload, staging));
      PinnedStagingArena::configure(staging, 256 << 10, 2, 1);
      auto input = makeDirectInput(inputData, &executor);
      if (preload) {
        input->preload();
      }
      auto ioStats = std::make_shared<facebook::velox::IoStats>();
      BufferedInputDataSource dataSource(input, ioStats);
      TestCudaStream stream;
      constexpr size_t kTotalSize = 2 * kReadSize + 13;
      rmm::device_buffer destination(
          kTotalSize, stream.view(), cudf::get_current_device_resource_ref());
      auto* dst = static_cast<uint8_t*>(destination.data());
      const std::vector<cudf::io::datasource::device_read_request> requests{
          {71, kReadSize, dst},
          {0, 13, dst + kReadSize},
          {71, kReadSize, dst + kReadSize + 13}};
      EXPECT_EQ(
          dataSource.device_read_batch_async(requests, stream.view()).get(),
          (std::vector<size_t>{kReadSize, 13, kReadSize}));
      std::string actual(kTotalSize, '\0');
      CUDF_CUDA_TRY(cudaMemcpy(
          actual.data(),
          destination.data(),
          actual.size(),
          cudaMemcpyDeviceToHost));
      EXPECT_EQ(
          actual,
          inputData.substr(71, kReadSize) + inputData.substr(0, 13) +
              inputData.substr(71, kReadSize));
      const auto metrics = ioStats->stats();
      const auto chunksPerRead =
          (kReadSize + input->loadQuantum() - 1) / input->loadQuantum();
      EXPECT_EQ(
          metrics.at("cudfBufferedDeviceReadPlannedRanges").sum,
          preload ? 3 : 2 * chunksPerRead + 1);
      EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, kTotalSize);
      EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
      EXPECT_EQ(metrics.count("cudfCacheBackedSourceBytes"), 0);
    }
  }
}

TEST_F(CudfSplitReaderHelpersTest, directBufferedReadsPrefetchEveryQuantum) {
  constexpr size_t kQuantum = 64 << 10;
  constexpr size_t kReadSize = 3 * kQuantum + 17;
  constexpr size_t kOffset = 31;
  std::string inputData(kOffset + kReadSize, '\0');
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<char>(i % 251);
  }
  auto readFile = std::make_shared<GatedReadFile>(inputData);
  folly::CPUThreadPoolExecutor executor(4);
  io::ReaderOptions options(pool_.get());
  options.setLoadQuantum(kQuantum);
  // Keep the four pieces in separate storage calls for the concurrency check.
  options.setMaxCoalesceBytes(1);
  auto input = std::make_shared<DirectBufferedInput>(
      readFile,
      dwio::common::MetricsLog::voidLog(),
      StringIdLease(fileIds(), "direct-buffered-prefetch-file"),
      nullptr,
      StringIdLease(fileIds(), "direct-buffered-prefetch-group"),
      std::make_shared<dwio::common::IoStatistics>(),
      nullptr,
      &executor,
      options);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());
  auto future = dataSource.device_read_async(
      kOffset,
      kReadSize,
      static_cast<uint8_t*>(destination.data()),
      stream.view());

  // All four chunks must begin before any storage read is allowed to finish.
  // Release the gate before asserting so the old serialized path fails rather
  // than hanging while its completion future is destroyed.
  const bool allReadsStarted = readFile->waitForReads(4);
  readFile->release();
  EXPECT_TRUE(allReadsStarted);
  EXPECT_EQ(future.get(), kReadSize);
  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData.substr(kOffset, kReadSize));
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadRequests").sum, 1);
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadPlannedRanges").sum, 4);
  EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
}

TEST_F(
    CudfSplitReaderHelpersTest,
    directBufferedBatchPreservesRangeBoundaries) {
  constexpr size_t kQuantum = 64 << 10;
  const auto maxSize = std::numeric_limits<size_t>::max();
  std::string inputData(3 * kQuantum + 31, '\0');
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<char>(i % 251);
  }
  // Out-of-order and overlapping ranges, with a request clamped at EOF.
  const std::string expected = inputData.substr(kQuantum + 7, kQuantum + 9) +
      inputData.substr(37, 2 * kQuantum) + inputData.substr(3 * kQuantum - 5);
  folly::CPUThreadPoolExecutor executor(1);
  for (const bool async : {false, true}) {
    SCOPED_TRACE(fmt::format("async={}", async));
    auto input = makeDirectInput(inputData, async ? &executor : nullptr);
    auto ioStats = std::make_shared<facebook::velox::IoStats>();
    BufferedInputDataSource dataSource(input, ioStats);
    TestCudaStream stream;
    rmm::device_buffer destination(
        expected.size(),
        stream.view(),
        cudf::get_current_device_resource_ref());
    auto* dst = static_cast<uint8_t*>(destination.data());
    const std::vector<cudf::io::datasource::device_read_request> requests{
        {kQuantum + 7, kQuantum + 9, dst},
        {0, 0, nullptr},
        {37, 2 * kQuantum, dst + kQuantum + 9},
        {3 * kQuantum - 5, maxSize, dst + 3 * kQuantum + 9},
        {maxSize, maxSize, dst}};
    EXPECT_EQ(
        dataSource.device_read_batch_async(requests, stream.view()).get(),
        (std::vector<size_t>{kQuantum + 9, 0, 2 * kQuantum, 36, 0}));
    std::string actual(expected.size(), '\0');
    CUDF_CUDA_TRY(cudaMemcpy(
        actual.data(),
        destination.data(),
        actual.size(),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
    const auto metrics = ioStats->stats();
    EXPECT_EQ(metrics.at("cudfBufferedDeviceReadRequests").sum, 5);
    EXPECT_EQ(metrics.at("cudfBufferedDeviceReadPlannedRanges").sum, 5);
    EXPECT_EQ(
        metrics.at("cudfBufferedRetainedSourceBytes").sum, expected.size());
    EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
  }
}

TEST_F(CudfSplitReaderHelpersTest, uncachedBatchMixesLoadedAndNewRanges) {
  const std::string inputData = std::string(256, 'a') + std::string(256, 'b');
  auto input = std::make_shared<BufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData), *pool_);
  input->enqueue({0, 128});
  input->load(dwio::common::LogType::FILE);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      256, stream.view(), cudf::get_current_device_resource_ref());
  auto* dst = static_cast<uint8_t*>(destination.data());
  const std::vector<cudf::io::datasource::device_read_request> requests{
      {0, 128, dst}, {256, 128, dst + 128}};
  EXPECT_EQ(
      dataSource.device_read_batch_async(requests, stream.view()).get(),
      (std::vector<size_t>{128, 128}));
  std::string actual(256, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, std::string(128, 'a') + std::string(128, 'b'));
  EXPECT_EQ(ioStats->stats().at("cudfBufferedRetainedSourceBytes").sum, 256);
  EXPECT_EQ(ioStats->stats().count("cudfCopiedSourceBytes"), 0);
}

TEST_F(CudfSplitReaderHelpersTest, streamOwnedInputKeepsCopyingFallback) {
  constexpr size_t kReadSize = (1 << 20) + 19;
  PinnedStagingArena::configure(true, 256 << 10, 2, 1);
  const std::string inputData(kReadSize, 's');
  auto input = std::make_shared<StreamOnlyBufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData), *pool_);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());
  EXPECT_EQ(
      dataSource
          .device_read_async(
              0,
              kReadSize,
              static_cast<uint8_t*>(destination.data()),
              stream.view())
          .get(),
      kReadSize);
  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfCopiedSourceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfBufferedRetainedSourceBytes"), 0);
}

TEST_F(
    CudfSplitReaderHelpersTest,
    stagedDeviceReadPinsSourcesBeforeWaitingForArena) {
  constexpr size_t kLoadQuantum = 64 << 10;
  constexpr size_t kWindowBytes = 1 << 20;
  constexpr size_t kReadOffset = 43;
  constexpr size_t kReadSize = kWindowBytes + kLoadQuantum;
  PinnedStagingArena::configure(true, kWindowBytes, 2, 1);
  auto arenaBlocker = PinnedStagingArena::acquirePair();
  ASSERT_TRUE(arenaBlocker.has_value());

  std::string inputData(kReadOffset + kReadSize, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }

  auto allocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 32 << 20, .reservationByteLimit = 0});
  auto cache = AsyncDataCache::create(allocator.get());
  auto tracker = std::make_shared<ScanTracker>(
      "cudfStagedReadPinsBeforeArena", nullptr, kLoadQuantum);
  auto ioStatistics = std::make_shared<facebook::velox::io::IoStatistics>();
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  folly::CPUThreadPoolExecutor executor(2);
  facebook::velox::io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(kLoadQuantum);
  readerOptions.setCacheable(true);

  auto& ids = facebook::velox::fileIds();
  auto input = std::make_shared<CachedBufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData),
      facebook::velox::dwio::common::MetricsLog::voidLog(),
      StringIdLease(ids, "cudfStagedReadPinsBeforeArenaFile"),
      cache.get(),
      tracker,
      StringIdLease(ids, "cudfStagedReadPinsBeforeArenaGroup"),
      ioStatistics,
      ioStats,
      &executor,
      readerOptions);
  BufferedInputDataSource dataSource(input, ioStats);

  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());
  auto completion = dataSource.device_read_async(
      kReadOffset,
      kReadSize,
      static_cast<uint8_t*>(destination.data()),
      stream.view());
  auto getResult = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        return completion.get();
      });

  bool waitingForArena = false;
  const auto waitingDeadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < waitingDeadline) {
    const auto metrics = ioStats->stats();
    const auto attempt = metrics.find("cudfPinnedStagingAttempts");
    if (attempt != metrics.end() && attempt->second.sum == 1) {
      waitingForArena = true;
      break;
    }
    std::this_thread::sleep_for(1ms);
  }
  if (!waitingForArena) {
    arenaBlocker->release();
    std::ignore = getResult.get();
    FAIL() << "Device read did not reach the occupied staging arena";
  }

  // Reaching acquirePair means storage loading, Next(), and exact-region
  // retention are complete. Clearing the cache while the arena is occupied
  // must leave those entries pinned, proving no later storage retry can occur
  // under the arena lease.
  cache->clear();
  EXPECT_GT(cache->refreshStats().numEntries, 0);
  EXPECT_EQ(getResult.wait_for(0s), std::future_status::timeout);

  arenaBlocker->release();
  EXPECT_EQ(getResult.get(), kReadSize);
  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData.substr(kReadOffset, kReadSize));

  cache->clear();
  EXPECT_EQ(cache->refreshStats().numEntries, 0);
  const auto metrics = ioStats->stats();
  EXPECT_GT(metrics.at("cudfPinnedStagingAcquireNanos").sum, 0);
  EXPECT_EQ(metrics.at("cudfPinnedStagingContendedAcquisitions").sum, 1);
  const auto& activeLeaseSamples =
      metrics.at("cudfPinnedStagingActiveLeasesAtAcquireSamples");
  EXPECT_EQ(activeLeaseSamples.sum, 1);
  EXPECT_EQ(activeLeaseSamples.count, 1);
  const auto& capacitySamples =
      metrics.at("cudfPinnedStagingWindowSetCapacityAtAcquireSamples");
  EXPECT_EQ(capacitySamples.sum, 1);
  EXPECT_EQ(capacitySamples.count, 1);
  cache->shutdown();
}

TEST_F(
    CudfSplitReaderHelpersTest,
    stagedDeviceReadReleasesCachePinsBeforeH2DCompletes) {
  constexpr size_t kLoadQuantum = 64 << 10;
  constexpr size_t kWindowBytes = 1 << 20;
  constexpr size_t kReadOffset = 37;
  constexpr size_t kReadSize = kWindowBytes + kLoadQuantum;
  PinnedStagingArena::configure(true, kWindowBytes, 2, 1);

  std::string inputData(kReadOffset + kReadSize, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }

  auto allocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 32 << 20, .reservationByteLimit = 0});
  auto cache = AsyncDataCache::create(allocator.get());
  auto tracker = std::make_shared<ScanTracker>(
      "cudfStagedReadReleasesPins", nullptr, kLoadQuantum);
  auto ioStatistics = std::make_shared<facebook::velox::io::IoStatistics>();
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  folly::CPUThreadPoolExecutor executor(2);
  facebook::velox::io::ReaderOptions readerOptions(pool_.get());
  readerOptions.setLoadQuantum(kLoadQuantum);
  readerOptions.setCacheable(true);

  auto& ids = facebook::velox::fileIds();
  auto input = std::make_shared<CachedBufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData),
      facebook::velox::dwio::common::MetricsLog::voidLog(),
      StringIdLease(ids, "cudfStagedReadReleasesPinsFile"),
      cache.get(),
      tracker,
      StringIdLease(ids, "cudfStagedReadReleasesPinsGroup"),
      ioStatistics,
      ioStats,
      &executor,
      readerOptions);
  auto dataSource = std::make_shared<BufferedInputDataSource>(input, ioStats);

  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());
  const std::vector<cudf::io::datasource::device_read_request> requests{{
      kReadOffset,
      kReadSize,
      static_cast<uint8_t*>(destination.data()),
  }};

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));
  auto completion = dataSource->device_read_batch_async(
      cudf::host_span<cudf::io::datasource::device_read_request const>{
          requests.data(), requests.size()},
      stream.view());
  auto getResult = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        return completion.get();
      });

  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    std::ignore = getResult.get();
    FAIL() << "CUDA stream callback did not start";
  }

  const auto populatedDeadline = std::chrono::steady_clock::now() + 10s;
  while (cache->refreshStats().numEntries == 0 &&
         std::chrono::steady_clock::now() < populatedDeadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (cache->refreshStats().numEntries == 0) {
    releaseGate(gate);
    std::ignore = getResult.get();
    FAIL() << "Staged device read did not populate the cache";
  }

  // Both windows can hold this request, so every cache source can be packed
  // even though the stream gate prevents either H2D copy from completing.
  const auto releaseDeadline = std::chrono::steady_clock::now() + 10s;
  do {
    cache->clear();
    if (cache->refreshStats().numEntries == 0) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < releaseDeadline);
  if (cache->refreshStats().numEntries != 0) {
    releaseGate(gate);
    std::ignore = getResult.get();
    FAIL() << "Cache pins remained after their bytes were staged";
  }
  EXPECT_EQ(getResult.wait_for(0s), std::future_status::timeout);

  releaseGate(gate);
  EXPECT_EQ(getResult.get(), (std::vector<size_t>{kReadSize}));
  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData.substr(kReadOffset, kReadSize));
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadBatches").sum, 1);
  EXPECT_EQ(metrics.at("cudfBufferedDeviceReadBytes").sum, kReadSize);
  EXPECT_EQ(metrics.at("cudfCacheBackedSourceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.at("cudfPinnedStagingTransfers").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingBytes").sum, kReadSize);
  EXPECT_EQ(metrics.at("cudfPinnedStagingWindows").sum, 2);
  const auto& activeLeaseSamples =
      metrics.at("cudfPinnedStagingActiveLeasesAtAcquireSamples");
  EXPECT_EQ(activeLeaseSamples.sum, 1);
  EXPECT_EQ(activeLeaseSamples.count, 1);
  const auto& capacitySamples =
      metrics.at("cudfPinnedStagingWindowSetCapacityAtAcquireSamples");
  EXPECT_EQ(capacitySamples.sum, 1);
  EXPECT_EQ(capacitySamples.count, 1);
  EXPECT_EQ(metrics.count("cudfPinnedStagingFallbacks"), 0);
  cache->shutdown();
}

TEST_F(CudfSplitReaderHelpersTest, stagedDeviceReadRollsAcrossBothWindows) {
  constexpr size_t kWindowBytes = 256 << 10;
  constexpr size_t kReadSize = (2 << 20) + 113;
  PinnedStagingArena::configure(true, kWindowBytes, 4, 1);

  std::string inputData(kReadSize, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }
  auto input = std::make_shared<BufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData), *pool_);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());

  auto completion = dataSource.device_read_async(
      0, kReadSize, static_cast<uint8_t*>(destination.data()), stream.view());
  EXPECT_EQ(completion.get(), kReadSize);

  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfPinnedStagingAttempts").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingTransfers").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingBytes").sum, kReadSize);
  EXPECT_EQ(metrics.at("cudfPinnedStagingWindows").sum, 9);
  EXPECT_EQ(metrics.at("cudfPinnedStagingMemcpyBatchAttempts").sum, 9);
  EXPECT_EQ(metrics.at("cudfPinnedStagingMemcpyBatchCopies").sum, 9);
#if CUDART_VERSION >= 13000
  EXPECT_EQ(metrics.at("cudfPinnedStagingNativeMemcpyBatchAttempts").sum, 9);
  EXPECT_EQ(metrics.at("cudfPinnedStagingNativeMemcpyBatchCopies").sum, 9);
#else
  EXPECT_EQ(metrics.count("cudfPinnedStagingNativeMemcpyBatchAttempts"), 0);
  EXPECT_EQ(metrics.count("cudfPinnedStagingNativeMemcpyBatchCopies"), 0);
#endif
  EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
}

TEST_F(CudfSplitReaderHelpersTest, asyncLoadCompletesBeforePinnedStaging) {
  constexpr size_t kReadSize = (1 << 20) + 31;
  PinnedStagingArena::configure(true, 256 << 10, 2, 1);

  std::string inputData(kReadSize, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }
  folly::CPUThreadPoolExecutor executor(1);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(makeInput(inputData, &executor), ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());

  EXPECT_EQ(
      dataSource
          .device_read_async(
              0,
              kReadSize,
              static_cast<uint8_t*>(destination.data()),
              stream.view())
          .get(),
      kReadSize);

  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);

  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfPinnedStagingAttempts").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingTransfers").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingBytes").sum, kReadSize);
  EXPECT_EQ(metrics.at("cudfBufferedRetainedSourceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfCopiedSourceBytes"), 0);
  EXPECT_EQ(metrics.count("cudfDirectHostToDeviceBytes"), 0);
}

TEST_F(CudfSplitReaderHelpersTest, disabledStagingIsNotAFailure) {
  constexpr size_t kReadSize = (1 << 20) + 13;
  PinnedStagingArena::configure(false, 0, 0, 0);

  std::string inputData(kReadSize, 'd');
  auto input = std::make_shared<BufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData), *pool_);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());

  EXPECT_EQ(
      dataSource
          .device_read_async(
              0,
              kReadSize,
              static_cast<uint8_t*>(destination.data()),
              stream.view())
          .get(),
      kReadSize);

  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfPinnedStagingDisabledBypasses").sum, 1);
  EXPECT_EQ(metrics.at("cudfDirectHostToDeviceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfPinnedStagingAttempts"), 0);
  EXPECT_EQ(metrics.count("cudfPinnedStagingFallbacks"), 0);
}

TEST_F(CudfSplitReaderHelpersTest, stagingAllocationFailureFallsBack) {
  constexpr size_t kReadSize = (1 << 20) + 17;
  PinnedStagingArena::setAllocationFailureForTesting(true);
  PinnedStagingArena::configure(true, 256 << 10, 2, 1);

  std::string inputData(kReadSize, '\0');
  for (size_t index = 0; index < inputData.size(); ++index) {
    inputData[index] = static_cast<char>(index % 251);
  }
  auto input = std::make_shared<BufferedInput>(
      std::make_shared<InMemoryReadFile>(inputData), *pool_);
  auto ioStats = std::make_shared<facebook::velox::IoStats>();
  BufferedInputDataSource dataSource(input, ioStats);
  TestCudaStream stream;
  rmm::device_buffer destination(
      kReadSize, stream.view(), cudf::get_current_device_resource_ref());

  auto completion = dataSource.device_read_async(
      0, kReadSize, static_cast<uint8_t*>(destination.data()), stream.view());
  EXPECT_EQ(completion.get(), kReadSize);

  std::string actual(kReadSize, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
  const auto metrics = ioStats->stats();
  EXPECT_EQ(metrics.at("cudfPinnedStagingAttempts").sum, 1);
  EXPECT_EQ(metrics.at("cudfPinnedStagingFallbacks").sum, 1);
  EXPECT_EQ(metrics.at("cudfDirectHostToDeviceBytes").sum, kReadSize);
  EXPECT_EQ(metrics.count("cudfPinnedStagingBytes"), 0);
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadBatchRejectsNullDestination) {
  folly::CPUThreadPoolExecutor executor(1);
  BufferedInputDataSource dataSource(makeInput("buffered-input", &executor));
  const std::vector<cudf::io::datasource::device_read_request> requests{
      {0, 0, nullptr}, {0, 1, nullptr}};

  EXPECT_THROW(
      dataSource.device_read_batch_async(
          cudf::host_span<cudf::io::datasource::device_read_request const>{
              requests.data(), requests.size()},
          cudf::get_default_stream()),
      VeloxException);
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadBatchClampsOverflowingExtent) {
  folly::CPUThreadPoolExecutor executor(1);
  BufferedInputDataSource dataSource(makeInput("short-input", &executor));
  TestCudaStream stream;
  rmm::device_buffer destination(
      1, stream.view(), cudf::get_current_device_resource_ref());
  const std::vector<cudf::io::datasource::device_read_request> requests{{
      std::numeric_limits<size_t>::max() - 1,
      4,
      static_cast<uint8_t*>(destination.data()),
  }};

  auto completion = dataSource.device_read_batch_async(
      cudf::host_span<cudf::io::datasource::device_read_request const>{
          requests.data(), requests.size()},
      stream.view());
  EXPECT_EQ(completion.get(), (std::vector<size_t>{0}));
}

TEST_F(CudfSplitReaderHelpersTest, deviceReadFutureWaitsForDeviceCopy) {
  const std::string inputData = "buffered-input-device-read";
  folly::CPUThreadPoolExecutor executor(1);
  auto input = makeInput(inputData, &executor);
  BufferedInputDataSource dataSource(input);

  TestCudaStream stream;
  rmm::device_buffer destination(
      inputData.size(), stream.view(), cudf::get_current_device_resource_ref());

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));

  auto completion = dataSource.device_read_async(
      0,
      inputData.size(),
      static_cast<uint8_t*>(destination.data()),
      stream.view());

  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    std::ignore = completion.get();
    FAIL() << "CUDA stream callback did not start";
  }

  auto completionWaiter = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        return completion.get();
      });
  EXPECT_EQ(completionWaiter.wait_for(100ms), std::future_status::timeout);
  releaseGate(gate);

  EXPECT_EQ(completionWaiter.get(), inputData.size());
  std::string actual(inputData.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData);
}

TEST_F(CudfSplitReaderHelpersTest, synchronousDeviceReadVariants) {
  const std::string inputData = "synchronous-device-read";
  folly::CPUThreadPoolExecutor executor(1);
  BufferedInputDataSource dataSource(makeInput(inputData, &executor));
  TestCudaStream stream;

  rmm::device_buffer destination(
      inputData.size(), stream.view(), cudf::get_current_device_resource_ref());
  const auto bytesRead = dataSource.device_read(
      2,
      inputData.size(),
      static_cast<uint8_t*>(destination.data()),
      stream.view());
  ASSERT_EQ(bytesRead, inputData.size() - 2);
  std::string actual(bytesRead, '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData.substr(2));

  auto owning = dataSource.device_read(1, 7, stream.view());
  ASSERT_EQ(owning->size(), 7);
  actual.assign(owning->size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpy(
      actual.data(), owning->data(), actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, inputData.substr(1, 7));
}

TEST_F(CudfSplitReaderHelpersTest, discardedDeviceBatchWaitsForDeviceCopy) {
  const std::string inputData = "discarded-device-batch";
  folly::CPUThreadPoolExecutor executor(1);
  auto dataSource = std::make_shared<BufferedInputDataSource>(
      makeInput(inputData, &executor));

  TestCudaStream stream;
  rmm::device_buffer destination(
      inputData.size(), stream.view(), cudf::get_current_device_resource_ref());
  const std::vector<cudf::io::datasource::device_read_request> requests{{
      0,
      inputData.size(),
      static_cast<uint8_t*>(destination.data()),
  }};

  StreamGate gate;
  auto entered = gate.entered.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(stream.view().value(), waitForGate, &gate));
  auto completion = dataSource->device_read_batch_async(
      cudf::host_span<cudf::io::datasource::device_read_request const>{
          requests.data(), requests.size()},
      stream.view());
  dataSource.reset();

  auto discard = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        completion = std::future<std::vector<size_t>>{};
      });
  if (entered.wait_for(10s) != std::future_status::ready) {
    releaseGate(gate);
    discard.get();
    FAIL() << "CUDA stream callback did not start";
  }
  EXPECT_EQ(discard.wait_for(0s), std::future_status::timeout);
  releaseGate(gate);
  EXPECT_NO_THROW(discard.get());
}

TEST_F(CudfSplitReaderHelpersTest, fetchByteRangesAcceptsEmptyBatch) {
  folly::CPUThreadPoolExecutor executor(1);
  auto dataSource = std::make_shared<BufferedInputDataSource>(
      makeInput("empty-range-batch", &executor));
  const std::vector<cudf::io::text::byte_range_info> ranges;
  TestCudaStream stream;

  auto [buffers, spans, completion] = fetchByteRangesAsync(
      dataSource,
      cudf::host_span<const cudf::io::text::byte_range_info>{
          ranges.data(), ranges.size()},
      stream.view(),
      cudf::get_current_device_resource_ref());
  ASSERT_EQ(buffers.size(), 1);
  EXPECT_EQ(buffers.front().size(), 0);
  EXPECT_TRUE(spans.empty());
  EXPECT_NO_THROW(completion.get());
}

TEST_F(CudfSplitReaderHelpersTest, fetchByteRangesAcceptsZeroSizedRanges) {
  folly::CPUThreadPoolExecutor executor(1);
  auto dataSource = std::make_shared<BufferedInputDataSource>(
      makeInput("zero-sized-ranges", &executor));
  const std::vector<cudf::io::text::byte_range_info> ranges{
      {0, 0}, {std::numeric_limits<int64_t>::max(), 0}};
  TestCudaStream stream;

  auto [buffers, spans, completion] = fetchByteRangesAsync(
      dataSource,
      cudf::host_span<const cudf::io::text::byte_range_info>{
          ranges.data(), ranges.size()},
      stream.view(),
      cudf::get_current_device_resource_ref());
  ASSERT_EQ(buffers.size(), 1);
  EXPECT_EQ(buffers.front().size(), 0);
  ASSERT_EQ(spans.size(), ranges.size());
  EXPECT_TRUE(spans[0].empty());
  EXPECT_TRUE(spans[1].empty());
  EXPECT_NO_THROW(completion.get());
}

TEST_F(CudfSplitReaderHelpersTest, byteRangeInfoRejectsNegativeValues) {
  EXPECT_THROW((void)cudf::io::text::byte_range_info(-1, 1), cudf::logic_error);
  EXPECT_THROW((void)cudf::io::text::byte_range_info(0, -1), cudf::logic_error);
}

TEST_F(
    CudfSplitReaderHelpersTest,
    fetchByteRangesRejectsOverflowBeforeAllocation) {
  folly::CPUThreadPoolExecutor executor(1);
  auto dataSource = std::make_shared<BufferedInputDataSource>(
      makeInput("invalid-ranges", &executor));
  TestCudaStream stream;

  const auto fetch = [&](const auto& ranges) {
    return fetchByteRangesAsync(
        dataSource,
        cudf::host_span<const cudf::io::text::byte_range_info>{
            ranges.data(), ranges.size()},
        stream.view(),
        cudf::get_current_device_resource_ref());
  };

  constexpr auto kMaximumRange = std::numeric_limits<int64_t>::max();
  const std::vector<cudf::io::text::byte_range_info> totalSizeOverflow{
      {0, kMaximumRange}, {0, kMaximumRange}, {0, 2}};
  EXPECT_THROW((void)fetch(totalSizeOverflow), VeloxException);

  const std::vector<cudf::io::text::byte_range_info> paddedSizeOverflow{
      {0, kMaximumRange}, {0, kMaximumRange}, {0, 1}};
  EXPECT_THROW((void)fetch(paddedSizeOverflow), VeloxException);
}

TEST_F(CudfSplitReaderHelpersTest, discardedFutureDrainsBeforeDatasourceDies) {
  auto state = std::make_shared<PendingReadState>();
  auto entered = state->entered.get_future();
  auto dataSource = std::make_shared<PendingDeviceDataSource>(state);
  const std::vector<cudf::io::text::byte_range_info> ranges{{0, 16}};
  TestCudaStream stream;

  auto [buffers, spans, completion] = fetchByteRangesAsync(
      dataSource,
      cudf::host_span<const cudf::io::text::byte_range_info>{
          ranges.data(), ranges.size()},
      stream.view(),
      cudf::get_current_device_resource_ref());
  dataSource.reset();
  EXPECT_FALSE(state->destroyed);

  auto discard = std::async(
      std::launch::async, [completion = std::move(completion)]() mutable {
        completion = std::future<void>{};
      });

  if (entered.wait_for(10s) != std::future_status::ready) {
    state->release.set_value();
    discard.get();
    FAIL() << "device read did not start";
  }
  EXPECT_EQ(discard.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(state->destroyed);

  state->release.set_value();
  EXPECT_NO_THROW(discard.get());
  EXPECT_TRUE(state->destroyed);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
