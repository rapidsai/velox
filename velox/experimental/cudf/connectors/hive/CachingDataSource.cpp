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
#include "velox/experimental/cudf/connectors/hive/CachingDataSource.h"

#include "velox/common/caching/FileIds.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <folly/Conv.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <folly/system/HardwareConcurrency.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// Keep remote cache fills out of the connector I/O executor, as in #18941.
// The launcher supplies KVIKIO_NTHREADS for an explicitly sized experiment.
folly::Executor* remoteReadExecutor() {
  static auto* executor = [] {
    size_t threads = 5 * std::max<size_t>(1, folly::available_concurrency());
    if (const auto* value = std::getenv("KVIKIO_NTHREADS")) {
      threads = folly::to<size_t>(value);
      VELOX_USER_CHECK_GT(threads, 0, "KVIKIO_NTHREADS must be positive");
    }
    return new folly::CPUThreadPoolExecutor(
        threads, std::make_shared<folly::NamedThreadFactory>("CudfRemoteIO"));
  }();
  return executor;
}

// cuDF exposes std::future, not a completion callback. Keep outstanding host
// reads off the executor while their futures are pending. The single readiness
// thread never performs I/O, CUDA work, or a blocking future get. Deferred
// delegates are supported, but their get must still run on the executor.
// Admission precedes cache allocation, so a large pass cannot allocate host
// buffers for every queued request at once. This does not change GPU
// preloading.
class CacheReadScheduler {
 public:
  struct Read {
    virtual ~Read() = default;
    virtual bool ready() noexcept = 0;
    virtual void run() noexcept = 0;
    virtual void fail(std::exception_ptr error) noexcept = 0;
  };

  static CacheReadScheduler& instance() {
    // Like remoteReadExecutor, outlive CUDA/static teardown. No buffers are
    // retained once reads finish; callers must quiesce before cache shutdown.
    static auto* scheduler = new CacheReadScheduler;
    return *scheduler;
  }

  void add(std::shared_ptr<Read> read) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (admitted_ == limit_) {
        queued_.push_back(std::move(read));
        return;
      }
      ++admitted_;
    }
    dispatch(std::move(read));
  }

  void await(std::shared_ptr<Read> read) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      waiting_.push_back(std::move(read));
    }
    condition_.notify_one();
  }

  void finished() noexcept {
    std::shared_ptr<Read> next;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queued_.empty()) {
        --admitted_;
      } else {
        next = std::move(queued_.front());
        queued_.pop_front();
      }
    }
    if (next) {
      dispatch(std::move(next));
    }
  }

 private:
  CacheReadScheduler() : executor_(remoteReadExecutor()) {
    // Reuse the existing remote-I/O window. "Unlimited" transport admission
    // must not imply unbounded cache allocations; absent/zero uses 64 here.
    if (const auto* value =
            std::getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) {
      const auto configured = folly::to<size_t>(value);
      if (configured != 0) {
        limit_ = configured;
      }
    }
    std::thread([this] { poll(); }).detach();
  }

  void dispatch(std::shared_ptr<Read> read) noexcept {
    try {
      executor_->add([read] { read->run(); });
    } catch (...) {
      read->fail(std::current_exception());
    }
  }

  void poll() {
    for (;;) {
      std::list<std::shared_ptr<Read>> ready;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return !waiting_.empty(); });
        for (auto it = waiting_.begin(); it != waiting_.end();) {
          auto current = it++;
          if ((*current)->ready()) {
            ready.splice(ready.end(), waiting_, current);
          }
        }
        if (ready.empty()) {
          condition_.wait_for(lock, std::chrono::microseconds(100));
        }
      }
      for (auto& read : ready) {
        dispatch(std::move(read));
      }
    }
  }

  folly::Executor* const executor_;
  size_t limit_{64};
  size_t admitted_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::shared_ptr<Read>> queued_;
  std::list<std::shared_ptr<Read>> waiting_;
};

std::atomic<int64_t> activeCacheFills{0};

int streamDevice(rmm::cuda_stream_view stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.value(), &device));
#else
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

void finishDeviceRead(rmm::cuda_stream_view stream, int device) {
  const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
  try {
    stream.synchronize();
  } catch (...) {
    // Do not release a destination while a failed stream may still reference
    // it. This is the same last-resort fence as BufferedInputDataSource.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    throw;
  }
}

class PinnedStagingBuffer {
 public:
  explicit PinnedStagingBuffer(size_t size)
      : mr_(cudf::get_pinned_memory_resource()),
        size_(size),
        data_(static_cast<uint8_t*>(mr_.allocate_sync(size, 64))) {}
  ~PinnedStagingBuffer() {
    mr_.deallocate_sync(data_, size_, 64);
  }
  uint8_t* data() const {
    return data_;
  }

 private:
  rmm::host_device_async_resource_ref mr_;
  size_t size_;
  uint8_t* data_;
};

struct PinnedStagingSlot {
  PinnedStagingBuffer* buffer{nullptr};
  size_t capacity{0};
  cudaEvent_t event{nullptr};

  uint8_t* reserve(size_t size) {
    if (event != nullptr) {
      CUDF_CUDA_TRY(cudaEventSynchronize(event));
    }
    if (capacity < size) {
      delete std::exchange(buffer, nullptr);
      capacity = 0;
      // Leave an empty, reusable slot if allocation throws.
      buffer = new PinnedStagingBuffer(size);
      capacity = size;
    }
    return buffer->data();
  }

  void recordCopy(cudaStream_t stream) {
    if (event == nullptr) {
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  }
};

// The upstream experiment retains its high-water staging capacity until
// process exit. Do not invoke CUDA during thread-pool/static teardown: the
// runtime may already be gone. This is not the bounded BufferedInput arena.
PinnedStagingSlot& stagingSlot(int device) {
  static thread_local std::unordered_map<int, PinnedStagingSlot> slots;
  return slots[device];
}

// WXD's cuDF batch API permits destination destruction when its future is
// consumed or discarded. The executor's future alone only marks H2D
// submission, so the outer completion also fences the copy. Wait outside the
// remote executor rather than occupying an I/O thread during H2D completion.
struct SubmittedDeviceRead {
  size_t bytes;
  // End the active registration lease before releasing the transfer's pin.
  // Idle registrations retain independent ownership in the registry.
  cache::CachePin pin;
  std::optional<CacheHostRegistration::Lease> registration;
};

struct DeviceReadCompletion {
  std::future<SubmittedDeviceRead> submitted;
  rmm::cuda_stream_view stream;
  int device;
  std::shared_ptr<IoStats> stats;
  std::shared_ptr<std::atomic<bool>> published;

  DeviceReadCompletion(
      std::future<SubmittedDeviceRead> f,
      rmm::cuda_stream_view s,
      int d,
      std::shared_ptr<IoStats> ioStats,
      std::shared_ptr<std::atomic<bool>> ioPublished)
      : submitted(std::move(f)),
        stream(s),
        device(d),
        stats(std::move(ioStats)),
        published(std::move(ioPublished)) {}
  DeviceReadCompletion(DeviceReadCompletion&&) = default;
  DeviceReadCompletion(const DeviceReadCompletion&) = delete;
  ~DeviceReadCompletion() {
    if (submitted.valid() && published->load()) {
      try {
        get();
      } catch (...) {
      }
    }
  }
  static size_t finish(
      SubmittedDeviceRead result,
      rmm::cuda_stream_view stream,
      int device,
      const std::shared_ptr<IoStats>& stats) {
    const auto start = std::chrono::steady_clock::now();
    finishDeviceRead(stream, device);
    if (result.registration && stats) {
      // Host completion wait, not GPU event elapsed time; work may already
      // have completed before the caller consumes its future.
      const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
      stats->addCounter(
          "cudfCacheRegisteredH2DWaitNanos",
          RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
    }
    return result.bytes;
  }
  size_t get() {
    return finish(submitted.get(), stream, device, stats);
  }
};

} // namespace

struct CachingDataSource::State {
  struct AsyncRead;
  std::unique_ptr<cudf::io::datasource> delegate;
  cache::AsyncDataCache* cache;
  StringIdLease fileNum;
  std::shared_ptr<IoStats> ioStats;

  State(
      std::unique_ptr<cudf::io::datasource> source,
      std::string_view path,
      cache::AsyncDataCache* dataCache,
      std::shared_ptr<IoStats> stats)
      : delegate(std::move(source)),
        cache(dataCache),
        fileNum(fileIds(), std::string(path)),
        ioStats(std::move(stats)) {
    VELOX_CHECK_NOT_NULL(delegate);
    VELOX_CHECK_NOT_NULL(cache);
  }

  void addBytes(const std::string& name, size_t bytes) {
    if (ioStats) {
      ioStats->addCounter(
          name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
    }
  }

  size_t clamp(size_t offset, size_t size) const {
    const auto fileSize = delegate->size();
    return offset < fileSize ? std::min(size, fileSize - offset) : 0;
  }

  void addCount(const std::string& name, int64_t value = 1) {
    if (ioStats) {
      ioStats->addCounter(name, RuntimeCounter(value));
    }
  }

  void addTime(
      const std::string& name,
      std::chrono::steady_clock::time_point start) {
    if (ioStats) {
      const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
      ioStats->addCounter(
          name, RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
    }
  }

  cache::CachePin pinRange(size_t offset, size_t size) {
    // AsyncDataCache stores entry lengths in an int32_t. Large individual
    // reads remain legal datasource requests, but must bypass that cache.
    if (size > std::numeric_limits<int32_t>::max()) {
      return {};
    }
    const cache::RawFileCacheKey key{fileNum.id(), offset};
    for (int attempt = 0; attempt < 32; ++attempt) {
      folly::SemiFuture<bool> wait = folly::SemiFuture<bool>::makeEmpty();
      cache::CachePin pin;
      try {
        pin = cache->findOrCreate(key, size, /*contiguous=*/true, &wait);
      } catch (const VeloxRuntimeError& error) {
        if (error.errorCode() != error_code::kNoCacheSpace) {
          throw;
        }
        return {};
      }
      if (pin.empty()) {
        if (!wait.valid()) {
          return {};
        }
        std::move(wait).via(&folly::QueuedImmediateExecutor::instance()).wait();
        continue;
      }
      auto* entry = pin.checkedEntry();
      entry->getAndClearFirstUseFlag();
      if (!entry->isExclusive()) {
        addBytes("cudfKvikioCacheHitBytes", size);
        return pin;
      }
      if (!entry->hasContiguousData()) {
        return {};
      }
      const auto actual = delegate->host_read(
          offset, size, reinterpret_cast<uint8_t*>(entry->contiguousData()));
      // Do not publish uninitialized tail bytes after a short read. An
      // exception releases the exclusive pin and abandons the incomplete fill.
      VELOX_CHECK_EQ(actual, size, "Short KvikIO read while filling cache");
      entry->setExclusiveToShared();
      addBytes("cudfKvikioCacheMissBytes", size);
      return pin;
    }
    return {};
  }

  size_t readHost(size_t offset, size_t size, uint8_t* dst) {
    const auto bytes = clamp(offset, size);
    if (bytes == 0) {
      return 0;
    }
    VELOX_CHECK_NOT_NULL(dst);
    auto pin = pinRange(offset, bytes);
    if (pin.empty()) {
      const auto actual = delegate->host_read(offset, bytes, dst);
      addBytes("cudfKvikioCacheBypassBytes", actual);
      return actual;
    }
    copyCached(pin, bytes, dst);
    return bytes;
  }

  static void
  copyCached(const cache::CachePin& pin, size_t bytes, uint8_t* dst) {
    auto* entry = pin.checkedEntry();
    if (entry->hasContiguousData()) {
      std::memcpy(dst, entry->contiguousData(), bytes);
      return;
    }
    // Another reader may have populated this key using page-run allocation.
    // contiguous=true controls new allocations, not the layout of cache hits.
    for (const auto& range : entry->dataRanges(bytes)) {
      std::memcpy(dst, range.data(), range.size());
      dst += range.size();
    }
  }

  std::unique_ptr<cudf::io::datasource::buffer> readHost(
      size_t offset,
      size_t size) {
    std::vector<uint8_t> bytes(clamp(offset, size));
    bytes.resize(readHost(offset, bytes.size(), bytes.data()));
    return cudf::io::datasource::buffer::create(std::move(bytes));
  }

  SubmittedDeviceRead submitDeviceRead(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      rmm::cuda_stream_view stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    return submitReadyDeviceRead(
        pinRange(offset, bytes), offset, bytes, dst, stream, device);
  }

  SubmittedDeviceRead submitReadyDeviceRead(
      cache::CachePin pin,
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      rmm::cuda_stream_view stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    SubmittedDeviceRead result{0, std::move(pin), std::nullopt};
    const bool registrationEnabled = CacheHostRegistration::enabled();
    if (registrationEnabled) {
      if (!result.pin.empty()) {
        const auto ranges = result.pin.checkedEntry()->dataRanges(bytes);
        auto lease = CacheHostRegistration::tryAcquire(
            std::span<const cache::CachePin>(&result.pin, 1), ioStats);
        if (lease) {
          result.registration.emplace(std::move(*lease));
          try {
            size_t copied = 0;
            for (const auto& range : ranges) {
              CUDF_CUDA_TRY(cudaMemcpyAsync(
                  dst + copied,
                  range.data(),
                  range.size(),
                  cudaMemcpyHostToDevice,
                  stream.value()));
              copied += range.size();
            }
            result.bytes = copied;
            addBytes("cudfCacheRegisteredH2DBytes", copied);
            addBytes("cudfKvikioCacheHostToDeviceBytes", copied);
          } catch (...) {
            // A failed call may have submitted a prefix. Fence before result
            // releases its registration and cache pin during unwinding.
            finishDeviceRead(stream, device);
            throw;
          }
          return result;
        }
        addBytes("cudfCacheHostRegistrationFallbackBytes", bytes);
      }
    }
    auto& slot = stagingSlot(device);
    auto* staging = slot.reserve(bytes);
    if (!result.pin.empty()) {
      // A registration fallback must not look up/refetch the range again.
      copyCached(result.pin, bytes, staging);
      result.bytes = bytes;
      result.pin.clear();
    } else {
      result.bytes = delegate->host_read(offset, bytes, staging);
      addBytes("cudfKvikioCacheBypassBytes", result.bytes);
    }
    try {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          dst, staging, result.bytes, cudaMemcpyHostToDevice, stream.value()));
      slot.recordCopy(stream.value());
      addBytes("cudfKvikioCacheHostToDeviceBytes", result.bytes);
    } catch (...) {
      finishDeviceRead(stream, device);
      throw;
    }
    return result;
  }
};

struct CachingDataSource::State::AsyncRead final
    : CacheReadScheduler::Read,
      std::enable_shared_from_this<AsyncRead> {
  std::shared_ptr<State> state;
  std::shared_ptr<std::promise<SubmittedDeviceRead>> promise;
  size_t offset;
  size_t bytes;
  uint8_t* dst;
  rmm::cuda_stream_view stream;
  int device;
  CacheReadScheduler& scheduler;
  cache::CachePin pin;
  folly::SemiFuture<bool> cacheWait = folly::SemiFuture<bool>::makeEmpty();
  std::future<size_t> hostRead;
  const std::chrono::steady_clock::time_point queued =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point fillStart;
  std::chrono::steady_clock::time_point cacheWaitStart;
  int attempts{0};
  bool activeFill{false};

  AsyncRead(
      std::shared_ptr<State> s,
      std::shared_ptr<std::promise<SubmittedDeviceRead>> p,
      size_t o,
      size_t b,
      uint8_t* d,
      rmm::cuda_stream_view st,
      int dev,
      CacheReadScheduler& sched)
      : state(std::move(s)),
        promise(std::move(p)),
        offset(o),
        bytes(b),
        dst(d),
        stream(st),
        device(dev),
        scheduler(sched) {}

  bool ready() noexcept override {
    if (cacheWait.valid()) {
      return cacheWait.isReady();
    }
    return hostRead.wait_for(std::chrono::seconds(0)) !=
        std::future_status::timeout;
  }

  void endFill() {
    if (std::exchange(activeFill, false)) {
      --activeCacheFills;
      state->addTime("cudfKvikioCacheAsyncFillReadNanos", fillStart);
    }
  }

  void fail(std::exception_ptr error) noexcept override {
    // A std::future destructor need not wait for external writes. Even if
    // scheduling/allocation fails after dispatch, drain before abandoning the
    // exclusive cache allocation or allowing destination destruction.
    if (hostRead.valid()) {
      try {
        hostRead.get();
      } catch (...) {
      }
    }
    try {
      endFill();
      state->addCount("cudfKvikioCacheAsyncReadFailures");
    } catch (...) {
    }
    pin.clear();
    promise->set_exception(error);
    scheduler.finished();
  }

  void run() noexcept override {
    try {
      const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
      if (attempts == 0) {
        state->addTime("cudfKvikioCacheAsyncReadQueueNanos", queued);
      }
      if (hostRead.valid()) {
        const auto actual = hostRead.get();
        endFill();
        VELOX_CHECK_EQ(actual, bytes, "Short KvikIO read while filling cache");
        pin.checkedEntry()->setExclusiveToShared();
        state->addBytes("cudfKvikioCacheMissBytes", bytes);
        copy();
        return;
      }
      if (cacheWait.valid()) {
        std::move(cacheWait).get();
        cacheWait = folly::SemiFuture<bool>::makeEmpty();
        state->addTime("cudfKvikioCacheExclusiveWaitNanos", cacheWaitStart);
      }
      if (++attempts > 32 || bytes > std::numeric_limits<int32_t>::max()) {
        copy();
        return;
      }
      try {
        pin = state->cache->findOrCreate(
            {state->fileNum.id(), offset}, bytes, true, &cacheWait);
      } catch (const VeloxRuntimeError& error) {
        if (error.errorCode() != error_code::kNoCacheSpace) {
          throw;
        }
        copy();
        return;
      }
      if (pin.empty()) {
        if (cacheWait.valid()) {
          cacheWaitStart = std::chrono::steady_clock::now();
          scheduler.await(shared_from_this());
        } else {
          copy();
        }
        return;
      }
      auto* entry = pin.checkedEntry();
      entry->getAndClearFirstUseFlag();
      if (!entry->isExclusive()) {
        state->addBytes("cudfKvikioCacheHitBytes", bytes);
        copy();
        return;
      }
      if (!entry->hasContiguousData()) {
        pin.clear();
        copy();
        return;
      }
      fillStart = std::chrono::steady_clock::now();
      activeFill = true;
      const auto active = ++activeCacheFills;
      state->addCount("cudfKvikioCacheAsyncFillInFlightSamples", active);
      hostRead = state->delegate->host_read_async(
          offset, bytes, reinterpret_cast<uint8_t*>(entry->contiguousData()));
      VELOX_CHECK(hostRead.valid(), "KvikIO returned an invalid read future");
      state->addCount("cudfKvikioCacheAsyncFillSubmitted");
      if (hostRead.wait_for(std::chrono::seconds(0)) ==
          std::future_status::deferred) {
        state->addCount("cudfKvikioCacheDeferredHostReads");
      }
      scheduler.await(shared_from_this());
    } catch (...) {
      fail(std::current_exception());
    }
  }

  void copy() {
    if (pin.empty()) {
      state->addCount("cudfKvikioCacheAsyncReadSynchronousFallbacks");
    }
    promise->set_value(state->submitReadyDeviceRead(
        std::move(pin), offset, bytes, dst, stream, device));
    scheduler.finished();
  }
};

CachingDataSource::CachingDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    std::shared_ptr<IoStats> ioStats)
    : state_(
          std::make_shared<State>(
              std::move(delegate),
              path,
              cache,
              std::move(ioStats))) {}
CachingDataSource::~CachingDataSource() = default;

size_t CachingDataSource::size() const {
  return state_->delegate->size();
}
bool CachingDataSource::supports_device_read() const {
  return true;
}
bool CachingDataSource::is_device_read_preferred(size_t bytes) const {
  return state_->delegate->is_device_read_preferred(bytes);
}

std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::host_read(
    size_t offset,
    size_t bytes) {
  return state_->readHost(offset, bytes);
}
size_t CachingDataSource::host_read(size_t offset, size_t bytes, uint8_t* dst) {
  return state_->readHost(offset, bytes, dst);
}
std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CachingDataSource::host_read_async(size_t offset, size_t bytes) {
  return std::async(std::launch::deferred, [state = state_, offset, bytes] {
    return state->readHost(offset, bytes);
  });
}
std::future<size_t>
CachingDataSource::host_read_async(size_t offset, size_t bytes, uint8_t* dst) {
  return std::async(
      std::launch::deferred, [state = state_, offset, bytes, dst] {
        return state->readHost(offset, bytes, dst);
      });
}

std::future<size_t> CachingDataSource::device_read_async(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  bytes = state_->clamp(offset, bytes);
  if (bytes == 0) {
    std::promise<size_t> ready;
    ready.set_value(0);
    return ready.get_future();
  }
  VELOX_CHECK_NOT_NULL(dst);
  const auto device = streamDevice(stream);
  auto& scheduler = CacheReadScheduler::instance();
  auto promise = std::make_shared<std::promise<SubmittedDeviceRead>>();
  auto submitted = promise->get_future();
  auto read = std::make_shared<State::AsyncRead>(
      state_, promise, offset, bytes, dst, stream, device, scheduler);
  auto published = std::make_shared<std::atomic<bool>>(false);
  // Construct the destination-lifetime fence before publishing any I/O.
  auto completion = std::async(
      std::launch::deferred,
      [completion = DeviceReadCompletion(
           std::move(submitted),
           stream,
           device,
           state_->ioStats,
           published)]() mutable { return completion.get(); });
  published->store(true);
  try {
    scheduler.add(std::move(read));
  } catch (...) {
    promise->set_exception(std::current_exception());
    throw;
  }
  return completion;
}
size_t CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  bytes = state_->clamp(offset, bytes);
  if (bytes == 0) {
    return 0;
  }
  VELOX_CHECK_NOT_NULL(dst);
  const auto device = streamDevice(stream);
  // A synchronous caller may already occupy the connector executor. Queuing
  // the read back to that executor and waiting would deadlock when all its
  // threads are occupied by such callers. Keep scheduling in the async API;
  // perform synchronous reads inline, including the destination-lifetime fence.
  return DeviceReadCompletion::finish(
      state_->submitDeviceRead(offset, bytes, dst, stream, device),
      stream,
      device,
      state_->ioStats);
}
std::unique_ptr<cudf::io::datasource::buffer> CachingDataSource::device_read(
    size_t offset,
    size_t bytes,
    rmm::cuda_stream_view stream) {
  const auto readSize = state_->clamp(offset, bytes);
  rmm::device_buffer result(
      readSize, stream, cudf::get_current_device_resource_ref());
  const auto actual = device_read(
      offset, readSize, static_cast<uint8_t*>(result.data()), stream);
  result.resize(actual, stream);
  return datasource::buffer::create(std::move(result));
}

std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats) {
  if (cache == nullptr || !cacheable) {
    return delegate;
  }
  return std::make_unique<CachingDataSource>(
      std::move(delegate), path, cache, std::move(ioStats));
}

} // namespace facebook::velox::cudf_velox::connector::hive
