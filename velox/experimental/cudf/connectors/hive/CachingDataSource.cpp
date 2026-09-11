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
#include <cstdlib>
#include <cstring>
#include <limits>
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
struct DeviceReadCompletion {
  std::future<size_t> submitted;
  rmm::cuda_stream_view stream;
  int device;

  DeviceReadCompletion(std::future<size_t> f, rmm::cuda_stream_view s, int d)
      : submitted(std::move(f)), stream(s), device(d) {}
  DeviceReadCompletion(DeviceReadCompletion&&) = default;
  DeviceReadCompletion(const DeviceReadCompletion&) = delete;
  ~DeviceReadCompletion() {
    if (submitted.valid()) {
      try {
        get();
      } catch (...) {
      }
    }
  }
  size_t get() {
    const auto bytes = submitted.get();
    finishDeviceRead(stream, device);
    return bytes;
  }
};

} // namespace

struct CachingDataSource::State {
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
    auto* entry = pin.checkedEntry();
    if (entry->hasContiguousData()) {
      std::memcpy(dst, entry->contiguousData(), bytes);
      return bytes;
    }
    // Another reader may have populated this key using page-run allocation.
    // contiguous=true controls new allocations, not the layout of cache hits.
    for (const auto& range : entry->dataRanges(bytes)) {
      std::memcpy(dst, range.data(), range.size());
      dst += range.size();
    }
    return bytes;
  }

  std::unique_ptr<cudf::io::datasource::buffer> readHost(
      size_t offset,
      size_t size) {
    std::vector<uint8_t> bytes(clamp(offset, size));
    bytes.resize(readHost(offset, bytes.size(), bytes.data()));
    return cudf::io::datasource::buffer::create(std::move(bytes));
  }

  size_t submitDeviceRead(
      size_t offset,
      size_t bytes,
      uint8_t* dst,
      rmm::cuda_stream_view stream,
      int device) {
    const rmm::cuda_set_device_raii scope{rmm::cuda_device_id{device}};
    auto& slot = stagingSlot(device);
    auto* staging = slot.reserve(bytes);
    const auto actual = readHost(offset, bytes, staging);
    try {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          dst, staging, actual, cudaMemcpyHostToDevice, stream.value()));
      slot.recordCopy(stream.value());
      addBytes("cudfKvikioCacheHostToDeviceBytes", actual);
    } catch (...) {
      finishDeviceRead(stream, device);
      throw;
    }
    return actual;
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
  auto* executor = remoteReadExecutor();
  auto promise = std::make_shared<std::promise<size_t>>();
  auto submitted = promise->get_future();
  executor->add([state = state_, promise, offset, bytes, dst, stream, device] {
    try {
      promise->set_value(
          state->submitDeviceRead(offset, bytes, dst, stream, device));
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  });
  return std::async(
      std::launch::deferred,
      [completion = DeviceReadCompletion(
           std::move(submitted), stream, device)]() mutable {
        return completion.get();
      });
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
  const auto actual =
      state_->submitDeviceRead(offset, bytes, dst, stream, device);
  finishDeviceRead(stream, device);
  return actual;
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
