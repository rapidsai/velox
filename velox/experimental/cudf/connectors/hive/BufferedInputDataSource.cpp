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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/PinnedStagingArena.h"

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/common/DirectInputStream.h"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/datasource.hpp>

#include <rmm/cuda_device.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using DeviceReadRequest = cudf::io::datasource::device_read_request;
using facebook::velox::IoStats;
using facebook::velox::RuntimeCounter;
using facebook::velox::cudf_velox::connector::hive::PinnedStagingArena;
using facebook::velox::dwio::common::BufferedInput;
using facebook::velox::dwio::common::CachedRegion;
using facebook::velox::dwio::common::CacheInputStream;
using facebook::velox::dwio::common::DirectBufferedInput;
using facebook::velox::dwio::common::DirectInputStream;
using facebook::velox::dwio::common::RetainedBufferedRegion;

constexpr size_t kMaximumCopiesPerBatch = 32;
constexpr size_t kMinimumPinnedStagingBytes = 1ULL << 20;

const std::string kDeviceReadBatches = "cudfBufferedDeviceReadBatches";
const std::string kDeviceReadRequests = "cudfBufferedDeviceReadRequests";
const std::string kDeviceReadPlannedRanges =
    "cudfBufferedDeviceReadPlannedRanges";
const std::string kDeviceReadBytes = "cudfBufferedDeviceReadBytes";
const std::string kDeviceReadFragments = "cudfBufferedDeviceReadFragments";
const std::string kCacheBackedSourceBytes = "cudfCacheBackedSourceBytes";
const std::string kBufferedRetainedSourceBytes =
    "cudfBufferedRetainedSourceBytes";
const std::string kCopiedSourceBytes = "cudfCopiedSourceBytes";
const std::string kStagingAttempts = "cudfPinnedStagingAttempts";
const std::string kStagingTransfers = "cudfPinnedStagingTransfers";
const std::string kStagingBytes = "cudfPinnedStagingBytes";
const std::string kStagingWindows = "cudfPinnedStagingWindows";
const std::string kStagingAcquireNanos = "cudfPinnedStagingAcquireNanos";
const std::string kStagingContendedAcquisitions =
    "cudfPinnedStagingContendedAcquisitions";
// These are per-acquisition distribution samples. Their RuntimeMetric sum is
// not a gauge; use count/min/max (or sum/count) when interpreting them.
const std::string kStagingActiveLeasesAtAcquireSamples =
    "cudfPinnedStagingActiveLeasesAtAcquireSamples";
const std::string kStagingWindowSetCapacityAtAcquireSamples =
    "cudfPinnedStagingWindowSetCapacityAtAcquireSamples";
const std::string kStagingPackNanos = "cudfPinnedStagingPackNanos";
const std::string kStagingH2DWaitNanos = "cudfPinnedStagingH2DWaitNanos";
const std::string kStagingMemcpyBatchAttempts =
    "cudfPinnedStagingMemcpyBatchAttempts";
const std::string kStagingMemcpyBatchCopies =
    "cudfPinnedStagingMemcpyBatchCopies";
const std::string kStagingNativeMemcpyBatchAttempts =
    "cudfPinnedStagingNativeMemcpyBatchAttempts";
const std::string kStagingNativeMemcpyBatchCopies =
    "cudfPinnedStagingNativeMemcpyBatchCopies";
const std::string kStagingFallbacks = "cudfPinnedStagingFallbacks";
const std::string kStagingDisabledBypasses =
    "cudfPinnedStagingDisabledBypasses";
const std::string kStagingSmallReadBypasses =
    "cudfPinnedStagingSmallReadBypasses";
const std::string kDirectH2DBytes = "cudfDirectHostToDeviceBytes";

void addIoCounter(
    const std::shared_ptr<IoStats>& ioStats,
    const std::string& name,
    uint64_t value,
    RuntimeCounter::Unit unit = RuntimeCounter::Unit::kNone) {
  if (ioStats == nullptr) {
    return;
  }
  ioStats->addCounter(
      name, RuntimeCounter(facebook::velox::saturateCast(value), unit));
}

uint64_t elapsedNanos(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

int getStreamDevice(rmm::cuda_stream_view stream) {
  int device{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  CUDF_CUDA_TRY(cudaStreamGetDevice(stream.value(), &device));
#else
  // CUDA versions before 12.8 cannot query a stream's device. Retain the
  // historical requirement that the stream belongs to the current device.
  CUDF_CUDA_TRY(cudaGetDevice(&device));
#endif
  return device;
}

void synchronizeStream(rmm::cuda_stream_view stream, int device) {
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
  try {
    stream.synchronize();
  } catch (...) {
    const auto primaryError = std::current_exception();
    // Returning while a host buffer may still be in use would be unsafe. A
    // device-wide fence is the last recoverable fallback.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::terminate();
    }
    std::rethrow_exception(primaryError);
  }
}

// std::future returned by an executor does not wait when discarded. Wrapping
// it in a deferred future whose captured state drains in its destructor makes
// discard obey the datasource contract: once the outer future is gone, no
// destination or retained host source remains in use.
template <typename T>
class DrainingCompletion {
 public:
  explicit DrainingCompletion(std::future<T> completion)
      : completion_(std::move(completion)) {}

  DrainingCompletion(DrainingCompletion&& other) noexcept
      : completion_(std::move(other.completion_)) {}

  DrainingCompletion(const DrainingCompletion&) = delete;
  DrainingCompletion& operator=(const DrainingCompletion&) = delete;
  DrainingCompletion& operator=(DrainingCompletion&&) = delete;

  ~DrainingCompletion() noexcept {
    if (completion_.valid()) {
      try {
        std::ignore = completion_.get();
      } catch (...) {
      }
    }
  }

  T get() {
    return completion_.get();
  }

 private:
  std::future<T> completion_;
};

template <typename T>
std::future<T> makeDrainingFuture(std::future<T> completion) {
  auto state = DrainingCompletion<T>{std::move(completion)};
  return std::async(
      std::launch::deferred,
      [state = std::move(state)]() mutable { return state.get(); });
}

class CudaEvent {
 public:
  CudaEvent() {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CudaEvent() {
    if (event_ != nullptr) {
      // Event destruction releases bookkeeping only. Completion has already
      // been established before this object leaves the transfer routine.
      std::ignore = cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  cudaEvent_t get() const {
    return event_;
  }

  void record(rmm::cuda_stream_view stream) {
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
    recorded_ = true;
  }

  void synchronize() {
    if (!recorded_) {
      return;
    }
    CUDF_CUDA_TRY(cudaEventSynchronize(event_));
    recorded_ = false;
  }

 private:
  cudaEvent_t event_{nullptr};
  bool recorded_{false};
};

// Owns every host source referenced by a batch of H2D descriptors. Cache hits
// retain cache pins directly; non-cache reads retain loaded allocations when
// available, with copied pageable buffers as the generic fallback. The
// submission layer is intentionally independent of source ownership: the same
// prepared plan can use bounded double-buffered pinned staging or safely fall
// back to direct pageable copies without changing lifetime/completion rules.
class HostToDeviceTransferPlan {
 public:
  explicit HostToDeviceTransferPlan(std::shared_ptr<IoStats> ioStats)
      : ioStats_(std::move(ioStats)) {}

  void addCachedRegion(CachedRegion region, uint8_t* destination) {
    VELOX_CHECK_LE(
        region.size(),
        std::numeric_limits<size_t>::max() - cachedSourceBytes_,
        "Cached source byte count overflows size_t");
    cachedSourceBytes_ += region.size();
    const auto ownerIndex = retainedRegions_.size();
    retainedRegions_.emplace_back(std::move(region));
    retainedDescriptorCounts_.push_back(0);
    size_t destinationOffset = 0;
    for (const auto range : retainedRegions_.back()->ranges()) {
      addDescriptor(
          destination + destinationOffset,
          range.data(),
          range.size(),
          SourceOwner{SourceOwnerKind::kCached, ownerIndex});
      ++retainedDescriptorCounts_.back();
      destinationOffset += range.size();
    }
    VELOX_CHECK_EQ(
        destinationOffset,
        retainedRegions_.back()->size(),
        "Cached region ranges do not cover the retained region");
    VELOX_CHECK_GT(
        retainedDescriptorCounts_.back(),
        0,
        "A nonempty cached region must contain at least one range");
  }

  void addCopiedRegion(std::vector<uint8_t> region, uint8_t* destination) {
    if (region.empty()) {
      return;
    }
    VELOX_CHECK_LE(
        region.size(),
        std::numeric_limits<size_t>::max() - copiedSourceBytes_,
        "Copied source byte count overflows size_t");
    copiedSourceBytes_ += region.size();
    const auto ownerIndex = copiedRegions_.size();
    copiedRegions_.push_back(std::move(region));
    copiedDescriptorCounts_.push_back(1);
    addDescriptor(
        destination,
        copiedRegions_.back().data(),
        copiedRegions_.back().size(),
        SourceOwner{SourceOwnerKind::kCopied, ownerIndex});
  }

  void addBufferedRegion(RetainedBufferedRegion region, uint8_t* destination) {
    VELOX_CHECK_GT(region.size(), 0);
    VELOX_CHECK_LE(
        region.size(),
        std::numeric_limits<size_t>::max() - bufferedSourceBytes_,
        "Buffered source byte count overflows size_t");
    bufferedSourceBytes_ += region.size();
    const auto ownerIndex = bufferedRegions_.size();
    bufferedRegions_.emplace_back(std::move(region));
    const auto& retained = bufferedRegions_.back().value();
    addDescriptor(
        destination,
        retained.data(),
        retained.size(),
        SourceOwner{SourceOwnerKind::kBuffered, ownerIndex});
  }

  void submitAndWait(
      rmm::cuda_stream_view stream,
      int device,
      std::optional<PinnedStagingArena::WindowSetLease> windows) {
    if (destinations_.empty()) {
      return;
    }

    auto const deviceScope =
        rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
    addIoCounter(ioStats_, kDeviceReadFragments, destinations_.size());
    if (cachedSourceBytes_ != 0) {
      addIoCounter(
          ioStats_,
          kCacheBackedSourceBytes,
          cachedSourceBytes_,
          RuntimeCounter::Unit::kBytes);
    }
    if (copiedSourceBytes_ != 0) {
      addIoCounter(
          ioStats_,
          kCopiedSourceBytes,
          copiedSourceBytes_,
          RuntimeCounter::Unit::kBytes);
    }
    if (bufferedSourceBytes_ != 0) {
      addIoCounter(
          ioStats_,
          kBufferedRetainedSourceBytes,
          bufferedSourceBytes_,
          RuntimeCounter::Unit::kBytes);
    }
    if (windows.has_value()) {
      addIoCounter(ioStats_, kStagingTransfers, 1);
      addIoCounter(
          ioStats_, kStagingBytes, totalBytes_, RuntimeCounter::Unit::kBytes);
      submitStagedAndWait(*windows, stream, device);
      return;
    }

    addIoCounter(
        ioStats_, kDirectH2DBytes, totalBytes_, RuntimeCounter::Unit::kBytes);
    submitDirectAndWait(stream, device);
  }

 private:
  enum class SourceOwnerKind : uint8_t { kCached, kCopied, kBuffered };

  struct SourceOwner {
    SourceOwnerKind kind;
    size_t index;
  };

  struct Cursor {
    size_t descriptor{0};
    size_t offset{0};
  };

  struct WindowBatch {
    std::vector<PinnedStagingArena::Copy> packCopies;
    std::vector<void*> destinations;
    std::vector<const void*> sources;
    std::vector<size_t> sizes;
    std::vector<size_t> completedDescriptors;
    size_t bytes{0};
  };

  void submitDirectAndWait(rmm::cuda_stream_view stream, int device) {
    // Allocate the tail fence before submitting any copy. A failure here is
    // therefore a synchronous scheduling failure with no source lifetime to
    // drain.
    CudaEvent tailEvent;
    std::exception_ptr scheduleError;

    for (size_t begin = 0; begin < destinations_.size();
         begin += kMaximumCopiesPerBatch) {
      const auto count =
          std::min(kMaximumCopiesPerBatch, destinations_.size() - begin);
      try {
        CUDF_CUDA_TRY(
            cudf::detail::memcpy_batch_async(
                destinations_.data() + begin,
                sources_.data() + begin,
                sizes_.data() + begin,
                count,
                stream));
      } catch (...) {
        scheduleError = std::current_exception();
        break;
      }
    }

    // One event after the final successfully submitted group is sufficient to
    // keep every retained pin/buffer alive until all preceding copies finish.
    // Even after a scheduling error the CUDA call may have submitted a prefix,
    // so always attempt the tail fence and conservatively drain on failure.
    std::exception_ptr fenceError;
    const auto recordStatus = cudaEventRecord(tailEvent.get(), stream.value());
    if (recordStatus == cudaSuccess) {
      const auto synchronizeStatus = cudaEventSynchronize(tailEvent.get());
      if (synchronizeStatus != cudaSuccess) {
        try {
          CUDF_CUDA_TRY(synchronizeStatus);
        } catch (...) {
          fenceError = std::current_exception();
        }
      }
    } else {
      try {
        CUDF_CUDA_TRY(recordStatus);
      } catch (...) {
        fenceError = std::current_exception();
      }
    }

    if (fenceError != nullptr) {
      try {
        synchronizeStream(stream, device);
      } catch (...) {
        // synchronizeStream either established device completion before
        // throwing or terminated because source lifetime could not be proven.
      }
    }
    if (scheduleError != nullptr) {
      std::rethrow_exception(scheduleError);
    }
    if (fenceError != nullptr) {
      std::rethrow_exception(fenceError);
    }
  }

  WindowBatch makeWindowBatch(Cursor& cursor, uint8_t* staging, size_t capacity)
      const {
    WindowBatch batch;
    while (cursor.descriptor < sizes_.size() && batch.bytes < capacity) {
      const auto descriptor = cursor.descriptor;
      VELOX_DCHECK_LT(cursor.offset, sizes_[descriptor]);
      const auto bytes =
          std::min(sizes_[descriptor] - cursor.offset, capacity - batch.bytes);
      VELOX_CHECK_GT(bytes, 0);

      const auto* source =
          static_cast<const uint8_t*>(sources_[descriptor]) + cursor.offset;
      auto* destination =
          static_cast<uint8_t*>(destinations_[descriptor]) + cursor.offset;
      auto* stagedSource = staging + batch.bytes;
      batch.packCopies.push_back(
          PinnedStagingArena::Copy{source, batch.bytes, bytes});

      const auto canCoalesce = !batch.sizes.empty() &&
          static_cast<uint8_t*>(batch.destinations.back()) +
                  batch.sizes.back() ==
              destination &&
          static_cast<const uint8_t*>(batch.sources.back()) +
                  batch.sizes.back() ==
              stagedSource;
      if (canCoalesce) {
        batch.sizes.back() += bytes;
      } else {
        batch.destinations.push_back(destination);
        batch.sources.push_back(stagedSource);
        batch.sizes.push_back(bytes);
      }

      batch.bytes += bytes;
      cursor.offset += bytes;
      if (cursor.offset == sizes_[descriptor]) {
        batch.completedDescriptors.push_back(descriptor);
        ++cursor.descriptor;
        cursor.offset = 0;
      }
    }
    VELOX_CHECK_GT(batch.bytes, 0, "Pinned staging made no transfer progress");
    return batch;
  }

  void releasePackedSources(const std::vector<size_t>& descriptors) {
    for (const auto descriptor : descriptors) {
      VELOX_CHECK_LT(descriptor, owners_.size());
      const auto owner = owners_[descriptor];
      if (owner.kind == SourceOwnerKind::kCached) {
        VELOX_CHECK_LT(owner.index, retainedDescriptorCounts_.size());
        auto& remaining = retainedDescriptorCounts_[owner.index];
        VELOX_CHECK_GT(remaining, 0);
        if (--remaining == 0) {
          retainedRegions_[owner.index].reset();
        }
      } else if (owner.kind == SourceOwnerKind::kBuffered) {
        VELOX_CHECK_LT(owner.index, bufferedRegions_.size());
        bufferedRegions_[owner.index].reset();
      } else {
        VELOX_CHECK_LT(owner.index, copiedDescriptorCounts_.size());
        auto& remaining = copiedDescriptorCounts_[owner.index];
        VELOX_CHECK_GT(remaining, 0);
        if (--remaining == 0) {
          std::vector<uint8_t>{}.swap(copiedRegions_[owner.index]);
        }
      }
    }
  }

  void submitWindow(
      const WindowBatch& batch,
      CudaEvent& event,
      rmm::cuda_stream_view stream,
      bool& cudaWorkMayBePending) {
    VELOX_CHECK(!batch.destinations.empty());
    cudaWorkMayBePending = true;
    for (size_t begin = 0; begin < batch.destinations.size();
         begin += kMaximumCopiesPerBatch) {
      const auto count =
          std::min(kMaximumCopiesPerBatch, batch.destinations.size() - begin);
      addIoCounter(ioStats_, kStagingMemcpyBatchAttempts, 1);
      addIoCounter(ioStats_, kStagingMemcpyBatchCopies, count);
#if CUDART_VERSION >= 13000
      if (!stream.is_default()) {
        // This is the exact compile/runtime gate used by cuDF's
        // memcpy_batch_async wrapper. Recording it here lets a profile prove
        // that a CUDA 13 build reached the native API rather than its loop of
        // cudaMemcpyAsync fallbacks.
        addIoCounter(ioStats_, kStagingNativeMemcpyBatchAttempts, 1);
        addIoCounter(ioStats_, kStagingNativeMemcpyBatchCopies, count);
      }
#endif
      CUDF_CUDA_TRY(
          cudf::detail::memcpy_batch_async(
              batch.destinations.data() + begin,
              batch.sources.data() + begin,
              batch.sizes.data() + begin,
              count,
              stream));
    }
    event.record(stream);
  }

  void submitStagedAndWait(
      PinnedStagingArena::WindowSetLease& windows,
      rmm::cuda_stream_view stream,
      int device) {
    std::array<CudaEvent, PinnedStagingArena::kWindowCount> completionEvents;
    std::array<bool, PinnedStagingArena::kWindowCount> submitted{};
    Cursor cursor;
    bool cudaWorkMayBePending = false;

    try {
      size_t batchIndex = 0;
      while (cursor.descriptor < sizes_.size()) {
        const auto windowIndex = batchIndex % PinnedStagingArena::kWindowCount;
        // Before rolling onto a window again, establish that its previous H2D
        // copy is complete. The other window remains in flight while host pack
        // threads fill this one.
        if (submitted[windowIndex]) {
          const auto waitStart = std::chrono::steady_clock::now();
          completionEvents[windowIndex].synchronize();
          addIoCounter(
              ioStats_,
              kStagingH2DWaitNanos,
              elapsedNanos(waitStart),
              RuntimeCounter::Unit::kNanos);
          submitted[windowIndex] = false;
        }

        auto batch = makeWindowBatch(
            cursor,
            windows.data(windowIndex),
            static_cast<size_t>(windows.capacity()));
        const auto packStart = std::chrono::steady_clock::now();
        windows.pack(windowIndex, batch.packCopies);
        addIoCounter(
            ioStats_,
            kStagingPackNanos,
            elapsedNanos(packStart),
            RuntimeCounter::Unit::kNanos);
        addIoCounter(ioStats_, kStagingWindows, 1);
        // pack() is the source-lifetime boundary: cache pins and owned
        // pageable fallbacks are no longer needed once their last fragment is
        // resident in the leased pinned window.
        releasePackedSources(batch.completedDescriptors);
        submitWindow(
            batch, completionEvents[windowIndex], stream, cudaWorkMayBePending);
        submitted[windowIndex] = true;
        ++batchIndex;
      }

      for (uint32_t windowIndex = 0;
           windowIndex < PinnedStagingArena::kWindowCount;
           ++windowIndex) {
        if (submitted[windowIndex]) {
          const auto waitStart = std::chrono::steady_clock::now();
          completionEvents[windowIndex].synchronize();
          addIoCounter(
              ioStats_,
              kStagingH2DWaitNanos,
              elapsedNanos(waitStart),
              RuntimeCounter::Unit::kNanos);
        }
      }
      windows.release();
    } catch (...) {
      const auto primaryError = std::current_exception();
      if (cudaWorkMayBePending) {
        try {
          synchronizeStream(stream, device);
        } catch (...) {
          // synchronizeStream establishes device completion before throwing,
          // or terminates if safe window reuse cannot be proven.
        }
      }
      windows.release();
      std::rethrow_exception(primaryError);
    }
  }

  void addDescriptor(
      void* destination,
      const void* source,
      size_t size,
      SourceOwner owner) {
    if (size == 0) {
      return;
    }
    VELOX_CHECK_NOT_NULL(destination);
    VELOX_CHECK_NOT_NULL(source);
    VELOX_CHECK_LE(
        size,
        std::numeric_limits<size_t>::max() - totalBytes_,
        "Host-to-device transfer size overflows size_t");
    destinations_.push_back(destination);
    sources_.push_back(source);
    sizes_.push_back(size);
    owners_.push_back(owner);
    totalBytes_ += size;
  }

  std::vector<std::optional<CachedRegion>> retainedRegions_;
  std::vector<std::optional<RetainedBufferedRegion>> bufferedRegions_;
  std::vector<size_t> retainedDescriptorCounts_;
  std::vector<std::vector<uint8_t>> copiedRegions_;
  std::vector<size_t> copiedDescriptorCounts_;
  std::vector<void*> destinations_;
  std::vector<const void*> sources_;
  std::vector<size_t> sizes_;
  std::vector<SourceOwner> owners_;
  std::shared_ptr<IoStats> ioStats_;
  size_t cachedSourceBytes_{0};
  size_t bufferedSourceBytes_{0};
  size_t copiedSourceBytes_{0};
  size_t totalBytes_{0};
};

std::vector<size_t> executeDeviceReadBatch(
    const std::shared_ptr<BufferedInput>& input,
    const std::shared_ptr<std::mutex>& inputMutex,
    const std::vector<DeviceReadRequest>& requests,
    size_t fileSize,
    rmm::cuda_stream_view stream,
    int device,
    const std::shared_ptr<IoStats>& ioStats) {
  addIoCounter(ioStats, kDeviceReadBatches, 1);
  addIoCounter(ioStats, kDeviceReadRequests, requests.size());
  std::vector<size_t> results(requests.size());
  struct PendingRead {
    size_t offset;
    size_t size;
    uint8_t* dst;
    std::unique_ptr<facebook::velox::dwio::common::SeekableInputStream> stream;
  };
  std::vector<PendingRead> pendingReads;
  pendingReads.reserve(requests.size());
  HostToDeviceTransferPlan transfer(ioStats);
  std::optional<PinnedStagingArena::WindowSetLease> stagingWindows;
  size_t totalReadBytes = 0;

  {
    // BufferedInput stores enqueue/load bookkeeping in the input object. Keep
    // batches for the same input atomic while allowing unrelated files and
    // drivers to progress concurrently.
    std::lock_guard<std::mutex> lock(*inputMutex);
    size_t maxReadSize = std::numeric_limits<size_t>::max();
    if (auto* directInput = dynamic_cast<DirectBufferedInput*>(input.get());
        directInput != nullptr && !directInput->preloaded()) {
      // DirectBufferedInput normally prefetches only the first quantum of a
      // large region: selective CPU readers may never consume its tail. cuDF
      // needs every byte of each device read. Enqueue all required quanta so
      // the existing coalescer and I/O executor can load them concurrently,
      // rather than issuing synchronous tail reads while preparing H2D.
      VELOX_CHECK_GT(directInput->loadQuantum(), 0);
      maxReadSize = directInput->loadQuantum();
    }
    for (size_t index = 0; index < requests.size(); ++index) {
      const auto& request = requests[index];
      // Deliberately avoid offset + size: the mathematical end can exceed
      // size_t, while datasource semantics still allow a short (or empty)
      // result. Subtract only after proving that offset is inside the file.
      const auto readSize = request.offset < fileSize
          ? std::min(request.size, fileSize - request.offset)
          : 0;
      results[index] = readSize;
      if (readSize == 0) {
        continue;
      }
      VELOX_CHECK_LE(
          readSize,
          std::numeric_limits<size_t>::max() - totalReadBytes,
          "Total buffered device read size overflows size_t");
      totalReadBytes += readSize;
      // Preserve a region already in the input before loading any new ranges.
      // The next load can otherwise invalidate an enqueue() stream referring
      // to the previous load's buffers.
      if (auto retained =
              input->retainedBufferedRegion(request.offset, readSize)) {
        transfer.addBufferedRegion(std::move(*retained), request.dst);
        continue;
      }
      for (size_t consumed = 0; consumed < readSize;) {
        const auto chunkSize = std::min(readSize - consumed, maxReadSize);
        const auto offset = request.offset + consumed;
        auto inputStream = input->enqueue({offset, chunkSize});
        VELOX_CHECK_NOT_NULL(
            inputStream, "BufferedInput::enqueue returned null stream");
        pendingReads.push_back(
            {offset,
             chunkSize,
             request.dst + consumed,
             std::move(inputStream)});
        consumed += chunkSize;
      }
    }

    // Counts adapter-enqueued ranges, not storage requests: adjacent ranges
    // can still coalesce into one I/O. Original cuDF request counts and result
    // ordering are unchanged.
    addIoCounter(ioStats, kDeviceReadPlannedRanges, pendingReads.size());
    if (!pendingReads.empty()) {
      // CachedBufferedInput may schedule coalesced loads asynchronously, while
      // a singleton demand region may not be scheduled at all. The Next()/
      // readFully() materialization below is the authoritative completion
      // barrier for both cases, and it still runs before staging is acquired.
      input->load(facebook::velox::dwio::common::LogType::FILE);
    }

    addIoCounter(
        ioStats,
        kDeviceReadBytes,
        totalReadBytes,
        RuntimeCounter::Unit::kBytes);

    for (const auto& read : pendingReads) {
      const auto readSize = read.size;
      auto* cacheStream = dynamic_cast<CacheInputStream*>(read.stream.get());
      if (cacheStream == nullptr) {
        if (auto* directStream =
                dynamic_cast<DirectInputStream*>(read.stream.get())) {
          size_t retainedBytes = 0;
          while (retainedBytes < readSize) {
            auto retained = directStream->nextRetained();
            VELOX_CHECK(
                retained.has_value(),
                "Direct input ended after {} of {} bytes",
                retainedBytes,
                readSize);
            const auto bytes = retained->size();
            VELOX_CHECK_GT(bytes, 0, "Direct input returned an empty run");
            VELOX_CHECK_LE(bytes, readSize - retainedBytes);
            transfer.addBufferedRegion(
                std::move(*retained), read.dst + retainedBytes);
            retainedBytes += bytes;
          }
          continue;
        }
        if (auto retained =
                input->retainedBufferedRegion(read.offset, readSize)) {
          transfer.addBufferedRegion(std::move(*retained), read.dst);
          continue;
        }
        std::vector<uint8_t> copied(readSize);
        read.stream->readFully(
            reinterpret_cast<char*>(copied.data()), readSize);
        transfer.addCopiedRegion(std::move(copied), read.dst);
        continue;
      }

      size_t copiedBytes = 0;
      while (copiedBytes < readSize) {
        const void* data = nullptr;
        int runSize = 0;
        VELOX_CHECK(
            cacheStream->Next(&data, &runSize),
            "Cached input ended after {} of {} bytes",
            copiedBytes,
            readSize);
        VELOX_CHECK_GT(runSize, 0, "Cached input returned an empty run");
        VELOX_CHECK_LE(
            static_cast<size_t>(runSize),
            readSize - copiedBytes,
            "Cached input returned bytes beyond the requested region");

        auto retained = cacheStream->retainedRegionForLastNext();
        VELOX_CHECK_EQ(
            retained.size(),
            static_cast<size_t>(runSize),
            "Retained cache region does not match Next() result");
        VELOX_CHECK_EQ(
            retained.ranges().front().data(),
            data,
            "Retained cache region does not begin at the Next() result");
        transfer.addCachedRegion(std::move(retained), read.dst + copiedBytes);
        copiedBytes += runSize;
      }
    }

    // The transfer plan now owns an independent pin for every cache fragment
    // (or a retained allocation / owned copy for non-cache input). Release
    // the input streams and their original pins before H2D begins. Staged
    // transfers release each owner after its last fragment is packed; the
    // direct fallback retains the owners until its CUDA completion fence.
    pendingReads.clear();
  }

  // Only reserve the bounded pinned arena after all storage work is complete
  // and the transfer plan owns exact cache pins or retained/copied buffers.
  // AsyncDataCache entries can be exclusive, stale-sized, cancelled, or
  // evicted between a load barrier and Next(); preparing sources first closes
  // that residency gap and guarantees no remote I/O can hold both windows.
  if (totalReadBytes >= kMinimumPinnedStagingBytes &&
      PinnedStagingArena::enabled()) {
    addIoCounter(ioStats, kStagingAttempts, 1);
    const auto acquireStart = std::chrono::steady_clock::now();
    stagingWindows = PinnedStagingArena::acquirePair();
    addIoCounter(
        ioStats,
        kStagingAcquireNanos,
        elapsedNanos(acquireStart),
        RuntimeCounter::Unit::kNanos);
    if (!stagingWindows.has_value()) {
      addIoCounter(ioStats, kStagingFallbacks, 1);
    } else {
      addIoCounter(
          ioStats,
          kStagingActiveLeasesAtAcquireSamples,
          stagingWindows->activeLeasesAtAcquire());
      addIoCounter(
          ioStats,
          kStagingWindowSetCapacityAtAcquireSamples,
          stagingWindows->windowSetCount());
      if (stagingWindows->wasContended()) {
        addIoCounter(ioStats, kStagingContendedAcquisitions, 1);
      }
    }
  } else if (totalReadBytes >= kMinimumPinnedStagingBytes) {
    addIoCounter(ioStats, kStagingDisabledBypasses, 1);
  } else if (totalReadBytes != 0) {
    addIoCounter(ioStats, kStagingSmallReadBypasses, 1);
  }

  transfer.submitAndWait(stream, device, std::move(stagingWindows));
  return results;
}

} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

std::string normalizeKvikioUri(std::string_view path) {
  constexpr std::string_view kS3aPrefix = "s3a://";
  constexpr std::string_view kS3nPrefix = "s3n://";
  if (path.starts_with(kS3aPrefix) || path.starts_with(kS3nPrefix)) {
    return "s3://" + std::string(path.substr(kS3aPrefix.size()));
  }
  return std::string(path);
}

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<facebook::velox::IoStats> ioStats)
    : input_(std::move(input)),
      ioStats_(std::move(ioStats)),
      fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::device_read(
    size_t offset,
    size_t size,
    rmm::cuda_stream_view stream) {
  const auto readSize =
      offset < fileSize_ ? std::min(size, fileSize_ - offset) : 0;
  rmm::device_buffer result(
      readSize, stream, rmm::mr::get_current_device_resource_ref());
  const auto bytesRead = device_read(
      offset, readSize, static_cast<uint8_t*>(result.data()), stream);
  result.resize(bytesRead, stream);
  return datasource::buffer::create(std::move(result));
}

size_t BufferedInputDataSource::device_read(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  return device_read_async(offset, size, dst, stream).get();
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  const DeviceReadRequest request{offset, size, dst};
  auto completion = device_read_batch_async(
      cudf::host_span<DeviceReadRequest const>{&request, 1}, stream);
  return std::async(
      std::launch::deferred, [completion = std::move(completion)]() mutable {
        auto results = completion.get();
        VELOX_CHECK_EQ(results.size(), 1);
        return results.front();
      });
}

std::future<std::vector<size_t>>
BufferedInputDataSource::device_read_batch_async(
    cudf::host_span<DeviceReadRequest const> requests,
    rmm::cuda_stream_view stream) {
  // Validate the complete descriptor list before scheduling any destination
  // access, and copy it because the caller's span expires on return.
  std::vector<DeviceReadRequest> copiedRequests(
      requests.begin(), requests.end());
  for (const auto& request : copiedRequests) {
    VELOX_CHECK(
        request.size == 0 || request.dst != nullptr,
        "A nonempty device read requires a non-null destination");
  }
  if (copiedRequests.empty()) {
    return std::async(
        std::launch::deferred, [] { return std::vector<size_t>{}; });
  }

  const auto device = getStreamDevice(stream);
  auto const deviceScope =
      rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
  auto completion = cudf::detail::host_worker_pool().submit_task(
      [input = input_,
       inputMutex = inputMutex_,
       ioStats = ioStats_,
       requests = std::move(copiedRequests),
       fileSize = fileSize_,
       stream,
       device]() {
        auto const taskDeviceScope =
            rmm::cuda_set_device_raii{rmm::cuda_device_id{device}};
        return executeDeviceReadBatch(
            input, inputMutex, requests, fileSize, stream, device, ioStats);
      });
  return makeDrainingFuture(std::move(completion));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // read() consults BufferedInput's current merged-region state, which load()
  // replaces. Serialize host/footer reads with device batches for this input.
  std::lock_guard<std::mutex> lock(*inputMutex_);
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

} // namespace facebook::velox::cudf_velox::connector::hive
