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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderIOHelpers.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <algorithm>
#include <charconv>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using DeviceReadRequest = cudf::io::datasource::device_read_request;

// Keeps the datasource alive while cuDF drains its completion future. The
// explicit destructor is important when the outer deferred future is discarded:
// lambda-capture destruction order alone cannot provide this lifetime rule.
class RetainedReadCompletion {
 public:
  RetainedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      std::future<void> completion)
      : dataSource_(std::move(dataSource)),
        completion_(std::move(completion)) {}

  RetainedReadCompletion(RetainedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        completion_(std::move(other.completion_)) {}

  RetainedReadCompletion(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(const RetainedReadCompletion&) = delete;
  RetainedReadCompletion& operator=(RetainedReadCompletion&&) = delete;

  ~RetainedReadCompletion() noexcept {
    if (completion_.valid()) {
      try {
        completion_.get();
      } catch (...) {
      }
    }
  }

  void get() {
    completion_.get();
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::future<void> completion_;
};

class BufferedReadCompletion {
 public:
  BufferedReadCompletion(
      std::shared_ptr<cudf::io::datasource> dataSource,
      std::future<std::vector<size_t>> completion,
      std::vector<size_t> expectedSizes)
      : dataSource_(std::move(dataSource)),
        completion_(std::move(completion)),
        expectedSizes_(std::move(expectedSizes)) {}

  BufferedReadCompletion(BufferedReadCompletion&& other) noexcept
      : dataSource_(std::move(other.dataSource_)),
        completion_(std::move(other.completion_)),
        expectedSizes_(std::move(other.expectedSizes_)) {}

  BufferedReadCompletion(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(const BufferedReadCompletion&) = delete;
  BufferedReadCompletion& operator=(BufferedReadCompletion&&) = delete;

  ~BufferedReadCompletion() noexcept {
    if (completion_.valid()) {
      try {
        std::ignore = completion_.get();
      } catch (...) {
      }
    }
  }

  void get() {
    const auto actualSizes = completion_.get();
    VELOX_CHECK_EQ(
        actualSizes.size(),
        expectedSizes_.size(),
        "Buffered device batch returned the wrong number of results");
    for (size_t index = 0; index < actualSizes.size(); ++index) {
      VELOX_CHECK_EQ(
          actualSizes[index],
          expectedSizes_[index],
          "Buffered device read was unexpectedly short");
    }
  }

 private:
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::future<std::vector<size_t>> completion_;
  std::vector<size_t> expectedSizes_;
};

} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

std::optional<size_t> knownKvikioFileSize(const CudfHiveConnectorSplit& split) {
  // Presto forwards HiveFileSplit.fileSize as this synthesized column. It is
  // object metadata, not the byte range assigned to this particular split.
  const auto it = split.infoColumns.find("$file_size");
  if (it == split.infoColumns.end() || it->second.empty()) {
    return std::nullopt;
  }
  const auto& text = it->second;
  int64_t parsedSize;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsedSize);
  if (error != std::errc{} || end != text.data() + text.size() ||
      parsedSize < 0) {
    return std::nullopt;
  }
  const auto fileSize = static_cast<uint64_t>(parsedSize);
  if (fileSize > std::numeric_limits<size_t>::max() || split.start > fileSize ||
      (split.length != std::numeric_limits<uint64_t>::max() &&
       split.length > fileSize - split.start)) {
    return std::nullopt;
  }
  return static_cast<size_t>(fileSize);
}

void ByteRangeFetch::wait() {
  if (pending.valid()) {
    pending.get();
  }
}

void ByteRangeFetch::abandon() {
  wait();
}

std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info>
        pageIndexByteRanges) {
  std::vector<std::reference_wrapper<cudf::io::datasource>> dataSources{
      std::ref(*dataSource)};
  auto buffers = cudf::io::parquet::fetch_page_indexes_to_host(
      dataSources, pageIndexByteRanges);

  std::vector<cudf::host_span<const uint8_t>> spans;
  spans.reserve(buffers.size());
  std::transform(
      buffers.begin(),
      buffers.end(),
      std::back_inserter(spans),
      [](const auto& buffer) {
        return cudf::host_span<const uint8_t>{*buffer};
      });

  return {std::move(buffers), std::move(spans)};
}

ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto bufferedInput = dynamic_cast<BufferedInputDataSource*>(dataSource.get());
  if (bufferedInput == nullptr) {
    // cuDF owns the generic KvikIO/host implementation. Delegating avoids a
    // second copy of its coalescing, short-read validation, failure draining,
    // and host-buffer lifetime rules at the Velox boundary.
    auto [buffers, spans, completion] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
            *dataSource, byteRanges, stream, mr);
    auto completionState =
        RetainedReadCompletion{std::move(dataSource), std::move(completion)};
    auto retainedCompletion = std::async(
        std::launch::deferred,
        [completionState = std::move(completionState)]() mutable {
          completionState.get();
        });
    return {
        .buffers = std::move(buffers),
        .data = std::move(spans),
        .pending = std::move(retainedCompletion)};
  }

  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  std::vector<cudf::device_span<const uint8_t>> columnChunkData;
  columnChunkData.reserve(byteRanges.size());

  // Validate before arithmetic or allocation. byte_range_info uses signed
  // fields, so accumulating an invalid negative size into size_t would
  // otherwise underflow before the request builder rejected it.
  std::vector<size_t> rangeOffsets;
  std::vector<size_t> rangeSizes;
  rangeOffsets.reserve(byteRanges.size());
  rangeSizes.reserve(byteRanges.size());
  size_t totalSize = 0;
  for (const auto& byteRange : byteRanges) {
    const auto offset = byteRange.offset();
    const auto size = byteRange.size();
    VELOX_CHECK_GE(offset, 0, "Device read offset must be nonnegative");
    VELOX_CHECK_GE(size, 0, "Device read size must be nonnegative");
    VELOX_CHECK_LE(
        static_cast<uint64_t>(offset),
        std::numeric_limits<size_t>::max(),
        "Device read offset does not fit size_t");
    VELOX_CHECK_LE(
        static_cast<uint64_t>(size),
        std::numeric_limits<size_t>::max(),
        "Device read size does not fit size_t");
    const auto requestSize = static_cast<size_t>(size);
    VELOX_CHECK_LE(
        requestSize,
        std::numeric_limits<size_t>::max() - totalSize,
        "Total device read size overflows size_t");
    rangeOffsets.push_back(static_cast<size_t>(offset));
    rangeSizes.push_back(requestSize);
    totalSize += requestSize;
  }
  VELOX_CHECK_LE(
      totalSize,
      std::numeric_limits<size_t>::max() - (kBufferPaddingMultiple - 1),
      "Padded device read size overflows size_t");

  // Allocate one device buffer for all column chunks.
  std::vector<rmm::device_buffer> columnChunkBuffers;
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  auto* bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      rangeSizes.begin(),
      rangeSizes.end(),
      std::size_t{0},
      [&](auto offset, auto rangeSize) {
        // A nonempty list of empty ranges has a zero-byte allocation, whose
        // data pointer is null. Even adding zero to null is undefined.
        auto* rangeData = bufferData == nullptr ? nullptr : bufferData + offset;
        columnChunkData.emplace_back(rangeData, rangeSize);
        return offset + rangeSize;
      });

  // Submit one immutable batch so cached fragments share the pinned staging
  // transfer and completion lifetime.
  std::vector<DeviceReadRequest> requests;
  std::vector<size_t> expectedSizes;
  requests.reserve(byteRanges.size());
  expectedSizes.reserve(byteRanges.size());
  for (size_t index = 0; index < byteRanges.size(); ++index) {
    requests.push_back(
        {rangeOffsets[index],
         rangeSizes[index],
         const_cast<uint8_t*>(columnChunkData[index].data())});
    expectedSizes.push_back(rangeSizes[index]);
  }

  auto batchCompletion = cudf::io::device_read_batch_async(
      *bufferedInput,
      cudf::host_span<DeviceReadRequest const>{
          requests.data(), requests.size()},
      stream);
  auto completionState = BufferedReadCompletion{
      std::move(dataSource),
      std::move(batchCompletion),
      std::move(expectedSizes)};
  auto completion = std::async(
      std::launch::deferred,
      [completionState = std::move(completionState)]() mutable {
        completionState.get();
      });
  return {
      .buffers = std::move(columnChunkBuffers),
      .data = std::move(columnChunkData),
      .pending = std::move(completion)};
}

} // namespace facebook::velox::cudf_velox::connector::hive
