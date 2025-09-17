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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <vector>

namespace facebook::velox::cudf_velox {

std::unique_ptr<cudf::table> create_joined_table(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightJoinIndices,
    cudf::table_view const& leftTableView,
    cudf::table_view const& rightTableView,
    std::vector<cudf::size_type> const& leftColumnIndicesToGather_,
    std::vector<cudf::size_type> const& rightColumnIndicesToGather_,
    std::vector<size_t> const& leftColumnOutputIndices_,
    std::vector<size_t> const& rightColumnOutputIndices_,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
sort_join_indices(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightJoinIndices,
    rmm::cuda_stream_view stream);

std::vector<std::unique_ptr<cudf::column>> filter_left_joined_cols(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightJoinIndices,
    cudf::table_view const& leftTableView,
    cudf::table_view const& rightTableView,
    cudf::column_view const& filterColumn,
    rmm::cuda_stream_view stream);

rmm::device_uvector<cudf::size_type> filter_left_joined_cols(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    cudf::table_view const& leftTableView,
    cudf::column_view const& filterColumn,
    rmm::cuda_stream_view stream);
} // namespace facebook::velox::cudf_velox
