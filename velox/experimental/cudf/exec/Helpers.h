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

#include <cudf/types.hpp>
#include <rmm/device_uvector.hpp>

namespace facebook::velox::cudf_velox {
  
std::unique_ptr<cudf::table> create_joined_table(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> &&leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> &&rightJoinIndices,
    cudf::table_view const &leftTableView,
    cudf::table_view const &rightTableView,
    std::vector<cudf::size_type> const &leftColumnIndicesToGather_,
    std::vector<cudf::size_type> const &rightColumnIndicesToGather_,
    std::vector<size_t> const &leftColumnOutputIndices_,
    std::vector<size_t> const &rightColumnOutputIndices_,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
}
