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

#include "velox/experimental/cudf/exec/Helpers.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvtx3/nvtx3.hpp>
#include <thrust/count.h>
#include <thrust/sort.h>

namespace facebook::velox::cudf_velox {

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
sort_join_indices(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightJoinIndices,
    rmm::cuda_stream_view stream) {
  thrust::sort_by_key(
      rmm::exec_policy_nosync(stream),
      leftJoinIndices->begin(),
      leftJoinIndices->end(),
      rightJoinIndices->begin());
  return {std::move(leftJoinIndices), std::move(rightJoinIndices)};
}

rmm::device_uvector<cudf::size_type> filter_left_joined_cols(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    cudf::table_view const& leftTableView,
    cudf::column_view const& filterColumn,
    rmm::cuda_stream_view stream) {
  // 1. Remove all filtered rows
  // 2. Re insert rows from left table if they are missing
  auto mr = cudf::get_current_device_resource_ref();

  rmm::device_uvector<bool> unique_filter(leftTableView.num_rows(), stream, mr);
  thrust::reduce_by_key(
      rmm::exec_policy_nosync(stream),
      leftJoinIndices->begin(),
      leftJoinIndices->end(),
      filterColumn.begin<bool>(),
      thrust::make_discard_iterator(),
      unique_filter.begin(),
      cuda::std::equal_to{},
      cuda::std::logical_or{});
  auto num_extra_rows = thrust::count_if(
      rmm::exec_policy(stream),
      unique_filter.begin(),
      unique_filter.end(),
      [] __device__(auto b) { return !b; });

  // Identify rows from the left table that are false in unique_filter
  rmm::device_uvector<cudf::size_type> extra_rows(num_extra_rows, stream, mr);
  thrust::copy_if(
      rmm::exec_policy_nosync(stream),
      thrust::counting_iterator(0),
      thrust::counting_iterator(leftTableView.num_rows()),
      extra_rows.begin(),
      [unique_filter = unique_filter.begin()] __device__(auto i) {
        return !unique_filter[i];
      });
  return extra_rows;
}

} // namespace facebook::velox::cudf_velox
