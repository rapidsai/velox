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

#include <vector>

namespace facebook::velox::cudf_velox {

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
sort_join_indices(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& leftJoinIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>&& rightJoinIndices,
    rmm::cuda_stream_view stream) {
#if 0
  stream.synchronize();
  auto num_matches = leftJoinIndices->size();
  std::vector<cudf::size_type> h_leftJoinIndices(num_matches, -1);
  cudaMemcpyAsync(h_leftJoinIndices.data(), leftJoinIndices->data(), num_matches * sizeof(cudf::size_type), cudaMemcpyDefault, stream);
  stream.synchronize();
  std::cout << "unsorted h_leftJoinIndices = ";
  for(auto e : h_leftJoinIndices) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
#endif

  thrust::sort_by_key(
      rmm::exec_policy(stream),
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

#if 0
  auto num_matches = leftJoinIndices->size();
  std::vector<cudf::size_type> h_leftJoinIndices(num_matches, false);
  cudaMemcpyAsync(h_leftJoinIndices.data(), leftJoinIndices->data(), num_matches * sizeof(cudf::size_type), cudaMemcpyDefault, stream);
  stream.synchronize();
  std::cout << "sorted h_leftJoinIndices = ";
  for(auto e : h_leftJoinIndices) {
    std::cout << e << " ";
  }
  std::cout << std::endl;

  rmm::device_uvector<cudf::size_type> filter(num_matches, stream);
  thrust::copy_n(rmm::exec_policy(stream), thrust::make_transform_iterator(filterColumn.begin<bool>(), [] __device__(auto b) {return b ? 1 : 0; }), num_matches, filter.begin());
  std::vector<cudf::size_type> h_filter(num_matches, false);
  cudaMemcpyAsync(h_filter.data(), filter.data(), num_matches * sizeof(cudf::size_type), cudaMemcpyDefault, stream);
  stream.synchronize();
  std::cout << "sorted h_filter = ";
  for(auto e : h_filter) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
#endif

  rmm::device_uvector<int> unique_filter(leftTableView.num_rows(), stream, mr);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      leftJoinIndices->begin(),
      leftJoinIndices->end(),
      thrust::make_transform_iterator(
          filterColumn.begin<bool>(),
          [] __device__(auto b) { return b ? 1 : 0; }),
      thrust::make_discard_iterator(),
      unique_filter.begin());

#if 0
  std::vector<int> h_unique_filter(leftTableView.num_rows(), -1);
  cudaMemcpyAsync(h_unique_filter.data(), unique_filter.data(), leftTableView.num_rows() * sizeof(int), cudaMemcpyDefault, stream);
  stream.synchronize();
  std::cout << "h_unique_filter = ";
  for(auto e : h_unique_filter) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
#endif

  auto num_extra_rows = thrust::count_if(
      rmm::exec_policy(stream),
      unique_filter.begin(),
      unique_filter.end(),
      [] __device__(auto b) { return b == 0; });

  // Identify rows from the left table that are false in unique_filter
  rmm::device_uvector<cudf::size_type> extra_rows(num_extra_rows, stream, mr);
  thrust::copy_if(
      rmm::exec_policy(stream),
      thrust::counting_iterator(0),
      thrust::counting_iterator(leftTableView.num_rows()),
      extra_rows.begin(),
      [unique_filter = unique_filter.begin()] __device__(auto i) {
        return !unique_filter[i];
      });
  return extra_rows;
}

void printTable(cudf::table_view const& t, rmm::cuda_stream_view stream) {
  std::cout << t.num_rows() << " " << t.num_columns() << std::endl;
  for (auto i = 0; i < t.num_columns(); i++) {
    auto col = t.column(i);
    std::vector<cudf::size_type> h_col(col.size(), -1);
    cudaMemcpyAsync(
        h_col.data(),
        col.data<cudf::size_type>(),
        col.size() * sizeof(cudf::size_type),
        cudaMemcpyDefault,
        stream);
    stream.synchronize();
    for (auto e : h_col)
      std::cout << e << " ";
    std::cout << std::endl;
  }
}

} // namespace facebook::velox::cudf_velox
