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

#include <nvtx3/nvtx3.hpp>

#include <thrust/count.h>
#include <thrust/sort.h>

#include <rmm/exec_policy.hpp>
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
    rmm::device_async_resource_ref mr) {
  
  auto leftIndicesSpan = leftJoinIndices
      ? cudf::device_span<cudf::size_type const>{*leftJoinIndices}
      : cudf::device_span<cudf::size_type const>{};
  auto rightIndicesSpan = rightJoinIndices
      ? cudf::device_span<cudf::size_type const>{*rightJoinIndices}
      : cudf::device_span<cudf::size_type const>{};
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
  auto constexpr oobPolicy = cudf::out_of_bounds_policy::NULLIFY;

  std::vector<std::unique_ptr<cudf::column>> joinedCols;

  // for inner join, apply the filter if one exists
  if (joinNode_->filter() && joinNode_->isInnerJoin()) {
    auto leftResult =
        cudf::gather(leftTableView, leftIndicesCol, oobPolicy, stream);
    auto rightResult =
        cudf::gather(rightTableView, rightIndicesCol, oobPolicy, stream);
    auto leftColsSize = leftResult->num_columns();
    auto rightColsSize = rightResult->num_columns();

    joinedCols = leftResult->release();
    auto rightCols = rightResult->release();
    joinedCols.insert(
        joinedCols.end(),
        std::make_move_iterator(rightCols.begin()),
        std::make_move_iterator(rightCols.end()));

    auto probeType = joinNode_->sources()[0]->outputType();
    auto buildType = joinNode_->sources()[1]->outputType();
    std::vector<velox::RowTypePtr> rowTypes{probeType, buildType};

    exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);
    auto filterEvaluator = ExpressionEvaluator(
        {exprs.exprs()[0]}, facebook::velox::type::concatRowTypes(rowTypes));
    auto filterColumns = filterEvaluator.compute(
        joinedCols, stream, mr);
    auto filterColumn = filterColumns[0]->view();
    // is all true in filter_column
    auto isAllTrue = cudf::reduce(
        filterColumn,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    using ScalarType = cudf::scalar_type_t<bool>;
    auto result = static_cast<ScalarType*>(isAllTrue.get());

    // If filter is not all true, apply the filter
    if (!(result->is_valid(stream) && result->value(stream))) {
      // apply the filter
      auto filterTable = std::make_unique<cudf::table>(std::move(joinedCols));
      auto filteredTable =
          cudf::apply_boolean_mask(*filterTable, filterColumn, stream);
      joinedCols = filteredTable->release();
    }

    auto filteredjoinedCols =
        std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
    for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
      filteredjoinedCols[leftColumnOutputIndices_[i]] =
          std::move(joinedCols[leftColumnIndicesToGather_[i]]);
    }
    for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
      filteredjoinedCols[rightColumnOutputIndices_[i]] =
          std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
    }

    swap(joinedCols, filteredjoinedCols);
  }
  else if(joinNode_->filter() && joinNode_->isLeftJoin()) {
    // for left join, we need to ensure that all rows in the left table exist after the 
    // filter is applied
    thrust::sort_by_key(rmm::exec_policy_nosync(stream), leftJoinIndices->begin(), leftJoinIndices->end(), rightJoinIndices->begin());
    auto leftResult =
        cudf::gather(leftTableView, leftIndicesCol, oobPolicy, stream);
    auto rightResult =
        cudf::gather(rightTableView, rightIndicesCol, oobPolicy, stream);
    auto leftColsSize = leftResult->num_columns();
    auto rightColsSize = rightResult->num_columns();

    std::vector<std::unique_ptr<cudf::column>> joinedCols =
        leftResult->release();
    auto rightCols = rightResult->release();
    joinedCols.insert(
        joinedCols.end(),
        std::make_move_iterator(rightCols.begin()),
        std::make_move_iterator(rightCols.end()));

    auto probeType = joinNode_->sources()[0]->outputType();
    auto buildType = joinNode_->sources()[1]->outputType();
    std::vector<velox::RowTypePtr> rowTypes{probeType, buildType};

    exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);
    auto filterEvaluator = ExpressionEvaluator(
        {exprs.exprs()[0]}, facebook::velox::type::concatRowTypes(rowTypes));
    auto filterColumns = filterEvaluator.compute(
        joinedCols, stream, mr);
    auto filterColumn = filterColumns[0]->view();
    // is all true in filter_column
    auto isAllTrue = cudf::reduce(
        filterColumn,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    using ScalarType = cudf::scalar_type_t<bool>;
    auto result = static_cast<ScalarType*>(isAllTrue.get());

    // If filter is not all true, apply the filter
    if (!(result->is_valid(stream) && result->value(stream))) {
      // 1. Remove all filtered rows
      // 2. Re insert rows from left table if they are missing
      rmm::device_uvector<bool> unique_filter(leftTableView.num_rows(), stream, mr);
      cudf::device_span<bool const> filter_column_span = filterColumn;
      thrust::reduce_by_key(rmm::exec_policy_nosync(stream), leftJoinIndices->begin(), leftJoinIndices->end(), 
          filter_column_span.begin(), thrust::make_discard_iterator(), unique_filter.begin(), cuda::std::equal_to{}, 
          cuda::std::logical_or{});
      auto num_extra_rows = thrust::count_if(rmm::exec_policy(stream), unique_filter.begin(), unique_filter.end(), [] __device__ (auto b) {
            return !b;
          });
      
      // Identify rows from the left table that are false in unique_filter
      rmm::device_uvector<cudf::size_type> extra_rows(num_extra_rows, stream, mr);
      thrust::copy_if(rmm::exec_policy_nosync(stream), thrust::counting_iterator(0), thrust::counting_iterator(leftTableView.num_rows()), 
          extra_rows.begin(), 
          [unique_filter = unique_filter.begin()] __device__(auto i) {
            return !unique_filter[i];
          });
      cudf::device_span<cudf::size_type const> extra_rows_span{extra_rows};
      auto left_extra_result = cudf::gather(leftTableView, cudf::column_view{extra_rows_span}, oobPolicy, stream);
      auto extra_columns = left_extra_result->release();
      for(auto col = 0; col < rightColsSize; col++) {
        auto null_scalar = cudf::make_empty_scalar_like(joinedCols[col]->view(), stream);
        //auto res = cudf::copy_if_else(joinedCols[col]->view(), *null_scalar, filterColumn, stream);
        extra_columns.push_back(cudf::make_column_from_scalar(*null_scalar, num_extra_rows, stream, mr);
      }
      auto extra_table = std::make_unique<cudf::table>(std::move(extra_columns));
      
      // Apply the Filter
      auto filterTable = std::make_unique<cudf::table>(std::move(joinedCols));
      auto filteredTable =
          cudf::apply_boolean_mask(*filterTable, filterColumn, stream);
      std::vector<cudf::table_view> concat_table_views;
      concat_table_views.push_back(filteredTable->view());
      concat_table_views.push_back(extra_table->view());
      auto filteredLeftJoinTable = cudf::concatenate(concat_table_views, stream, mr);

      joinedCols = filteredLeftJoinTable->release();
    }

    auto filteredjoinedCols =
        std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
    for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
      filteredjoinedCols[leftColumnOutputIndices_[i]] =
          std::move(joinedCols[leftColumnIndicesToGather_[i]]);
    }
    for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
      filteredjoinedCols[rightColumnOutputIndices_[i]] =
          std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
    }
    
    swap(joinedCols, filteredjoinedCols);
  }
  else {
    auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
    auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
    auto leftResult = cudf::gather(leftInput, leftIndicesCol, oobPolicy, stream);
    auto rightResult =
        cudf::gather(rightInput, rightIndicesCol, oobPolicy, stream);

    auto leftCols = leftResult->release();
    auto rightCols = rightResult->release();
    auto joinedCols =
        std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
    for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
      joinedCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
    }
    for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
      joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
    }
  }

  auto cudfOutput =
      std::make_unique<cudf::table>(std::move(joinedCols));
  stream.synchronize();
  return std::move(cudfOutput);
}


}
