/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/core/Expressions.h"

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

namespace {

std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

cudf::size_type getLeadLagOffset(const core::WindowNode::Function& func) {
  const auto& args = func.functionCall->inputs();
  if (args.size() >= 2) {
    if (auto constExpr =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                args[1])) {
      if (constExpr->hasValueVector()) {
        return constExpr->valueVector()
            ->as<SimpleVector<int64_t>>()
            ->valueAt(0);
      }
      return constExpr->value().value<int64_t>();
    }
  }
  return 1;
}

cudf::rank_method toRankMethod(const std::string& baseName) {
  if (baseName == "row_number") {
    return cudf::rank_method::FIRST;
  } else if (baseName == "rank") {
    return cudf::rank_method::MIN;
  } else if (baseName == "dense_rank") {
    return cudf::rank_method::DENSE;
  }
  VELOX_FAIL("toRankMethod called on non-rank function: {}", baseName);
}

} // namespace

cudf::column_view CudfWindow::multiSortKeyStructView(
    cudf::table_view const& sortedInput) const {
  VELOX_CHECK_GE(
      sortKeyIndices_.size(), 2,
      "multiSortKeyStructView requires >= 2 sort keys");
  sortKeyStructChildren_.clear();
  sortKeyStructChildren_.reserve(sortKeyIndices_.size());
  for (auto ch : sortKeyIndices_) {
    sortKeyStructChildren_.push_back(sortedInput.column(ch));
  }
  return cudf::column_view(
      cudf::data_type{cudf::type_id::STRUCT},
      sortedInput.num_rows(),
      nullptr,
      nullptr,
      0,
      0,
      sortKeyStructChildren_);
}

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    const std::string& baseName,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  auto method = toRankMethod(baseName);

  // Build the "values" column for rank tie detection.
  cudf::column_view valuesCol = [&]() -> cudf::column_view {
    if (sortKeyIndices_.empty()) {
      return sortedInput.column(0);
    }
    if (sortKeyIndices_.size() == 1 || baseName == "row_number") {
      return sortedInput.column(sortKeyIndices_[0]);
    }
    return multiSortKeyStructView(sortedInput);
  }();

  auto colOrder = sortKeyIndices_.empty()
      ? cudf::order::ASCENDING
      : sortOrders_[0];
  auto nullOrd = sortKeyIndices_.empty()
      ? cudf::null_order::BEFORE
      : nullOrders_[0];

  if (partitionKeyIndices_.empty()) {
    auto agg = cudf::make_rank_aggregation<cudf::scan_aggregation>(
        method, colOrder, cudf::null_policy::INCLUDE, nullOrd);
    return cudf::scan(
        valuesCol, *agg, cudf::scan_type::INCLUSIVE,
        cudf::null_policy::INCLUDE, stream, mr);
  }

  auto partCols = sortedInput.select(partitionKeyIndices_);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = valuesCol;
  requests[0].aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
          method, colOrder, cudf::null_policy::INCLUDE, nullOrd));

  cudf::groupby::groupby grouper(
      cudf::table_view(partCols),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyIndices_.size(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyIndices_.size(), cudf::null_order::BEFORE));

  auto scanResult = grouper.scan(requests, stream, mr);
  auto& aggResults = scanResult.second;
  VELOX_CHECK_EQ(aggResults.size(), 1);
  VELOX_CHECK_EQ(aggResults[0].results.size(), 1);
  return std::move(aggResults[0].results[0]);
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : exec::Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "CudfWindow"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0},
          operatorId,
          fmt::format("[{}]", windowNode->id())),
      windowNode_(windowNode) {
  const auto& inputType = windowNode->inputType();

  for (const auto& key : windowNode->partitionKeys()) {
    partitionKeyIndices_.push_back(inputType->getChildIdx(key->name()));
  }

  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    sortKeyIndices_.push_back(
        inputType->getChildIdx(windowNode->sortingKeys()[i]->name()));
    const auto& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING
                            : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfWindow::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "CudfWindow expects CudfVector input");

  stream_ = cudfInput->stream();
  pool_ = cudfInput->pool();
  auto mr = cudf::get_current_device_resource_ref();

  if (!accumulatedData_) {
    accumulatedData_ =
        std::make_unique<cudf::table>(cudfInput->getTableView(), stream_, mr);
  } else {
    std::vector<cudf::table_view> views = {
        accumulatedData_->view(), cudfInput->getTableView()};
    accumulatedData_ = cudf::concatenate(views, stream_, mr);
  }
}

void CudfWindow::noMoreInput() {
  Operator::noMoreInput();
  if (!accumulatedData_ || accumulatedData_->num_rows() == 0) {
    finished_ = true;
  }
}

bool CudfWindow::isFinished() {
  return finished_;
}

RowVectorPtr CudfWindow::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (!accumulatedData_ || accumulatedData_->num_rows() == 0) {
    finished_ = true;
    return nullptr;
  }

  auto mr = cudf::get_current_device_resource_ref();
  auto allView = accumulatedData_->view();

  // 1. Sort by partition keys + sort keys if not already sorted.
  std::unique_ptr<cudf::table> sortedData;
  cudf::table_view sortedView;

  if (!windowNode_->inputsSorted()) {
    std::vector<cudf::size_type> allSortKeys;
    std::vector<cudf::order> allOrders;
    std::vector<cudf::null_order> allNullOrders;

    for (auto idx : partitionKeyIndices_) {
      allSortKeys.push_back(idx);
      allOrders.push_back(cudf::order::ASCENDING);
      allNullOrders.push_back(cudf::null_order::BEFORE);
    }
    for (size_t i = 0; i < sortKeyIndices_.size(); ++i) {
      allSortKeys.push_back(sortKeyIndices_[i]);
      allOrders.push_back(sortOrders_[i]);
      allNullOrders.push_back(nullOrders_[i]);
    }

    auto keyTable = allView.select(allSortKeys);
    auto indices = cudf::stable_sorted_order(
        keyTable, allOrders, allNullOrders, stream_, mr);
    sortedData = cudf::detail::gather(
        allView,
        indices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::detail::negative_index_policy::NOT_ALLOWED,
        stream_,
        mr);
    sortedView = sortedData->view();
  } else {
    sortedView = allView;
  }

  // 2. Build partition key table for grouped_rolling_window.
  auto partKeys = sortedView.select(partitionKeyIndices_);

  // 3. Evaluate each window function and collect result columns.
  std::vector<std::unique_ptr<cudf::column>> windowResultCols;
  const auto& funcs = windowNode_->windowFunctions();

  for (const auto& func : funcs) {
    const auto baseName = getBaseFunctionName(func.functionCall->name());

    cudf::size_type inputColIdx = 0;
    if (!func.functionCall->inputs().empty()) {
      if (auto field =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  func.functionCall->inputs()[0])) {
        inputColIdx = windowNode_->inputType()->getChildIdx(field->name());
      }
    }
    auto inputCol = sortedView.column(inputColIdx);

    if (baseName == "lag") {
      auto offset = getLeadLagOffset(func);
      auto agg = cudf::make_lag_aggregation<cudf::rolling_aggregation>(offset);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, offset + 1, 0, offset + 1, *agg, stream_, mr));
    } else if (baseName == "lead") {
      auto offset = getLeadLagOffset(func);
      auto agg =
          cudf::make_lead_aggregation<cudf::rolling_aggregation>(offset);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, 0, offset + 1, offset + 1, *agg, stream_, mr));
    } else if (baseName == "first_value") {
      auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
          0,
          func.ignoreNulls ? cudf::null_policy::EXCLUDE
                           : cudf::null_policy::INCLUDE);
      auto unbounded = cudf::window_bounds::unbounded();
      auto current = cudf::window_bounds::get(1);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, unbounded, current, 1, *agg, stream_, mr));
    } else if (baseName == "last_value") {
      auto agg = cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
          -1,
          func.ignoreNulls ? cudf::null_policy::EXCLUDE
                           : cudf::null_policy::INCLUDE);
      auto unbounded = cudf::window_bounds::unbounded();
      auto current = cudf::window_bounds::get(1);
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, current, unbounded, 1, *agg, stream_, mr));
    } else if (
        baseName == "row_number" || baseName == "rank" ||
        baseName == "dense_rank") {
      windowResultCols.push_back(
          computeRankColumn(sortedView, baseName, stream_));
    } else if (
        baseName == "sum" || baseName == "min" || baseName == "max" ||
        baseName == "count" || baseName == "avg") {
      std::unique_ptr<cudf::rolling_aggregation> agg;
      if (baseName == "sum") {
        agg = cudf::make_sum_aggregation<cudf::rolling_aggregation>();
      } else if (baseName == "min") {
        agg = cudf::make_min_aggregation<cudf::rolling_aggregation>();
      } else if (baseName == "max") {
        agg = cudf::make_max_aggregation<cudf::rolling_aggregation>();
      } else if (baseName == "count") {
        agg = cudf::make_count_aggregation<cudf::rolling_aggregation>(
            cudf::null_policy::EXCLUDE);
      } else {
        agg = cudf::make_mean_aggregation<cudf::rolling_aggregation>();
      }
      auto bounds = cudf::window_bounds::unbounded();
      windowResultCols.push_back(cudf::grouped_rolling_window(
          partKeys, inputCol, bounds, bounds, 1, *agg, stream_, mr));
    } else {
      VELOX_FAIL("Unsupported window function for GPU: {}", baseName);
    }
  }

  // 4. Build the output table: input columns + window result columns.
  auto& dataOwner = sortedData ? sortedData : accumulatedData_;
  auto sortedCols = dataOwner->release();
  for (auto& wc : windowResultCols) {
    sortedCols.push_back(std::move(wc));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(sortedCols));
  auto resultSize = resultTable->num_rows();

  accumulatedData_.reset();
  finished_ = true;
  return std::make_shared<CudfVector>(
      pool_, outputType_, resultSize, std::move(resultTable), stream_);
}

} // namespace facebook::velox::cudf_velox
