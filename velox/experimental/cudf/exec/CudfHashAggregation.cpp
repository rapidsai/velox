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

#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

namespace {

using namespace facebook::velox;

#define DEFINE_SIMPLE_AGGREGATOR(Name, name, KIND)                            \
  struct Name##Aggregator : cudf_velox::CudfHashAggregation::Aggregator {     \
    Name##Aggregator(                                                         \
        core::AggregationNode::Step step,                                     \
        uint32_t inputIndex,                                                  \
        VectorPtr constant,                                                   \
        bool is_global)                                                       \
        : Aggregator(                                                         \
              step,                                                           \
              cudf::aggregation::KIND,                                        \
              inputIndex,                                                     \
              constant,                                                       \
              is_global) {}                                                   \
                                                                              \
    void addGroupbyRequest(                                                   \
        cudf::table_view const& tbl,                                          \
        std::vector<cudf::groupby::aggregation_request>& requests) override { \
      VELOX_CHECK(                                                            \
          constant == nullptr,                                                \
          #Name "Aggregator does not yet support constant input");            \
      auto& request = requests.emplace_back();                                \
      output_idx = requests.size() - 1;                                       \
      request.values = tbl.column(inputIndex);                                \
      request.aggregations.push_back(                                         \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>());      \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> makeOutputColumn(                           \
        std::vector<cudf::groupby::aggregation_result>& results,              \
        rmm::cuda_stream_view stream) override {                              \
      return std::move(results[output_idx].results[0]);                       \
    }                                                                         \
                                                                              \
    std::unique_ptr<cudf::column> doReduce(                                   \
        cudf::table_view const& input,                                        \
        TypePtr const& output_type,                                           \
        rmm::cuda_stream_view stream) override {                              \
      auto const agg_request =                                                \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>();        \
      auto const cudf_output_type =                                           \
          cudf::data_type(cudf_velox::velox_to_cudf_type_id(output_type));    \
      auto const result_scalar = cudf::reduce(                                \
          input.column(inputIndex), *agg_request, cudf_output_type, stream);  \
      return cudf::make_column_from_scalar(*result_scalar, 1, stream);        \
    }                                                                         \
                                                                              \
   private:                                                                   \
    uint32_t output_idx;                                                      \
  };

DEFINE_SIMPLE_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_AGGREGATOR(Max, max, MAX)

struct CountAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  CountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool is_global)
      : Aggregator(
            step,
            cudf::aggregation::COUNT_VALID,
            inputIndex,
            constant,
            is_global) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    auto& request = requests.emplace_back();
    output_idx = requests.size() - 1;
    request.values = tbl.column(constant == nullptr ? inputIndex : 0);
    std::unique_ptr<cudf::groupby_aggregation> agg_request =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              constant == nullptr ? cudf::null_policy::EXCLUDE
                                  : cudf::null_policy::INCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    request.aggregations.push_back(std::move(agg_request));
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& output_type,
      rmm::cuda_stream_view stream) override {
    if (exec::isRawInput(step)) {
      // For raw input, implement count using size + null count
      auto input_col = input.column(constant == nullptr ? inputIndex : 0);

      // count_valid: size - null_count, count_all: just the size
      int64_t count = constant == nullptr
          ? input_col.size() - input_col.null_count()
          : input_col.size();

      auto result_scalar = cudf::numeric_scalar<int64_t>(count);

      return cudf::make_column_from_scalar(result_scalar, 1, stream);
    } else {
      // For non-raw input (intermediate/final), use sum aggregation
      auto const agg_request =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto const cudf_output_type = cudf::data_type(cudf::type_id::INT64);
      auto const result_scalar = cudf::reduce(
          input.column(inputIndex), *agg_request, cudf_output_type, stream);
      return cudf::make_column_from_scalar(*result_scalar, 1, stream);
    }
    return nullptr;
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    // cudf produces int32 for count(0) but velox expects int64
    auto col = std::move(results[output_idx].results[0]);
    if (constant != nullptr &&
        col->type() == cudf::data_type(cudf::type_id::INT32)) {
      col = cudf::cast(*col, cudf::data_type(cudf::type_id::INT64), stream);
    }
    return col;
  }

 private:
  uint32_t output_idx;
};

struct MeanAggregator : cudf_velox::CudfHashAggregation::Aggregator {
  MeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      bool is_global)
      : Aggregator(
            step,
            cudf::aggregation::MEAN,
            inputIndex,
            constant,
            is_global) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto& request = requests.emplace_back();
        mean_idx = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto& request = requests.emplace_back();
        sum_idx = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        break;
      }
      case core::AggregationNode::Step::kFinal: {
        // In final aggregation, the previously computed sum and count are in
        // the child columns of the input column.
        auto& request = requests.emplace_back();
        sum_idx = requests.size() - 1;
        request.values = tbl.column(inputIndex).child(0);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto& request2 = requests.emplace_back();
        count_idx = requests.size() - 1;
        request2.values = tbl.column(inputIndex).child(1);
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        request2.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        // We don't know how to handle kIntermediate step for mean
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[mean_idx].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sum_idx].results[0]);
        auto count = std::move(results[sum_idx].results[1]);

        auto const size = sum->size();

        auto count_int64 =
            cudf::cast(*count, cudf::data_type(cudf::type_id::INT64), stream);

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count_int64));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = std::move(results[sum_idx].results[0]);
        auto count = std::move(results[count_idx].results[0]);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            // TODO: Change the output type to be dependent on the input type
            // like in the cudf groupby implementation.
            cudf::data_type(cudf::type_id::FLOAT64),
            stream);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& output_type,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto const agg_request =
            cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        auto const cudf_output_type =
            cudf::data_type(cudf_velox::velox_to_cudf_type_id(output_type));
        auto const result_scalar = cudf::reduce(
            input.column(inputIndex), *agg_request, cudf_output_type, stream);
        return cudf::make_column_from_scalar(*result_scalar, 1, stream);
      }
      case core::AggregationNode::Step::kPartial: {
        VELOX_CHECK(output_type->isRow());
        auto const& row_type = output_type->asRow();
        auto const sum_type = row_type.childAt(0);
        auto const count_type = row_type.childAt(1);
        auto const cudf_sum_type =
            cudf::data_type(cudf_velox::velox_to_cudf_type_id(sum_type));
        auto const cudf_count_type =
            cudf::data_type(cudf_velox::velox_to_cudf_type_id(count_type));

        // sum
        auto const agg_request =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sum_result_scalar = cudf::reduce(
            input.column(inputIndex), *agg_request, cudf_sum_type, stream);
        auto sum_col =
            cudf::make_column_from_scalar(*sum_result_scalar, 1, stream);

        // libcudf doesn't have a count agg for reduce. What we want is to
        // count the number of valid rows.
        auto count_col = cudf::make_column_from_scalar(
            cudf::numeric_scalar<int64_t>(
                input.column(inputIndex).size() -
                input.column(inputIndex).null_count()),
            1,
            stream);

        // Assemble into struct as expected by velox.
        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum_col));
        children.push_back(std::move(count_col));
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            1,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        // Input column has two children: sum and count
        auto const sum_col = input.column(inputIndex).child(0);
        auto const count_col = input.column(inputIndex).child(1);

        // sum the sums
        auto const sum_agg_request =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sum_result_scalar =
            cudf::reduce(sum_col, *sum_agg_request, sum_col.type(), stream);
        auto sum_result_col =
            cudf::make_column_from_scalar(*sum_result_scalar, 1, stream);

        // sum the counts
        auto const count_agg_request =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const count_result_scalar = cudf::reduce(
            count_col, *count_agg_request, count_col.type(), stream);

        // divide the sums by the counts
        auto const cudf_output_type =
            cudf::data_type(cudf_velox::velox_to_cudf_type_id(output_type));
        return cudf::binary_operation(
            *sum_result_col,
            *count_result_scalar,
            cudf::binary_operator::DIV,
            cudf_output_type,
            stream);
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  uint32_t mean_idx;
  uint32_t sum_idx;
  uint32_t count_idx;
};

std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator> createAggregator(
    core::AggregationNode::Step step,
    std::string const& kind,
    uint32_t inputIndex,
    VectorPtr constant,
    bool is_global) {
  if (kind == "sum") {
    return std::make_unique<SumAggregator>(
        step, inputIndex, constant, is_global);
  } else if (kind == "count") {
    return std::make_unique<CountAggregator>(
        step, inputIndex, constant, is_global);
  } else if (kind == "min") {
    return std::make_unique<MinAggregator>(
        step, inputIndex, constant, is_global);
  } else if (kind == "max") {
    return std::make_unique<MaxAggregator>(
        step, inputIndex, constant, is_global);
  } else if (kind == "avg") {
    return std::make_unique<MeanAggregator>(
        step, inputIndex, constant, is_global);
  } else {
    VELOX_NYI("Aggregation not yet supported");
  }
}

auto toAggregators(
    core::AggregationNode const& aggregationNode,
    exec::OperatorCtx const& operatorCtx) {
  auto const step = aggregationNode.step();
  bool const isGlobal = aggregationNode.groupingKeys().empty();
  auto const& inputRowSchema = aggregationNode.sources()[0]->outputType();

  std::vector<std::unique_ptr<cudf_velox::CudfHashAggregation::Aggregator>>
      aggregators;
  for (auto const& aggregate : aggregationNode.aggregates()) {
    std::vector<column_index_t> agg_inputs;
    std::vector<VectorPtr> agg_constants;
    for (auto const& arg : aggregate.call->inputs()) {
      if (auto const field =
              dynamic_cast<core::FieldAccessTypedExpr const*>(arg.get())) {
        agg_inputs.push_back(inputRowSchema->getChildIdx(field->name()));
      } else if (
          auto constant =
              dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
        agg_inputs.push_back(kConstantChannel);
        agg_constants.push_back(constant->toConstantVector(operatorCtx.pool()));
      } else {
        VELOX_NYI("Constants and lambdas not yet supported");
      }
    }
    // The loop on aggregate.call->inputs() is taken from
    // AggregateInfo.cpp::toAggregateInfo(). It seems to suggest that there can
    // be multiple inputs to an aggregate.
    // We're postponing properly supporting this for now because the currently
    // supported aggregation functions in cudf_velox don't use it.
    VELOX_CHECK(agg_inputs.size() == 1);

    if (aggregate.distinct) {
      VELOX_NYI("De-dup before aggregation is not yet supported");
    }

    auto const kind = aggregate.call->name();
    auto const inputIndex = agg_inputs[0];
    auto const constant = agg_constants.empty() ? nullptr : agg_constants[0];
    aggregators.push_back(
        createAggregator(step, kind, inputIndex, constant, isGlobal));
  }
  return aggregators;
}

} // namespace

namespace facebook::velox::cudf_velox {

CudfHashAggregation::CudfHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          aggregationNode->step() == core::AggregationNode::Step::kPartial
              ? "CudfPartialAggregation"
              : "CudfAggregation",
          aggregationNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt),
      NvtxHelper(
          nvtx3::rgb{34, 139, 34}, // Forest Green
          operatorId,
          fmt::format("[{}]", aggregationNode->id())),
      aggregationNode_(aggregationNode),
      isPartialOutput_(exec::isPartialOutput(aggregationNode->step())),
      isGlobal_(aggregationNode->groupingKeys().empty()),
      isDistinct_(!isGlobal_ && aggregationNode->aggregates().empty()) {}

void CudfHashAggregation::initialize() {
  Operator::initialize();

  auto const& inputType = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      groupingKeyInputChannels_, groupingKeyOutputChannels_);

  auto const numGroupingKeys = groupingKeyOutputChannels_.size();

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  aggregators_ = toAggregators(*aggregationNode_, *operatorCtx_);

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfHashAggregation::setupGroupingKeyChannelProjections(
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) const {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  auto const& inputType = aggregationNode_->sources()[0]->outputType();
  auto const& groupingKeys = aggregationNode_->groupingKeys();
  // The map from the grouping key output channel to the input channel.
  //
  // NOTE: grouping key output order is specified as 'groupingKeys' in
  // 'aggregationNode_'.
  std::vector<exec::IdentityProjection> groupingKeyProjections;
  groupingKeyProjections.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyProjections.emplace_back(
        exec::exprToChannel(groupingKeys[i].get(), inputType), i);
  }

  groupingKeyInputChannels.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyInputChannels.push_back(groupingKeyProjections[i].inputChannel);
  }

  groupingKeyOutputChannels.resize(groupingKeys.size());

  std::iota(
      groupingKeyOutputChannels.begin(), groupingKeyOutputChannels.end(), 0);
}

void CudfHashAggregation::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudf_input = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudf_input);
    inputs_.push_back(std::move(cudf_input));
  }
}

RowVectorPtr CudfHashAggregation::doGroupByAggregation(
    std::unique_ptr<cudf::table> tbl,
    rmm::cuda_stream_view stream) {
  auto groupby_key_view = tbl->select(
      groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end());

  size_t const num_grouping_keys = groupby_key_view.num_columns();

  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby group_by_owner(
      groupby_key_view,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& aggregator : aggregators_) {
    aggregator->addGroupbyRequest(tbl->view(), requests);
  }

  auto [group_keys, results] = group_by_owner.aggregate(requests, stream);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> result_columns;

  // first fill the grouping keys
  auto group_keys_columns = group_keys->release();
  result_columns.insert(
      result_columns.begin(),
      std::make_move_iterator(group_keys_columns.begin()),
      std::make_move_iterator(group_keys_columns.end()));

  // then fill the aggregation results
  for (auto& aggregator : aggregators_) {
    result_columns.push_back(aggregator->makeOutputColumn(results, stream));
  }

  // make a cudf table out of columns
  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));

  // velox expects nullptr instead of a table with 0 rows
  if (result_table->num_rows() == 0) {
    return nullptr;
  }

  auto num_rows = result_table->num_rows();

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, num_rows, std::move(result_table), stream);
}

RowVectorPtr CudfHashAggregation::doGlobalAggregation(
    std::unique_ptr<cudf::table> tbl,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(aggregators_.size());
  for (auto i = 0; i < aggregators_.size(); i++) {
    result_columns.push_back(aggregators_[i]->doReduce(
        tbl->view(), outputType_->childAt(i), stream));
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(result_columns)),
      stream);
}

RowVectorPtr CudfHashAggregation::getDistinctKeys(
    std::unique_ptr<cudf::table> tbl,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::size_type> key_indices(
      groupingKeyInputChannels_.begin(), groupingKeyInputChannels_.end());
  auto result = cudf::distinct(
      tbl->view(),
      key_indices,
      cudf::duplicate_keep_option::KEEP_FIRST,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream);

  auto num_rows = result->num_rows();

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, num_rows, std::move(result), stream);
}

RowVectorPtr CudfHashAggregation::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (inputs_.empty()) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto tbl = getConcatenatedTable(inputs_, stream);

  // Release input data after synchronizing
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  if (!isGlobal_) {
    return doGroupByAggregation(std::move(tbl), stream);
  } else if (isDistinct_) {
    return getDistinctKeys(std::move(tbl), stream);
  } else {
    return doGlobalAggregation(std::move(tbl), stream);
  }
}

void CudfHashAggregation::noMoreInput() {
  Operator::noMoreInput();
  if (inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfHashAggregation::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
