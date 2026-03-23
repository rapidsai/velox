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
#pragma once

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated Window operator using cuDF's grouped_rolling_window API.
///
/// Each incoming batch is immediately concatenated into an accumulated cudf
/// table on the GPU in addInput(), avoiding the need to hold separate batch
/// pointers and perform a bulk concatenation in getOutput(). Once all input
/// has arrived, getOutput() sorts (if needed), evaluates the window functions,
/// and returns the result.
///
/// Currently supports: LAG, LEAD, ROW_NUMBER, FIRST_VALUE, LAST_VALUE,
/// and aggregate window functions (SUM, MIN, MAX, COUNT, AVG).
class CudfWindow : public exec::Operator, public NvtxHelper {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  std::shared_ptr<const core::WindowNode> windowNode_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  // Accumulated input data on the GPU. Each addInput() call concatenates
  // the new batch into this table immediately rather than deferring to
  // getOutput().
  std::unique_ptr<cudf::table> accumulatedData_;
  rmm::cuda_stream_view stream_{cudf::get_default_stream()};
  memory::MemoryPool* pool_{nullptr};

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
