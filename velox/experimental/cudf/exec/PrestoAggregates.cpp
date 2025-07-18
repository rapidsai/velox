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
#include "velox/experimental/cudf/exec/PrestoAggregates.h"

#include "velox/functions/prestosql/aggregates/AggregateNames.h"

namespace facebook::velox::cudf_velox::presto {

namespace {
// Use constants from AggregateNames.h
using facebook::velox::aggregate::kAvg;
using facebook::velox::aggregate::kCount;
using facebook::velox::aggregate::kMax;
using facebook::velox::aggregate::kMin;
using facebook::velox::aggregate::kSum;
} // namespace

void registerPrestoAggregate(
    const std::string& prefix,
    const std::string& aggregateName,
    bool overwrite) {
  auto name = prefix + aggregateName;
  registerAggregator(
      name,
      [aggregateName](
          core::AggregationNode::Step step,
          uint32_t inputIndex,
          VectorPtr constant,
          bool isGlobal) -> std::unique_ptr<CudfHashAggregation::Aggregator> {
        return facebook::velox::cudf_velox::createAggregator(
            aggregateName, step, inputIndex, constant, isGlobal);
      },
      overwrite);
}

void registerPrestoSumAggregate(const std::string& prefix, bool overwrite) {
  registerPrestoAggregate(prefix, kSum, overwrite);
}

void registerPrestoCountAggregate(const std::string& prefix, bool overwrite) {
  registerPrestoAggregate(prefix, kCount, overwrite);
}

void registerPrestoMinAggregate(const std::string& prefix, bool overwrite) {
  registerPrestoAggregate(prefix, kMin, overwrite);
}

void registerPrestoMaxAggregate(const std::string& prefix, bool overwrite) {
  registerPrestoAggregate(prefix, kMax, overwrite);
}

void registerPrestoAvgAggregate(const std::string& prefix, bool overwrite) {
  registerPrestoAggregate(prefix, kAvg, overwrite);
}

void registerAllPrestoAggregates(const std::string& prefix, bool overwrite) {
  registerPrestoSumAggregate(prefix, overwrite);
  registerPrestoCountAggregate(prefix, overwrite);
  registerPrestoMinAggregate(prefix, overwrite);
  registerPrestoMaxAggregate(prefix, overwrite);
  registerPrestoAvgAggregate(prefix, overwrite);
}

} // namespace facebook::velox::cudf_velox::presto
