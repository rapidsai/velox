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

#include <string>

namespace facebook::velox::cudf_velox::presto {

/// Register presto-style aggregate functions that use CUDF aggregators
/// with the specified prefix.
void registerPrestoSumAggregate(
    const std::string& prefix,
    bool overwrite = false);

void registerPrestoCountAggregate(
    const std::string& prefix,
    bool overwrite = false);

void registerPrestoMinAggregate(
    const std::string& prefix,
    bool overwrite = false);

void registerPrestoMaxAggregate(
    const std::string& prefix,
    bool overwrite = false);

void registerPrestoAvgAggregate(
    const std::string& prefix,
    bool overwrite = false);

/// Register all presto-style CUDF aggregators with the specified prefix
void registerAllPrestoAggregates(
    const std::string& prefix,
    bool overwrite = false);

} // namespace facebook::velox::cudf_velox::presto
