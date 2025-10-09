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

#include "velox/connectors/Connector.h"
#include "velox/dwio/common/Options.h"

namespace cudf {
namespace io {
struct source_info;
}
} // namespace cudf

#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {

struct CudfHiveConnectorSplit
    : public facebook::velox::connector::ConnectorSplit {
  const std::string filePath;
  const facebook::velox::dwio::common::FileFormat fileFormat{
      facebook::velox::dwio::common::FileFormat::PARQUET};
  const std::unique_ptr<cudf::io::source_info> cudfSourceInfo;
  const uint64_t start;
  const uint64_t length;

  /// These represent columns like $file_size, $file_modified_time that are
  /// associated with the CudfHiveConnectorSplit.
  std::unordered_map<std::string, std::string> infoColumns = {};

  CudfHiveConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      uint64_t _start,
      uint64_t _length,
      int64_t _splitWeight = 0,
      const std::unordered_map<std::string, std::string>& _infoColumns = {})
      : facebook::velox::connector::ConnectorSplit(connectorId, _splitWeight),
        filePath(_filePath),
        cudfSourceInfo(std::make_unique<cudf::io::source_info>(filePath)),
        start(_start),
        length(_length),
        infoColumns(_infoColumns) {}

  uint64_t size() const override;

  std::string toString() const override;

  std::string getFileName() const;

  const cudf::io::source_info& getCudfSourceInfo() const;

  static std::shared_ptr<CudfHiveConnectorSplit> create(
      const folly::dynamic& obj);
};

class CudfHiveConnectorSplitBuilder {
 public:
  explicit CudfHiveConnectorSplitBuilder(std::string filePath)
      : filePath_{std::move(filePath)} {}

  CudfHiveConnectorSplitBuilder& splitWeight(int64_t splitWeight) {
    splitWeight_ = splitWeight;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& connectorId(const std::string& connectorId) {
    connectorId_ = connectorId;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& start(uint64_t start) {
    start_ = start;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& length(uint64_t length) {
    length_ = length;
    return *this;
  }

  CudfHiveConnectorSplitBuilder& infoColumn(
      const std::string& name,
      const std::string& value) {
    infoColumns_.emplace(std::move(name), std::move(value));
    return *this;
  }

  std::shared_ptr<CudfHiveConnectorSplit> build() const {
    return std::make_shared<CudfHiveConnectorSplit>(
        connectorId_, filePath_, start_, length_, infoColumns_, splitWeight_);
  }

 private:
  const std::string filePath_;
  std::string connectorId_;
  uint64_t start_{0};
  uint64_t length_{std::numeric_limits<uint64_t>::max()};
  int64_t splitWeight_{0};
  std::unordered_map<std::string, std::string> infoColumns_ = {};
};

} // namespace facebook::velox::cudf_velox::connector::hive
