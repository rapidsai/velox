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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"

#include <cudf/types.hpp>

#include <algorithm>
#include <cctype>
#include <optional>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

std::optional<cudf::data_type> parseTimestampPushdownType(
    std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (value.empty() || value == "none" || value == "off" || value == "false") {
    return std::nullopt;
  }
  if (value == "ms" || value == "millis" || value == "milliseconds") {
    return cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
  }
  if (value == "us" || value == "micros" || value == "microseconds") {
    return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  }
  if (value == "ns" || value == "nanos" || value == "nanoseconds") {
    return cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
  }
  VELOX_USER_FAIL(
      "Invalid cudf.hive.timestamp-pushdown-type '{}'. "
      "Use ms|us|ns (or none).",
      value);
}

} // namespace

std::size_t CudfHiveConfig::maxChunkReadLimit() const {
  // chunk read limit = 0 means no limit
  return config_->get<std::size_t>(kMaxChunkReadLimit, 0);
}

std::size_t CudfHiveConfig::maxChunkReadLimitSession(
    const config::ConfigBase* session) const {
  // pass read limit = 0 means no limit
  return session->get<std::size_t>(
      kMaxChunkReadLimitSession,
      config_->get<std::size_t>(kMaxChunkReadLimit, 0));
}

std::size_t CudfHiveConfig::maxPassReadLimit() const {
  // pass read limit = 0 means no limit
  return config_->get<std::size_t>(kMaxPassReadLimit, 0);
}

std::size_t CudfHiveConfig::maxPassReadLimitSession(
    const config::ConfigBase* session) const {
  // pass read limit = 0 means no limit
  return session->get<std::size_t>(
      kMaxPassReadLimitSession,
      config_->get<std::size_t>(kMaxPassReadLimit, 0));
}

bool CudfHiveConfig::isConvertStringsToCategories() const {
  return config_->get<bool>(kConvertStringsToCategories, false);
}

bool CudfHiveConfig::isConvertStringsToCategoriesSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kConvertStringsToCategoriesSession,
      config_->get<bool>(kConvertStringsToCategories, false));
}

bool CudfHiveConfig::isUsePandasMetadata() const {
  return config_->get<bool>(kUsePandasMetadata, true);
}

bool CudfHiveConfig::isUsePandasMetadataSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUsePandasMetadataSession, config_->get<bool>(kUsePandasMetadata, true));
}

bool CudfHiveConfig::isUseArrowSchema() const {
  return config_->get<bool>(kUseArrowSchema, true);
}

bool CudfHiveConfig::isUseArrowSchemaSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUseArrowSchemaSession, config_->get<bool>(kUseArrowSchema, true));
}

bool CudfHiveConfig::isAllowMismatchedCudfHiveSchemas() const {
  return config_->get<bool>(kAllowMismatchedCudfHiveSchemas, false);
}

bool CudfHiveConfig::isAllowMismatchedCudfHiveSchemasSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kAllowMismatchedCudfHiveSchemasSession,
      config_->get<bool>(kAllowMismatchedCudfHiveSchemas, false));
}

cudf::data_type CudfHiveConfig::timestampType() const {
  const auto unit = config_->get<cudf::type_id>(
      kTimestampType, cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/);
  VELOX_CHECK(
      unit == cudf::type_id::TIMESTAMP_DAYS /*days*/ ||
          unit == cudf::type_id::TIMESTAMP_SECONDS /*seconds*/ ||
          unit == cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/ ||
          unit == cudf::type_id::TIMESTAMP_MICROSECONDS /*micro*/ ||
          unit == cudf::type_id::TIMESTAMP_NANOSECONDS /*nano*/,
      "Invalid timestamp unit.");
  return cudf::data_type(cudf::type_id{unit});
}

cudf::data_type CudfHiveConfig::timestampTypeSession(
    const config::ConfigBase* session) const {
  const auto unit = session->get<cudf::type_id>(
      kTimestampTypeSession,
      config_->get<cudf::type_id>(
          kTimestampType, cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/));
  VELOX_CHECK(
      unit == cudf::type_id::TIMESTAMP_DAYS /*days*/ ||
          unit == cudf::type_id::TIMESTAMP_SECONDS /*seconds*/ ||
          unit == cudf::type_id::TIMESTAMP_MILLISECONDS /*milli*/ ||
          unit == cudf::type_id::TIMESTAMP_MICROSECONDS /*micro*/ ||
          unit == cudf::type_id::TIMESTAMP_NANOSECONDS /*nano*/,
      "Invalid timestamp unit.");
  return cudf::data_type(cudf::type_id{unit});
}

std::optional<cudf::data_type> CudfHiveConfig::timestampPushdownType() const {
  auto value = config_->get<std::string>(kTimestampPushdownType, "");
  return parseTimestampPushdownType(value);
}

std::optional<cudf::data_type> CudfHiveConfig::timestampPushdownTypeSession(
    const config::ConfigBase* session) const {
  auto value = session->get<std::string>(
      kTimestampPushdownTypeSession,
      config_->get<std::string>(kTimestampPushdownType, ""));
  return parseTimestampPushdownType(value);
}

bool CudfHiveConfig::useBufferedInput() const {
  return config_->get<bool>(kUseBufferedInput, true);
}

bool CudfHiveConfig::useBufferedInputSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUseBufferedInputSession, config_->get<bool>(kUseBufferedInput, true));
}

bool CudfHiveConfig::useExperimentalCudfReader() const {
  return config_->get<bool>(kUseExperimentalCudfReader, false);
}

bool CudfHiveConfig::useExperimentalCudfReaderSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kUseExperimentalCudfReaderSession,
      config_->get<bool>(kUseExperimentalCudfReader, false));
}

bool CudfHiveConfig::immutableFiles() const {
  return config_->get<bool>(kImmutableFiles, false);
}

uint64_t CudfHiveConfig::sortWriterFinishTimeSliceLimitMs(
    const config::ConfigBase* session) const {
  return session->get<uint64_t>(
      kSortWriterFinishTimeSliceLimitMsSession,
      config_->get<uint64_t>(kSortWriterFinishTimeSliceLimitMs, 5'000));
}

bool CudfHiveConfig::writeTimestampsAsUTC() const {
  return config_->get<bool>(kWriteTimestampsAsUTC, true);
}

bool CudfHiveConfig::writeTimestampsAsUTCSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kWriteTimestampsAsUTCSession,
      config_->get<bool>(kWriteTimestampsAsUTC, true));
}

bool CudfHiveConfig::writeArrowSchema() const {
  return config_->get<bool>(kWriteArrowSchema, false);
}

bool CudfHiveConfig::writeArrowSchemaSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kWriteArrowSchemaSession, config_->get<bool>(kWriteArrowSchema, false));
}

bool CudfHiveConfig::writev2PageHeaders() const {
  return config_->get<bool>(kWritev2PageHeaders, false);
}

bool CudfHiveConfig::writev2PageHeadersSession(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kWritev2PageHeadersSession,
      config_->get<bool>(kWritev2PageHeaders, false));
}

} // namespace facebook::velox::cudf_velox::connector::hive
