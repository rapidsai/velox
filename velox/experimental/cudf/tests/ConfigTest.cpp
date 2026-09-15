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

#include "velox/experimental/cudf/CudfConfig.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, defaults) {
  CudfConfig config;
  ASSERT_FALSE(config.exchangeCompressionAllowCudaIpc);
  ASSERT_TRUE(config.hostToDeviceStagingEnabled);
  ASSERT_EQ(config.hostToDeviceStagingWindowBytes, 128ULL << 20);
  ASSERT_EQ(config.hostToDeviceStagingPackThreads, 4);
  ASSERT_EQ(config.hostToDeviceStagingWindowSets, 2);
  ASSERT_FALSE(config.cacheHostRegistrationEnabled);
  ASSERT_EQ(config.cacheHostRegistrationMaxBytes, 32ULL << 30);
}

TEST(ConfigTest, CudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfHostToDeviceStagingEnabled, "false"},
      {CudfConfig::kCudfHostToDeviceStagingWindowBytes, "67108864"},
      {CudfConfig::kCudfHostToDeviceStagingPackThreads, "3"},
      {CudfConfig::kCudfHostToDeviceStagingWindowSets, "5"},
      {CudfConfig::kCudfCacheHostRegistrationEnabled, "true"},
      {CudfConfig::kCudfCacheHostRegistrationMaxBytes, "68719476736"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfStreamingGroupbyApiEnabled, "true"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kUcxExchange, "true"},
      {CudfConfig::kUcxxErrorHandling, "false"},
      {CudfConfig::kUcxIntraNodeExchange, "true"},
      {CudfConfig::kUcxxBlockingPolling, "false"},
      {CudfConfig::kUcxExchangeLogLevel, "2"},
      {CudfConfig::kUcxPartitionedOutputBatchRows, "100000"},
      {CudfConfig::kUcxExchangeCompression, "column-adaptive-freq-pfor-min128"},
      {CudfConfig::kUcxExchangeCompressionAllowCudaIpc, "true"},
      {CudfConfig::kUcxExchangeCompressionPipeline, "true"},
      {CudfConfig::kUcxExchangeCompressionPipelineThreads, "2"},
      {CudfConfig::kUcxExchangeCompressionMinBytes, "268435456"},
      {CudfConfig::kUcxExchangeCompressionSafetyMargin, "1.5"}};

  CudfConfig config;
  ASSERT_FALSE(config.streamingGroupbyApiEnabled);
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_FALSE(config.hostToDeviceStagingEnabled);
  ASSERT_EQ(config.hostToDeviceStagingWindowBytes, 67'108'864);
  ASSERT_EQ(config.hostToDeviceStagingPackThreads, 3);
  ASSERT_EQ(config.hostToDeviceStagingWindowSets, 5);
  ASSERT_TRUE(config.cacheHostRegistrationEnabled);
  ASSERT_EQ(config.cacheHostRegistrationMaxBytes, 64ULL << 30);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.streamingGroupbyApiEnabled, true);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_TRUE(config.exchange);
  ASSERT_FALSE(config.ucxxErrorHandling);
  ASSERT_TRUE(config.intraNodeExchange);
  ASSERT_FALSE(config.ucxxBlockingPolling);
  ASSERT_EQ(config.exchangeLogLevel, 2);
  ASSERT_EQ(config.partitionedOutputBatchRows, 100000);
  ASSERT_EQ(config.exchangeCompression, "column-adaptive-freq-pfor-min128");
  ASSERT_TRUE(config.exchangeCompressionAllowCudaIpc);
  ASSERT_TRUE(config.exchangeCompressionPipeline);
  ASSERT_EQ(config.exchangeCompressionPipelineThreads, 2);
  ASSERT_EQ(config.exchangeCompressionMinBytes, 268435456);
  ASSERT_DOUBLE_EQ(config.exchangeCompressionSafetyMargin, 1.5);
}

TEST(ConfigTest, exchangeCompressionCudaIpcOptIn) {
  CudfConfig config;
  config.initialize(
      {{CudfConfig::kUcxExchangeCompressionAllowCudaIpc, "true"}});
  EXPECT_TRUE(config.exchangeCompressionAllowCudaIpc);
  config.initialize(
      {{CudfConfig::kUcxExchangeCompressionAllowCudaIpc, "false"}});
  EXPECT_FALSE(config.exchangeCompressionAllowCudaIpc);
  EXPECT_ANY_THROW(config.initialize(
      {{CudfConfig::kUcxExchangeCompressionAllowCudaIpc, "invalid"}}));
  EXPECT_FALSE(config.exchangeCompressionAllowCudaIpc);
}
} // namespace facebook::velox::cudf_velox::test
