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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, wxdReaderDefaultAndOverrides) {
  using connector::hive::CudfHiveConfig;
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  CudfHiveConfig reader(properties);
  EXPECT_FALSE(reader.useBufferedInput());
  config::ConfigBase session(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseBufferedInputSession, "true"}});
  EXPECT_TRUE(reader.useBufferedInputSession(&session));
  auto bufferedProperties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseBufferedInput, "true"}});
  CudfHiveConfig buffered(bufferedProperties);
  EXPECT_TRUE(buffered.useBufferedInput());
}

TEST(ConfigTest, defaults) {
  CudfConfig config;
  ASSERT_TRUE(config.exchangeCompressionAllowCudaIpc);
  ASSERT_TRUE(config.hostToDeviceStagingEnabled);
  ASSERT_EQ(config.hostToDeviceStagingWindowBytes, 128ULL << 20);
  ASSERT_EQ(config.hostToDeviceStagingPackThreads, 4);
  ASSERT_EQ(config.hostToDeviceStagingWindowSets, 2);
  ASSERT_TRUE(config.cacheHostRegistrationEnabled);
  ASSERT_EQ(config.cacheHostRegistrationMaxBytes, 32ULL << 30);
  ASSERT_TRUE(config.exchange);
  ASSERT_FALSE(config.ucxxErrorHandling);
  ASSERT_FALSE(config.ucxxBlockingPolling);
  ASSERT_TRUE(config.intraNodeExchange);
  ASSERT_EQ(config.memoryResource, "async");
  ASSERT_TRUE(config.outputMemoryResource.empty());
  ASSERT_EQ(config.exchangeCompression, "column-adaptive-freq-pfor-min128");
  ASSERT_TRUE(config.exchangeCompressionPipeline);
  ASSERT_EQ(config.exchangeCompressionPipelineThreads, 1);
  ASSERT_EQ(config.exchangeCompressionMinBytes, 16LL << 20);
  ASSERT_DOUBLE_EQ(config.exchangeCompressionSafetyMargin, 1.5);
  ASSERT_EQ(config.partitionedOutputBatchRows, 10'000'000);
  ASSERT_TRUE(config.concatOptimizationEnabled);
  ASSERT_EQ(config.batchSizeMinThreshold, 40'000'000);
  ASSERT_TRUE(config.streamingGroupbyApiEnabled);
  ASSERT_FALSE(config.jitExpressionEnabled);
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
  ASSERT_TRUE(config.streamingGroupbyApiEnabled);
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

TEST(ConfigTest, wxdDefaultsRemainOverridable) {
  CudfConfig config;
  config.initialize(
      {{CudfConfig::kUcxExchange, "false"},
       {CudfConfig::kUcxxErrorHandling, "true"},
       {CudfConfig::kUcxxBlockingPolling, "true"},
       {CudfConfig::kUcxIntraNodeExchange, "false"},
       {CudfConfig::kUcxExchangeCompression, "none"},
       {CudfConfig::kUcxExchangeCompressionAllowCudaIpc, "false"},
       {CudfConfig::kUcxExchangeCompressionPipeline, "false"},
       {CudfConfig::kCudfCacheHostRegistrationEnabled, "false"},
       {CudfConfig::kCudfStreamingGroupbyApiEnabled, "false"},
       {CudfConfig::kCudfConcatOptimizationEnabled, "false"},
       {CudfConfig::kCudfJitExpressionEnabled, "true"},
       {CudfConfig::kUcxPartitionedOutputBatchRows, "10000"},
       {CudfConfig::kCudfBatchSizeMinThreshold, "100000"}});
  EXPECT_FALSE(config.exchange);
  EXPECT_TRUE(config.ucxxErrorHandling);
  EXPECT_TRUE(config.ucxxBlockingPolling);
  EXPECT_FALSE(config.intraNodeExchange);
  EXPECT_EQ(config.exchangeCompression, "none");
  EXPECT_FALSE(config.exchangeCompressionAllowCudaIpc);
  EXPECT_FALSE(config.exchangeCompressionPipeline);
  EXPECT_FALSE(config.cacheHostRegistrationEnabled);
  EXPECT_FALSE(config.streamingGroupbyApiEnabled);
  EXPECT_FALSE(config.concatOptimizationEnabled);
  EXPECT_TRUE(config.jitExpressionEnabled);
  EXPECT_EQ(config.partitionedOutputBatchRows, 10000);
  EXPECT_EQ(config.batchSizeMinThreshold, 100000);
}
} // namespace facebook::velox::cudf_velox::test
