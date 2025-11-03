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

#include "PythonBenchmarkBridge.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <memory>
#include <sstream>

// Include base headers first to help clangd resolve namespaces
#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/benchmarks/QueryBenchmarkBase.h"

// Include the existing CudfTpchBenchmark implementation  
// Note: We include the .cpp to reuse the CudfTpchBenchmark class definition
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.cpp"

// Declare FLAGS that we'll use from other benchmark files
// These are defined in TpchBenchmark.cpp
DECLARE_string(data_path);
DECLARE_int32(run_query_verbose);

// These are defined in QueryBenchmarkBase.cpp
DECLARE_string(data_format);
DECLARE_int32(num_drivers);
DECLARE_int32(num_splits_per_file);
DECLARE_bool(include_results);

// These are defined in CudfTpchBenchmark.cpp (included above)
// Already available through the include

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

// No need for an anonymous namespace here if we want Python bindings to access the symbols.

// Minimal extension - just adds statistics collection to the EXISTING CudfTpchBenchmark
class PythonCudfBenchmark : public CudfTpchBenchmark {
 public:
  void initialize() override {
    // Validate FLAGS before initialization
    if (FLAGS_data_path.empty()) {
      throw std::runtime_error("FLAGS_data_path is empty during initialization");
    }
    LOG(INFO) << "Initializing with FLAGS_data_path=" << FLAGS_data_path 
              << ", FLAGS_data_format=" << FLAGS_data_format;
    
    // Call parent initialize - it creates and initializes queryBuilder_
    CudfTpchBenchmark::initialize();
    
    // Verify the inherited queryBuilder_ is initialized
    if (!queryBuilder_) {
      throw std::runtime_error("queryBuilder_ was not initialized by parent");
    }
    
    LOG(INFO) << "Benchmark initialized successfully, queryBuilder_ is ready";
  }

  BenchmarkResult runQueryWithStats(int32_t queryId) {
    BenchmarkResult result = {};
    result.error_message = nullptr;

    try {
      LOG(INFO) << "Running query " << queryId;
      
      // Get query plan using parent's queryBuilder_
      auto queryPlan = queryBuilder_->getQueryPlan(queryId);
      
      // Run the query (inherited from QueryBenchmarkBase)
      auto [cursor, actualResults] = run(queryPlan, queryConfigs_);
      
      if (!cursor) {
        result.error_message = strdup("Query cursor is null");
        return result;
      }
      
      // Wait for completion and collect stats
      auto task = cursor->task();
      ensureTaskCompletion(task.get());
      
      const auto stats = task->taskStats();
      
      // Populate result
      result.execution_time_ms = static_cast<double>(
          stats.executionEndTimeMs - stats.executionStartTimeMs);
      
      // Sum raw input bytes from all TableScan operators
      int64_t rawInputBytes = 0;
      for (auto& pipeline : stats.pipelineStats) {
        for (auto& opStats : pipeline.operatorStats) {
          if (opStats.operatorType == "TableScan") {
            rawInputBytes += opStats.rawInputBytes;
          }
        }
      }
      result.raw_input_bytes = rawInputBytes;
      result.num_total_splits = stats.numTotalSplits;
      result.num_finished_splits = stats.numFinishedSplits;
      
      LOG(INFO) << "Query " << queryId << " completed: " 
                << result.execution_time_ms << "ms, "
                << result.raw_input_bytes << " bytes";
      
    } catch (const std::exception& e) {
      LOG(ERROR) << "Query " << queryId << " failed: " << e.what();
      result.error_message = strdup(e.what());
    } catch (...) {
      result.error_message = strdup("Unknown error occurred");
    }

    return result;
  }
};  // End of PythonCudfBenchmark class

static bool runtime_initialized = false;

extern "C" {

void initialize_runtime(int argc, char** argv) {
  if (!runtime_initialized) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    folly::Init init{&argc, &argv, false};
    runtime_initialized = true;
  }
}

BenchmarkHandle create_benchmark(const BenchmarkConfig* config) {
  if (!config) {
    return nullptr;
  }

  try {
    // Set gflags values from config
    FLAGS_data_path = config->data_path ? config->data_path : "";
    FLAGS_data_format = config->data_format ? config->data_format : "parquet";
    FLAGS_num_drivers = config->num_drivers;
    FLAGS_num_splits_per_file = config->num_splits_per_file;
    FLAGS_include_results = config->include_results;
    FLAGS_cudf_chunk_read_limit = config->cudf_chunk_read_limit;
    FLAGS_cudf_pass_read_limit = config->cudf_pass_read_limit;
    FLAGS_cudf_gpu_batch_size_rows = config->cudf_gpu_batch_size_rows;
    FLAGS_velox_cudf_table_scan = config->velox_cudf_table_scan;

    // Use our minimal extension which just adds stats collection
    auto* benchmark = new PythonCudfBenchmark();
    benchmark->initialize();
    return static_cast<BenchmarkHandle>(benchmark);
  } catch (...) {
    return nullptr;
  }
}

BenchmarkResult run_query_with_stats(BenchmarkHandle handle, int32_t query_id) {
  BenchmarkResult result = {};
  result.error_message = nullptr;

  if (!handle) {
    result.error_message = strdup("Invalid benchmark handle");
    return result;
  }

  if (query_id < 1 || query_id > 22) {
    result.error_message = strdup("Query ID must be between 1 and 22");
    return result;
  }

  auto* benchmark = static_cast<PythonCudfBenchmark*>(handle);
  return benchmark->runQueryWithStats(query_id);
}

void free_result(BenchmarkResult* result) {
  if (result && result->error_message) {
    free(result->error_message);
    result->error_message = nullptr;
  }
}

void destroy_benchmark(BenchmarkHandle handle) {
  if (handle) {
    auto* benchmark = static_cast<PythonCudfBenchmark*>(handle);
    benchmark->shutdown();
    delete benchmark;
  }
}

void shutdown_runtime() {
  runtime_initialized = false;
}

} // extern "C"

