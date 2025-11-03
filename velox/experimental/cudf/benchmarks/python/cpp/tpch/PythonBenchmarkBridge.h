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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result structure for Python
typedef struct {
  double execution_time_ms;
  int64_t raw_input_bytes;
  int32_t num_total_splits;
  int32_t num_finished_splits;
  char* error_message;
} BenchmarkResult;

// Configuration structure
typedef struct {
  const char* data_path;
  const char* data_format;
  int32_t num_drivers;
  int32_t num_splits_per_file;
  bool include_results;
  uint64_t cudf_chunk_read_limit;
  uint64_t cudf_pass_read_limit;
  int32_t cudf_gpu_batch_size_rows;
  bool velox_cudf_table_scan;
} BenchmarkConfig;

// Opaque handle 
typedef void* BenchmarkHandle;

// Minimal C API - just wraps existing CudfTpchBenchmark
void initialize_runtime(int argc, char** argv);
BenchmarkHandle create_benchmark(const BenchmarkConfig* config);
BenchmarkResult run_query_with_stats(BenchmarkHandle handle, int32_t query_id);
void free_result(BenchmarkResult* result);
void destroy_benchmark(BenchmarkHandle handle);
void shutdown_runtime();

#ifdef __cplusplus
}
#endif

