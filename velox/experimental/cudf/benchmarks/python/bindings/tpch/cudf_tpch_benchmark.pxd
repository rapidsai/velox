# distutils: language = c++
# cython: language_level = 3

from libc.stdint cimport int32_t, int64_t, uint64_t
from libcpp cimport bool

cdef extern from "PythonBenchmarkBridge.h":
    ctypedef void* BenchmarkHandle
    
    ctypedef struct BenchmarkResult:
        double execution_time_ms
        int64_t raw_input_bytes
        int32_t num_total_splits
        int32_t num_finished_splits
        char* error_message
    
    ctypedef struct BenchmarkConfig:
        const char* data_path
        const char* data_format
        int32_t num_drivers
        int32_t num_splits_per_file
        bool include_results
        uint64_t cudf_chunk_read_limit
        uint64_t cudf_pass_read_limit
        int32_t cudf_gpu_batch_size_rows
        bool velox_cudf_table_scan

    void initialize_runtime(int argc, char** argv)
    BenchmarkHandle create_benchmark(const BenchmarkConfig* config)
    BenchmarkResult run_query_with_stats(BenchmarkHandle handle, int32_t query_id)
    void free_result(BenchmarkResult* result)
    void destroy_benchmark(BenchmarkHandle handle)
    void shutdown_runtime()
