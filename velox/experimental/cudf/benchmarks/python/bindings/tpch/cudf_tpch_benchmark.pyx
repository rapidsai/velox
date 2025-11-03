# distutils: language = c++
# cython: language_level = 3

from libc.stdint cimport int32_t, int64_t, uint64_t
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy
cimport cudf_tpch_benchmark

import sys
import os

class BenchmarkError(Exception):
    """Exception raised when benchmark execution fails."""
    pass

class QueryResult:
    """Results from running a TPC-H query benchmark."""
    
    def __init__(self, execution_time_ms, raw_input_bytes, 
                 num_total_splits, num_finished_splits):
        self.execution_time_ms = execution_time_ms
        self.raw_input_bytes = raw_input_bytes
        self.num_total_splits = num_total_splits
        self.num_finished_splits = num_finished_splits
        self.throughput_mbps = (raw_input_bytes / (1024 * 1024)) / (execution_time_ms / 1000.0) if execution_time_ms > 0 else 0
    
    def __repr__(self):
        return (f"QueryResult(execution_time={self.execution_time_ms:.2f}ms, "
                f"raw_input={self.raw_input_bytes} bytes, "
                f"throughput={self.throughput_mbps:.2f} MB/s, "
                f"splits={self.num_finished_splits}/{self.num_total_splits})")

cdef class CudfTpchBenchmark:
    """
    Python wrapper for Velox CUDF TPC-H Benchmark.
    
    This class provides Python bindings to the Velox CUDF-accelerated TPC-H
    benchmark suite, enabling programmatic access from Python for
    other Python-based benchmarking frameworks.
    
    Example:
        >>> benchmark = CudfTpchBenchmark(
        ...     data_path="/path/to/tpch/data",
        ...     data_format="parquet",
        ...     cudf_gpu_batch_size_rows=100000
        ... )
        >>> result = benchmark.run_query(1)
        >>> print(f"Query 1 took {result.execution_time_ms:.2f}ms")
        >>> benchmark.close()
    """
    
    cdef cudf_tpch_benchmark.BenchmarkHandle handle
    cdef bool initialized
    
    def __init__(self, 
                 data_path,
                 data_format="parquet",
                 num_drivers=4,
                 num_splits_per_file=10,
                 include_results=False,
                 cudf_chunk_read_limit=0,
                 cudf_pass_read_limit=0,
                 cudf_gpu_batch_size_rows=100000,
                 velox_cudf_table_scan=True):
        """
        Initialize the CUDF TPC-H Benchmark.
        
        Args:
            data_path (str): Path to the TPC-H data directory
            data_format (str): Data format, e.g., "parquet" or "orc"
            num_drivers (int): Number of driver threads
            num_splits_per_file (int): Number of splits per file
            include_results (bool): Whether to include query results
            cudf_chunk_read_limit (int): Output table chunk read limit for cudf
            cudf_pass_read_limit (int): Pass read limit for cudf
            cudf_gpu_batch_size_rows (int): Preferred output batch size in rows
            velox_cudf_table_scan (bool): Enable cuDF table scan
        """
        # Convert Python strings to bytes first
        data_path_bytes = data_path.encode('utf-8')
        data_format_bytes = data_format.encode('utf-8')
        
        print(f"[DEBUG] Initializing with data_path: {data_path}")
        print(f"[DEBUG] data_format: {data_format}")
        
        # Initialize runtime with data_path flag to satisfy the validator
        cdef int argc = 2
        cdef char** argv = <char**>malloc(3 * sizeof(char*))
        cdef bytes data_path_flag = b"--data_path=" + data_path_bytes
        argv[0] = b"python"
        argv[1] = data_path_flag
        argv[2] = NULL
        
        print(f"[DEBUG] Calling initialize_runtime with argc={argc}, argv[1]={data_path_flag}")
        cudf_tpch_benchmark.initialize_runtime(argc, argv)
        free(argv)
        print("[DEBUG] Runtime initialized successfully")
        
        # Create config
        cdef cudf_tpch_benchmark.BenchmarkConfig config
        config.data_path = data_path_bytes
        config.data_format = data_format_bytes
        config.num_drivers = num_drivers
        config.num_splits_per_file = num_splits_per_file
        config.include_results = include_results
        config.cudf_chunk_read_limit = cudf_chunk_read_limit
        config.cudf_pass_read_limit = cudf_pass_read_limit
        config.cudf_gpu_batch_size_rows = cudf_gpu_batch_size_rows
        config.velox_cudf_table_scan = velox_cudf_table_scan
        
        # Create benchmark
        print(f"[DEBUG] Creating benchmark with config:")
        print(f"[DEBUG]   data_path: {data_path}")
        print(f"[DEBUG]   data_format: {data_format}")
        print(f"[DEBUG]   num_drivers: {num_drivers}")
        print(f"[DEBUG]   cudf_gpu_batch_size_rows: {cudf_gpu_batch_size_rows}")
        
        self.handle = cudf_tpch_benchmark.create_benchmark(&config)
        if self.handle == NULL:
            raise BenchmarkError("Failed to create benchmark instance")
        
        print("[DEBUG] Benchmark instance created successfully")
        self.initialized = True
    
    def run_query(self, query_id):
        """
        Run a specific TPC-H query.
        
        Args:
            query_id (int): Query number (1-22)
            
        Returns:
            QueryResult: Object containing benchmark results
            
        Raises:
            BenchmarkError: If benchmark execution fails
            ValueError: If query_id is not in range 1-22
        """
        if not self.initialized:
            raise BenchmarkError("Benchmark not initialized")
        
        if not (1 <= query_id <= 22):
            raise ValueError(f"Query ID must be between 1 and 22, got {query_id}")
        
        print(f"[DEBUG] Running query {query_id}...")
        cdef cudf_tpch_benchmark.BenchmarkResult result = \
            cudf_tpch_benchmark.run_query_with_stats(self.handle, query_id)
        print(f"[DEBUG] Query {query_id} completed")
        
        # Check for errors
        if result.error_message != NULL:
            error_msg = result.error_message.decode('utf-8')
            cudf_tpch_benchmark.free_result(&result)
            raise BenchmarkError(f"Query {query_id} failed: {error_msg}")
        
        # Create Python result object
        py_result = QueryResult(
            execution_time_ms=result.execution_time_ms,
            raw_input_bytes=result.raw_input_bytes,
            num_total_splits=result.num_total_splits,
            num_finished_splits=result.num_finished_splits
        )
        
        cudf_tpch_benchmark.free_result(&result)
        return py_result
    
    def run_all_queries(self):
        """
        Run all 22 TPC-H queries.
        
        Returns:
            dict: Dictionary mapping query_id (1-22) to QueryResult objects
            
        Raises:
            BenchmarkError: If benchmark execution fails
        """
        if not self.initialized:
            raise BenchmarkError("Benchmark not initialized")
        
        # Run each query individually
        py_results = {}
        for i in range(1, 23):
            try:
                result = self.run_query(i)
                py_results[i] = result
            except BenchmarkError as e:
                py_results[i] = e
        
        return py_results
    
    def close(self):
        """Close and cleanup benchmark resources."""
        if self.initialized:
            cudf_tpch_benchmark.destroy_benchmark(self.handle)
            self.initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __dealloc__(self):
        """Cleanup when object is destroyed."""
        self.close()

def shutdown():
    """Shutdown the runtime. Call this at program exit."""
    cudf_tpch_benchmark.shutdown_runtime()

