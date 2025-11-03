# Type stub for cudf_tpch_benchmark
# This file provides type hints for IDEs and type checkers

from typing import Optional

class BenchmarkError(Exception):
    """Exception raised when benchmark execution fails."""
    ...

class QueryResult:
    """Results from running a TPC-H query benchmark."""
    
    execution_time_ms: float
    raw_input_bytes: int
    num_total_splits: int
    num_finished_splits: int
    throughput_mbps: float
    
    def __init__(
        self,
        execution_time_ms: float,
        raw_input_bytes: int,
        num_total_splits: int,
        num_finished_splits: int
    ) -> None: ...
    
    def __repr__(self) -> str: ...

class CudfTpchBenchmark:
    """
    Python wrapper for Velox CUDF TPC-H Benchmark.
    
    This class provides Python bindings to the Velox CUDF-accelerated TPC-H
    benchmark suite, enabling programmatic access from Python for
    other Python-based benchmarking frameworks.
    """
    
    def __init__(
        self,
        data_path: str,
        data_format: str = "parquet",
        num_drivers: int = 4,
        num_splits_per_file: int = 10,
        include_results: bool = False,
        cudf_chunk_read_limit: int = 0,
        cudf_pass_read_limit: int = 0,
        cudf_gpu_batch_size_rows: int = 100000,
        velox_cudf_table_scan: bool = True
    ) -> None:
        """
        Initialize the CUDF TPC-H Benchmark.
        
        Args:
            data_path: Path to the TPC-H data directory
            data_format: Data format, e.g., "parquet" or "orc"
            num_drivers: Number of driver threads
            num_splits_per_file: Number of splits per file
            include_results: Whether to include query results
            cudf_chunk_read_limit: Output table chunk read limit for cudf
            cudf_pass_read_limit: Pass read limit for cudf
            cudf_gpu_batch_size_rows: Preferred output batch size in rows
            velox_cudf_table_scan: Enable cuDF table scan
        """
        ...
    
    def run_query(self, query_id: int) -> QueryResult:
        """
        Run a specific TPC-H query.
        
        Args:
            query_id: Query number (1-22)
            
        Returns:
            Object containing benchmark results
            
        Raises:
            BenchmarkError: If benchmark execution fails
            ValueError: If query_id is not in range 1-22
        """
        ...
    
    def run_all_queries(self) -> dict[int, QueryResult]:
        """
        Run all 22 TPC-H queries.
        
        Returns:
            Dictionary mapping query_id (1-22) to QueryResult objects
            
        Raises:
            BenchmarkError: If benchmark execution fails
        """
        ...
    
    def close(self) -> None:
        """Close and cleanup benchmark resources."""
        ...
    
    def __enter__(self) -> 'CudfTpchBenchmark':
        """Context manager entry."""
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        ...

def shutdown() -> None:
    """Shutdown the runtime. Call this at program exit."""
    ...
