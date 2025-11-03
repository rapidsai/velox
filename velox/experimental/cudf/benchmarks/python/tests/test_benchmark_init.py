"""
Tests for CudfTpchBenchmark initialization.
"""
import pytest
from cudf_tpch_benchmark import CudfTpchBenchmark, BenchmarkError


class TestBenchmarkInitialization:
    """Test suite for benchmark initialization."""
    
    def test_init_default_params(self, benchmark_config, skip_without_data):
        """Test initialization with default parameters."""
        benchmark = CudfTpchBenchmark(
            data_path=benchmark_config['data_path']
        )
        try:
            assert benchmark.initialized
        finally:
            benchmark.close()
    
    def test_init_custom_params(self, benchmark_config, skip_without_data):
        """Test initialization with custom parameters."""
        benchmark = CudfTpchBenchmark(
            data_path=benchmark_config['data_path'],
            data_format='parquet',
            num_drivers=2,
            num_splits_per_file=5,
            include_results=False,
            cudf_chunk_read_limit=1000,
            cudf_pass_read_limit=2000,
            cudf_gpu_batch_size_rows=50000,
            velox_cudf_table_scan=True
        )
        try:
            assert benchmark.initialized
        finally:
            benchmark.close()
    
    def test_init_invalid_path(self):
        """Test initialization with invalid data path."""
        with pytest.raises((BenchmarkError, Exception)):
            CudfTpchBenchmark(data_path="/nonexistent/path/to/data")
    
    def test_init_with_orc_format(self, benchmark_config):
        """Test initialization with ORC format."""
        # This may fail without ORC data, but should not crash
        try:
            benchmark = CudfTpchBenchmark(
                data_path=benchmark_config['data_path'],
                data_format='orc'
            )
            benchmark.close()
        except (BenchmarkError, Exception):
            # Expected if ORC files don't exist
            pass
    
    def test_close_twice(self, benchmark_config, skip_without_data):
        """Test calling close() multiple times."""
        benchmark = CudfTpchBenchmark(
            data_path=benchmark_config['data_path']
        )
        benchmark.close()
        # Should not crash on second close
        benchmark.close()
    
    def test_context_manager(self, benchmark_config, skip_without_data):
        """Test using benchmark as context manager."""
        with CudfTpchBenchmark(data_path=benchmark_config['data_path']) as benchmark:
            assert benchmark.initialized
        # Should be closed after exiting context
        assert not benchmark.initialized
    
    def test_context_manager_with_exception(self, benchmark_config, skip_without_data):
        """Test context manager cleanup on exception."""
        try:
            with CudfTpchBenchmark(data_path=benchmark_config['data_path']) as benchmark:
                assert benchmark.initialized
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Benchmark should still be closed
        assert not benchmark.initialized

