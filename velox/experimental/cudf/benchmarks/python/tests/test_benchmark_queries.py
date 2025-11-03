"""
Tests for running TPC-H queries.
"""
import pytest
from cudf_tpch_benchmark import (
    CudfTpchBenchmark, 
    BenchmarkError, 
    QueryResult
)


class TestBenchmarkQueries:
    """Test suite for query execution."""
    
    @pytest.fixture
    def benchmark(self, benchmark_config, skip_without_data):
        """Create a benchmark instance for tests."""
        bench = CudfTpchBenchmark(**benchmark_config)
        yield bench
        bench.close()
    
    def test_run_query_valid_id(self, benchmark):
        """Test running a query with valid ID."""
        result = benchmark.run_query(1)
        
        assert isinstance(result, QueryResult)
        assert result.execution_time_ms >= 0
        assert result.raw_input_bytes >= 0
        assert result.num_total_splits >= 0
        assert result.num_finished_splits >= 0
        assert result.throughput_mbps >= 0
    
    def test_run_query_invalid_id_zero(self, benchmark):
        """Test running query with ID 0 (invalid)."""
        with pytest.raises(ValueError, match="Query ID must be between 1 and 22"):
            benchmark.run_query(0)
    
    def test_run_query_invalid_id_negative(self, benchmark):
        """Test running query with negative ID."""
        with pytest.raises(ValueError, match="Query ID must be between 1 and 22"):
            benchmark.run_query(-1)
    
    def test_run_query_invalid_id_too_high(self, benchmark):
        """Test running query with ID > 22."""
        with pytest.raises(ValueError, match="Query ID must be between 1 and 22"):
            benchmark.run_query(23)
    
    def test_run_query_boundary_min(self, benchmark):
        """Test running query with minimum valid ID (1)."""
        result = benchmark.run_query(1)
        assert isinstance(result, QueryResult)
    
    def test_run_query_boundary_max(self, benchmark):
        """Test running query with maximum valid ID (22)."""
        result = benchmark.run_query(22)
        assert isinstance(result, QueryResult)
    
    def test_run_query_not_initialized(self, benchmark_config, skip_without_data):
        """Test running query on closed benchmark."""
        benchmark = CudfTpchBenchmark(**benchmark_config)
        benchmark.close()
        
        with pytest.raises(BenchmarkError, match="Benchmark not initialized"):
            benchmark.run_query(1)
    
    def test_run_multiple_queries(self, benchmark):
        """Test running multiple queries sequentially."""
        results = []
        for query_id in [1, 6, 10]:
            result = benchmark.run_query(query_id)
            results.append(result)
            assert isinstance(result, QueryResult)
        
        assert len(results) == 3
    
    def test_run_same_query_twice(self, benchmark):
        """Test running the same query multiple times."""
        result1 = benchmark.run_query(1)
        result2 = benchmark.run_query(1)
        
        assert isinstance(result1, QueryResult)
        assert isinstance(result2, QueryResult)
        # Both should complete successfully
        assert result1.execution_time_ms >= 0
        assert result2.execution_time_ms >= 0
    
    def test_run_all_queries(self, benchmark):
        """Test running all queries."""
        results = benchmark.run_all_queries()
        
        assert isinstance(results, dict)
        assert len(results) == 22
        
        # Check all query IDs present
        for i in range(1, 23):
            assert i in results
            result = results[i]
            # Result should be either QueryResult or BenchmarkError
            assert isinstance(result, (QueryResult, BenchmarkError))
    
    def test_run_all_queries_returns_dict(self, benchmark):
        """Test that run_all_queries returns proper dict structure."""
        results = benchmark.run_all_queries()
        
        # Verify it's a dict with integer keys
        assert all(isinstance(k, int) for k in results.keys())
        assert all(1 <= k <= 22 for k in results.keys())
    
    def test_run_all_queries_not_initialized(self, benchmark_config, skip_without_data):
        """Test run_all_queries on closed benchmark."""
        benchmark = CudfTpchBenchmark(**benchmark_config)
        benchmark.close()
        
        with pytest.raises(BenchmarkError, match="Benchmark not initialized"):
            benchmark.run_all_queries()


class TestQueryResultProperties:
    """Test query result properties and calculations."""
    
    @pytest.fixture
    def benchmark(self, benchmark_config, skip_without_data):
        """Create a benchmark instance for tests."""
        bench = CudfTpchBenchmark(**benchmark_config)
        yield bench
        bench.close()
    
    def test_result_has_all_fields(self, benchmark):
        """Test that result has all expected fields."""
        result = benchmark.run_query(1)
        
        assert hasattr(result, 'execution_time_ms')
        assert hasattr(result, 'raw_input_bytes')
        assert hasattr(result, 'num_total_splits')
        assert hasattr(result, 'num_finished_splits')
        assert hasattr(result, 'throughput_mbps')
    
    def test_result_values_non_negative(self, benchmark):
        """Test that result values are non-negative."""
        result = benchmark.run_query(1)
        
        assert result.execution_time_ms >= 0
        assert result.raw_input_bytes >= 0
        assert result.num_total_splits >= 0
        assert result.num_finished_splits >= 0
        assert result.throughput_mbps >= 0
    
    def test_result_splits_consistency(self, benchmark):
        """Test that finished splits <= total splits."""
        result = benchmark.run_query(1)
        assert result.num_finished_splits <= result.num_total_splits
    
    def test_result_repr_readable(self, benchmark):
        """Test that result has readable string representation."""
        result = benchmark.run_query(1)
        repr_str = repr(result)
        
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
        assert 'QueryResult' in repr_str

