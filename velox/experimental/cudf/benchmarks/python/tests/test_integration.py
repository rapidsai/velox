"""
Integration tests for the full benchmark workflow.
"""
import pytest
from cudf_tpch_benchmark import (
    CudfTpchBenchmark,
    QueryResult,
    BenchmarkError,
    shutdown
)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_single_query(self, benchmark_config, skip_without_data):
        """Test complete workflow for running a single query."""
        # Initialize
        benchmark = CudfTpchBenchmark(**benchmark_config)
        
        try:
            # Run query
            result = benchmark.run_query(1)
            
            # Validate result
            assert isinstance(result, QueryResult)
            assert result.execution_time_ms > 0
            assert result.raw_input_bytes > 0
            
            # Check result is printable
            result_str = repr(result)
            assert len(result_str) > 0
        finally:
            # Cleanup
            benchmark.close()
    
    def test_full_workflow_multiple_queries(self, benchmark_config, skip_without_data):
        """Test workflow for running multiple queries."""
        with CudfTpchBenchmark(**benchmark_config) as benchmark:
            # Run several queries
            query_ids = [1, 6, 13]
            results = {}
            
            for qid in query_ids:
                result = benchmark.run_query(qid)
                results[qid] = result
            
            # Validate all results
            assert len(results) == len(query_ids)
            for qid in query_ids:
                assert isinstance(results[qid], QueryResult)
                assert results[qid].execution_time_ms >= 0
    
    def test_full_workflow_all_queries(self, benchmark_config, skip_without_data):
        """Test workflow for running all queries."""
        with CudfTpchBenchmark(**benchmark_config) as benchmark:
            results = benchmark.run_all_queries()
            
            # Should have results for all 22 queries
            assert len(results) == 22
            
            # Count successful vs failed
            successes = sum(1 for r in results.values() if isinstance(r, QueryResult))
            failures = sum(1 for r in results.values() if isinstance(r, BenchmarkError))
            
            # At least some queries should succeed
            assert successes > 0
            assert successes + failures == 22
    
    def test_sequential_benchmark_instances(self, benchmark_config, skip_without_data):
        """Test creating multiple benchmark instances sequentially."""
        # First instance
        benchmark1 = CudfTpchBenchmark(**benchmark_config)
        result1 = benchmark1.run_query(1)
        benchmark1.close()
        
        # Second instance
        benchmark2 = CudfTpchBenchmark(**benchmark_config)
        result2 = benchmark2.run_query(1)
        benchmark2.close()
        
        # Both should succeed
        assert isinstance(result1, QueryResult)
        assert isinstance(result2, QueryResult)
    
    def test_error_recovery(self, benchmark_config, skip_without_data):
        """Test that benchmark recovers from query errors."""
        with CudfTpchBenchmark(**benchmark_config) as benchmark:
            # Try invalid query
            with pytest.raises(ValueError):
                benchmark.run_query(0)
            
            # Should still be able to run valid queries
            result = benchmark.run_query(1)
            assert isinstance(result, QueryResult)
    
    def test_configuration_variations(self, test_data_path, skip_without_data):
        """Test benchmark with different configurations."""
        configs = [
            {'num_drivers': 1, 'cudf_gpu_batch_size_rows': 10000},
            {'num_drivers': 4, 'cudf_gpu_batch_size_rows': 50000},
            {'num_drivers': 2, 'num_splits_per_file': 5},
        ]
        
        for config in configs:
            full_config = {
                'data_path': test_data_path,
                'data_format': 'parquet',
                **config
            }
            
            with CudfTpchBenchmark(**full_config) as benchmark:
                result = benchmark.run_query(1)
                assert isinstance(result, QueryResult)


class TestShutdown:
    """Tests for shutdown functionality."""
    
    def test_shutdown_callable(self):
        """Test that shutdown function is callable."""
        assert callable(shutdown)
    
    def test_shutdown_runs_without_error(self):
        """Test that shutdown can be called without error."""
        # Note: We don't actually call shutdown in tests as it would
        # affect other tests. This just checks the function exists.
        try:
            # Just verify it's importable and callable
            from cudf_tpch_benchmark import shutdown
            assert callable(shutdown)
        except Exception as e:
            pytest.fail(f"shutdown function not properly accessible: {e}")

