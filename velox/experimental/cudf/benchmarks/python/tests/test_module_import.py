"""
Tests for module imports and basic structure.
"""
import pytest


class TestModuleImport:
    """Test suite for module imports."""
    
    def test_import_module(self):
        """Test that the module can be imported."""
        import cudf_tpch_benchmark
        assert cudf_tpch_benchmark is not None
    
    def test_import_benchmark_class(self):
        """Test importing CudfTpchBenchmark class."""
        from cudf_tpch_benchmark import CudfTpchBenchmark
        assert CudfTpchBenchmark is not None
    
    def test_import_query_result_class(self):
        """Test importing QueryResult class."""
        from cudf_tpch_benchmark import QueryResult
        assert QueryResult is not None
    
    def test_import_benchmark_error(self):
        """Test importing BenchmarkError exception."""
        from cudf_tpch_benchmark import BenchmarkError
        assert BenchmarkError is not None
    
    def test_import_shutdown_function(self):
        """Test importing shutdown function."""
        from cudf_tpch_benchmark import shutdown
        assert callable(shutdown)
    
    def test_all_exports(self):
        """Test that all expected exports are available."""
        import cudf_tpch_benchmark as module
        
        expected_exports = [
            'CudfTpchBenchmark',
            'QueryResult',
            'BenchmarkError',
            'shutdown'
        ]
        
        for export in expected_exports:
            assert hasattr(module, export), f"Missing export: {export}"


class TestClassStructure:
    """Test suite for class structure and methods."""
    
    def test_benchmark_has_required_methods(self):
        """Test that CudfTpchBenchmark has all required methods."""
        from cudf_tpch_benchmark import CudfTpchBenchmark
        
        required_methods = [
            '__init__',
            'run_query',
            'run_all_queries',
            'close',
            '__enter__',
            '__exit__'
        ]
        
        for method in required_methods:
            assert hasattr(CudfTpchBenchmark, method), f"Missing method: {method}"
    
    def test_query_result_has_required_attributes(self):
        """Test that QueryResult has all required attributes."""
        from cudf_tpch_benchmark import QueryResult
        
        result = QueryResult(
            execution_time_ms=100.0,
            raw_input_bytes=1000,
            num_total_splits=5,
            num_finished_splits=5
        )
        
        required_attrs = [
            'execution_time_ms',
            'raw_input_bytes',
            'num_total_splits',
            'num_finished_splits',
            'throughput_mbps'
        ]
        
        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"

