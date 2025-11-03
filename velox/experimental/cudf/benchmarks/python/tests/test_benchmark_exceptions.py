"""
Tests for exception handling in benchmark.
"""
import pytest
from cudf_tpch_benchmark import BenchmarkError


class TestBenchmarkError:
    """Test suite for BenchmarkError exception."""
    
    def test_benchmark_error_is_exception(self):
        """Test that BenchmarkError is an Exception."""
        assert issubclass(BenchmarkError, Exception)
    
    def test_benchmark_error_can_be_raised(self):
        """Test that BenchmarkError can be raised."""
        with pytest.raises(BenchmarkError):
            raise BenchmarkError("Test error")
    
    def test_benchmark_error_with_message(self):
        """Test BenchmarkError with custom message."""
        msg = "Custom error message"
        with pytest.raises(BenchmarkError, match=msg):
            raise BenchmarkError(msg)
    
    def test_benchmark_error_can_be_caught(self):
        """Test that BenchmarkError can be caught."""
        try:
            raise BenchmarkError("Test error")
        except BenchmarkError as e:
            assert str(e) == "Test error"
    
    def test_benchmark_error_inheritance(self):
        """Test that BenchmarkError can be caught as Exception."""
        try:
            raise BenchmarkError("Test error")
        except Exception as e:
            assert isinstance(e, BenchmarkError)

