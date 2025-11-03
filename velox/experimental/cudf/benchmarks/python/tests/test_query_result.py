"""
Tests for QueryResult class.
"""
import pytest
from cudf_tpch_benchmark import QueryResult


class TestQueryResult:
    """Test suite for QueryResult class."""
    
    def test_init_basic(self):
        """Test basic initialization of QueryResult."""
        result = QueryResult(
            execution_time_ms=1000.0,
            raw_input_bytes=1024 * 1024 * 100,  # 100 MB
            num_total_splits=10,
            num_finished_splits=10
        )
        
        assert result.execution_time_ms == 1000.0
        assert result.raw_input_bytes == 1024 * 1024 * 100
        assert result.num_total_splits == 10
        assert result.num_finished_splits == 10
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        # 100 MB processed in 1 second = 100 MB/s
        result = QueryResult(
            execution_time_ms=1000.0,
            raw_input_bytes=1024 * 1024 * 100,
            num_total_splits=10,
            num_finished_splits=10
        )
        assert abs(result.throughput_mbps - 100.0) < 0.01
    
    def test_throughput_zero_time(self):
        """Test throughput when execution time is zero."""
        result = QueryResult(
            execution_time_ms=0.0,
            raw_input_bytes=1024 * 1024 * 100,
            num_total_splits=10,
            num_finished_splits=10
        )
        assert result.throughput_mbps == 0.0
    
    def test_throughput_small_time(self):
        """Test throughput with small execution time."""
        # 100 MB processed in 100ms = 1000 MB/s
        result = QueryResult(
            execution_time_ms=100.0,
            raw_input_bytes=1024 * 1024 * 100,
            num_total_splits=5,
            num_finished_splits=5
        )
        assert abs(result.throughput_mbps - 1000.0) < 0.01
    
    def test_repr(self):
        """Test string representation."""
        result = QueryResult(
            execution_time_ms=1234.56,
            raw_input_bytes=1024 * 1024 * 50,
            num_total_splits=8,
            num_finished_splits=8
        )
        
        repr_str = repr(result)
        assert "QueryResult" in repr_str
        assert "1234.56ms" in repr_str
        assert "8/8" in repr_str
        assert "MB/s" in repr_str
    
    def test_partial_splits(self):
        """Test with partial splits completed."""
        result = QueryResult(
            execution_time_ms=500.0,
            raw_input_bytes=1024 * 1024,
            num_total_splits=10,
            num_finished_splits=5
        )
        
        assert result.num_total_splits == 10
        assert result.num_finished_splits == 5
        assert result.throughput_mbps > 0

