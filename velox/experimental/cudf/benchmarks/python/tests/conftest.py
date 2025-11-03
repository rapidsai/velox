"""
Pytest configuration and fixtures for cudf_tpch_benchmark tests.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_path():
    """
    Fixture that provides a path to test data.
    
    By default, looks for TPCH_DATA_PATH environment variable.
    If not set, uses a temporary directory (tests will be limited).
    """
    data_path = os.environ.get('TPCH_DATA_PATH')
    if data_path and os.path.exists(data_path):
        return data_path
    
    # Create temporary directory for tests without real data
    temp_dir = tempfile.mkdtemp(prefix="tpch_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def has_real_data(test_data_path):
    """Check if real TPC-H data is available."""
    # Check for common TPC-H table files
    required_tables = ['customer', 'lineitem', 'nation', 'orders', 'part', 
                       'partsupp', 'region', 'supplier']
    
    for table in required_tables:
        parquet_file = Path(test_data_path) / f"{table}.parquet"
        if not parquet_file.exists():
            return False
    return True


@pytest.fixture
def benchmark_config(test_data_path):
    """Provides default configuration for benchmark tests."""
    return {
        'data_path': test_data_path,
        'data_format': 'parquet',
        'num_drivers': 2,
        'num_splits_per_file': 2,
        'include_results': False,
        'cudf_chunk_read_limit': 0,
        'cudf_pass_read_limit': 0,
        'cudf_gpu_batch_size_rows': 10000,
        'velox_cudf_table_scan': True
    }


@pytest.fixture
def skip_without_data(has_real_data):
    """Skip tests that require real TPC-H data."""
    if not has_real_data:
        pytest.skip("Real TPC-H data not available. Set TPCH_DATA_PATH environment variable.")

