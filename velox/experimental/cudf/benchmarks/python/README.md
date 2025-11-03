# Velox CUDF TPC-H Python Bindings

Python bindings for the Velox CUDF TPC-H benchmarks using Cython.

## Overview

This package provides Python bindings to the Velox CUDF-accelerated TPC-H benchmark suite, allowing you to:

- Run TPC-H queries (Q1-Q22) from Python
- Collect detailed performance metrics (execution time, throughput, I/O statistics)
- Integrate with custom benchmarking and analysis workflows
- Programmatically execute queries and collect results

## Prerequisites

1. **Build Velox with CUDF support**:
   - Follow the Velox build instructions
   - Ensure CUDF is enabled in the build

2. **Python requirements**:
   - Python 3.7 or higher
   - Cython 0.29 or higher

3. **TPC-H data**:
   - Generate TPC-H data in Parquet format
   - Set the data path when running benchmarks

## Installation

### Quick Start with build.sh

The easiest way to build everything:

```bash
cd /path/to/velox/velox/experimental/cudf/benchmarks/python

# Set build directory (if not default)
export VELOX_ROOT=/path/to/velox
export VELOX_BUILD_DIR=/path/to/velox/_build/release

# Build C++ wrapper and Python bindings
./build.sh
```

The `build.sh` script will:
1. Build the C++ wrapper library (`libpython_benchmark_bridge_tpch.so`)
2. Build Python bindings in-place (no installation)
3. Verify imports work correctly

**Note**: Bindings are built in-place, not installed. Set `PYTHONPATH` to use them:
```bash
export PYTHONPATH=/path/to/velox/velox/experimental/cudf/benchmarks/python:$PYTHONPATH
```

### Manual Installation (Advanced)

If you prefer to install the package:

#### Step 1: Build Velox with Python bindings

```bash
cd /path/to/velox
mkdir -p _build/release
cd _build/release

cmake ../.. \
  -DCMAKE_BUILD_TYPE=Release \
  -DVELOX_ENABLE_CUDF=ON

make -j$(nproc)
```

#### Step 2: Install Python bindings

```bash
cd /path/to/velox/velox/experimental/cudf/benchmarks/python

export VELOX_ROOT=/path/to/velox
export VELOX_BUILD_DIR=/path/to/velox/_build/release

pip install -e .
```

## Usage

### Basic Usage

```python
from cudf_tpch_benchmark import CudfTpchBenchmark

# Create benchmark instance
benchmark = CudfTpchBenchmark(
    data_path="/path/to/tpch/data",
    data_format="parquet",
    num_drivers=4,
    cudf_gpu_batch_size_rows=100000
)

# Run a single query
result = benchmark.run_query(1)
print(f"Query 1 took {result.execution_time_ms:.2f}ms")
print(f"Throughput: {result.throughput_mbps:.2f} MB/s")

# Run all queries
results = benchmark.run_all_queries()
for query_id, result in results.items():
    print(f"Query {query_id}: {result}")

# Clean up
benchmark.close()
```

### Using Context Manager

```python
with CudfTpchBenchmark(data_path="/path/to/tpch/data") as benchmark:
    result = benchmark.run_query(6)
    print(result)
# Automatically cleaned up
```

### Example Script

An example script is provided:

```bash
python example_usage.py --data-path /path/to/tpch/data --query 1
```

Run all queries:
```bash
python example_usage.py --data-path /path/to/tpch/data
```

## API Reference

### `CudfTpchBenchmark`

Main class for running benchmarks.

**Constructor Parameters:**
- `data_path` (str): Path to TPC-H data directory
- `data_format` (str): Data format ("parquet", "orc", etc.). Default: "parquet"
- `num_drivers` (int): Number of driver threads. Default: 4
- `num_splits_per_file` (int): Number of splits per file. Default: 10
- `include_results` (bool): Whether to include query results. Default: False
- `cudf_chunk_read_limit` (int): Chunk read limit for CUDF. Default: 0
- `cudf_pass_read_limit` (int): Pass read limit for CUDF. Default: 0
- `cudf_gpu_batch_size_rows` (int): GPU batch size in rows. Default: 100000
- `velox_cudf_table_scan` (bool): Enable CUDF table scan. Default: True

**Methods:**
- `run_query(query_id: int) -> QueryResult`: Run a specific query (1-22)
- `run_all_queries() -> dict`: Run all 22 queries, returns dict of results
- `close()`: Clean up resources

### `QueryResult`

Contains results from a query execution.

**Attributes:**
- `execution_time_ms` (float): Execution time in milliseconds
- `raw_input_bytes` (int): Total bytes read
- `num_total_splits` (int): Total number of splits
- `num_finished_splits` (int): Number of completed splits
- `throughput_mbps` (float): Throughput in MB/s (calculated)

## Performance Tuning

### GPU Batch Size

Adjust the `cudf_gpu_batch_size_rows` parameter to optimize for your GPU:

```python
# Smaller batches (better for limited GPU memory)
benchmark = CudfTpchBenchmark(
    data_path="/path/to/data",
    cudf_gpu_batch_size_rows=50000
)

# Larger batches (better for high-end GPUs)
benchmark = CudfTpchBenchmark(
    data_path="/path/to/data",
    cudf_gpu_batch_size_rows=200000
)
```

### Number of Drivers

Adjust parallelism with `num_drivers`:

```python
# More parallelism
benchmark = CudfTpchBenchmark(
    data_path="/path/to/data",
    num_drivers=8
)
```

### Data Format

Different formats can have different performance characteristics:

```python
# Parquet (recommended)
benchmark = CudfTpchBenchmark(
    data_path="/path/to/data",
    data_format="parquet"
)

# ORC
benchmark = CudfTpchBenchmark(
    data_path="/path/to/data",
    data_format="orc"
)
```

## Running in Docker

If you're running inside a Docker container:

### 1. Mount Your Repository

```bash
docker run -it \
  -v /host/path/to/velox:/workspace/velox \
  -p 8080:8080 \
  your-image
```

### 2. Run Examples in Docker

```bash
export PYTHONPATH=/workspace/velox/velox/experimental/cudf/benchmarks/python:$PYTHONPATH
python3 /workspace/velox/velox/experimental/cudf/benchmarks/python/examples/example_usage.py \
  --data-path /data/tpch --query 6
```

### 3. Git Safe Directory (if needed)

If you get git errors about "dubious ownership", add the directory as safe:

```bash
git config --global --add safe.directory /workspace/velox
```

## Troubleshooting

### Import Error: Cannot find shared library

Make sure the C++ wrapper library is in your library path:

```bash
export LD_LIBRARY_PATH=/path/to/velox/_build/release/velox/experimental/cudf/benchmarks/python:$LD_LIBRARY_PATH
```

Or add it to the setup.py rpath.

### Query Execution Fails

- Verify data path is correct and contains TPC-H tables
- Check that data is in the correct format
- Ensure you have sufficient GPU memory
- Try reducing `cudf_gpu_batch_size_rows`

## Development

### Building for Development

Use the provided build script to build the C++ wrapper and Python bindings:

```bash
cd velox/experimental/cudf/benchmarks/python
./build.sh
```

This builds everything in-place without installation. Set `PYTHONPATH` to use the bindings:

```bash
export PYTHONPATH=/path/to/velox/velox/experimental/cudf/benchmarks/python:$PYTHONPATH
python3 -c "import cudf_tpch_benchmark; print('Import successful')"
```

### Rebuilding After C++ or Python Changes

The `build.sh` script handles rebuilding both C++ and Python components:

```bash
cd velox/experimental/cudf/benchmarks/python
./build.sh
```

**What it does:**
1. Rebuilds the C++ wrapper library (`libpython_benchmark_bridge_tpch.so`)
2. Cleans and rebuilds Python bindings in-place
3. Verifies the import works

**Manual rebuild (if needed):**

```bash
# If you only need to rebuild Python bindings (C++ unchanged)
cd velox/experimental/cudf/benchmarks/python
python3 setup.py build_ext --inplace --force

# If you need to rebuild C++ wrapper
cd /path/to/velox/_build/release
ninja python_benchmark_bridge_tpch  # or: make python_benchmark_bridge_tpch -j8
```

## Testing

A comprehensive test suite is provided for the Python bindings.

### Installing Test Dependencies

```bash
# Install test dependencies
pip install -e .[test]

# Or manually
pip install pytest pytest-cov pytest-xdist
```

### Running Tests

#### Quick Start

Use the provided test runner script:

```bash
# Run all tests
./run_tests.sh

# Run with coverage report
./run_tests.sh --coverage

# Run specific test subset
./run_tests.sh --subset unit        # Only unit tests (no data required)
./run_tests.sh --subset integration # Integration tests (requires data)

# Run with TPC-H data
./run_tests.sh --data-path /path/to/tpch/data

# Verbose output
./run_tests.sh --verbose
```

#### Using pytest Directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_benchmark_queries.py -v

# Run specific test class
pytest tests/test_query_result.py::TestQueryResult -v

# Run specific test method
pytest tests/test_benchmark_queries.py::TestBenchmarkQueries::test_run_query_valid_id -v

# Run with coverage
pytest tests/ --cov=cudf_tpch_benchmark --cov-report=html
```

#### Running Without TPC-H Data

Some tests don't require actual TPC-H data:

```bash
# Tests that work without data
pytest tests/test_module_import.py tests/test_benchmark_exceptions.py -v
```

Tests requiring data will be automatically skipped if `TPCH_DATA_PATH` is not set:

```bash
# Set data path for tests that need it
export TPCH_DATA_PATH=/path/to/tpch/data
pytest tests/ -v
```

### Test Structure

The test suite includes:

- **`test_module_import.py`**: Module import and structure tests
- **`test_query_result.py`**: QueryResult class tests
- **`test_benchmark_init.py`**: Benchmark initialization tests
- **`test_benchmark_queries.py`**: Query execution tests
- **`test_benchmark_exceptions.py`**: Exception handling tests
- **`test_integration.py`**: End-to-end integration tests

### Test Coverage

View coverage report after running with `--coverage`:

```bash
# Generate HTML coverage report
pytest tests/ --cov=cudf_tpch_benchmark --cov-report=html

# Open in browser
firefox htmlcov/index.html  # or your browser of choice
```

For more details, see the [test documentation](tests/README.md).

## License

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the Apache License, Version 2.0. See the LICENSE file in the Velox repository for details.

## Contributing

Contributions are welcome! Please follow the Velox contribution guidelines.

## Support

For issues and questions:
- Velox Issues: https://github.com/facebookincubator/velox/issues
- Velox Discussions: https://github.com/facebookincubator/velox/discussions
