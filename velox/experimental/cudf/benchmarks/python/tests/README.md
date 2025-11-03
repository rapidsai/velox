# Tests for cudf_tpch_benchmark Python Bindings

This directory contains comprehensive test suites for the Python bindings of the Velox CUDF TPC-H Benchmark.

## Test Organization

### Test Files

- **`test_module_import.py`**: Tests for module imports and basic structure
- **`test_query_result.py`**: Tests for the `QueryResult` class
- **`test_benchmark_init.py`**: Tests for `CudfTpchBenchmark` initialization
- **`test_benchmark_queries.py`**: Tests for query execution functionality
- **`test_benchmark_exceptions.py`**: Tests for exception handling
- **`test_integration.py`**: Integration tests for complete workflows

### Configuration

- **`conftest.py`**: Pytest configuration and shared fixtures
- **`__init__.py`**: Package marker

## Running Tests

### Prerequisites

1. **Install test dependencies:**
   ```bash
   pip install pytest pytest-cov pytest-xdist
   ```

2. **Build the extension:**
   ```bash
   cd /raid/avinash/projects/velox/velox/experimental/cudf/benchmarks/python
   python setup.py build_ext --inplace
   ```

### Running All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cudf_tpch_benchmark --cov-report=html
```

### Running Specific Test Files

```bash
# Run only import tests
pytest tests/test_module_import.py -v

# Run only query tests
pytest tests/test_benchmark_queries.py -v

# Run only integration tests
pytest tests/test_integration.py -v
```

### Running Specific Test Classes or Methods

```bash
# Run specific test class
pytest tests/test_query_result.py::TestQueryResult -v

# Run specific test method
pytest tests/test_benchmark_queries.py::TestBenchmarkQueries::test_run_query_valid_id -v
```

## Test Data

Many tests require actual TPC-H data to run properly. Set the `TPCH_DATA_PATH` environment variable to point to your TPC-H dataset:

```bash
export TPCH_DATA_PATH=/path/to/tpch/data
pytest tests/ -v
```

If `TPCH_DATA_PATH` is not set, tests requiring data will be automatically skipped.

### Running Without Data

Some tests (like import tests, exception tests, and basic structure tests) don't require real data:

```bash
# Run tests that don't need data
pytest tests/test_module_import.py tests/test_benchmark_exceptions.py -v
```

## Test Coverage

The test suite covers:

### Unit Tests
- Module imports and structure
- `QueryResult` class initialization and calculations
- `CudfTpchBenchmark` initialization with various parameters
- Query execution (single and batch)
- Error handling and validation
- Context manager usage
- Resource cleanup

### Integration Tests
- Complete benchmark workflows
- Multiple query execution
- Sequential benchmark instances
- Configuration variations
- Error recovery

### Edge Cases
- Invalid query IDs
- Zero and negative values
- Boundary conditions
- Partial splits
- Empty or missing data

## Continuous Integration

To integrate these tests into CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Python binding tests
  env:
    TPCH_DATA_PATH: ${{ secrets.TPCH_DATA_PATH }}
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=cudf_tpch_benchmark --cov-report=xml
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Using Fixtures

```python
def test_my_feature(benchmark_config, skip_without_data):
    """Test description."""
    benchmark = CudfTpchBenchmark(**benchmark_config)
    # ... test code ...
    benchmark.close()
```

### Skipping Tests Without Data

```python
@pytest.mark.skipif(not has_real_data(), reason="Requires TPC-H data")
def test_with_data(benchmark):
    # Test that requires real data
    pass
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the extension is built with `python setup.py build_ext --inplace`

2. **Tests skipped**: Set `TPCH_DATA_PATH` to run data-dependent tests

3. **Segmentation faults**: Check that C++ library dependencies are correctly linked

4. **GPU errors**: Ensure CUDA is properly installed and GPU is available

### Debug Mode

Run tests with additional debug output:

```bash
pytest tests/ -v -s  # Don't capture output
pytest tests/ -v --log-cli-level=DEBUG  # Show debug logs
```

## Performance Testing

For performance benchmarking, use pytest-benchmark:

```bash
pip install pytest-benchmark
pytest tests/ --benchmark-only
```

## Contact

For issues or questions about the tests, please refer to the main project documentation or open an issue in the project repository.

