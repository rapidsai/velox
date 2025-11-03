#!/bin/bash
# Script to run Python binding tests for cudf_tpch_benchmark
# Copyright (c) Facebook, Inc. and its affiliates.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== CUDF TPC-H Benchmark - Python Tests ===${NC}"
echo ""

# Check if pytest is installed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}pytest not found. Installing test dependencies...${NC}"
    pip install -e .[test]
fi

# Check if the extension is built
if [ ! -f "cudf_tpch_benchmark*.so" ]; then
    echo -e "${YELLOW}Extension module not found. Building...${NC}"
    python setup.py build_ext --inplace
fi

# Default test options
TEST_ARGS=""
COVERAGE=false
SUBSET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --subset|-s)
            SUBSET="$2"
            shift 2
            ;;
        --data-path|-d)
            export TPCH_DATA_PATH="$2"
            shift 2
            ;;
        --verbose|-v)
            TEST_ARGS="$TEST_ARGS -vv"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage          Run tests with coverage report"
            echo "  -s, --subset <name>     Run specific test subset (import|unit|integration|all)"
            echo "  -d, --data-path <path>  Set TPC-H data path"
            echo "  -v, --verbose           Verbose output"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                  # Run all tests"
            echo "  $0 --coverage                       # Run with coverage"
            echo "  $0 --subset unit                    # Run only unit tests"
            echo "  $0 --data-path /path/to/data       # Run with TPC-H data"
            exit 0
            ;;
        *)
            TEST_ARGS="$TEST_ARGS $1"
            shift
            ;;
    esac
done

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
if [ -n "$TPCH_DATA_PATH" ]; then
    echo "  TPC-H Data Path: $TPCH_DATA_PATH"
else
    echo -e "  TPC-H Data Path: ${YELLOW}Not set (some tests will be skipped)${NC}"
fi
echo "  Coverage: $COVERAGE"
echo ""

# Determine which tests to run
if [ -n "$SUBSET" ]; then
    case $SUBSET in
        import)
            echo -e "${GREEN}Running import tests...${NC}"
            TEST_FILES="tests/test_module_import.py"
            ;;
        unit)
            echo -e "${GREEN}Running unit tests...${NC}"
            TEST_FILES="tests/test_module_import.py tests/test_query_result.py tests/test_benchmark_exceptions.py"
            ;;
        integration)
            echo -e "${GREEN}Running integration tests...${NC}"
            TEST_FILES="tests/test_benchmark_init.py tests/test_benchmark_queries.py tests/test_integration.py"
            ;;
        all|*)
            echo -e "${GREEN}Running all tests...${NC}"
            TEST_FILES="tests/"
            ;;
    esac
else
    echo -e "${GREEN}Running all tests...${NC}"
    TEST_FILES="tests/"
fi

# Run tests
if [ "$COVERAGE" = true ]; then
    echo ""
    pytest $TEST_FILES $TEST_ARGS \
        --cov=cudf_tpch_benchmark \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-report=xml
    
    echo ""
    echo -e "${GREEN}Coverage report generated:${NC}"
    echo "  HTML: file://$(pwd)/htmlcov/index.html"
    echo "  XML:  $(pwd)/coverage.xml"
else
    pytest $TEST_FILES $TEST_ARGS
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi

