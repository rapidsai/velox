#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Script to build (or rebuild) the Velox CUDF TPC-H Python bindings
# This builds the bindings in-place without installing them

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "Building Velox CUDF TPC-H Python Bindings"
echo "(In-place build - no installation)"
echo "================================================"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VELOX_ROOT="${VELOX_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
BUILD_DIR="${VELOX_BUILD_DIR:-/opt/velox-build/release}"

echo -e "${YELLOW}VELOX_ROOT:${NC} $VELOX_ROOT"
echo -e "${YELLOW}BUILD_DIR:${NC} $BUILD_DIR"

# Check if Velox is built
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory does not exist: $BUILD_DIR${NC}"
    echo "Please build Velox first with CUDF support enabled"
    exit 1
fi

# Detect build system (Ninja or Make)
if [ -f "$BUILD_DIR/build.ninja" ]; then
    BUILD_TOOL="ninja"
    echo -e "${YELLOW}Build system:${NC} Ninja"
elif [ -f "$BUILD_DIR/Makefile" ]; then
    BUILD_TOOL="make"
    echo -e "${YELLOW}Build system:${NC} Make"
else
    echo -e "${RED}Error: No build system found (neither build.ninja nor Makefile)${NC}"
    exit 1
fi

# Build the C++ Python bridge library
echo ""
echo "Building C++ Python bridge library..."
WRAPPER_LIB="$BUILD_DIR/velox/experimental/cudf/benchmarks/python/cpp/tpch/libpython_benchmark_bridge_tpch.so"

cd "$BUILD_DIR"
if [ "$BUILD_TOOL" = "ninja" ]; then
    ninja python_benchmark_bridge_tpch
elif [ "$BUILD_TOOL" = "make" ]; then
    make python_benchmark_bridge_tpch -j8
fi

if [ ! -f "$WRAPPER_LIB" ]; then
    echo -e "${RED}Error: Failed to build C++ wrapper library: $WRAPPER_LIB${NC}"
    exit 1
else
    echo -e "${GREEN}✓ C++ wrapper library built successfully${NC}"
fi

# Set environment variables for setup.py
export VELOX_ROOT
export VELOX_BUILD_DIR="$BUILD_DIR"

# Check if Python and pip are available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip not found${NC}"
    exit 1
fi

PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
$PIP_CMD install -q cython numpy setuptools wheel

# Build the Python bindings (in-place, no installation)
echo ""
echo "Building Python bindings..."
cd "$SCRIPT_DIR"

# Clean previous builds
rm -rf build/ dist/ *.egg-info bindings/tpch/*.cpp
echo "Cleaned previous build artifacts"

# Build extensions in-place
echo ""
echo "Building Cython extensions in-place..."
# --inplace: Build extension modules directly in the source tree
# --force: Force rebuild even if files haven't changed
python3 setup.py build_ext --inplace --force

echo ""
echo -e "${GREEN}✓ Python bindings built successfully!${NC}"

# Test import
echo ""
echo "Testing import..."
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python3 -c "import cudf_tpch_benchmark; print('✓ Import successful - cudf_tpch_benchmark module loaded')"

echo ""
echo "=========================================="
echo -e "${GREEN}Build completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Bindings built in-place at: $SCRIPT_DIR"
echo ""
echo "To run the example:"
echo "  export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH"
echo "  python3 $SCRIPT_DIR/examples/example_usage.py --data-path /path/to/tpch/data --query 1"
echo ""
echo "Note: The bindings are built in-place and NOT installed."
echo "      Add $SCRIPT_DIR to PYTHONPATH to use them."
echo ""
echo "For performance benchmarking with ASV, see:"
echo "  $VELOX_ROOT/asv_benchmarks/README.md"
echo ""

