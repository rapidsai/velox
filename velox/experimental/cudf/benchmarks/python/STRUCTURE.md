# Directory Structure

This document describes the organization of the Velox CUDF Python bindings.

## Overview

The codebase is organized into a clean, modular structure that separates concerns and makes it easy to add new benchmark suites (like TPC-DS in the future).

## Directory Layout

```
velox/experimental/cudf/benchmarks/python/
    ├── cpp/                      # C++ bridge/wrapper code
    │   └── tpch/                 # TPC-H specific bridge
    │       ├── PythonBenchmarkBridge.cpp
    │       ├── PythonBenchmarkBridge.h
    │       └── CMakeLists.txt
    │
    ├── bindings/                 # Python Cython bindings
    │   └── tpch/                 # TPC-H specific bindings
    │       ├── cudf_tpch_benchmark.pyx
    │       ├── cudf_tpch_benchmark.pxd
    │       └── __init__.py
    │
    ├── examples/                 # Example usage scripts
    │   ├── example_usage.py
    │   └── __init__.py
    │
    ├── docs/                     # Documentation
    │   ├── INSTALL.md           # Installation instructions
    │   ├── QUICKSTART.md        # Quick start guide
    │   ├── ARCHITECTURE.md      # Architecture overview
    │   └── CHANGES.md           # Changelog
    │
    ├── CMakeLists.txt           # Top-level CMake configuration
    ├── setup.py                 # Python package setup
    ├── pyproject.toml          # Python project metadata
    ├── requirements.txt        # Python dependencies
    ├── Makefile                # Build automation
    ├── build.sh                # Build script
    ├── README.md               # Main documentation
    ├── PROJECT_SUMMARY.md      # Project summary
    └── STRUCTURE.md            # This file
```

## Component Descriptions

### cpp/
Contains C++ bridge libraries that wrap native Velox benchmark code and expose a C API for Python.

- **tpch/** - TPC-H benchmark bridge
  - Creates `libpython_benchmark_bridge_tpch.so`
  - Provides C API functions: `initialize_runtime()`, `create_benchmark()`, `run_query_with_stats()`, etc.

### bindings/
Contains Cython bindings that wrap the C APIs into Python classes.

- **tpch/** - TPC-H benchmark Python bindings
  - `cudf_tpch_benchmark.pyx` - Main Cython implementation
  - `cudf_tpch_benchmark.pxd` - C declarations for Cython
  - Provides Python classes: `CudfTpchBenchmark`, `QueryResult`, etc.

### examples/
Example scripts demonstrating how to use the Python bindings.

### docs/
All documentation files organized in one place.

## Performance Benchmarking

For automated performance benchmarking with ASV (Airspeed Velocity), see the `asv_benchmarks` directory at the Velox project root (`/path/to/velox/asv_benchmarks/`). This contains benchmarks that use these Python bindings.

## Adding New Benchmark Suites (e.g., TPC-DS)

To add TPC-DS support in the future, follow this pattern:

1. **Create C++ bridge:**
   ```
   cpp/tpcds/
   ├── PythonBenchmarkBridge.cpp
   ├── PythonBenchmarkBridge.h
   └── CMakeLists.txt
   ```

2. **Create Python bindings:**
   ```
   bindings/tpcds/
   ├── cudf_tpcds_benchmark.pyx
   ├── cudf_tpcds_benchmark.pxd
   └── __init__.py
   ```

3. **Update CMakeLists.txt:**
   ```cmake
   add_subdirectory(cpp/tpch)
   add_subdirectory(cpp/tpcds)  # Add this line
   ```

4. **Update setup.py:**
   Add a new Extension for the TPC-DS bindings.

5. **Add examples:**
   Create `examples/example_tpcds_usage.py`

## Benefits of This Structure

1. **Separation of Concerns:** C++ bridges, Python bindings, examples, and docs are clearly separated
2. **Scalability:** Easy to add new benchmark suites without restructuring
3. **Maintainability:** Each component has a clear purpose and location
4. **Discoverability:** Developers can quickly find what they need
5. **Build Modularity:** Each benchmark suite can be built independently

## Build Artifacts

When built, the structure in the build directory mirrors the source:
```
_build/release/velox/experimental/cudf/benchmarks/python/
└── cpp/
    └── tpch/
        └── libpython_benchmark_bridge_tpch.so
```

The Python extension modules are built in-place at the project root:
```
cudf_tpch_benchmark.cpython-*.so
```

