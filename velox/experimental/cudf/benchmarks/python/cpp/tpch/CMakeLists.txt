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

# Build the Python bridge library for TPC-H benchmarks
# This wraps the existing CudfTpchBenchmark with a C API for Python bindings
add_library(python_benchmark_bridge_tpch SHARED
  PythonBenchmarkBridge.cpp
)

target_link_libraries(
  python_benchmark_bridge_tpch
  velox_cudf_exec
  velox_cudf_exec_test_lib
  velox_tpch_benchmark_lib
  velox_exec
  velox_core
  velox_common_base
  Folly::folly
  gflags::gflags
)

# Install the shared library
install(TARGETS python_benchmark_bridge_tpch
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Install headers for Python bindings
install(FILES PythonBenchmarkBridge.h
  DESTINATION include/velox/experimental/cudf/benchmarks/python/cpp/tpch
)
