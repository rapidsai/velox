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

include_guard(GLOBAL)

# 4.0 is the minimum version required by cudf
cmake_minimum_required(VERSION 4.0)

# rapids_cmake commit 9c0829e from 2026-07-23 (release/26.08 branch)
set(VELOX_rapids_cmake_VERSION 26.08)
set(VELOX_rapids_cmake_COMMIT 9c0829ec73702b3df8a5c2ec43f6aaabe5f1e5ec)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  3582f621f3b3d63952aafd716985901a075f2ad85602073973f3619761723962
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 1a39f9e from 2026-07-24 (release/26.08 branch)
set(VELOX_rmm_VERSION 26.08)
set(VELOX_rmm_COMMIT 1a39f9e81c467b1a9522a4dcec8b5581ae165fef)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  d8b82bc491a2c5b093cdfb4e1fb99630ccf16d1cbd69bc74451fbb56efc3e68c
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# Caller-owned S3 receive requires the KvikIO research implementation
# for bounded pinned-host staging, event-fenced H2D copies, and strict
# path accounting, and bounded adaptive TCP MSS reuse. Preserve
# network-monitor pin for ordinary cuDF builds.
if(VELOX_ENABLE_S3_DIRECT_RECEIVE)
  set(VELOX_kvikio_VERSION 26.10)
  set(VELOX_kvikio_COMMIT 7bb1345a4078e47e1490031c7b5bd93042185b8d)
  set(
    VELOX_kvikio_BUILD_SHA256_CHECKSUM
    617eafffccfdd6b3f1f404b87890469f46aec5a3e3813a5322bdebfe9b6c3c18
  )
  set(
    VELOX_kvikio_SOURCE_URL
    "https://github.com/kjmph/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
  )
else()
  # kvikio PR 999 head from 2026-08-19 (network-monitor branch)
  set(VELOX_kvikio_VERSION 26.10)
  set(VELOX_kvikio_COMMIT 496acd854f0e49cf212e6c6c6ed2e8c945650b20)
  set(
    VELOX_kvikio_BUILD_SHA256_CHECKSUM
    691be85e9cc454eb833f93d1bd6588f7d654153c04cdfc7b37474cc73d7de2e8
  )
  set(
    VELOX_kvikio_SOURCE_URL
    "https://github.com/kingcrimsontianyu/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
  )
endif()
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
# GPU input uses cuDF's lifetime-safe asynchronous reads and batched datasource
# interface, including its exported host worker pool, regardless of the selected
# S3 reader mode. Keep one pin so both build modes expose the same datasource ABI.
set(VELOX_cudf_COMMIT 7e91464b11d68a13d36500d9f2b4bc067a24154e)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  13632b88eaf2b012174faa8c65f3b84444a18e107a4c02a104f9b06a3404c599
)
set(VELOX_cudf_SOURCE_URL "https://github.com/kjmph/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a CUDA-enabled system UCX install, to pick the default for
# VELOX_ENABLE_UCX_EXCHANGE below. velox_ucx_exchange runs its own
# find_package(ucx REQUIRED); this probe only decides whether we opt in by
# default and whether ucxx is fetched. libucp alone is insufficient: a distro
# can ship a CPU-only UCX that accepts CUDA pointers at compile time but fails
# transfers at runtime because the dynamically loaded cuda_copy transport is
# absent.
find_library(UCX_LIBRARY NAMES ucp)
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
unset(VELOX_UCX_CUDA_LIBRARY CACHE)
if(UCX_LIBRARY)
  get_filename_component(UCX_LIBRARY_DIR "${UCX_LIBRARY}" DIRECTORY)
  find_library(VELOX_UCX_CUDA_LIBRARY NAMES uct_cuda PATHS "${UCX_LIBRARY_DIR}/ucx" NO_DEFAULT_PATH)
endif()
if(UCX_LIBRARY AND UCX_INCLUDE_DIR AND VELOX_UCX_CUDA_LIBRARY)
  set(UCX_FOUND TRUE)
else()
  set(UCX_FOUND FALSE)
endif()

# Whether to build the experimental UCX GPU exchange transport
# (velox/experimental/ucx-exchange) and the cuDF-side registration that selects
# it. Defaults to whether a system UCX was found, which reproduces the earlier
# implicit behaviour, but can be forced either way from the command line --
# -DVELOX_ENABLE_UCX_EXCHANGE=OFF is how the no-UCX configuration is exercised
# on a host that does have UCX. Declared here rather than next to the other
# options because the default depends on the probe above; cache variables are
# global, so every subdirectory sees it. Requires VELOX_ENABLE_CUDF, since this
# file is only reached when cuDF is enabled and the transport links cudf::cudf.
option(
  VELOX_ENABLE_UCX_EXCHANGE
  "Build the experimental UCX GPU exchange transport. Requires a CUDA-enabled system UCX install."
  ${UCX_FOUND}
)
if(VELOX_ENABLE_UCX_EXCHANGE AND NOT UCX_FOUND)
  message(
    FATAL_ERROR
    "VELOX_ENABLE_UCX_EXCHANGE=ON but no CUDA-enabled system UCX was found "
    "(need libucp, ucp/api/ucp.h, and the libuct_cuda transport module)."
  )
endif()

if(VELOX_ENABLE_UCX_EXCHANGE)
  message(
    STATUS
    "UCX exchange enabled with ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}; "
    "CUDA transport: ${VELOX_UCX_CUDA_LIBRARY}) -- ucxx will be fetched"
  )
  # ucxx commit b7faed1 from 2026-07-23 (release/0.51 branch)
  set(VELOX_ucxx_VERSION 0.51)
  set(VELOX_ucxx_COMMIT b7faed1a2e8038f63676183cdb056c3b69daa15d)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    3eb5ff5459dde31edf344f24f0b3086550be70961038b69229b05973b6f37524
  )
  set(VELOX_ucxx_SOURCE_URL "https://github.com/rapidsai/ucxx/archive/${VELOX_ucxx_COMMIT}.tar.gz")
  velox_resolve_dependency_url(ucxx)
else()
  message(STATUS "UCX exchange disabled -- ucxx will not be fetched")
endif()

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
  set(BUILD_SHARED_LIBS ON)
  if(VELOX_ENABLE_S3_DIRECT_RECEIVE)
    # KvikIO 26.10 enables its Nsight plugin by default. It is not part of the
    # runtime data path and would add an unnecessary build/runtime dependency.
    set(KvikIO_BUILD_NSYS_PLUGIN OFF)
  endif()

  # TODO(mh,bd): Remove this once we have a permanent solution for the spdlog/fmt
  # incompatibility.

  # cuDF (via rapids_logger) pins spdlog 1.14.1, which is incompatible with
  # the fmt 11.2.0 that Velox builds. Override the rapids-cmake/CPM spdlog
  # version to 1.15.3, which is fmt 11.2 compatible.
  # RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE is honored by every rapids_cpm_init,
  # so the override applies before rapids_logger fetches spdlog.
  set(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/cudf-cpm-overrides.json")

  FetchContent_Declare(
    rapids-cmake
    URL ${VELOX_rapids_cmake_SOURCE_URL}
    URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    rmm
    URL ${VELOX_rmm_SOURCE_URL}
    URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    kvikio
    URL ${VELOX_kvikio_SOURCE_URL}
    URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    cudf
    URL ${VELOX_cudf_SOURCE_URL}
    URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  if(VELOX_ENABLE_UCX_EXCHANGE)
    FetchContent_Declare(
      ucxx
      URL ${VELOX_ucxx_SOURCE_URL}
      URL_HASH ${VELOX_ucxx_BUILD_SHA256_CHECKSUM}
      SOURCE_SUBDIR
      cpp
      UPDATE_DISCONNECTED 1
    )
  endif()

  FetchContent_MakeAvailable(cudf)

  if(VELOX_ENABLE_UCX_EXCHANGE)
    FetchContent_MakeAvailable(ucxx)
  endif()

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
  unset(BUILD_TESTING CACHE)
endblock()
