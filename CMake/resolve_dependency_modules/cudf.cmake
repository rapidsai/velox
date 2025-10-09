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

# 3.30.4 is the minimum version required by cudf
cmake_minimum_required(VERSION 3.30.4)

set(VELOX_rapids_cmake_VERSION 25.10)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  5c882a6ec2df1b8a17cd6aaa68d7a56f6df674de24d8ce86de62ad2270ac442d
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/b435ca821fbc08162937071a4b5ac41d4cdb5af3.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

set(VELOX_rmm_VERSION 25.10)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  c8ca7e1c20106cb0cb5caf626fa6a0b7e66428787286185ea3ac414bb2b43bcf
)
set(
  VELOX_rmm_SOURCE_URL
  "https://github.com/rapidsai/rmm/archive/7aaad1dee0690a48db8c92210593f6c70f6f7648.tar.gz"
)
velox_resolve_dependency_url(rmm)

set(VELOX_kvikio_VERSION 25.10)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  2c2f1833119de0c6103e250a622689ff75c6ec92d596ac265e08ea2bb72000ab
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/fb6220c4aa2fd2d07a8fbb6578f8447a9efe08e9.tar.gz"
)
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 25.10 CACHE STRING "cudf version")

set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  21200e1f8b3b140dc4dd45214e515368c5c713c39168c05e4c1755c2541bf727
)
set(
  VELOX_cudf_SOURCE_URL
  "https://github.com/rapidsai/cudf/archive/f4e35ca02118eada383e7417273c6cb1857ec66e.tar.gz"
)
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(BUILD_SHARED_LIBS ON)

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

  FetchContent_MakeAvailable(cudf)

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
endblock()
