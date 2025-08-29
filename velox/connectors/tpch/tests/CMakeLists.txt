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
add_executable(velox_tpch_connector_test TpchConnectorTest.cpp)

add_test(velox_tpch_connector_test velox_tpch_connector_test)

target_link_libraries(
  velox_tpch_connector_test
  velox_tpch_connector
  velox_vector_test_lib
  velox_exec_test_lib
  velox_aggregates
  GTest::gtest
  GTest::gtest_main)

add_executable(velox_tpch_speed_test SpeedTest.cpp)

target_link_libraries(velox_tpch_speed_test PRIVATE
  velox_tpch_connector
  velox_exec
  velox_exec_test_lib_cudf
  velox_memory
  velox_cudf_parquet_connector
  fmt::fmt)
# Fix curl_url_strerror linker error for this cudf-enabled test
if(VELOX_ENABLE_CUDF)
  # Find the bundled curl library 
  set(BUNDLED_CURL_LIB_VERSIONED "${CMAKE_BINARY_DIR}/_deps/curl-build/lib/libcurl.so.4.8.0")
  set(BUNDLED_CURL_LIB "${CMAKE_BINARY_DIR}/_deps/curl-build/lib/libcurl.so")
  set(BUNDLED_CURL_DIR "${CMAKE_BINARY_DIR}/_deps/curl-build/lib")
  
  if(EXISTS "${BUNDLED_CURL_LIB_VERSIONED}")
    set(SELECTED_CURL_LIB "${BUNDLED_CURL_LIB_VERSIONED}")
  elseif(EXISTS "${BUNDLED_CURL_LIB}")
    set(SELECTED_CURL_LIB "${BUNDLED_CURL_LIB}")
  else()
    message(WARNING "[velox_tpch_speed_test] No bundled curl found, curl_url_strerror error likely")
    set(SELECTED_CURL_LIB "")
  endif()
  
  if(SELECTED_CURL_LIB)
    message(STATUS "[velox_tpch_speed_test] Using bundled curl: ${SELECTED_CURL_LIB}")
    # Set rpath so bundled curl is found at runtime
    target_link_options(velox_tpch_speed_test PRIVATE "LINKER:-rpath,${BUNDLED_CURL_DIR}")
    # Link bundled curl with whole-archive to force curl_url_strerror symbol availability
    target_link_libraries(velox_tpch_speed_test PRIVATE 
      -Wl,--whole-archive "${SELECTED_CURL_LIB}" -Wl,--no-whole-archive)
    message(STATUS "[velox_tpch_speed_test] Bundled curl linked successfully with --whole-archive")
  endif()
endif()

