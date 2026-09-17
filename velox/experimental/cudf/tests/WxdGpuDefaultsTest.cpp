/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/WxdGpuDefaults.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <string>

namespace facebook::velox::cudf_velox::test {

// Fork each case: never change the test runner's environment or initialize
// UCX/CUDA, and no GPU is required to exercise precedence and profile
// isolation.
TEST(WxdGpuDefaultsTest, explicitEnvironmentWins) {
  ASSERT_EXIT(
      {
        setenv("UCX_TLS", "tcp,cuda_copy", 1);
        setenv("KVIKIO_REMOTE_IO_BACKEND", "EASY_THREADPOOL", 1);
        setenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS", "64", 1);
        setenv("KVIKIO_TASK_SIZE", "", 1);
        applyWxdGpuEnvironmentDefaults(true, true);
        const bool valid = std::string(getenv("UCX_TLS")) == "tcp,cuda_copy" &&
            std::string(getenv("KVIKIO_REMOTE_IO_BACKEND")) ==
                "EASY_THREADPOOL" &&
            std::string(getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) ==
                "64" &&
            std::string(getenv("KVIKIO_TASK_SIZE")).empty();
        _exit(valid ? 0 : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(WxdGpuDefaultsTest, gpuDefaultsWithoutLauncher) {
  ASSERT_EXIT(
      {
        unsetenv("UCX_TLS");
        unsetenv("UCX_MAX_RNDV_RAILS");
        unsetenv("UCX_RNDV_PIPELINE_ERROR_HANDLING");
        unsetenv("UCX_RNDV_FRAG_SIZE");
        unsetenv("UCX_RNDV_FRAG_MEM_TYPES");
        unsetenv("UCX_RNDV_FRAG_ALLOC_COUNT");
        unsetenv("UCX_SOCKADDR_TLS_PRIORITY");
        unsetenv("UCX_CUDA_IPC_CACHE");
        unsetenv("UCX_CUDA_IPC_CACHE_MAX_REGIONS");
        unsetenv("UCX_CUDA_IPC_CACHE_MAX_SIZE");
        unsetenv("UCX_CUDA_IPC_ENABLE_GET_ZCOPY");
        unsetenv("UCX_CUDA_COPY_ASYNC_MEM_TYPE");
        unsetenv("UCX_NET_DEVICES");
        unsetenv("KVIKIO_TASK_SIZE");
        unsetenv("KVIKIO_REMOTE_IO_BACKEND");
        unsetenv("KVIKIO_REMOTE_DIRECT_RECEIVE");
        unsetenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS");
        unsetenv("KVIKIO_REMOTE_IO_NUM_REACTORS");
        unsetenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH");
        unsetenv("KVIKIO_NTHREADS");
        unsetenv("KVIKIO_NUM_THREADS");
        applyWxdGpuEnvironmentDefaults(true, true);
        const bool valid =
            std::string(getenv("UCX_TLS")) == "tcp,srd,cuda_copy,cuda_ipc" &&
            std::string(getenv("UCX_MAX_RNDV_RAILS")) == "1" &&
            std::string(getenv("UCX_RNDV_PIPELINE_ERROR_HANDLING")) == "n" &&
            std::string(getenv("UCX_RNDV_FRAG_SIZE")) == "cuda:32M" &&
            std::string(getenv("UCX_RNDV_FRAG_MEM_TYPES")) == "cuda" &&
            std::string(getenv("UCX_RNDV_FRAG_ALLOC_COUNT")) ==
                "host:128,cuda:32" &&
            std::string(getenv("UCX_SOCKADDR_TLS_PRIORITY")) == "tcp" &&
            std::string(getenv("UCX_CUDA_IPC_CACHE")) == "y" &&
            std::string(getenv("UCX_CUDA_IPC_CACHE_MAX_REGIONS")) == "inf" &&
            std::string(getenv("UCX_CUDA_IPC_CACHE_MAX_SIZE")) == "16G" &&
            std::string(getenv("KVIKIO_TASK_SIZE")) == "33554432" &&
            std::string(getenv("KVIKIO_REMOTE_IO_BACKEND")) == "MULTI_POLL" &&
            std::string(getenv("KVIKIO_REMOTE_DIRECT_RECEIVE")) == "REQUIRE" &&
            std::string(getenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS")) ==
                "128" &&
            std::string(getenv("KVIKIO_REMOTE_IO_NUM_REACTORS")) == "4" &&
            std::string(getenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH")) ==
                "PER_CHUNK" &&
            std::string(getenv("KVIKIO_NTHREADS")) == "16" &&
            getenv("UCX_CUDA_IPC_ENABLE_GET_ZCOPY") == nullptr &&
            getenv("UCX_CUDA_COPY_ASYNC_MEM_TYPE") == nullptr &&
            getenv("UCX_NET_DEVICES") == nullptr;
        _exit(valid ? 0 : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(WxdGpuDefaultsTest, explicitThreadCountAliasWins) {
  ASSERT_EXIT(
      {
        unsetenv("KVIKIO_NTHREADS");
        setenv("KVIKIO_NUM_THREADS", "24", 1);
        applyWxdGpuEnvironmentDefaults(false, true);
        _exit(
            getenv("KVIKIO_NTHREADS") == nullptr &&
                    std::string(getenv("KVIKIO_NUM_THREADS")) == "24"
                ? 0
                : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(WxdGpuDefaultsTest, disabledPathsDoNotSetEnvironment) {
  ASSERT_EXIT(
      {
        unsetenv("UCX_TLS");
        unsetenv("KVIKIO_REMOTE_IO_BACKEND");
        applyWxdGpuEnvironmentDefaults(false, false);
        _exit(
            getenv("UCX_TLS") == nullptr &&
                    getenv("KVIKIO_REMOTE_IO_BACKEND") == nullptr
                ? 0
                : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(WxdGpuDefaultsTest, directS3WithoutExchange) {
  ASSERT_EXIT(
      {
        unsetenv("UCX_TLS");
        unsetenv("KVIKIO_REMOTE_IO_BACKEND");
        applyWxdGpuEnvironmentDefaults(false, true);
        _exit(
            getenv("UCX_TLS") == nullptr &&
                    std::string(getenv("KVIKIO_REMOTE_IO_BACKEND")) ==
                        "MULTI_POLL"
                ? 0
                : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(WxdGpuDefaultsTest, exchangeWithoutDirectS3) {
  ASSERT_EXIT(
      {
        unsetenv("UCX_TLS");
        unsetenv("KVIKIO_REMOTE_IO_BACKEND");
        applyWxdGpuEnvironmentDefaults(true, false);
        _exit(
            std::string(getenv("UCX_TLS")) == "tcp,srd,cuda_copy,cuda_ipc" &&
                    getenv("KVIKIO_REMOTE_IO_BACKEND") == nullptr
                ? 0
                : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}

} // namespace facebook::velox::cudf_velox::test
