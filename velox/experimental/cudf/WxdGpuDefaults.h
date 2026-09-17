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

#pragma once

#include <cerrno>
#include <cstdlib>
#include <system_error>

namespace facebook::velox::cudf_velox {

// WXD/IBM ONLY — DO NOT UPSTREAM / DO NOT OPEN AS AN UPSTREAM PR.
// Called during single-threaded GPU registration, before UCXX/KvikIO startup.
// Do not set NIC addresses, NUMA bindings, credentials, or host kernel tuning.
// Existing environment values (including explicit empty values) always win.
inline void applyWxdGpuEnvironmentDefaults(bool exchange, bool directS3) {
  const auto setDefault = [](const char* key, const char* value) {
    if (::setenv(key, value, 0) != 0) {
      throw std::system_error(errno, std::generic_category(), key);
    }
  };
  if (exchange) {
    setDefault("UCX_TLS", "tcp,srd,cuda_copy,cuda_ipc");
    setDefault("UCX_MAX_RNDV_RAILS", "1");
    setDefault("UCX_RNDV_PIPELINE_ERROR_HANDLING", "n");
    setDefault("UCX_RNDV_FRAG_SIZE", "cuda:32M");
    setDefault("UCX_RNDV_FRAG_MEM_TYPES", "cuda");
    setDefault("UCX_RNDV_FRAG_ALLOC_COUNT", "host:128,cuda:32");
    setDefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp");
    setDefault("UCX_CUDA_IPC_CACHE", "y");
    setDefault("UCX_CUDA_IPC_CACHE_MAX_REGIONS", "inf");
    setDefault("UCX_CUDA_IPC_CACHE_MAX_SIZE", "16G");
    // Deliberately do not force GET_ZCOPY or private-async local access:
    // those belong to the parked raw/PULL experiment, not this baseline.
  }
  if (directS3) {
    // KvikIO accepts NUM_THREADS as an alias when NTHREADS is absent.
    if (::getenv("KVIKIO_NUM_THREADS") == nullptr) {
      setDefault("KVIKIO_NTHREADS", "16");
    }
    setDefault("KVIKIO_TASK_SIZE", "33554432");
    setDefault("KVIKIO_REMOTE_IO_BACKEND", "MULTI_POLL");
    setDefault("KVIKIO_REMOTE_DIRECT_RECEIVE", "REQUIRE");
    setDefault("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS", "128");
    setDefault("KVIKIO_REMOTE_IO_NUM_REACTORS", "4");
    setDefault("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", "PER_CHUNK");
  }
}

} // namespace facebook::velox::cudf_velox
