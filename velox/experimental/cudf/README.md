# Velox-cuDF

Velox-cuDF is a Velox extension module that uses the cuDF library to implement a GPU-accelerated backend for executing Velox plans. [cuDF](https://github.com/rapidsai/cudf) is an open source library for GPU data processing, and Velox-cuDF integrates with "[libcudf](https://github.com/rapidsai/cudf/tree/main/cpp)", the CUDA C++ core of cuDF. libcudf uses [Arrow](https://arrow.apache.org)-compatible data layouts and includes single-node, single-GPU algorithms for data processing.

## How Velox and cuDF work together

Velox-cuDF implements the Velox [DriverAdapter](https://github.com/facebookincubator/velox/blob/d9f953cd23880f29593534f1ba9031c6cea8ba06/velox/exec/Driver.h#L695) interface as [CudfDriverAdapter](https://github.com/facebookincubator/velox/blob/226b92cefedce4b8a484bfc351260edbd3d2e501/velox/experimental/cudf/exec/ToCudf.cpp#L301) to rewrite query plans for GPU execution. Generally the cuDF DriverAdapter replaces operators one-to-one. For end-to-end GPU execution where cuDF replaces all of the Velox CPU operators, cuDF relies on Velox's [pipeline-based execution model](https://facebookincubator.github.io/velox/develop/task.html) to separate stages of execution, partition the work across drivers, and schedule concurrent work on the GPU.

For more information please refer to our blog: "[Extending Velox - GPU Acceleration with cuDF](https://velox-lib.io/blog/extending-velox-with-cudf)."

## Getting started with Velox-cuDF

cuDF supports Linux and WSL2 but not Windows or MacOS. cuDF also has minimum CUDA version, NVIDIA driver and GPU architecture requirements which can be found in the [RAPIDS Installation Guide](https://docs.rapids.ai/install/). Please refer to cuDF's [readme](https://github.com/rapidsai/cudf) and [developer guide](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md) for more information.

### Building Velox with cuDF

The cuDF backend is included in Velox builds when the [VELOX_ENABLE_CUDF](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/CMakeLists.txt#L455) CMake option is set. The `adapters-cuda` service in Velox's [docker-compose.yml](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/docker-compose.yml#L69) is an excellent starting point for Velox builds with cuDF.

1. Use `docker compose` to run an `adapters-cuda` image.
```shell
docker compose -f docker-compose.yml run -e NUM_THREADS=8 --rm -v "$(pwd):/velox" adapters-cuda /bin/bash
```
2. Once inside the image, build cuDF with the following flags:
```shell
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_ARROW=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_ENABLE_BENCHMARKS_BASIC=ON" make cudf
```
3. After cuDF is built, verify the build by running the unit tests.
```shell
cd _build/release
ctest -R cudf -V
```

Velox-cuDF builds are included in Velox CI as part of the [adapters build](https://github.com/facebookincubator/velox/blob/de31a3eb07b5ec3cbd1e6320a989fcb2ee1a95a7/.github/workflows/linux-build-base.yml#L85). The build step for cuDF does not require the worker to have a GPU, so adding a Velox-cuDF build step to Velox CI is compatible with the existing runners.

### Configuring Velox-cuDF

Velox-cuDF provides several configuration properties to control GPU execution behavior, memory management, and debugging. These configurations are available when compiled with cuDF support and can be set via Velox's configuration system. For a complete list of cuDF-specific configuration properties and their descriptions, see the [Cudf-specific Configuration section](https://facebookincubator.github.io/velox/configs.html#cudf-specific-configuration-experimental) in the Velox configuration documentation.

#### Asynchronous KvikIO cache fills

The KvikIO read-through cache submits GPU scan cache misses through the cuDF
datasource's `host_read_async` API. Executor threads perform lookup, submission,
and H2D preparation, but do not wait for eager host-I/O futures or another
reader's exclusive cache entry. A process-wide readiness thread checks pending
futures, without blocking I/O or CUDA calls; cuDF's `std::future` interface does
not expose completion callbacks. It sleeps when idle and checks outstanding
reads at 100-microsecond intervals. Ready ranges can submit H2D independently
of earlier pending ranges. A deferred-only delegate remains supported via the
executor and is counted separately.

The cache-read admission window uses a positive
`KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS`, or 64 when unset/zero. It bounds
admitted logical reads **before cache allocation**, not the number of TCP GETs;
KvikIO may split a large logical read into multiple transport requests. This
is a count limit, not a new host-memory budget. Existing cache/pinned-pool
budgets still apply. `KVIKIO_NTHREADS` sizes the executor, not the number of
pending asynchronous cache fills. No driver count, split preload depth, GPU
destination size, or worker placement is changed. This is not yet host-only
split prefetch or subrange H2D before a logical host read completes.

Use query-level runtime counters to verify dispatch:

- `cudfKvikioCacheAsyncFillSubmitted`: logical asynchronous fill calls.
- `cudfKvikioCacheAsyncFillInFlightSamples`: outstanding logical fills across
  the worker process at submission; inspect `max`, not the sum of samples.
- `cudfKvikioCacheAsyncReadQueueNanos`: admission/executor delay before lookup.
- `cudfKvikioCacheAsyncFillReadNanos`: elapsed host-fill work, including
  completion observation/scheduling; **not** pure network service time.
- `cudfKvikioCacheExclusiveWaitNanos`: waiting for another cache fill.
- `cudfKvikioCacheDeferredHostReads`: delegates that cannot start eagerly.
- `cudfKvikioCacheAsyncReadSynchronousFallbacks`: uncached fallback reads
  when a cache entry cannot be admitted (including oversized ranges).
- `cudfKvikioCacheAsyncReadFailures`: failed asynchronous read operations.

Time counters sum concurrent work and are not query critical-path durations.
Cache-off KvikIO, synchronous host/device callers, and BufferedInput/AWS SDK
dispatch are unchanged. Exclusive entries remain unpublished until a complete
successful fill; failure paths drain outstanding writes before freeing storage.
The device-read completion still fences H2D and retains cache ownership through
completion, including when the caller discards its future.

#### Experimental registered cache backing

`cudf.cache_host_registration_enabled=true` enables a registered backing pool
for AsyncDataCache in both the BufferedInput and KvikIO GPU readers. It is
disabled by default; ordinary CPU and cache-off paths are unchanged. Cache
entries suballocate page-aligned slices of stable slabs charged to the existing
Velox allocator. A new slab is first-touched and registered once before its
slices are filled. S3/SSD fills write directly into these slices and H2D reads
the same storage: there is no additional cache-to-staging copy for admitted
data, and no GPU-to-host cache population step.

Regular slabs grow from 64 MiB to 128 MiB to 256 MiB, then remain at 256 MiB.
This sizes backing blocks, **not individual entries**. Larger contiguous
requests use dedicated page-rounded blocks. Existing allocations never move.
Slab preparation inherits the allocating worker's CPU/memory policy and reuse
is separated by CPU NUMA node; production workers should bind CPU and memory
together. Root-allocator huge-page policy is unchanged. This does not migrate
already resident malloc backing to another NUMA node.

`cudf.cache_host_registration_max_bytes` limits full registered slab capacity
across one worker **process**, including free space and pending registrations.
The default is 34359738368 bytes (32 GiB), not a hardware limit; other CUDA/UCX
registrations and staging buffers are outside this budget. All slab backing is
also charged to Velox's root allocator. Tiny entries and admissions that exceed
the budget, fail allocation, or encounter a recoverable CUDA registration error
use ordinary cache storage and staging without refetching bytes. Existing
pageable cache entries are not copied or registered individually. Empty slabs
can be reclaimed to make room; this version does not evict cached data or unpin
live slabs to admit a different working set.

Idle registrations retain no cache pins. Eviction, file invalidation, and
failed fills return slices to the pool; adjacent free slices can be reused as
a larger range without copying live data. DMA leases independently hold shared
cache pins and exclude unregister until the reader's completion fence. Under
root-memory pressure, or explicit cache shrink, empty slabs are unregistered
before their backing is freed. Partially live slabs cannot release root
capacity, and shrink does not report their returned slices as freed memory.
Cache shutdown/destruction frees all backing while CUDA and the root allocator
are alive; readers must be quiescent. CUDA calls and root backing frees run
outside cache shard and pool mutexes.

Cache clear drops unpinned **data** but retains reusable prepared slabs. A
data-cache-cold run with a prepared pool is therefore different from a fresh
worker run, which also pays allocation, page faults, and registration. Report
both separately when testing. Restart for registration on/off comparisons and
keep the reader, native exchange, drivers, and split policy fixed. New slab
preparation is synchronous; overlapping it with I/O is a separate experiment.

`cudfCacheRegisteredH2DBytes` proves registered transfers occurred.
`cudfCacheHostRegisterCalls`, `cudfCacheHostRegisteredBytes`,
`cudfCacheHostPoolPrepareNanos`, and `cudfCacheHostRegisterNanos` report each slab's
creation once, to its first GPU acquisition with query stats. This may be a
different query than its preloader. Preparation includes backing allocation and
first touch; registration time covers the CUDA API call. Repeated acquisitions
should report zero new calls/time while their backing survives, including after
entry eviction and reuse. `cudfCacheHostRegistrationFallbackBytes` records H2D
bytes that instead use staging.

Counters ending in `Samples` are process snapshots: use **max**, not sum, for
footprint gauges, or successive per-worker differences for cumulative events.
`cudfCacheHostPoolSlabsSamples`, `cudfCacheHostPoolUsedBytesSamples`, and
`cudfCacheHostPoolFreeBytesSamples` expose pool occupancy.
`cudfCacheHostRegistrationReservedBytesSamples` measures registered/pending
capacity, and `cudfCacheHostRegistrationRetainedBytesSamples` measures backing
capacity. `cudfCacheHostPoolRegisterCallsSamples` and
`cudfCacheHostPoolUnregisterCallsSamples` are cumulative registration attempts
and successful unregisters, including maintenance; `cudfCacheHostPoolBudgetFallbacksSamples`,
`cudfCacheHostPoolAllocationFallbacksSamples`, and
`cudfCacheHostPoolRegistrationFailuresSamples` distinguish admission failures.
Query-stat owners are not retained by idle slabs. Submission/wait counters
(`cudfCacheRegisteredH2DSubmitWaitNanos` and `cudfCacheRegisteredH2DWaitNanos`)
measure host intervals, **not** CUDA-event elapsed time or query critical path.

### Testing Velox with cuDF

Tests with Velox-cuDF can only be run on GPU-enabled hardware. The Velox-cuDF tests in [experimental/cudf/tests](https://github.com/facebookincubator/velox/blob/main/velox/experimental/cudf/tests) include several types of tests:
* operator tests
* function tests
* fuzz tests (not yet implemented)

The repo [rapidsai/velox-testing](https://github.com/rapidsai/velox-testing/) includes standard scripts for testing Velox-cuDF. Please refer to the [test_velox.sh](https://github.com/rapidsai/velox-testing/blob/main/velox/scripts/test_velox.sh) for running the Velox-cuDF unit tests. We plan to first develop GitHub Actions for GPU CI in [rapidsai/velox-testing](https://github.com/rapidsai/velox-testing/), and then later transition GPU-enabled GitHub Actions to Velox mainline.

#### Operator tests

Many of the tests for cuDF are "operator tests" which confirm correct execution of simple query plans. cuDF's operator tests use `CudfDriverAdapter` to modify the test plan with GPU operators before executing it. The operator tests for cuDF include both tests that assert successful GPU operator replacement, and tests that pass with CPU fallback.

#### Function tests

Velox-cuDF also includes "function tests" which cover the behavior of shared functions that could be called in multiple operators. Velox-cuDF function tests assess the correctness of functions using one or more cuDF API calls to provide the output. [SubfieldFilterAstTest](https://github.com/facebookincubator/velox/blob/99a04b94eed42d1c35ae99101da3bf77b31652e8/velox/experimental/cudf/tests/SubfieldFilterAstTest.cpp#L158) includes several examples of function tests. Please note that unit tests for cuDF APIs are included in [cudf/cpp/tests](https://github.com/rapidsai/cudf/tree/branch-25.10/cpp/tests) rather than Velox.

#### Fuzz tests

Velox includes components for "fuzz testing" to ensure robustness of Velox operators. For instance, the [Join Fuzzer](https://github.com/facebookincubator/velox/blob/99a04b94eed42d1c35ae99101da3bf77b31652e8/velox/docs/develop/testing/join-fuzzer.rst) executes a random join type with random inputs and compares the Velox results with a reference query engine. Fuzz testing tools have been used for cuDF operator development, but fuzz testing for cuDF is not yet integrated into Velox mainline.

### Benchmarking Velox with cuDF

Velox's [TpchBenchmark](https://github.com/facebookincubator/velox/blob/d9f953cd23880f29593534f1ba9031c6cea8ba06/velox/benchmarks/tpch/TpchBenchmark.cpp) is derived from [TPC-H](https://www.tpc.org/tpch/) and provides a convenient tool for  benchmarking Velox's performance with OLAP (Online Analytical Processing) workloads. Velox-cuDF includes GPU operators for the hand-built query plans located in [TpchQueryBuilder](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/velox/exec/tests/utils/TpchQueryBuilder.cpp). Velox [PR 13695](https://github.com/facebookincubator/velox/pull/13695) extends Velox's TpchBenchmark to the cuDF backend.

Please note that Velox's hand-built query plans require the data set to have floating-point types in place of the fixed-point types defined in the standard. Further development of Velox's TpchBenchmark could allow correct behavior with both fixed-point and floating-point types.

## Contributing

Velox-cuDF's development priorities are documented as Velox issues using the "[cuDF]" prefix. Please check out the [open issues](https://github.com/facebookincubator/velox/issues?q=is%3Aissue%20state%3Aopen%20%5BcuDF%5D) to learn more.

We would love to hear from you in Velox's Slack workspace, please see Velox discussion [11348](https://github.com/facebookincubator/velox/discussions/11348) for information on joining.
