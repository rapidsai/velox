# WXD/IBM ONLY — DO NOT UPSTREAM

**Do not open this profile as an upstream PR.** These are deployment-specific
defaults for the WXD AWS benchmark branch, not general Velox defaults or a
recommendation for other hardware. The reference deployment is eight workers,
one 96-GiB GPU per worker, on g7e.48xlarge with EFA, the WXD UCX 1.22 build and
CUDA 13.2. It has unusually large host-memory and network budgets.

This profile follows the published IPC compression fix. No IPC speedup is promised.

## Runtime defaults

Explicit `cudf.*` and `ucxx.*` properties still override these values.

| Area | WXD default |
| --- | --- |
| Exchange | `cudf.exchange=true`, `cudf.intra_node_exchange=true` |
| UCXX | Error handling and blocking polling disabled |
| Compression | `column-adaptive-freq-pfor-min128`, allowed on IPC endpoints |
| Codec pipeline | Enabled, one thread, 16 MiB minimum, 1.50 safety margin |
| Batches | Partitioned output: 10 million rows; concat minimum: 40 million |
| Operators | Concat and eligible streaming groupby enabled; JIT expressions off |
| Reader | KvikIO (`cudf.hive.use-buffered-input=false`); AsyncDataCache remains available |
| Cache registration | Enabled, **32 GiB per process** maximum registered slab capacity |

The registered-cache limit is not a host-memory allocation at startup and is
not the total cache size. Host cache and query budgets still need deployment
sizing. Registration failures/capacity limits retain the existing staging
fallback. Eight workers can collectively retain up to 256 GiB of registered
slabs; this is not suitable for arbitrary hosts or containers.

`registerCudf()` supplies the following missing environment settings before
UCXX or KvikIO initialization. Existing values, including explicit empty ones,
are not overwritten. This is worker startup behavior, not a runtime retuning API.
Startup logs include `WXD GPU effective settings` with the actual selected
codec, allocator, UCX transports, KvikIO backend, direct-receive mode, thread
counts, request slots, task size and reactor dispatch. Since missing
environment values are supplied inside the worker process, a separate
`docker exec ... printenv` does not show these newly supplied defaults.

```text
UCX_TLS=tcp,srd,cuda_copy,cuda_ipc
UCX_MAX_RNDV_RAILS=1
UCX_RNDV_PIPELINE_ERROR_HANDLING=n
UCX_RNDV_FRAG_SIZE=cuda:32M
UCX_RNDV_FRAG_MEM_TYPES=cuda
UCX_RNDV_FRAG_ALLOC_COUNT=host:128,cuda:32
UCX_SOCKADDR_TLS_PRIORITY=tcp
UCX_CUDA_IPC_CACHE=y
UCX_CUDA_IPC_CACHE_MAX_REGIONS=inf
UCX_CUDA_IPC_CACHE_MAX_SIZE=16G

KVIKIO_NTHREADS=16
KVIKIO_TASK_SIZE=33554432
KVIKIO_REMOTE_IO_BACKEND=MULTI_POLL
KVIKIO_REMOTE_DIRECT_RECEIVE=REQUIRE
KVIKIO_REMOTE_ADAPTIVE_TCP_MSS=ON
KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS=128
KVIKIO_REMOTE_IO_NUM_REACTORS=4
KVIKIO_REMOTE_IO_REACTOR_DISPATCH=PER_CHUNK
```

UCX defaults are applied when GPU exchange is enabled. KvikIO defaults are
applied only in a direct-receive build. **128 is the per-process request concurrency
limit, not a thread count**; there are four reactor threads using `MULTI_POLL`
and `PER_CHUNK` dispatch, plus the 16-thread KvikIO worker pool. 32 MiB is the task/chunk
size, not a promise that every HTTP request will be that size.

Adaptive TCP MSS is enabled only by this direct-S3 WXD profile; KvikIO's
general default remains off. The matching pinned curl provides the experimental
connection-reuse callback. The policy observes completed HTTP/1.1 GET responses,
prefers peers with observed jumbo evidence, and bounds connection retirement
with expiring evidence and a shared budget. If alternatives cannot connect
before any HTTP request, it falls back to ordinary peer selection so a
small-MSS-only service remains usable. It does not change DNS, MTU, TCP buffers,
credentials or TLS verification, or replay a successful response.

Set `KVIKIO_REMOTE_ADAPTIVE_TCP_MSS=OFF` before worker startup to disable it.
The effective setting is included in the existing startup log. An explicit
empty environment value is also preserved, not silently replaced with ON;
KvikIO rejects that invalid value, so use OFF rather than empty to disable it.
Custom dependency overrides must supply the matching callback API or explicitly
disable the policy; patched headers alone do not prove runtime support.

Mixed transports make IPC eligible; they do not prove a particular peer used
IPC or guarantee IPC-only traffic for a pair. Existing UCX protocol selection
still decides the path. This profile does not force GET_ZCOPY, keepalive tuning,
protocol tracing, NIC selection, or any custom local-access patch.

BufferedInput remains selectable with `cudf.hive.use-buffered-input=true`.
Its AWS SDK configuration is separate: the tested direct-receive alternative
also used `hive.s3.direct-receive-mode=required` and
`hive.s3.adaptive-tcp-mss-enabled=true`. Generic CPU/AWS SDK defaults are not
silently changed by this GPU profile.

## Dependencies and limits

The CentOS adapter installer defaults to CUDA 13.2 and
`kjmph/ucx@462c56777aaf268d7daf1b5d43e6f49e69b0207e`: UCX 1.22 with
the Blackwell RTX IPC bandwidth fix from **openucx/ucx#11865** and the GDA
build fixes. CUDA and EFA transport modules and UCX CMake metadata are required.
An explicit `UCX_VERSION` or local source still overrides the default; stock
1.22.0 is not the same dependency. The pinned checkout refuses an existing
different or dirty source tree instead of silently using it.

The direct S3 build pins curl `1c7a9684406103c9ab226ad7671f46322799653f`
and KvikIO `7bb1345a4078e47e1490031c7b5bd93042185b8d` for the matched
connection-reuse API; AWS SDK and cuDF revisions are unchanged. KvikIO's
bundled curl fallback pins the same curl revision. Ordinary non-direct-S3
cuDF builds retain their existing KvikIO revision.
A normal upstream curl/KvikIO build is not an
equivalent runtime for `REQUIRE`. Build-time API checks and runtime strict
receive failures are intentional; the profile does not make an unsupported
kernel or TLS route support direct receive.

The MSS policy was measured with SRD-only exchange. This dependency/default
update does not fix the separate mixed SRD/IPC decompression illegal-access
failure. Keep `UCX_TLS=tcp,srd,cuda_copy` when reproducing those MSS measurements;
no UCX defaults or transport code are changed here.

The coordinator, native-driver and direct-build defaults are documented in
Presto's `presto-native-execution/WXD_IBM_DEFAULTS.md`. Rebuild and restart the
worker to use this profile; existing generated configuration overrides still
win. Reduced fault tolerance, large batches, pinned host memory and a large
IPC mapping cache are deliberate experimental tradeoffs, not production-safe
defaults. SF3K Q18/Q21 memory failures are not fixed by this change.
