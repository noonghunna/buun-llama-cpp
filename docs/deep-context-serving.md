# Deep-context serving with the CUDA MoE cache

This runbook covers long-lived `llama-server` processes that combine a 262,144-token context with the [CUDA MoE expert cache](backend/CUDA-MOE-CACHE.md). It is based on measurements from two 24 GiB CUDA devices. Treat the numeric settings as a validated operating point for that regime, not as a universal sizing formula.

## Measured operating points

| Workload | Cache reserve, per device | Context checkpoints, per slot | Result |
|---|---:|---:|---|
| Shallow decode | 512 MiB | not isolated | Worked in the shallow regime only. This does not establish deep-context safety. |
| Repeated prompts around 100K tokens inside a 262K context | 512 MiB | not isolated | Unsafe: VRAM reached the limit and behavior degraded across repeated runs. |
| Same deep-context regime | 2048 MiB | 2 | Stable loaded decode: 26.5 tokens/s on each of three consecutive runs. |
| Large prompts with the server default | reserve not isolated | 32 | Large host-side checkpoint state accumulated. |

The implementation default cache reserve is 3072 MiB per device. The 2048 MiB value above is the smallest reserve validated by this measurement; it is not proof that 2048 MiB is sufficient for every model, quant, placement, batch size, slot count, or CUDA device.

## Serving policy

Use this policy when selecting the reserve:

| Regime | Policy |
|---|---|
| Unknown or mixed prompt lengths | Keep the 3072 MiB implementation default until the exact deployment is measured. |
| Repeated deep prompts near 100K tokens on 2 x 24 GiB | Use at least the measured 2048 MiB reserve and `--ctx-checkpoints 2`. Prefer 3072 MiB if the smaller cache still gives useful hit coverage. |
| Strictly shallow, bounded workload | A 512 MiB reserve can be evaluated as a capacity optimization, but only with a hard prompt ceiling and repeated-load validation. Do not carry it into a 262K service merely because boot and shallow decode succeed. |

The reserve is VRAM left outside the expert cache on each device. It is read when a cache session is created and cannot be repaired in place by changing the environment of an already-running process.

A representative deep-context launch is:

```sh
export GGML_CUDA_MOE_CACHE_RESERVE_MB=2048

./build/bin/llama-server \
    -m MODEL.gguf \
    -c 262144 \
    --ctx-checkpoints 2 \
    --moe-cache MODE \
    OTHER_VALIDATED_ARGUMENTS
```

Replace `MODE` and the other placeholders with the already-validated placement and cache mode. Do not change repacking, expert placement, batch size, slot count, or cache mode while evaluating only the reserve; those change both memory demand and the execution path.

### Why two checkpoints

`--ctx-checkpoints N` limits the number of context checkpoints retained per server slot. The default is 32. At very large prompts those checkpoints can retain substantial host state, and the cost multiplies across active slots. Two checkpoints were stable in the measured deep-context profile while preserving checkpoint-based reuse/rollback behavior.

Do not set the value to zero as a reflexive memory fix. That changes server behavior and was not the measured configuration. If two is unsuitable for a workload, measure checkpoint reuse, host memory, and latency together.

## Cold-pool behavior

After a cold process start, the first deep decode showed a one-time fill storm lasting about 160 generated tokens at roughly 1 token/s. Later requests on the same warm pools did not show it, and the cache-off control did not show it.

This behavior follows the demand-fill design:

- normal prompt-processing nodes do not populate the cache under the default one-token eligibility limit;
- early decode misses still run on CPU while fills are queued for later tokens;
- pools and demand history are scheduler-local and do not persist across process restarts.

A health check or a prompt-only request therefore does not prove that the expert pools are warm.

### Optional post-boot warm-up

There is no dedicated pool warm-up command. If first-user latency matters, use a sacrificial request after boot:

1. Use a prompt with length and routing characteristics representative of production deep requests.
2. Allow generation to extend beyond the observed fill-storm window; 192 to 256 generated tokens gives margin around the measured approximately 160-token event.
3. Discard the response and keep the same server process and scheduler alive.
4. Send the same request class again and verify that the slow window is absent and cache counters show hits with no growing queue.
5. Mark the service ready only after that validation completes.

This is an operational workaround, not a guarantee of full coverage. A different request distribution can route to experts that the warm-up never touched. Do not use a tiny synthetic prompt and call the cache warm.

## Cold/warm validation protocol

For a new model, quant, placement, or context shape, validate before exposing the service:

1. Start from a fresh process with the intended reserve, checkpoint count, slot count, and all production arguments.
2. Capture per-device used/free VRAM, process RSS and swap, cache logs, prompt-evaluation time, and decode throughput.
3. Send one representative deep request and retain its per-token timing. Treat it as the cold arm.
4. Without restarting, repeat the same loaded request three times. Verify stable VRAM headroom, host memory, and decode throughput across all three.
5. Run an otherwise identical cache-off control from a fresh process. Keep placement and repacking identical or explicitly label the result as an end-to-end configuration comparison.
6. Restart and repeat once to prove that any one-time slow window follows cold pool state rather than request variance.

Useful observations include:

```sh
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv -l 1

pidstat -r -p SERVER_PID 1

cat /proc/SERVER_PID/status | command grep -E 'VmRSS|VmSwap|VmLck'
```

Set `GGML_CUDA_MOE_CACHE_STATS` to a suitable nonzero collection interval when periodic cache counters are needed. Also retain the teardown summary. A valid run records the exact revision, model and quant, placement, cache mode and budget, reserve, context length, checkpoint count and spacing, slot count, batch settings, prompt token count, and generated token count.

Pass criteria for the measured deep regime are:

- enough VRAM headroom remains after prompt evaluation and throughout decode;
- used VRAM does not creep toward the limit across the three warm repetitions;
- process RSS and swap do not grow without a workload explanation;
- warm loaded decode remains near its established baseline rather than degrading run by run;
- cache queues drain and fill/eviction activity settles instead of continuously accelerating.

## Failure signatures and response

| Signature | Likely interpretation | Response |
|---|---|---|
| Shallow requests work, but repeated approximately 100K-token loads drive VRAM to the limit | Reserve is too small for deep-context transient and persistent allocations | Stop the service and restart with a larger reserve. Use 2048 MiB as the measured floor for this profile; return to the 3072 MiB default if uncertain. |
| First deep decode alone is very slow for about 160 tokens, then later deep requests are normal | Cold expert-pool fill storm | Complete a representative post-boot warm-up or accept/document cold-start latency. Confirm with a fresh cache-off control. |
| Every deep request has the same slow window | Not explained by the measured one-time fill storm | Inspect whether sessions are being recreated, whether pools allocate, whether the queue drains, and whether routing differs. Do not mask it with a longer warm-up. |
| Host RSS grows substantially with successive large prompts at 32 checkpoints | Checkpoint accumulation | Restart with `--ctx-checkpoints 2`, then repeat the cold/warm protocol. Also account for the number of server slots. |
| CUDA allocation pressure trims the cache | The cache is disabled on that device for the rest of the session | Restart after correcting reserve/context/placement. Do not treat the post-trim process as a steady-state cache measurement. |
| No `[moe-cache] enabled` pool message appears | The cache never became active | Check eligible CPU-resident expert placement, selected devices, repacking/cache mode, and available budget before interpreting throughput. |

After allocator failure or a cache trim, collect the logs before restarting. The fallback may keep requests correct on CPU while making the performance state incomparable to the healthy cache run.

## `--mlock`: evaluation only

`--mlock` asks the host to lock model mappings in RAM. It does not create GPU headroom and does not replace the cache reserve. It may help a shared-memory-hosted expert estate avoid major faults or swap, but that benefit has not been measured for this deep-context profile. It can also fail because of the process memlock limit or pin enough RAM to harm the rest of the host.

Do not make `--mlock` the default yet. Evaluate it as a separate A/B:

1. Confirm physical RAM headroom and the effective `ulimit -l`/service memlock limit.
2. Run the complete cold/warm protocol with `--mlock` off, then on, changing no other argument.
3. Compare prompt time, cold and warm decode, major faults, `VmRSS`, `VmSwap`, `VmLck`, and system-wide memory pressure.
4. Reject it if locking fails, swap or pressure moves to another process, or throughput/latency does not improve reproducibly.
5. Record the result for the exact model estate; do not generalize it to other mmap/shmem layouts.

Until that A/B is complete, the supported deep-context controls are reserve sizing, checkpoint limiting, a representative warm-up when required, and repeated-load validation.
