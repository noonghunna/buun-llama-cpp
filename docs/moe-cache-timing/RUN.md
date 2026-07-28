# MoE cache three-arm timing discriminator

This experiment separates observer/layout cost from the cost of actually reading
the clock. Build all three arms from the same clean commit. The only intended
difference is `GGML_CUDA_MOE_TIMING_ARM`:

| Arm | Definition | Behavior |
|---|---:|---|
| uninstrumented | `0` | timing guards compile away |
| guards-disabled | `1` | guard objects and a volatile disabled check remain; no clock reads or timing records |
| sampled | `2` | each phase is sampled independently at 1/1024 with fenced `RDTSCP`; migrated samples are discarded |

## Build

Run from the repository root. Keep the common configure arguments byte-for-byte
identical; use a distinct build directory for each arm.

```bash
COMMON_CMAKE=(
  -DGGML_CUDA=ON
  -DGGML_CUDA_FA=ON
  -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DCMAKE_CUDA_ARCHITECTURES=86
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_TESTS=ON
  -DLLAMA_OPENSSL=ON
)

cmake -S . -B build-moe-timing-0 "${COMMON_CMAKE[@]}" -DGGML_CUDA_MOE_TIMING_ARM=0
cmake --build build-moe-timing-0 --parallel --target llama-server llama-bench test-moe-cache

cmake -S . -B build-moe-timing-1 "${COMMON_CMAKE[@]}" -DGGML_CUDA_MOE_TIMING_ARM=1
cmake --build build-moe-timing-1 --parallel --target llama-server llama-bench test-moe-cache

cmake -S . -B build-moe-timing-2 "${COMMON_CMAKE[@]}" -DGGML_CUDA_MOE_TIMING_ARM=2
cmake --build build-moe-timing-2 --parallel --target llama-server llama-bench test-moe-cache
```

Each configure writes `moe-timing-build-manifest.txt`. Require `source_dirty=no`
in all three. Apart from `timing_arm`, the manifests must be identical:

```bash
for arm in 0 1 2; do
  sed '/^timing_arm=/d' "build-moe-timing-${arm}/moe-timing-build-manifest.txt" |
    sha256sum
done
```

## Boot and bench

Use the exact Laguna placement, cache size, context, KV, prompt, warmup, and
sampling arguments from the money bench. Replace `LAGUNA_ARGS` below with that
unchanged argument vector; do not tune between arms. Restart the process for
each arm, warm the same expert set, and rotate arm order between repetitions.

```bash
MODEL=<models>/Laguna-68B-GGUF/<exact-money-bench.gguf>
LAGUNA_ARGS=(<exact unchanged money-bench arguments, including --moe-cache>)

./build-moe-timing-0/bin/llama-server -m "$MODEL" "${LAGUNA_ARGS[@]}"
./build-moe-timing-0/bin/llama-bench  -m "$MODEL" "${LAGUNA_ARGS[@]}" -n 256 -r 5

./build-moe-timing-1/bin/llama-server -m "$MODEL" "${LAGUNA_ARGS[@]}"
./build-moe-timing-1/bin/llama-bench  -m "$MODEL" "${LAGUNA_ARGS[@]}" -n 256 -r 5

./build-moe-timing-2/bin/llama-server -m "$MODEL" "${LAGUNA_ARGS[@]}"
./build-moe-timing-2/bin/llama-bench  -m "$MODEL" "${LAGUNA_ARGS[@]}" -n 256 -r 5
```

Do not run server and bench simultaneously; the paired lines show the exact
binary for each alternative harness. Capture external wall-clock TPS and the
same internal generation-throughput field used by the money bench. Arm 2 emits
per-thread decode/fill phase totals only when the cache session is destroyed.

## Decision table

| Observation | Interpretation | Next action |
|---|---|---|
| arm 1 regresses versus arm 0 | guard scaffold/code layout is perturbing the hot path | treat phase prices as suspect; reduce or relocate scaffold before further profiling |
| arm 1 matches arm 0; arm 2 regresses | sampled clock reads/fences are the observer | increase sample interval or price only in an isolated diagnostic build |
| arms 0 and 1 recover prior TPS; arm 2 is within noise | the former atomics/`steady_clock` instrumentation caused the regression | use arm 2 phase totals to price the fusion work |
| all three match the prior slow result | instrumentation was not the primary cause | investigate the mechanism or run-state delta, not the observer |
| ordering changes across repetitions or variance dominates | cache age, placement, thermals, or workload state is uncontrolled | discard the run and repeat with identical warm state and rotated order |

Compare medians and dispersion, not a single best run. Record CPU migration
reject counts from arm 2; a large reject rate invalidates its per-phase estimate.
