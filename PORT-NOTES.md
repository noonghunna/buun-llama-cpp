# MoE expert-cache port notes

> **Status: experimental port** — leloch's MoE expert cache transplanted onto the buun fork; benchmarked on 2×RTX 3090 (sm_86). Not a supported product; validation record below and in the fork's issues.

## Result

The faithful four-commit MoE expert-cache series is ported onto buun `master`
and builds successfully with the production CUDA shape for this rig. The work
remains isolated and unmerged:

| Item | Value |
|---|---|
| Worktree | `<port-worktree>` |
| Branch | `moe-cache-port` |
| Target base | `7939b6c4705a40cba56cf25633f2e111d9ffabef` |
| Source | local fetch of `<local-leloch-checkout>`, branch `moe-cache-laguna` |
| Source tip | `a01eb26468a6cb2bf2c3bfd1cc3b5d9e64daefcd` |
| Runtime/test surface | 18 files, +4,481/-39 relative to target `master` |
| GPU execution | None |

The source series itself is +4,482/-37. The two-line delta relative to that
baseline comes from resolving newer buun code and adapting two buun-only
quant dispatches in `mmvq.cu`; the port did not grow beyond the authorized
18-file surface.

## Cherry-picks

Each source commit was applied in order with `git cherry-pick -x`:

| Source | Port commit | Purpose |
|---|---|---|
| `618cdb4cc` | `c3165af2a` | cache API plus CPU `mul_mat_id` and scheduler integration |
| `708c18c2b` | `83ecab84d` | CUDA cache implementation and backend hooks |
| `8853f0535` | `25fb4dfb9` | `--moe-cache` CLI and fit placement |
| `a01eb2646` | `37121166b` | blocker-fix execution rework, tests, and benchmark telemetry |

The source commits carry the placeholder author identity
`Your Name <you@example.com>`; the cherry-picks preserve it rather than
rewriting attribution. Each port commit also contains the exact source hash
in its `-x` trailer.

## Final cache architecture

The final blocker-fix design is a dynamic scheduler-scoped VRAM cache: hot expert
rows stay resident in GPU slots, cache hits run CUDA matvec concurrently with
normal CPU miss-row computation, and collection merges the hit rows into the
CPU result. The cache survives across graphs until its owning scheduler is
destroyed.

**It does not write or load a hot-set persistence file.** The blocker-fix commit
explicitly removed cross-session/process persistence along with fused GLU and
GPU-output handoff. That differs from the earlier source implementation and
from the persistence-file aspect of the engineering twin; it is intentional in
the exact authorized source tip, not a conflict-resolution drop.

## Conflict record

### `618cdb4cc` — API, CPU path, and scheduler

Applied cleanly. In particular, the scheduler session lifecycle, invalidation
hooks, and CPU `mul_mat_id` begin/plan/dispatch/collect/end weave landed at
buun current locations without manual conflict resolution.

### `708c18c2b` — initial CUDA implementation

| Path | Resolution |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Preserved buun newer virtual-device enumeration and physical-device-ID mapping. At this intermediate commit, inserted the source cache registration at the corresponding registry site. The two allocation-retry trim hooks applied. |

`moe-cache.cu/.cuh` were new files. `mmvq.cu/.cuh` merged textually without a
conflict; see the high-attention audit below.

### `8853f0535` — CLI and fit placement

| Path | Resolution |
|---|---|
| `common/arg.cpp` | The conflict was only in the include block. Kept buun current includes and retained the source `--moe-cache` handler and mode/budget environment plumbing. |
| `common/fit.cpp` | Kept buun newer `common_get_device_memory_data` wrapper and placed the source MoE-cache fit helper immediately after it. No fit-placement hunk was dropped at this stage. |

A mechanical conflict edit briefly displaced a brace in `common/fit.cpp`; it
was caught during inspection and corrected before continuing the cherry-pick
series, so it is not present in the port commit.

### `a01eb2646` — blocker-fix rework

| Path | Resolution |
|---|---|
| `common/arg.cpp` | Kept buun current include set and model-handler flow, added the required `cerrno` include, and inserted the explicit-cache post-parse repack guard after the initial parse. Did not copy the obsolete leloch remote-preset flow. |
| `common/fit.cpp` | Followed the blocker-fix intent: removed the earlier unsafe automatic GGUF scanning/cache-aware layer fit while preserving buun wrapper. The final three-line diff is include-level residue from the series, not active cache placement policy. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Kept buun virtual/physical device enumeration. Removed the obsolete early cache registration from the previous commit and retained the rework registration after `reg` is initialized, guarded consistently with the CUDA-only backend code. |
| `tests/test-arg-parser.cpp` | Kept buun VBR cases and current include set, then appended the MoE-cache and repack interaction cases. |
| `tools/llama-bench/llama-bench.cpp` | Combined buun `offline` initialization with the rework warmup-count, repack, and explicit-cache fields. Retained the complete cache telemetry and A/B controls. |

No fit-placement escape-hatch reduction was needed. The final series has no
automatic cache-driven `-ngl` placement because the required blocker-fix
commit intentionally removes that unsafe mechanism; explicit `-ngl`/`-ot`
placement remains the supported path.

## High-attention `mmvq.cu/.cuh` audit

**This is the one semantic adaptation in the kernel pair.**

Both source CUDA commits merged `mmvq.cu` and `mmvq.cuh` without textual
conflicts, but buun has two quant cases absent from the leloch base:
`GGML_TYPE_Q2_0` and `GGML_TYPE_Q2_0_G128`. The blocker-fix changes
`mul_mat_vec_q_switch_ncols_dst` to require the new `allow_small_k`
argument. Git clean merge could not update the two buun-only call sites, so
both were manually adapted to pass `allow_small_k`.

All calls to that dispatcher were audited after the merge. `mmvq.cuh`
required no target-specific edit beyond the clean source merge. This proves
the pair compiles for sm_86, but not runtime correctness or performance for
the buun-only Q2 variants; that remains part of the GPU validation ladder.

## Repacking contract

The required canonical-host-layout safeguard is preserved:

- `--moe-cache auto` leaves buun normal repack/default behavior intact.
- Explicit `--moe-cache on` and fixed numeric budgets set
  `moe_cache_force`.
- After argument parsing, an explicit cache request sets
  `no_extra_bufts` when needed, which disables the extra-buffer/repacking
  path before model loading.
- `--moe-cache off` disables the cache.

The accompanying argument-parser cases compile into `test-arg-parser`.

## Turbo-path assessment

The port adds cache handling around host-resident MoE expert weights and the
CUDA MMVQ dispatch path. Buun Turbo changes are primarily KV-cache and
flash-attention paths; there are no direct MoE-cache hooks in the Turbo
attention implementation. The full build compiled and linked the cache
alongside the complete Turbo template matrix.

What remains unverified without GPU execution is the important runtime
interaction: VRAM allocation/trim behavior with Turbo KV, output correctness,
cache hit/miss merging, and throughput at short and loaded context. In
particular, build success does not establish correctness for the two
buun-only Q2 MMVQ cases adapted above. Those uncertainties should remain
prominent in Stage 1 and the subsequent ON/OFF benchmark ladder.

## Build validation

Configured a fresh `build/` directory with the production-relevant settings:

```sh
PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_OPENSSL=ON
cmake --build build --parallel 16
```

Result: **green, 100%**, using CUDA 13.2.86 and sm_86. The following relevant
targets compiled and linked:

- `libggml-cuda.so`, including `mmvq.cu` and `moe-cache.cu`
- `test-moe-cache`
- `test-arg-parser`
- `llama-bench`
- the full server/CLI application build

The build emitted recurring unused-variable warnings from existing Turbo
flash-attention templates (`nstages` and `is_turbo_kv`) plus ordinary
upstream deprecation warnings. No warning identified a MoE-cache compile
failure, and the build exited zero.

## Test execution boundary

`ctest --test-dir build -N -R moe-cache` lists exactly
`Test #41: test-moe-cache`. The target was compiled and linked, but was
**not executed**. Static inspection shows its `main()` calls
`ggml_backend_load_all()`, finds a CUDA device, and invokes
`ggml_backend_dev_init()` before exercising cache hits, fault injection,
lifecycle, invalidation, admission, and multi-device scenarios. Executing it
would therefore initialize a CUDA context and violate the hard constraint.

No other test or llama binary was executed. No model was loaded, and no GPU
context was created. Stage 1 should run `test-moe-cache` on-device before any
model smoke test, as planned by the brief owner.

## Handoff

The branch is deliberately not merged into `master`, and the worktree is
left in place. Recommended owner-run validation order remains:

1. Execute `test-moe-cache` on the intended CUDA device.
2. Run the 8K `-lv 4` smoke test.
3. Run short-prompt and loaded-context Turbo4 cache ON/OFF pairs.
4. Sweep explicit placement/budget combinations only after correctness is
   established.

| Delta | Why | Upstream status |
|---|---|---|
| `AGENTS.md` (new) + 3-line pointer prepended to upstream `CLAUDE.md` | fork process contract for agents (policy #10 distillation); pointer keeps single-discovery-path — CLAUDE.md remains buun's content otherwise | ours (never upstreamed; drop pointer if buun adds his own AGENTS.md) |


## Round-2 fusion branch conflict warning

The rejected device gate exposed a ceiling/admission coupling in the first
fusion prototype: raising the hook-array ceiling from 64 to 192 also reserved
192 rows of CUDA scratch for every legacy `begin`, which made the narrow
8 MiB route-override and 64-slot admission fixtures ineligible. Meanwhile a
paired route could dispatch twice its per-projection id count. The follow-up
keeps `begin` source-compatible at its original 64-row reservation, appends
`begin_rows` for the CPU caller's actual route size, and reserves `2 * n_ids`
for paired dispatch. This restores legacy admission while protecting the
fused DFlash transaction's full scratch headroom.

The separate `moe-cache-fusion` prototype intentionally edits
`ggml/src/ggml-cuda/mmvq.cu` and `mmvq.cuh` again. These are the highest-risk
files when applying its plain diff to the leloch tree: the patch adds a
cache-only per-hit activation-index pointer, changes the cache MMV signature,
and changes the ids-branch activation-row mapping. This sits directly beside
buun's Q2/small-K adaptations documented above.

Treat a textual clean apply as insufficient. Preserve leloch's kernel work,
verify every `ggml_cuda_moe_cache_mmv` call and the Q2_0/Q2_0_G128 switch arms,
then compile sm_86 and run the dedicated Q2 tail/last-slot cases before the
Laguna benchmark. See `docs/moe-cache-fusion/DESIGN.md` for the paired route
contract and fallback semantics.
