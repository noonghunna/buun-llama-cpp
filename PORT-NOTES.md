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
| `common/speculative.cpp`: fall back from the draft-hparams target-layer getters to the architecture-`dflash` model-vector getters, normalize that one-based convention for zero-based capture, and emit an error-level warning when no layers resolve | poolside-convention DFlash models populated `target_layer_ids` but the cross-ring read only the `dflash-draft` getter family, silently binding a zero-layer ring | ours (fork issue #4; upstream candidate after original-drafter device acceptance) |
| DFlash poolside mask-token metadata aliases + explicit runtime override/warning | prevents missing metadata from silently becoming token 0 and poisoning every masked draft position; accepts both deployed key spellings and provides `--dflash-mask-token` / `LLAMA_ARG_DFLASH_MASK_TOKEN` for mask-less exports | ours (fork issue #4; integrated in `dflash-set`; device acceptance pending) |
| `dflash-set` constructor/factory integration | resolves the ring-layer and mask-token constructor conflict by forwarding the explicit override from the sole factory call, initializing pointer members before any model/context access, resolving mask state independently of capture-layer selection, and retaining the zero-layer diagnostic without an unsafe early return | ours (fork issue #4 integration branch; device acceptance pending) |
| Official-DFlash forward-contract routing + Laguna parity | `architecture=dflash` now uses the reference encoder → decoder-KV injection driver instead of the incompatible `dflash-draft` cross-ring path; restores pre-final-norm target capture, Laguna causal noise attention, normalized injected K/V, and per-slot accept notifications. The legacy cross-ring path remains selected for `architecture=dflash-draft` | ours (fork issue #4; source-parity fix against the poolside reference, device acceptance pending) |
| Laguna residual-stream graph outputs for DFlash | registers every layer input in `t_layer_inp`, publishes the pre-final-norm `t_h_nextn` capture, and defers final output-row gathering when unmasked nextn extraction is enabled; this prevents the first reserve/decode graph from asserting on a null capture tensor and preserves every-token feature rows | ours (fork issue #4 round-4 repair; source parity with the working reference, device acceptance pending) |
| Official-DFlash reinjection rewind | removes the stale speculative suffix from the drafter KV before target-conditioned encoder output is injected at the same positions; without the rewind, the batch allocator rejects the first verification-cycle injection as a non-consecutive position rewind | ours (fork issue #4 round-4 follow-up; device acceptance pending) |
| Official-DFlash position lifecycle + trace diagnostics | rewinds the target-conditioned suffix again before steady-state re-drafting after partial acceptance, validates consecutive positions before both drafter decoder calls, logs KV min/max plus injection offset/token count at `-lv 4`, and replaces opaque decode failures with expected-vs-actual position diagnostics | ours (fork issue #4 round-5 repair; device acceptance pending) |

---

## Delta manifest — 2026-07-30 (`moe-cache-engine-v1`)

Recorded late: these five deltas were authored during the engine-lock session and
were **not** entered here at the time, which violates AGENTS.md rule 3 ("a delta
without an upstream story is a defect of the delta"). Entered now, with the
upstream story each one will need.

Tag `moe-cache-engine-v1` = `43c43a5e8`. The tag predates this section; it is the
validated engine snapshot, this is living documentation of it.

### ⚠️ Attribution — read before opening any PR to buun

**The MoE expert cache is [leloch](https://github.com/leloch/llama.cpp)'s work, not
ours.** We ported it (see the Cherry-picks table above; the `-x` trailers carry the
exact source hashes and the original author identity is preserved). Any upstream PR
that carries the cache must either come from leloch or credit leloch explicitly and
link `leloch:moe-cache-pr`. Our deltas below are ours; the substrate they sit on is
not, and a PR that blurs that would be misattribution.

### Our deltas

| Delta | PR-unit branch | Why | Upstream status |
|---|---|---|---|
| `speculative: route single-seq draft to the caller's drafter sequence` | `fix/draft-dflash-seq-routing` (`66f9c7c55`) | `common_speculative_draft()` hard-coded `spec->dparams[0]`; contracts sharing one `common_speculative` across slots (DRAFT_DFLASH) had every slot draft into drafter sequence 0 — guaranteed KV-lineage collision at N>=2, taking down all in-flight slots | **ours — PR-ready to buun independently.** Touches buun's own DFlash contract, no moe-cache dependency. Highest-value single PR we hold. |
| `server: honour --dflash-max-slots for the DRAFT_DFLASH contract` | `fix/draft-dflash-slot-cap` (`5c5449f9f`) | `dflash_slots_cap` is triple-purpose (slot cap, shared drafter-ctx creation, AND the gate for shared `spec` init). Naive gate widening disables speculation entirely; this adds a separate counter | **ours — PR-ready independently.** buun's server code. |
| `cuda: warn when MAX_BATCH silently bypasses the MoE expert cache` | `fix/moe-cache-batch-bypass-warn` (`12be46004`) | Violating `MAX_BATCH >= n-max+1` makes the cache refuse every decode with no diagnostic — cost two runs read as -30% / -48% | **ours — BLOCKED on the cache landing upstream first.** Meaningless in a tree without moe-cache. |
| `moe-cache: make the cache-off states impossible to miss` (C/D/E) | `fix/moe-cache-observability` (`bf3da5261`) | Bypass ceiling 32->128 (ngram drafters emit 48-64 token drafts, so the guard was blind to them); one-shot flag process-global -> per-session; "no cache budget" INFO -> WARN with remedy (it disables the cache entirely yet logged quieter than the partial bypass) | **ours — BLOCKED on the cache landing upstream first.** |
| `clean the disclosed local probe patches out of the serving path` | `chore/drop-local-probe-patches` (`43c43a5e8`) | Split delta: **`common/arg.cpp`** removes upstream's hard 256-token DFlash draft-ctx default (the drafter tracks absolute positions, so a fixed cap truncates at depth) — that half is a real fix to buun's code. **`src/llama-context.cpp`** deletes our own R5 dormancy probe — nothing to upstream | **arg.cpp half: ours, PR-ready independently. llama-context half: local-only, nothing to send.** Split the commit before PRing. |

### Held back deliberately

| Item | Status |
|---|---|
| L2 `feat/spec-act-dedup` (`1f49319a4`) | **excluded from the tag.** Fails `cache-fill-invalidate` where base and L1+L3 pass (bisected 2026-07-30). Also measured ~10-14% slower. Do not upstream; do not fold into the engine until diagnosed. |

### PR ordering when the time comes

1. `fix/draft-dflash-seq-routing` — independent, and a crash fix. Send first.
2. `fix/draft-dflash-slot-cap` — independent.
3. `chore/drop-local-probe-patches`, **arg.cpp half only** — independent.
4. The moe-cache port itself — **leloch's call / leloch's credit**.
5. `fix/moe-cache-batch-bypass-warn` and `fix/moe-cache-observability` — only after (4).

Maintainer decision 2026-07-30: nothing goes upstream until the moe-cache work has
community validation, and we wait for buun to sync with mainline first (he is 108
commits behind ggml-org as of 2026-07-30, last mainline merge ~2026-07-21).
