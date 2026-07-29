# The spec-dec × CPU-offload challenge — measured bottleneck ledger

**Mission:** make speculative decoding (DFlash) net-positive on this stack — single-stream and, if economically viable, multi-stream — when most expert weights live in CPU RAM. Reference platform: 2×3090 PCIe (no NVLink/P2P), Laguna-S-2.1 (118B/8B-active, top-10 of 256 experts, 48 layers), expert layers CPU-offloaded, VRAM expert cache, turbo4 KV, 262K ctx, DFlash cross-attention drafter (GPU-resident, SWA-512).

> **REVISION 2 (2026-07-29).** A three-way blind audit plus a device campaign overturned much of revision 1. Two engine bugs were found and fixed, several "measurements" turned out to be artifacts, and the headline cause was wrong. Corrections are marked ⛔ inline — **read them before proposing anything**, because revision 1's framing sent three auditors after the wrong term. Full history: `specconc-reconciliation.md`. **Current answer to the mission: on this platform, no.** Spec-dec is net-negative at every draft depth and every concurrency tested. The remaining question is whether the dominant fixed cost is removable.

## Status quo — all first-party measured

Single-stream, N=1, 262K, 19L CPU offload, median of rounds 2–7 (2026-07-29):

| Configuration | TPS | Verdict |
|---|---|---|
| no-spec | **39.0** | the bar spec must beat |
| drafter loaded, `--draft-max 0` (runs, emits nothing) | **30.6** | **−21.5% — pure overhead, zero payoff.** The single most important number here |
| spec draft-1 | 34.3 | −12.1% |
| spec draft-2 | 33.4 | −14.4% |
| spec draft-3 | 29.4 | −24.6% |

Concurrency, aggregate tok/s, spec arms use the default slot cap (only slot 0 speculates):

| N | no-spec | spec | Δ |
|---|---|---|---|
| 1 | 31.6 | 24.0 | −24% |
| 2 | 49.9 | 28.6 | −43% |
| 4 | 53.7 | 34.3 | −36% |
| 6 | 59.5 | 38.8 | −35% |

⛔ **Corrections to revision 1's table:**
- ~~"spec `-np {2,3}` aggregate 3.9–10.6 → collapse 5–13×"~~ — **not a performance result.** All 35 concurrent requests died HTTP 500 from a drafter-state bug (below). The figure was one surviving stream's partial tokens ÷ a wall that was ~80% prefill. Decomposed exactly: 2.0× (errored streams zeroed) × 4.6× (wall/decode) = 9.2× vs the 8.8× observed.
- ~~"−9% shallow, spec loses"~~ — directionally right, wrong attribution. The −9% figure was draft-3-specific.
- ~~"no-spec `-np 4` = 74.3, multi-lane crown"~~ — **treat as unproven.** Independent runs give 53.7 and 60.2. It was a single last-round headline (see house rules on probe noise). Do not quote it as a baseline.
- "+7% at ~99K depth" (26.5 vs 28.3) stands but is a single pair on a config whose shallow numbers have since moved; crossover still unmeasured.

## The core physics — REVISED

Speculation pays when verifying K tokens costs ≈ one token, and when the drafter's own forward is cheap relative to the target's. On GPU-resident models both hold. Here **neither** does, and revision 1 named only the second one.

**Term one — drafter + capture overhead (DFlash-specific).** The `--draft-max 0` arm carries *no* activation duplication and *no* verify widening — the draft is empty, so the target batch is 1 token, **identical target work to no-spec** — yet it still costs 21.5%. ⚠️ **This is NOT an expert-path cost.** The drafter is dense and GPU-resident (no `ffn_*_exps` tensors); it never runs an offloaded expert and cannot compete with expert streaming. The 21.5% is the drafter's own dense GPU forward serially added to each step, plus the capture path on the target (hidden-state extraction every decode). d0 does not separate those two. Acceptance buys back roughly half (30.6 → 34.3 at draft-1). **Model-free drafters pay ~none of this term** — and are still net-negative, which is what identifies term two as the binding constraint.

**Second term — expert-union cost (revision 1's constraint #1, demoted).** A K-token verify batch routes to the union of ~K×10 experts, and CPU miss streaming scales with that union while acceptance returns ~2.1 tokens/round at draft-3. This is what makes depth *additionally* unprofitable (34.3 → 29.4 going draft-1 → draft-3), but it is **not** why spec loses at draft-1.

**Only the TARGET touches offloaded experts.** Every target forward routes the MoE layers: a no-spec decode routes top-10 for one token, a K-token verify routes the union of ~K×10. That is the whole of the CPU-offload penalty, and it is paid on the **verify**, per round. **Therefore wins can only come from: cheaper verify (lower C(k)), higher α at fixed verify width, or not verifying wide multi-token batches.** ⛔ "A drafter that avoids the experts" is NOT a lever — no drafter ever touched them. (DFlash is +27% on prose GPU-resident on this same rig because term one is cheap relative to a fast target there, not because of anything expert-related.)

Cost curve anchored on the measured α=1 point (better than a fitted intercept): `C(k) = TPS(d0)·α / TPS(d)` → **C(2)=1.41, C(3)=1.70, C(4)=2.19**. ⚠️ α values (1.58/1.85/2.10) are imported from earlier log measurements, not re-measured on this config.

**Any design that increases tokens-per-verify without addressing expert-union cost is still pre-refuted — but a design that only addresses the union cannot make spec net-positive here either.** Break-even against no-spec needs ~45% of the verify-batch cost multiplier removed at draft-3 (~41% at draft-1).

## Model-free drafters — measured, and they settle where the binding constraint is

Eight drafters that run **no forward pass at all** (`suffix`, `copyspec`, `recycle`, four `ngram-*`). They pay ~none of term one. Laguna supports exactly this family and no other: it has **no `nextn`/MTP head**, no EAGLE3 drafter, and no small vocab-compatible draft model exists. Single-stream N=1, prose workload, vs no-spec 38.68:

| drafter | tok/s | vs no-spec | acceptance |
|---|---|---|---|
| ngram-map-k | 37.41 | −3.3% | 0.04 |
| ngram-simple | 35.54 | −8.1% | **0.00** |
| ngram-map-k4v | 35.46 | −8.3% | 0.02 |
| copyspec | 35.22 | −8.9% | **0.45** (issues drafts on only ~1% of steps) |
| ngram-mod | 33.36 | −13.8% | 0.19 |
| suffix | 27.79 | −28.2% | 0.11 (drafts on nearly every step) |
| ngram-cache | 27.70 | −28.4% | 0.13 |

**All net-negative, and throughput tracks INACTIVITY rather than skill** — the best-looking arm accepted nothing at all. Removing the drafter cost entirely is therefore *not sufficient*: the binding constraint is verify economics, exactly as term two predicts. Only `copyspec` shows real signal, and it does so by **abstaining** — 45% acceptance on the rare prose steps that offer a copy match.

**Break-even for a zero-cost drafter is `α ≥ C(k)`.** At the measured C(2)=1.41 that is **41% acceptance at draft-1**, which only copyspec approaches. ⚡ **This makes cheaper verify the multiplier on the whole family:** if act-dedup drives C(2) toward ~1.15, break-even falls to ~15% acceptance — a bar `suffix` (11%), `ngram-cache` (13%) and `ngram-mod` (19%) already sit at or above. Lowering C(k) does not merely improve its own arm; it decides whether model-free speculation becomes viable at all.

## Measured constraint ledger

1. **Acceptance:** draft-1 ≈ 51–61%, draft-2 ≈ 40–47%, draft-3 mean accepted length ≈ 2.1 (incl. bonus). Prose accepts better than code.
2. **Drafter residency and forward cost (NEW, dominant).** BF16 drafter ≈ 2.1 GB, layer-split 1346 MiB CUDA0 + 781 MiB CUDA1. Costs **−21.5% before producing a single usable token**. It also displaces expert-cache pool (2117 → 647 slots at N=1) — but ⛔ **pool displacement is NOT the throughput driver**: restoring the pool (619→1201 via `RESERVE_MB=1024`, or 828 via `-devd CUDA1`) left throughput flat at 29–31. ⛔ Nor is it bandwidth competition with expert streaming (the drafter is dense and GPU-resident and never enters that path). The cost is the drafter's own forward + the target-side capture path.
3. **Cache hook ceiling:** `MOE_CACHE_MAX_TOPK=64` at `ggml-cpu.c:1495` → verify batch ≤ 6 tokens at top-10. ⛔ **Correction:** the live constraint is `(draft+1)·top_k ≤ 64` → draft ≤ 5, with **no `N_slots` factor** — `force_split_seq` keeps ubatches per-sequence, so the `×N_slots` form only becomes true *if* cross-slot verify batching is ever built. Raising to 192 costs a measured **−4.5% base decode**.
4. **A second, independent refusal path** at `moe-cache.cu:1316`: `n_tokens > max_batch` (env clamps [1,8]). **`MAX_BATCH` must be ≥ `draft_max+1`**, or the hook bypasses every decode, no pool is ever created, and the run silently measures *spec with no expert cache*. This cost a full benchmark arm (read −30%). Now warns once — `fix/moe-cache-batch-bypass-warn`.
5. **Spec-only activation duplication (~8×):** with >1 token per verify the `shared_activation` fast path fails and up to n_hits activation rows are gathered/uploaded/quantized where only n_tokens are unique. Fix in flight on `feat/spec-act-dedup` (issue #12), failing two device tests. Targets term two, **not** the dominant term.
6. **Sync/serialization (partially fixed):** was ~84 blocking collect syncs per forward step; D2H-at-dispatch + parallel scatter (L1+L3, `stack/sync-stage2`) recovered ~+5% single-stream and +23% at the no-spec batching knee. Remaining: ~500 CUDA API calls/step. Fixed costs are per-STEP → spec's per-token share divides by α.
7. ⛔ **Multi-slot "collapse" (issue #13) — was a crash, now FIXED.** `common_speculative_draft()` hard-coded `spec->dparams[0]` while `common_speculative_set_seq_id()` was a no-op for the `DRAFT_DFLASH` contract that a drafter GGUF declaring `general.architecture=dflash` auto-selects. Every slot drafted into drafter sequence 0 → guaranteed KV-lineage collision at N≥2 → abort within 9–80 decoded tokens. Fixed by `fix/draft-dflash-seq-routing` + `fix/draft-dflash-slot-cap`. Validated with a pre-fix control: **0/8 requests survived before, 8/8 after**, drafter seq usage 119/0 → 383/376, and 0 errors across all 8 arms of the concurrency sweep. Multi-slot spec is now *correct*; it is still **−12.5% vs the one-slot floor**, i.e. correctness, not a win.

## Refuted / dead (measured or source-proven — do not respend)

**From revision 1, still dead:** event-ring non-blocking collect · CUDA graphs on this path · cross-device layer parallelism · drafter device pinning · wider PCIe transfers · draft-max ≥ 4 · built-in MTP drafters on MoE at TP=2 · threads ≠ 28.

**Newly refuted (2026-07-29):**
- ⛔ **"Serial per-slot draft→verify with no cross-slot batching."** Half-true and mislocated. `common_speculative_draft_batch()` and `llama_set_force_split_seq(ctx,false)` both exist — but are gated on the fork `DFLASH` type, so they are dead code for `DRAFT_DFLASH`. Not physics; a type gate.
- ⛔ **The "~7× unexplained multiplier."** Does not exist (see status-quo corrections).
- ⛔ **Pool starvation as the cause of the spec penalty.** Refuted by two mitigation arms.
- ⛔ **`dflash_tape_active` global-OR** (non-speculating slots paying the tape write). Gated on `type() == COMMON_SPECULATIVE_TYPE_DFLASH` at `server-context.cpp:4771`; never fires for Laguna.
- ⛔ **"draft-1/draft-2 at parity with no-spec."** Derived from shallow warm-up requests; does **not** reproduce on a 262K CPU-offload serving config, where draft-1 is −12.1% with non-overlapping ranges.
- ⛔ **`--spec-p-min` confidence truncation as a win.** Its best case degenerates toward draft-1, which is already a 12% loss. It cannot beat a negative number. (The feature is real and already implemented at `speculative.cpp:1321`, shipped disabled — it just cannot help *here*.)
- ⛔ **Depth-adaptive spec policy.** Best available pick is draft-1, still −12.1%. No depth schedule is net-positive on this platform.
- ⛔ **Union-aware / cache-resident-aware draft selection, as specified.** A token's expert route depends on the target's residual stream at that position; DFlash is a *dense* 6-block decoder with no router to read. Only surviving path is an offline discriminator on `routing_trace.pt` (is a token's realised expert set predictable from token id alone?) — zero rig time, answers whether the branch is real.
- **Cross-slot verify batching: DON'T BUILD** — unanimous across three independent audits. Five prerequisites including a measured −4.5% rig-wide array tax, for an envelope that cannot clear +10% against the no-spec baseline it must beat.

## The open design space

Only two levers remain that could flip the sign, and they are unequal:

- **Reduce the drafter's per-step cost (attacks the dominant −21.5%).** Drafter Q8_0 is the cheap first probe — roughly halves bytes moved per forward. Beyond that: fewer drafter blocks, shorter cross-attention window, or skipping the drafter forward on steps unlikely to accept. **This is where the mission lives now.**
- **Reduce verify-batch cost (attacks the second term).** `feat/spec-act-dedup` — must remove ~45% of the multiplier at draft-3 to reach break-even. Worth landing for correctness and for deeper drafts; **cannot on its own make spec net-positive.**

**Economics gate:** every proposal states its envelope against **39.0** (single-stream no-spec) and the concurrency table above. Score proposals on Δ(unique activation rows) and Δ(drafter forward cost), not Δ(tokens). "Don't build X" with arithmetic is a first-class deliverable.

## House rules for any resulting work

One mechanism per branch; no inserted timers on hot paths (−22% A/A-proven observer effect); end-to-end TPS only; suites must pass on the branch's own base before stacking; **compile-clean is not evidence** — resource-lifetime bugs on this path only appear on device.

**Measurement rules (learned expensively):**
- `MAX_BATCH ≥ draft_max+1` or you are benching a disabled cache. Check for `[moe-cache] … pool[…] slots=` lines in the boot log; their absence means the cache never engaged. Requires `-lv 4`.
- The concurrency probe reports `agg_by_round[-1]` — **last round only**. Measured round-to-round spread: **~29% for multi-stream aggregates**, ~6% at N=1. Two boots of an identical no-spec N=2 config gave 44.6 and 49.9. Use median-of-rounds; only deltas >12% are real from single-boot multi-stream data.
- An arm with `clean=0` is a hard stop, never a TPS row. The probe now emits `NaN` when any round errors (club-3090 #820).
- The probe's N streams are near-duplicate prompts at temperature 0 — maximal expert-union overlap across slots. Every multi-stream number here is a best case.
