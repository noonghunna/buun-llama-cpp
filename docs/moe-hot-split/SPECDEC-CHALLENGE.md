# The spec-dec × CPU-offload challenge — measured bottleneck ledger

**Mission:** make speculative decoding (DFlash) net-positive on this stack — single-stream and, if economically viable, multi-stream — when most expert weights live in CPU RAM. Reference platform: 2×3090 PCIe (no NVLink/P2P), Laguna-S-2.1 (118B/8B-active, top-10 of 256 experts, 48 layers), 28/48 expert layers CPU-offloaded, VRAM expert cache ~80% hits, turbo4 KV, 262K ctx, DFlash cross-attention drafter (GPU-resident, SWA-512).

**Status quo (all first-party measured, 2026-07-28/29):**

| Configuration | TPS | Verdict |
|---|---|---|
| no-spec single-stream, shallow | **38.8** | the bar spec must beat |
| spec draft-3 single-stream, shallow | ~34.8 (33–37 by acceptance draw) | **−9% — spec LOSES shallow** |
| no-spec vs spec at ~99K depth | 26.5 vs **28.3** | **+7% — spec wins loaded**; crossover depth unmeasured |
| no-spec `-np 4` aggregate | **74.3** (knee) | multi-lane crown |
| spec `-np {2,3}` aggregate | 3.9–10.6 | **collapse 5–13×** (fork issue #13) |

## The core physics (constraint #1 — every failed idea below died on it)

Speculation pays when verifying K tokens costs ≈ one token. On GPU-resident models that holds. Here it does not: a K-token verify batch routes to the **union** of ~K×10 experts, and CPU miss streaming + cache pressure scale with that union while acceptance returns only ~2.1 tokens/round (draft-3). Measured casualties of this one curve: draft-5 (27.1) < draft-3 (33.9); draft-≥4 monotonic decay; spec < no-spec at shallow; spec×concurrency collapse. Depth flips the sign because attention cost grows with context while expert cost stays flat.

**Any design that increases tokens-per-verify without addressing expert-union cost is pre-refuted.**

## Measured constraint ledger

1. **Acceptance:** draft-1 ≈ 51–61%, draft-2 ≈ 40–47%, draft-3 mean accepted length ≈ 2.1 (incl. bonus). Prose accepts better than code. Drafter itself is cheap (GPU, ~2.1 GB, KV ~112 MiB even at 262K — SWA ring).
2. **Cache hook ceiling:** routing array `MOE_CACHE_MAX_TOPK=64` → verify batch ≤ 6 tokens at top-10 ((draft+1)×N_slots ≤ 6). Oversize = silent whole-cache bypass (now warns). Raising to 192 costs a **measured −4.5% base decode** (stack-array L1 pressure; the fusion post-mortem shows gate/up sharing can avoid part of it). `MAX_BATCH` env clamps at 8.
3. **Spec-only activation duplication (~8×):** with >1 token per verify, the `shared_activation` fast path fails and up to n_hits activation rows are gathered/uploaded/quantized where only n_tokens are unique. Fix (token-minor padded grid) is in flight on `feat/spec-act-dedup` — currently failing its suite on the oversize-refusal boundary case (issue #12).
4. **Sync/serialization background (partially fixed):** was ~84 blocking collect syncs per forward step; D2H-at-dispatch + parallel scatter (L1+L3, merged into `stack/sync-stage2`) recovered ~+5% single-stream and +23% at the no-spec batching knee. Remaining: ~500 CUDA API calls/step, host orchestration ≤7.8 ms/step upper bound. Fixed costs are per-STEP → spec's per-token share divides by α ≈ 2.1.
5. **Multi-slot spec collapse (issue #13):** hypothesis = serial per-slot draft→verify in the server loop; **the hypothesis under-explains the data by ~7×** (arm C: 3.9 aggregate vs ~29 perfect-serialization arithmetic at its own 34.4 per-stream rate) — a second multiplier (drafter ring/capture thrash per slot switch? re-prefill? scheduler pathology?) is unidentified.
6. **Regime dependence:** −9% shallow / +7% @99K; crossover unmeasured; continuous-soak p50 35.4 on the spec recipe (growing agentic sessions) — no-spec soak on the same protocol not yet run.

## Refuted / dead (measured or source-proven — do not respend)

- Event-ring non-blocking collect (zero deferral slack; consumer is the next CPU node).
- CUDA graphs on this path (CPU-backend hook; multi-device early-return).
- Cross-device layer parallelism (serial layer chain); drafter device pinning (pool redistribution eats the PCIe-hop saving).
- Wider PCIe transfers (KB-scale, latency-dominated).
- draft-max ≥ 4 (expert-union decay); MTP-style built-in drafters on MoE at TP=2 (−45/−51% history).
- More CPU threads (t=28 optimal on 32C; contention beyond).

## The open design space (where the audits should live)

- **α improvement at fixed batch cost:** dynamic draft length by rolling acceptance; entropy/temperature-aware drafting; stop-draft-on-low-confidence. Raises tokens/round without widening the union.
- **Union-aware speculation:** can the drafter (or a router-lookahead) prefer draft tokens whose expert routes overlap the batch's existing union or the cache's resident set? Speculative tokens are *optional* work — choosing cheaper-to-verify tokens is legal in a way normal decode can't.
- **Depth-adaptive spec:** auto-enable past the measured crossover (needs the crossover arm); trivially shippable as policy.
- **Multi-slot:** verify/refute the serialization + find the 7× multiplier; cross-slot verify batching design + honest envelope (prereqs: topk-192 with array-tax mitigation, MAX_BATCH raise, unique-activation indexing); cheaper intermediates (persistent per-slot drafter state, spec-on-one-slot, auto spec-off fallback ≥2 active slots as the do-no-harm floor).
- **Economics gate:** every proposal must state its expected envelope against 38.8 (single shallow) / 26.5→28.3 (loaded) / 74.3 (multi-lane no-spec). "Don't build X" with arithmetic is a first-class deliverable.

House rules for any resulting work: one mechanism per branch; no inserted timers on hot paths (−22% A/A-proven observer effect); end-to-end TPS only, ≥10% margin or A/B/A (±10–12% between-boot variance under spec; a 3-boot A/A band exists); suites must pass on the branch's own base before stacking.
