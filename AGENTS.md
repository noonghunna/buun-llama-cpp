# Agent operating rules for this fork

This fork stages MoE-expert-cache work on top of `spiritbuun/buun-llama-cpp`. The full sync/branching policy is pinned issue #10 — this file is its agent-facing distillation. **Read it before any work here.**

## Non-negotiables

1. **Never commit to `master`.** It is a fast-forward-only mirror of buun's master. All work branches off `moe-cache-port` (the integration trunk).
2. **Ticket-first, single purpose per branch** (`feat/<name>` / `fix/<name>`). Branches double as future PR units to buun — a branch that mixes concerns can't be upstreamed.
3. **PORT-NOTES.md is the delta manifest.** Any change to engine code adds/updates its entry: what, why, upstream status (`ours` / `PR-open` / `merged-by-buun` / `superseded`). A delta without an upstream story is a defect of the delta.
4. **Expect rebases.** The trunk and active branches rebase onto buun's master on sync (tags preserve old SHAs). Consequences for authors:
   - keep commits small and self-contained (they must survive replay);
   - export `git apply --check`-able diffs alongside branches;
   - after a sync you may be asked to re-deliver conflicted work — the hand-inspection surfaces are `mmvq.cu`, `ggml-cpu.c` (`mul_mat_id`), `ggml-backend.cpp` scheduler sites, `arg.cpp`, repack wiring, and the dflash code (where buun's evolution WINS and our delta re-applies on top).
5. **No GPU execution by authoring agents.** Compile-only validation (sm_86). Device runs — tests, smokes, benches — are performed by the maintainer session; state in your completion note which device run your change owes.
6. **No internal rig paths, hostnames, or usernames** in committed files or issues. Use `<models>/`-style placeholders.
7. **Build etiquette:** your compiles share CPU with measured benches on the same rig. Batch compiles; a stray parallel build has silently cost a bench −22% before.
8. **Results go to issues**, not local notes: completion comments carry branch + commit + owed device run; measurements that gate merges are posted on the ticket they gate.

## Sync execution (when explicitly asked to assist a sync)

Follow issue #10's checklist exactly: ff `master`, tag tips, rebase trunk then branches, hand-inspect the surfaces above even on clean applies, update PORT-NOTES statuses, and STOP before the device gate — build green is not the gate; the maintainer runs `test-moe-cache` + smoke + bench-sanity before any force-push.
