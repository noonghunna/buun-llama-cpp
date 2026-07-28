# Gate/up MoE cache route fusion prototype

## Objective

The prototype turns the two independent gate and up `MUL_MAT_ID` operations
into one paired route. The selected-expert ids are read once, both cache plans
are populated in one outer traversal, the compact activation working set is
uploaded and quantized once, and both MMVQ batches run on one CUDA stream. One
combined D2H copy and stream synchronization collects both results. The down
projection remains a second transaction because its input depends on SWiGLU.
The intended steady-state target is therefore about two waits per MoE layer.

## Graph and API shape

`ggml_mul_mat_id_pair()` reuses `GGML_OP_MUL_MAT_ID` and marks the paired form
with `src[3]`. Its F32 result packs `[up, gate]` along dimension zero; graph
views restore the two original tensors without a copy. The graph selects it
only when the separate weights have equal shape/type and there are no expert
scales, LoRAs, or CPU-repacked tensor traits. All other models retain the
existing two-node graph exactly.

The legacy cache callbacks are unchanged. Four appended callbacks implement
the paired seam:

- `route_begin`: begins both sibling tensors and fills both slot maps while
  traversing the ids array once;
- `route_dispatch`: accepts both compact hit lists and returns success only
  after both launches have been accepted;
- `route_collect`: downloads/scatters both result sets with one wait;
- `route_end`: releases all pins and the route-owned device exclusion.

The C-facing descriptors are POD so the CPU backend does not depend on CUDA or
C++ types.

## Unique activation indexing

Both the legacy dispatch and paired dispatch compact activation rows by exact
host pointer identity. Each hit carries a device-side activation index in
addition to its slot index. `mmvq` uses that index instead of the old
`hit % act_rows` assumption. This handles partially duplicated multi-token
routes correctly and is a prerequisite for raising the 64-hit ceiling later.

For the paired route, compaction spans both projections, so a gate/up hit pair
for the same token maps to one uploaded activation row. The device ids scratch
layout is `[slot indices][activation indices]`; no new allocation class is
introduced.

## Failure semantics

The blocker-fix contract is preserved for the whole pair:

- If begin/plan cannot form a same-device pair, both projections execute on
  the canonical CPU path.
- If fused dispatch fails, every skipped id in both filtered maps is restored
  before CPU workers start, then both projections execute normally on CPU.
- If fused collect fails, both full projections are replayed in parallel with
  the canonical CPU kernel before the operation returns.
- `route_end` is called exactly once for every successful `route_begin`, and a
  still-in-flight route synchronizes once before releasing pins.

`test-moe-cache` adds fused hit, dispatch-fault, and collect-fault scenarios
and compares the packed result with a cache-disabled CPU reference.

## Deliberate prototype limits

- Down remains on the legacy per-node cache seam.
- The graph does not pair scaled expert weights, LoRA-adapted experts,
  pre-merged gate/up weights, or repacked CPU weights.
- The canonical CPU miss path is invoked twice, so activation conversion on
  CPU is not yet shared. The CUDA hit path is the mechanism under test.
- Route scratch growth is checked against the existing session budget at
  dispatch. A future production version can reserve the paired maximum during
  census to remove that late growth decision.
- The public operation marker uses `src[3]` rather than adding an enum value;
  this keeps scheduler/backend plumbing small for the prototype.

## Validation boundary

Author validation is compile-only for CUDA sm_86. No CUDA context, test binary,
model, or benchmark is executed. The maintainer should run, in order:

```bash
./build-fusion/bin/test-moe-cache
```

Then run cache OFF/ON Laguna money-bench pairs with the three-arm timing build.
Require the fused fault tests to pass before interpreting throughput.

## High-conflict files

`ggml/src/ggml-cuda/mmvq.cu` and `mmvq.cuh` are the high-risk apply points on
the leloch tree. This prototype changes the cache MMV signature and the
ids-branch channel-to-activation mapping next to buun-only Q2 and small-K
dispatch work. Resolve those files semantically, then compile both Q2 variants;
do not accept a clean textual apply as validation.
