#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef GGML_MOE_CACHE_MAX_TOPK
#define GGML_MOE_CACHE_MAX_TOPK 192
#endif

#if GGML_MOE_CACHE_MAX_TOPK < 1
#error "GGML_MOE_CACHE_MAX_TOPK must be positive"
#endif

#ifdef __cplusplus
extern "C" {
#endif
// The paired route API is deliberately POD-only: ggml-cpu is C, while the CUDA
// provider is C++. It extends the legacy per-node seam without changing it.
struct ggml_moe_cache_tensor_desc {
    const char * name;
    const void * host_base;
    size_t expert_size;
    int64_t n_in;
    int64_t n_out;
    int wtype;
    int64_t n_expert;
    int64_t n_tokens;
};

struct ggml_moe_cache_dispatch_desc {
    int n_hits;
    const int32_t * slot_idx;
    const float * const * act_rows;
};

struct ggml_moe_cache_result_desc {
    int n_hits;
    float * const * dst_rows;
    int64_t n_out;
};


struct ggml_moe_cache_api {
    const void * owner;

    // The scheduler owns one cache session. backends contains the scheduler's
    // actual backend set, so the provider can use only selected CUDA devices.
    void * (*session_create)(void * const * backends, int n_backends);
    void   (*session_destroy)(void * session);
    // NULL and dormant sessions still create a suppressing thread-local scope.
    void   (*session_enter)(void * session);
    void   (*session_leave)(void * session);

    // Begin one CPU MUL_MAT_ID node. Returns an opaque plan, or NULL when the
    // stock CPU path should handle the complete node.
    void * (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                    int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // Mark cache hits and enqueue bounded demand fills for misses. A nonnegative
    // slot index means that the row may be omitted from CPU work only if
    // dispatch subsequently succeeds.
    int (*plan)(void * node, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // Dispatch all planned hit rows. Returns 1 only after the complete GPU
    // operation has been accepted. On 0, the caller must restore every row to
    // the normal CPU mapping before worker threads start.
    int (*dispatch)(void * node, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                    const int32_t * slot_idx, const float * const * act_rows);

    // Copy GPU results into dst_rows. On 0, the caller must recompute every
    // skipped row on the CPU.
    int (*collect)(void * node, int n_hits, float * const * dst_rows, int64_t n_out);

    // Releases slot pins and all per-node ownership. Must be called exactly
    // once for every non-NULL begin result, on every success or failure path.
    void (*end)(void * node);

    // Host buffer mutation or teardown notification. Sessions cancel or finish
    // any fill that still reads the supplied range before this call returns.
    void (*invalidate)(const void * base, size_t size);

    // Begin and plan the two sibling projections of a gate/up route. The ids
    // array is traversed once and both slot maps are populated together.
    // Returns NULL unless both tensors can share one device transaction.
    void * (*route_begin)(
            const struct ggml_moe_cache_tensor_desc tensors[2],
            const int32_t * ids, int n_ids, int32_t * slot_idx[2]);

    // Upload/quantize the compact unique activation set once and launch both
    // expert matvec batches on the same stream. On 0, restore all skipped rows
    // for both projections before CPU workers start.
    int (*route_dispatch)(
            void * route, const struct ggml_moe_cache_dispatch_desc pairs[2]);

    // Download both result sets and wait once. On 0, recompute all skipped rows
    // for both projections on CPU.
    int (*route_collect)(
            void * route, const struct ggml_moe_cache_result_desc pairs[2]);

    void (*route_end)(void * route);

    // Sized extension of begin(). n_route_rows is the maximum number of hit
    // rows this node may dispatch. Providers predating this callback retain
    // the legacy 64-row reservation through begin().
    void * (*begin_rows)(
            const char * tensor_name, const void * host_base, size_t expert_size,
            int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
            int64_t n_tokens, int64_t n_route_rows);
};

extern struct ggml_moe_cache_api ggml_moe_cache;
void ggml_moe_cache_unregister(const void * owner);

#ifdef __cplusplus
}
#endif
