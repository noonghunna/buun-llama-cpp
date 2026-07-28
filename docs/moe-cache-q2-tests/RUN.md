# Q2 MoE cache runtime coverage

The synthetic matrix covers `Q2_0` and `Q2_0_G128` with 65 output rows, 64
resident experts, and a 64-wide selected-expert list. A successful case therefore
uses a shape not divisible by the Q2 small-K rows per block and necessarily
includes slot 63 in the hit dispatch. This catches accidental small-K enablement
and tail-bound errors. Output is compared with the canonical CPU result every step.

Build for Ampere without executing during the port authoring session:

```bash
cmake -S . -B build-q2-cache \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build-q2-cache --parallel --target test-moe-cache
```

Maintainer device run:

```bash
./build-q2-cache/bin/test-moe-cache
```

Compute Sanitizer run (keep cache fault injection enabled by the test itself):

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build-q2-cache/bin/test-moe-cache
```

Require both `cache-q2_0-tail-last-slot: OK` and
`cache-q2_0_g128-tail-last-slot: OK`, plus a zero Compute Sanitizer exit status.
