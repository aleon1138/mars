# TODO

Ideas and notes for future performance work. Not planned for a specific release.

## BF16 support for X

**Goal:** ~2× speedup on the forward pass by halving memory bandwidth for the X
  matrix.

**Why it should help:** `covariates()` is ~1 FLOP/byte, which puts it near
  memory-bound territory at multi-core. Halving X bytes ≈ halving the load
  volume.

**Design:**
- bf16 only for X storage and as inputs to FMA. Accumulators stay fp32 — bf16's
  8-bit mantissa is not enough to accumulate n squared residuals without
  precision loss.
- Template `MarsAlgo` on input dtype (fp32 or bf16), dispatch from pybind based
  on numpy dtype.
- Most of the non-hot-loop code stays dtype-agnostic via templates. The real
  work is in `covariates()`.

**Complications:**
- numpy has no native bf16; need `ml_dtypes.bfloat16` or torch tensors on the
  Python side.
- Native bf16 FMA (`VDPBF16PS`) requires AVX-512 BF16 — i.e., Intel Sapphire
  Rapids+ or AMD Zen 4+. On older hardware, load bf16, convert to fp32 via
  widening cast, then normal FMA. Still wins on memory but not on compute.

## AVX-512 path

**When it makes sense:** only together with bf16, on hardware that supports
AVX-512 BF16 natively (e.g., EPYC 9845 / Zen 5c, Sapphire Rapids+). Plain
AVX-512 without bf16 is probably a wash because:
- Memory bandwidth is the real bottleneck at multi-core.
- Frequency downclocking on Skylake-X/Cascade Lake eats most of the width gain.
- AVX-512 absent from most consumer Intel (12th–14th gen).

**Design:**
- Keep current AVX2 kernel as the default path — portable and fast enough on
  most hardware.
- Add an AVX-512 + bf16 kernel behind `#ifdef __AVX512BF16__` or a runtime CPUID
  check.
- Repo stays public-friendly: anyone can compile with `-march=native` and get
  the best their CPU supports.

**Dev note:** i7-7820X (Skylake-X) has AVX-512F but no BF16 and heavy
  downclocking, so it's dev/test only. EPYC 9845 is where performance testing
  happens.

## CUDA port

**Realistic speedup:** 10–30× on the forward pass vs. current multi-core AVX2.
  Not 100× — MARS isn't pure matmul; there's sorting, dependency chains between
  epochs, and control flow for cut-point selection. Your current code already
  hits decent CPU throughput.

**Why it could still be worth it:**
- 96GB VRAM is enough to keep the entire working set (X, B, Bo, y, internals)
  resident — no PCIe transfers during a fit. That alone could be 2–3× if
  current runs are memory-bandwidth-limited.
- Tensor cores eat bf16×bf16 → fp32 natively, which is exactly the precision
  regime we want.

**Design:**
- Port incrementally. Keep CPU orchestration, move only the hottest inner loop
  (`covariates()`) to CUDA first. Measure before doing more.
- `covariates()` maps well to segmented scan primitives (CUB / thrust).
- Orthonormalization is straightforward cuBLAS.
- Per-epoch dependencies prevent parallelism across epochs, but within an epoch
  the (x_col, basis_col) candidate grid is embarrassingly parallel — same
  parallelism OpenMP is already exploiting.

**Effort estimate:** multi-week project, not a weekend.
