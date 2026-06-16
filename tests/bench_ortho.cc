/*
 *  Standalone microbench for the forward-pass orthonormalize, isolated from the
 *  full fit() search. No MarsAlgo, no epoch loop, no prune, no Python: just
 *  build a pre-populated m-wide (orthonormal-ish) Bo plus B / candidate X / y,
 *  and time the orthonormalize directly --
 *
 *    CPU: mars::orthonormalize() per X-column (OpenMP over columns, like the
 *         real eval binding), then the linear-ΔSSE dot.
 *    GPU: mars::cuda::orthonormalize_batch() over a block of candidate columns
 *         against a resident Bo (what eval(cuda=True) drives).
 *
 *  Bo columns are unit-norm random vectors: at n >> m they are mutually
 *  near-orthogonal (off-diagonal ~ 1/sqrt(n)), which is enough for a faithful
 *  timing -- the GEMM/reduce work is independent of exact orthonormality, and
 *  the DGKS gate stays quiet (as it does on a real orthonormal Bo).
 *
 *  Usage:  bench_ortho [n] [m] [p] [gpu_cols] [cpu_calls] [repeat]
 *    n         rows                          (default 300000)
 *    m         basis width (Bo columns)      (default 400)
 *    p         candidate X-columns           (default 1000)
 *    gpu_cols  candidate cols per GPU block  (default 8192, clamped to capacity;
 *              tune the cap with MARS_CUDA_BLOCK_GB)
 *    cpu_calls X-column orthonormalize calls (default 128; 0 = skip the CPU
 *              baseline). Each call is O(n*m^2) and the CPU side is single-pass
 *              (not repeated), so at large m it dominates the bench wall time --
 *              keep it small (~one wave per core) or 0 for GPU-only iteration.
 *    repeat    timed GPU repeats             (default 10; GPU only)
 *
 *  GFLOP/s counts the two GEMMs (T = Boᵀ·Bx and the projection): 4·n·m per
 *  candidate column.
 */
#include "kernels.h"
#include "cuda/mars_cuda.h"

#include <atomic>
#include <chrono>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {

double now_s()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

size_t arg(int argc, char **argv, int i, size_t dflt)
{
    return (argc > i) ? (size_t)std::strtoull(argv[i], nullptr, 10) : dflt;
}

}  // namespace

int main(int argc, char **argv)
{
    const size_t n        = arg(argc, argv, 1, 300000);
    const size_t m        = arg(argc, argv, 2, 400);
    const size_t p        = arg(argc, argv, 3, 1000);
    size_t       gpu_cols = arg(argc, argv, 4, 8192);
    const size_t cpu_calls = arg(argc, argv, 5, 128);
    const int    repeat   = (int)arg(argc, argv, 6, 10);
    const double tol      = (n * 0.02) * DBL_EPSILON;

    // Line-buffer stdout so the GPU result streams before the (slower) CPU run.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const char *block_gb = std::getenv("MARS_CUDA_BLOCK_GB");
    std::printf("# n=%zu m=%zu p=%zu gpu_cols=%zu cpu_calls=%zu repeat=%d BLOCK_GB=%s\n",
                n, m, p, gpu_cols, cpu_calls, repeat, block_gb ? block_gb : "default");

    // ---- build data (host) -------------------------------------------------
    std::mt19937 rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Bo: row-major (n, m), each column unit-norm -> near-orthonormal at n>>m.
    std::vector<float> Bo((size_t)n * m);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            Bo[i * m + j] = gauss(rng);
        }
    }
    for (size_t j = 0; j < m; ++j) {
        double sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = Bo[i * m + j];
            sq += v * v;
        }
        const float inv = (float)(1.0 / std::sqrt(sq));
        for (size_t i = 0; i < n; ++i) {
            Bo[i * m + j] *= inv;
        }
    }

    // B: col-major (n, m); X: col-major (n, p); y: (n); s: ones.
    std::vector<float> B((size_t)n * m), X((size_t)n * p), y(n);
    for (auto &v : B) v = gauss(rng);
    for (auto &v : X) v = gauss(rng);
    for (auto &v : y) v = gauss(rng);
    std::vector<float> s(p, 1.0f);

    // ---- GPU ---------------------------------------------------------------
  try {
    mars::cuda::Context *ctx = mars::cuda::context_create(n, m, tol);
    mars::cuda::context_set_target(ctx, y.data());
    mars::cuda::context_sync_basis(ctx, m, B.data(), n, Bo.data(), m);
    std::vector<int> cols(p);
    std::iota(cols.begin(), cols.end(), 0);
    mars::cuda::context_sync_xcols(ctx, p, X.data(), n, s.data(), cols.data(), p);

    const size_t cap = mars::cuda::context_batch_capacity(ctx);
    if (gpu_cols > cap) {
        std::printf("# gpu_cols clamped %zu -> %zu (batch capacity; raise MARS_CUDA_BLOCK_GB)\n",
                    gpu_cols, cap);
        gpu_cols = cap;
    }
    std::vector<int> src_xcol(gpu_cols), src_basis(gpu_cols);
    for (size_t j = 0; j < gpu_cols; ++j) {
        src_xcol[j]  = (int)(j % p);
        src_basis[j] = (int)(j % m);
    }
    std::vector<double> ybx(gpu_cols);

    mars::cuda::orthonormalize_batch(ctx, m, gpu_cols, src_xcol.data(),
                                     src_basis.data(), ybx.data());  // warmup
    double t0 = now_s();
    for (int r = 0; r < repeat; ++r) {
        mars::cuda::orthonormalize_batch(ctx, m, gpu_cols, src_xcol.data(),
                                         src_basis.data(), ybx.data());
    }
    const double gpu = (now_s() - t0) / repeat;
    mars::cuda::context_destroy(ctx);

    const double per_col_flop = 4.0 * (double)n * (double)m;  // 2 GEMMs
    std::printf("GPU  %8.2f ms/block  %8.0f cols/s  %7.1f GFLOP/s  (%.3f us/col)\n",
                gpu * 1e3, gpu_cols / gpu, per_col_flop * gpu_cols / gpu / 1e9,
                gpu * 1e6 / gpu_cols);

    // ---- CPU (OpenMP over X-columns, like the eval binding) ----------------
    // Each call is O(n*m^2) and there are `cpu_calls` of them (single pass, not
    // repeated) -- at large m this is the slow part, so it's skippable (0) and
    // kept small by default. cpu_calls=0 -> GPU-only.
    if (cpu_calls == 0) {
        std::printf("# CPU baseline skipped (cpu_calls=0)\n");
        return 0;
    }
    std::printf("# CPU baseline: %zu calls (O(n*m^2) each) -- the slow part...\n",
                cpu_calls);
    std::vector<int> mask(m);
    std::iota(mask.begin(), mask.end(), 0);
    auto cpu_run = [&]() {
        #pragma omp parallel
        {
            std::vector<float>  Bx((size_t)n * m), x(n);
            std::vector<double> T((size_t)m * m), s_buf(m);
            #pragma omp for schedule(static)
            for (long c = 0; c < (long)cpu_calls; ++c) {
                const float *xc = X.data() + (size_t)(c % p) * n;
                const float  sc = s[c % p];
                for (size_t i = 0; i < n; ++i) x[i] = xc[i] * sc;
                mars::orthonormalize(n, m, m, B.data(), n, x.data(), mask.data(),
                                     Bo.data(), m, Bx.data(), n, T.data(), m,
                                     s_buf.data(), tol);
                for (size_t j = 0; j < m; ++j) {
                    volatile double d = mars::dot_widen(Bx.data() + j * n, y.data(), n);
                    (void)d;
                }
            }
        }
    };
    // Single timed pass (no warmup/repeat -- it's already O(n*m^2)*cpu_calls).
    t0 = now_s();
    cpu_run();
    const double cpu = now_s() - t0;
    const double cpu_cols = (double)cpu_calls * m;
    std::printf("CPU  %8.2f ms       %8.0f cols/s  %7.1f GFLOP/s  (%.3f us/col)\n",
                cpu * 1e3, cpu_cols / cpu, per_col_flop * cpu_cols / cpu / 1e9,
                cpu * 1e6 / cpu_cols);

    std::printf("speedup (GPU/CPU cols/s): %.2fx\n",
                (gpu_cols / gpu) / (cpu_cols / cpu));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "bench_ortho: %s\n", e.what());
    std::fprintf(stderr, "  (out of memory? lower MARS_CUDA_BLOCK_GB, n, or m -- "
                         "batch scratch is ~16·n·p_cap bytes)\n");
    return 1;
  }
    return 0;
}
