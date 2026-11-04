/*
 *  Tests for the CUDA orthonormalize path (cuda/orthonormalize.cu). Validates
 *  the GPU result against the same Eigen oracle the CPU kernel is checked
 *  against (tests/test_kernels.cc) and against the CPU kernel itself. Host C++
 *  -- calls only the mars::cuda::* host API.
 *
 *  All checks run at the f32 storage floor (~1e-5): Bo and Bx are f32, the
 *  projection arithmetic is f64.
 */
#include "kernels.h"
#include "cuda/mars_cuda.h"

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <atomic>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdC;
typedef Matrix<float,  Dynamic, Dynamic, RowMajor> MatrixXfC;

namespace {

bool cuda_available()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

double invnorm(VectorXd x)
{
    const double s = x.norm();
    return s > 1e-14 ? 1.0 / s : 0.0;
}

// Build a strictly orthonormal Bo (row-major f32) via modified Gram-Schmidt in
// f64, then cast to f32 -- matches what MarsData holds after append(). Mirrors
// the helper in tests/test_kernels.cc.
MatrixXfC make_orthonormal_basis(int n, int m, int bad_col, std::mt19937 &rng)
{
    MatrixXf B = MatrixXf::Random(n, m);
    if (bad_col >= 0 && bad_col < m && bad_col > 0) {
        B.col(bad_col) = B.col(bad_col - 1);  // collinear
    }
    MatrixXdC Bo = B.cast<double>();
    Bo.col(0) *= invnorm(Bo.col(0));
    for (int j = 1; j < Bo.cols(); ++j) {
        Bo.col(j) *= invnorm(Bo.col(j));
        for (int k = 0; k < j; ++k) {
            Bo.col(j) -= (Bo.col(k).transpose() * Bo.col(j)) * Bo.col(k);
            Bo.col(j) *= invnorm(Bo.col(j));
        }
    }
    (void)rng;
    return Bo.cast<float>();
}

ArrayXi random_mask(int m, int p, std::mt19937 &rng)
{
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::sort(idx.begin(), idx.begin() + p);
    return Map<ArrayXi>(idx.data(), p);
}

// Eigen reference of what mars::orthonormalize computes (single projection, no
// DGKS). Copied from tests/test_kernels.cc so the two test files stay decoupled.
MatrixXd eigen_reference(int n, int m, int p,
                         const MatrixXf  &B,
                         const ArrayXf   &x,
                         const ArrayXi   &mask,
                         const MatrixXfC &Bo,
                         double tol)
{
    MatrixXd Bx(n, p);
    for (int j = 0; j < p; ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }
    const MatrixXd Bo_d = Bo.leftCols(m).cast<double>();
    Bx -= Bo_d * (Bo_d.transpose() * Bx);
    const ArrayXd s = Bx.colwise().squaredNorm().array();
    Bx *= (s > tol).select(1 / (s + tol).sqrt(), 0).matrix().asDiagonal();
    return Bx;
}

// Run the CUDA orthonormalize: create a context, sync the basis, return Bx.
MatrixXf run_cuda(int n, int m, int p,
                  const MatrixXf &B, const ArrayXf &x,
                  const ArrayXi &mask, const MatrixXfC &Bo,
                  double tol, std::atomic<long> *counter = nullptr)
{
    mars::cuda::Context *ctx = mars::cuda::context_create(n, m, tol);
    mars::cuda::context_sync_basis(ctx, m,
                                   B.data(),  (size_t)B.outerStride(),
                                   Bo.data(), (size_t)Bo.outerStride());
    MatrixXf Bx(n, p);
    Bx.setZero();
    mars::cuda::orthonormalize(ctx, m, p, x.data(), mask.data(),
                               Bx.data(), (size_t)Bx.outerStride(),
                               /*ybx=*/nullptr, counter);
    mars::cuda::context_destroy(ctx);
    return Bx;
}

}  // namespace

/*
 *  Math correctness: GPU Bx is orthogonal to Bo, unit-norm columns, and matches
 *  the Eigen reference within the f32 floor. Mirrors KernelsTest.OrthonormalizeMatchesEigen.
 */
TEST(CudaKernelsTest, OrthonormalizeMatchesEigen)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 89;
    const int m = 13;
    const int p = 7;
    constexpr int BAD_COL = 2;
    constexpr double TOL = 1e-14;
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(0xC0FFEE);
    MatrixXfC Bo = make_orthonormal_basis(n, m, BAD_COL, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    B.col(BAD_COL) = B.col(BAD_COL - 1);  // also collinear in B
    ArrayXf   x  = ArrayXf::Random(n) * 10;
    ArrayXi   mask = random_mask(m, p, rng);

    MatrixXd Bxd = run_cuda(n, m, p, B, x, mask, Bo, TOL).cast<double>();
    MatrixXd Bod = Bo.cast<double>();

    ASSERT_TRUE((Bod.transpose() * Bxd).isZero(F32_TOL));
    ASSERT_TRUE(Bxd.colwise().norm().isOnes(F32_TOL));

    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo, TOL);
    ASSERT_TRUE(Bxd.isApprox(Bx_ref, F32_TOL));
}

/*
 *  DGKS fires on a near-collinear column (95%/5% energy split) and the retry
 *  still leaves Bx orthogonal and unit-norm. Mirrors
 *  KernelsTest.OrthonormalizeFiresDgksOnSevereCancellation.
 */
TEST(CudaKernelsTest, OrthonormalizeFiresDgksOnSevereCancellation)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 200;
    const int m = 4;
    const int p = 1;
    constexpr double TOL = 1e-14;
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(0xBADCAFE);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    VectorXd v_perp = VectorXd::Random(n);
    for (int k = 0; k < m; ++k) {
        v_perp -= (Bod.col(k).dot(v_perp)) * Bod.col(k);
    }
    v_perp.normalize();

    MatrixXf B(n, m);
    B.setZero();
    B.col(0) = (std::sqrt(0.95) * Bod.col(0) + std::sqrt(0.05) * v_perp).cast<float>();

    ArrayXf x = ArrayXf::Ones(n);  // x=1 so B*x just selects col 0
    ArrayXi mask(p);
    mask[0] = 0;

    std::atomic<long> counter{0};
    MatrixXd Bxd = run_cuda(n, m, p, B, x, mask, Bo, TOL, &counter).cast<double>();

    ASSERT_EQ(counter.load(), 1);
    ASSERT_TRUE((Bod.transpose() * Bxd).isZero(F32_TOL));
    ASSERT_TRUE(Bxd.colwise().norm().isOnes(F32_TOL));
}

/*
 *  DGKS does not fire on well-conditioned input. Mirrors
 *  KernelsTest.OrthonormalizeDoesNotFireDgksOnWellConditioned.
 */
TEST(CudaKernelsTest, OrthonormalizeDoesNotFireDgksOnWellConditioned)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 89;
    const int m = 4;
    const int p = 3;
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0xFEED);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    ArrayXf   x  = ArrayXf::Random(n);
    ArrayXi   mask = random_mask(m, p, rng);

    std::atomic<long> counter{0};
    run_cuda(n, m, p, B, x, mask, Bo, TOL, &counter);
    ASSERT_EQ(counter.load(), 0);
}

/*
 *  Win #1: the device-side ybx = Bxᵀ·y output (linear ΔSSE input). It must
 *  match dot_widen(Bx, y) computed on the host from the same Bx, so the
 *  per-eval Bx download can be dropped in linear_only mode.
 */
TEST(CudaKernelsTest, YbxMatchesBxTransposeY)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 4096;
    const int m = 32;
    const int p = 12;
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0x59BC);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    ArrayXf   x  = ArrayXf::Random(n) * 3;
    ArrayXi   mask = random_mask(m, p, rng);
    ArrayXf   y  = ArrayXf::Random(n);

    mars::cuda::Context *ctx = mars::cuda::context_create(n, m, TOL);
    mars::cuda::context_set_target(ctx, y.data());
    mars::cuda::context_sync_basis(ctx, m,
                                   B.data(),  (size_t)B.outerStride(),
                                   Bo.data(), (size_t)Bo.outerStride());
    MatrixXf Bx(n, p);
    Bx.setZero();
    std::vector<double> ybx(p, 0.0);
    mars::cuda::orthonormalize(ctx, m, p, x.data(), mask.data(),
                               Bx.data(), (size_t)Bx.outerStride(), ybx.data());
    mars::cuda::context_destroy(ctx);

    // Reference: f64 dot of the returned (normalized) Bx columns with y. ybx is
    // computed as scale·Σ(unnormalized Bx)·y (the scale is folded in f64 instead
    // of applied per-element in f32), so it matches this reference only to the
    // f32 storage floor -- it is in fact marginally *more* accurate than the
    // per-element-normalized dot. ‖y‖~√n, so the absolute floor is ~ε_f32·√n.
    MatrixXd Bxd = Bx.cast<double>();
    VectorXd yd  = y.cast<double>();
    const double floor = 1e-6 * yd.norm();
    for (int j = 0; j < p; ++j) {
        const double ref = Bxd.col(j).dot(yd);
        EXPECT_NEAR(ybx[j], ref, floor + 1e-6 * std::abs(ref));
    }
}

/*
 *  CPU vs CUDA parity at a larger size with strides > logical sizes and a mix of
 *  well-conditioned, near-collinear and degenerate columns -- the two paths
 *  must agree within the f32 floor. (n=8192, m=128, p=32 matches the CPU
 *  precision-stress baseline.)
 */
TEST(CudaKernelsTest, MatchesCpuKernel)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 8192;
    const int m = 128;
    const int p = 32;
    constexpr double TOL = 1e-14;
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(0xBAD5EED);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/3, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    B.col(7) = B.col(6);            // collinear in B -> DGKS pressure
    B.col(10).setZero();            // degenerate column
    ArrayXf   x  = ArrayXf::Random(n) * 4;
    ArrayXi   mask = random_mask(m, p, rng);

    // CPU reference.
    MatrixXf Bx_cpu(n, p);
    MatrixXd T(m, p);
    VectorXd s(p);
    Bx_cpu.setZero();
    T.setZero();
    std::atomic<long> cpu_counter{0};
    mars::orthonormalize(
        n, m, p,
        B.data(),    (size_t)B.outerStride(),
        x.data(),
        mask.data(),
        Bo.data(),   (size_t)Bo.outerStride(),
        Bx_cpu.data(), (size_t)Bx_cpu.outerStride(),
        T.data(),    (size_t)T.outerStride(),
        s.data(),
        TOL, &cpu_counter);

    std::atomic<long> cuda_counter{0};
    MatrixXf Bx_cuda = run_cuda(n, m, p, B, x, mask, Bo, TOL, &cuda_counter);

    MatrixXd Bod = Bo.cast<double>();
    MatrixXd Bxd = Bx_cuda.cast<double>();

    // GPU result is internally consistent (orthogonal + unit/zero norm)...
    ASSERT_TRUE((Bod.transpose() * Bxd).isZero(F32_TOL));
    // ...and agrees with the CPU kernel within the f32 floor.
    ASSERT_TRUE(Bxd.isApprox(Bx_cpu.cast<double>(), F32_TOL));
    // Both gate DGKS on the same near-collinear column.
    ASSERT_EQ(cuda_counter.load(), cpu_counter.load());
}

/*
 *  Batched orthonormalize (many X-columns stacked into one set of GEMMs) must
 *  produce the same ybx as the per-column orthonormalize for each (X-column,
 *  basis) pair. Both fold the scale and accumulate in f64, so they agree to the
 *  f64 GEMM-order floor. P (= n_x·m) stays within the default batch capacity --
 *  the host-side blocking across multiple calls is exercised by the Python
 *  end-to-end test (eval_batch), which owns the blocking.
 */
TEST(CudaKernelsTest, BatchMatchesPerColumn)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 4096;
    const int m = 24;
    const int n_x = 5;                 // X-columns
    const int P = n_x * m;             // all (xcol, basis) pairs
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0x6A7C);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    ArrayXf   y  = ArrayXf::Random(n);
    MatrixXf  X(n, n_x);
    X.setRandom();
    std::vector<float> s(n_x);
    for (int c = 0; c < n_x; ++c) {
        s[c] = 0.5f + 0.25f * (float)(c + 1);  // arbitrary positive scales
    }

    std::vector<int> src_xcol(P), src_basis(P);
    for (int j = 0, xc = 0; xc < n_x; ++xc) {
        for (int k = 0; k < m; ++k, ++j) {
            src_xcol[j]  = xc;
            src_basis[j] = k;
        }
    }

    mars::cuda::Context *ctx = mars::cuda::context_create(n, m, TOL);
    mars::cuda::context_set_target(ctx, y.data());
    mars::cuda::context_sync_basis(ctx, m,
                                   B.data(),  (size_t)B.outerStride(),
                                   Bo.data(), (size_t)Bo.outerStride());

    // Make all n_x scaled candidate columns resident, then batch (src_xcol are
    // global X-column indices into the resident candidates).
    std::vector<int> all_cols(n_x);
    std::iota(all_cols.begin(), all_cols.end(), 0);
    mars::cuda::context_sync_xcols(ctx, (size_t)n_x, X.data(),
                                   (size_t)X.outerStride(), s.data(),
                                   all_cols.data(), (size_t)n_x);

    std::vector<double> ybx_batch(P, 0.0);
    mars::cuda::orthonormalize_batch(ctx, m, P, src_xcol.data(), src_basis.data(),
                                     ybx_batch.data());

    // Per-column reference on the same context: scale the candidate (f32), then
    // orthonormalize the single column and read ybx.
    for (int j = 0; j < P; ++j) {
        const int xc = src_xcol[j];
        const int k  = src_basis[j];
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = X(i, xc) * s[xc];
        }
        double ref = 0.0;
        mars::cuda::orthonormalize(ctx, m, 1, x.data(), &k,
                                   /*Bx=*/nullptr, 0, &ref);
        EXPECT_NEAR(ybx_batch[j], ref, 1e-6 * (1.0 + std::abs(ref)));
    }
    mars::cuda::context_destroy(ctx);
}

/*
 *  Surgical DGKS retry through the BATCHED path. Stack many (X-column, basis)
 *  pairs where a controlled subset is near-dependent on Bo (fires DGKS) while the
 *  rest are well-conditioned (do not). The retry must gather ONLY the flagged
 *  columns, re-orthogonalize them in f64 at width `fired`, and scatter the refined
 *  ybx back to the correct global slots interleaved among the untouched ones. So
 *  the batched ybx must still match the per-column orthonormalize for every pair,
 *  and the same pairs must fire in both paths. This is the production path the
 *  random-Bo benches never exercised (their gate stays quiet), which is why the
 *  whole-block f64 retry cost went unnoticed.
 */
TEST(CudaKernelsTest, BatchDgksRetryMatchesPerColumn)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const int n = 2048;
    const int m = 8;
    const int n_x = 4;
    const int P = n_x * m;
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0xD6C5);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    // Even basis columns are near-dependent on Bo (98%/2% energy split -> DGKS
    // fires); odd ones are well-conditioned random (no fire). x is constant per
    // X-column, so the candidate B[:,k]*x preserves that span relationship and
    // firing is deterministic and interleaved across the P stacked columns.
    MatrixXf B(n, m);
    B.setZero();
    for (int k = 0; k < m; ++k) {
        if (k % 2 == 0) {
            VectorXd v_perp = VectorXd::Random(n);
            for (int t = 0; t < m; ++t) {
                v_perp -= (Bod.col(t).dot(v_perp)) * Bod.col(t);
            }
            v_perp.normalize();
            B.col(k) = (std::sqrt(0.98) * Bod.col(k) +
                        std::sqrt(0.02) * v_perp).cast<float>();
        }
        else {
            B.col(k) = VectorXf::Random(n);
        }
    }

    ArrayXf  y = ArrayXf::Random(n);
    MatrixXf X = MatrixXf::Ones(n, n_x);   // constant cols keep span alignment
    std::vector<float> s(n_x);
    for (int c = 0; c < n_x; ++c) {
        s[c] = 0.5f + 0.5f * (float)(c + 1);  // distinct positive scales
    }

    std::vector<int> src_xcol(P), src_basis(P);
    for (int j = 0, xc = 0; xc < n_x; ++xc) {
        for (int k = 0; k < m; ++k, ++j) {
            src_xcol[j]  = xc;
            src_basis[j] = k;
        }
    }

    mars::cuda::Context *ctx = mars::cuda::context_create(n, m, TOL);
    mars::cuda::context_set_target(ctx, y.data());
    mars::cuda::context_sync_basis(ctx, m,
                                   B.data(),  (size_t)B.outerStride(),
                                   Bo.data(), (size_t)Bo.outerStride());
    std::vector<int> all_cols(n_x);
    std::iota(all_cols.begin(), all_cols.end(), 0);
    mars::cuda::context_sync_xcols(ctx, (size_t)n_x, X.data(),
                                   (size_t)X.outerStride(), s.data(),
                                   all_cols.data(), (size_t)n_x);

    std::atomic<long> batch_counter{0};
    std::vector<double> ybx_batch(P, 0.0);
    mars::cuda::orthonormalize_batch(ctx, m, P, src_xcol.data(), src_basis.data(),
                                     ybx_batch.data(), &batch_counter);

    std::atomic<long> percol_counter{0};
    for (int j = 0; j < P; ++j) {
        const int xc = src_xcol[j];
        const int k  = src_basis[j];
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) {
            x[i] = X(i, xc) * s[xc];
        }
        double ref = 0.0;
        mars::cuda::orthonormalize(ctx, m, 1, x.data(), &k,
                                   /*Bx=*/nullptr, 0, &ref, &percol_counter);
        EXPECT_NEAR(ybx_batch[j], ref, 1e-6 * (1.0 + std::abs(ref)));
    }
    mars::cuda::context_destroy(ctx);

    // The retry actually fired (else this proves nothing), and the SAME pairs
    // fired in the batched and per-column paths (the gate is per-column, so the
    // surgical gather/scatter must not change which columns are refined).
    ASSERT_GT(batch_counter.load(), 0);
    ASSERT_EQ(batch_counter.load(), percol_counter.load());
}

// ---------------------------------------------------------------------------
// GPU hinge sweep vs the CPU MarsAlgo::eval() (the oracle). The basis is grown
// the way fit() grows it (see tests/bench_sweep.cc) so it is as well
// conditioned as a real fit's; MARS_CUDA_KCHUNK=0 selects the native-f64
// orthonormalize so the comparison isolates the sweep.
// ---------------------------------------------------------------------------
#include "marsalgo.h"
#include <cstdlib>
#include <string>

namespace {

struct SweepProblem {
    size_t n, px;
    std::vector<float> X, y, w;
};

// Random inputs with planted hinge structure. Column 0 is quantized (ties in
// the sort); column 1 is exactly zero on a block of rows (a zero-support tail).
SweepProblem make_sweep_problem(size_t n, size_t px, unsigned seed)
{
    SweepProblem P;
    P.n  = n;
    P.px = px;
    P.X.resize(n * px);
    P.y.resize(n);
    P.w.assign(n, 1.0f);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (auto &v : P.X) {
        v = gauss(rng);
    }
    for (size_t i = 0; i < n; ++i) {
        P.X[i] = std::round(P.X[i] * 4.0f) / 4.0f;      // ties
        if (i % 7 == 3) {
            P.X[n + i] = 0.0f;                          // exact zeros
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const float x0 = P.X[i], x1 = P.X[n + i], x2 = P.X[2 * n + i],
                    x3 = P.X[3 * n + i], x4 = P.X[4 * n + i];
        P.y[i] = std::max(x0 - 0.3f, 0.0f) - 0.8f * std::max(0.5f - x1, 0.0f)
                 + 0.6f * x2 * std::max(x3, 0.0f) + 0.4f * x4 + 0.3f * gauss(rng);
    }
    return P;
}

// Grow the basis to m columns like fit(): degree <= 3, no input repeated in a
// term, alternating linear terms and mirrored hinge pairs; a rejected
// (collinear) append is skipped.
void grow_basis(MarsAlgo &algo, size_t px, size_t m, std::mt19937 &rng)
{
    std::vector<std::vector<int>> inputs(1);
    std::vector<char> used(px * (m + 2), 0);
    std::uniform_real_distribution<float> cut(-1.0f, 1.0f);
    int step = 0;
    while ((size_t)algo.nbasis() < m) {
        int xcol, bcol;
        do {
            xcol = (int)(rng() % px);
            bcol = (int)(rng() % algo.nbasis());
        } while (inputs[bcol].size() >= 3 || used[xcol * (m + 2) + bcol] ||
                 std::find(inputs[bcol].begin(), inputs[bcol].end(), xcol) != inputs[bcol].end());
        used[xcol * (m + 2) + bcol] = 1;
        const bool  hinge = (step++ % 2) == 1;
        const float h     = cut(rng);
        for (char type : (hinge ? std::string("+-") : std::string("l"))) {
            if ((size_t)algo.nbasis() >= m) {
                break;
            }
            if (algo.append(type, xcol, bcol, h) < 0) {
                break;
            }
            inputs.push_back(inputs[bcol]);
            inputs.back().push_back(xcol);
        }
    }
}

// Three-way comparison over several X columns, grid spacings and masks:
//   ref : CPU orthonormalize + CPU sweep            (eval(cuda=false))
//   hyb : GPU orthonormalize + CPU sweep            (cuda=true, MARS_CUDA_SWEEP=0)
//   gpu : GPU orthonormalize + GPU sweep            (cuda=true)
// hyb and gpu sweep the SAME device Bx, so they must agree to f64
// reassociation error (tight). gpu vs ref additionally carries the GPU
// orthonormalize's f32 projection (Bx differs by ~1 f32 ulp), which the score
// amplifies by h2/den near the HINGE_DEN_REL gate -- so that comparison is
// loose and reported, not the correctness gate for the sweep.
void check_sweep(size_t n, size_t px, size_t m, unsigned seed, const char *block)
{
    setenv("MARS_CUDA_KCHUNK", "0", 1);
    if (block) {
        setenv("MARS_CUDA_SWEEP_BLOCK", block, 1);
    }
    else {
        unsetenv("MARS_CUDA_SWEEP_BLOCK");
    }

    SweepProblem P = make_sweep_problem(n, px, seed);
    MarsAlgo algo(P.X.data(), P.y.data(), P.w.data(), n, px, m + 2, n);
    std::mt19937 rng(seed + 1);
    grow_basis(algo, px, m, rng);
    ASSERT_EQ((size_t)algo.nbasis(), m);

    const int end_span = (int)(0.02 * n);
    std::vector<double> lin_r(m), hin_r(m), cut_r(m);
    std::vector<double> lin_h(m), hin_h(m), cut_h(m);
    std::vector<double> lin_g(m), hin_g(m), cut_g(m);
    int    n_found = 0, n_cut_hg = 0, n_cut_gr = 0, n_rel_gr_1e3 = 0;
    double max_rel_hg = 0.0, max_rel_gr = 0.0, max_abs_gr = 0.0;
    for (int xcol : {0, 1, 3, (int)px - 1}) {
        for (int min_span : {1, 7}) {
            for (int masked : {0, 1}) {
                std::vector<char> mask(m, 1);
                if (masked) {
                    for (size_t j = 0; j < m; j += 3) {
                        mask[j] = 0;
                    }
                }
                const bool *bm = (const bool *)mask.data();
                algo.eval(lin_r.data(), hin_r.data(), cut_r.data(), xcol, bm,
                          min_span, end_span, false, false);
                setenv("MARS_CUDA_SWEEP", "0", 1);
                algo.eval(lin_h.data(), hin_h.data(), cut_h.data(), xcol, bm,
                          min_span, end_span, false, true);
                setenv("MARS_CUDA_SWEEP", "1", 1);
                algo.eval(lin_g.data(), hin_g.data(), cut_g.data(), xcol, bm,
                          min_span, end_span, false, true);
                for (size_t j = 0; j < m; ++j) {
                    SCOPED_TRACE("xcol=" + std::to_string(xcol) + " min_span=" +
                                 std::to_string(min_span) + " masked=" + std::to_string(masked) +
                                 " basis=" + std::to_string(j));
                    if (!mask[j]) {
                        EXPECT_EQ(hin_g[j], 0.0);
                        EXPECT_TRUE(std::isnan(cut_g[j]));
                        continue;
                    }
                    // Same device Bx up to the f64-ULP run-to-run noise of the
                    // orthonormalize's atomic reductions (TODO.md): the linear
                    // term agrees to ULPs, not bit-exactly.
                    EXPECT_NEAR(lin_g[j], lin_h[j], 1e-15 + 1e-12 * std::abs(lin_h[j]));
                    // The sweep itself: f64 reassociation, amplified by h2/den
                    // near the gate (observed <= 7e-9 at m=260; a logic bug shows
                    // as >= 1e-4).
                    const double rel_hg = std::abs(hin_g[j] - hin_h[j]) /
                                          std::max(1e-300, std::abs(hin_h[j]));
                    max_rel_hg = std::max(max_rel_hg, rel_hg);
                    EXPECT_NEAR(hin_g[j], hin_h[j], 1e-12 + 1e-7 * std::abs(hin_h[j]));
                    EXPECT_EQ(std::isnan(cut_g[j]), std::isnan(cut_h[j]));
                    if (!std::isnan(cut_h[j]) && !std::isnan(cut_g[j])) {
                        ++n_found;
                        if (cut_g[j] != cut_h[j]) {
                            ++n_cut_hg;
                            // A different cut is only acceptable as a tie: the
                            // two scores must then agree to the same tolerance.
                            EXPECT_NEAR(hin_g[j], hin_h[j], 1e-12 + 1e-7 * std::abs(hin_h[j]))
                                    << "cut mismatch vs the CPU sweep on the same Bx: gpu "
                                    << cut_g[j] << " hyb " << cut_h[j];
                        }
                    }
                    // Against the all-CPU eval: loose (f32 projection in the GPU
                    // orthonormalize), with a gross-error bound.
                    EXPECT_NEAR(lin_g[j], lin_r[j], 1e-8 + 1e-6 * std::abs(lin_r[j]));
                    const double rel_gr = std::abs(hin_g[j] - hin_r[j]) /
                                          std::max(1e-300, std::abs(hin_r[j]));
                    max_rel_gr = std::max(max_rel_gr, rel_gr);
                    max_abs_gr = std::max(max_abs_gr, std::abs(hin_g[j] - hin_r[j]));
                    n_rel_gr_1e3 += (rel_gr > 1e-3);
                    EXPECT_NEAR(hin_g[j], hin_r[j], 1e-6 + 0.1 * std::abs(hin_r[j]));
                    if (!std::isnan(cut_r[j]) && !std::isnan(cut_g[j]) && cut_g[j] != cut_r[j]) {
                        ++n_cut_gr;
                    }
                }
            }
        }
    }
    EXPECT_GT(n_found, 0);
    std::printf("[ sweep A/B ] n=%zu m=%zu: gpu vs hyb (same Bx): max rel %.2e, %d/%d cuts differ;"
                " gpu vs cpu: max rel %.2e (abs %.2e), %d pairs > 1e-3, %d cuts differ\n",
                n, m, max_rel_hg, n_cut_hg, n_found, max_rel_gr, max_abs_gr,
                n_rel_gr_1e3, n_cut_gr);
}

}  // namespace

TEST(CudaSweepTest, MatchesCpuEval)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    check_sweep(30011, 12, 40, 0x5EED, nullptr);   // one lane per thread (CPT=1)
}

TEST(CudaSweepTest, MatchesCpuEvalTwoLanesPerThread)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    check_sweep(30011, 12, 40, 0x5EEE, "32");      // m+3 = 43 lanes on 32 threads (CPT=2)
}

TEST(CudaSweepTest, MatchesCpuEvalFourLanesPerThread)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    check_sweep(12007, 20, 100, 0x5EEF, "32");     // 103 lanes on 32 threads (CPT=4)
}

TEST(CudaSweepTest, MatchesCpuEvalLargeBasis)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    // 260 columns: m + 1 + 2*JB lanes exceed one 256-thread block, so the host
    // picks two lanes per thread on a block sized to the lane count.
    check_sweep(8009, 24, 260, 0x5EF1, nullptr);
}

TEST(CudaSweepTest, MatchesCpuEvalSmallN)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    // Fewer rows than one chunk (tests the single-chunk / short-tile paths).
    check_sweep(601, 8, 12, 0x5EF0, nullptr);
}

// ---------------------------------------------------------------------------
// Linear delta-SSE of the GPU paths against the f64 oracle, in the regime where
// the DGKS retry fires on most candidates (uncentered, correlated inputs, a
// large basis; tests/linear_problem.h). The f32 projection of the first pass
// leaves out-of-span noise ~eps_f32·sqrt(m)·||T|| on a cancelling column, so a
// retry that restarted from that residual was off by up to tens of percent at
// m=400 (numpy model); the retry now restarts from the f64 B·x. Runs with the
// default blocked-f32 T GEMM (MARS_CUDA_KCHUNK unset) -- the production path.
// ---------------------------------------------------------------------------
#include "linear_problem.h"

TEST(CudaKernelsTest, LinearDsseMatchesOracleCollinearLargeBasis)
{
    if (!cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    unsetenv("MARS_CUDA_KCHUNK");
    const size_t N = 6000, PX = 6, M = 400;
    linear_problem::Problem P = linear_problem::make(N, PX, 3.0f, 0xD6C6);
    MarsAlgo algo(P.X.data(), P.y.data(), P.w.data(), N, PX, M + 2, N);
    linear_problem::grow(algo, P, M, 0xD6C7);
    ASSERT_EQ((size_t)algo.nbasis(), M);

    std::vector<double> ref, kappa;
    linear_problem::oracle(algo, P, ref, kappa);

    std::vector<double> cpu(PX * M), gpu(PX * M), hinge(PX * M), cut(PX * M);
    std::vector<char> mask(PX * M, 1);
    for (size_t c = 0; c < PX; ++c) {
        algo.eval(cpu.data() + c * M, hinge.data(), cut.data(), (int)c,
                  (const bool *)mask.data(), 1, 0, /*linear_only=*/true, /*cuda=*/false);
        algo.eval(gpu.data() + c * M, hinge.data(), cut.data(), (int)c,
                  (const bool *)mask.data(), 1, 0, /*linear_only=*/true, /*cuda=*/true);
    }
    std::vector<double> batch(PX * M);
    algo.eval_batch(batch.data(), hinge.data(), cut.data(),
                    (const bool *)mask.data(), (int)PX, (int)M);

    const char *names[] = {"cpu", "gpu eval", "gpu eval_batch"};
    const std::vector<double> *ests[] = {&cpu, &gpu, &batch};
    for (int t = 0; t < 3; ++t) {
        SCOPED_TRACE(names[t]);
        const auto S = linear_problem::compare(*ests[t], ref, kappa, 1e-10, 1e4);
        std::printf("[ linear dsse ] n=%zu m=%zu %-15s: %d candidates, rel err median %.1e max %.1e\n",
                    N, M, names[t], S.count, S.med_rel, S.max_rel);
        ASSERT_GT(S.count, (int)(PX * M / 4));
        EXPECT_LT(S.med_rel, 3e-7);
        EXPECT_LT(S.max_rel, 5e-4);
    }
}
