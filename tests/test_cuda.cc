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
                               Bx.data(), (size_t)Bx.outerStride(), counter);
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
