#include "../kernels.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdC;

namespace {

double invnorm(VectorXd x)
{
    const double s = x.norm();
    return s > 1e-14 ? 1.0 / s : 0.0;
}

// Build a strictly orthonormal Bo from a random matrix using
// modified Gram-Schmidt. Column `bad_col` is forced to be collinear with the
// previous column so it gets zeroed out -- exercises the degenerate path.
MatrixXdC make_orthonormal_basis(int n, int m, int bad_col, std::mt19937 &rng)
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
    return Bo;
}

ArrayXi random_mask(int m, int p, std::mt19937 &rng)
{
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    std::sort(idx.begin(), idx.begin() + p);
    return Map<ArrayXi>(idx.data(), p);
}

// Eigen reference of what mars::orthonormalize computes -- used as ground
// truth in the test.
MatrixXd eigen_reference(int n, int m, int p,
                         const MatrixXf  &B,
                         const ArrayXf   &x,
                         const ArrayXi   &mask,
                         const MatrixXdC &Bo,
                         double tol)
{
    MatrixXd Bx(n, p);
    for (int j = 0; j < p; ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }
    Bx -= Bo.leftCols(m) * (Bo.leftCols(m).transpose() * Bx);
    const ArrayXd s = Bx.colwise().squaredNorm().array();
    Bx *= (s > tol).select(1 / (s + tol).sqrt(), 0).matrix().asDiagonal();
    return Bx;
}

} // namespace

// ---------------------------------------------------------------------------
// Math correctness: the kernel produces a Bx orthogonal to Bo, with unit
// column norms (or zero for degenerate columns), and matches an Eigen
// reference implementation within FP tolerance.
// ---------------------------------------------------------------------------
TEST(KernelsTest, OrthonormalizeMatchesEigen)
{
    const int n = 89;
    const int m = 13;
    const int p = 7;
    constexpr int BAD_COL = 2;
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0xC0FFEE);
    MatrixXdC Bo = make_orthonormal_basis(n, m, BAD_COL, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    B.col(BAD_COL) = B.col(BAD_COL - 1);  // also collinear in B
    ArrayXf   x  = ArrayXf::Random(n) * 10;
    ArrayXi   mask = random_mask(m, p, rng);

    // Sanity: the manually built Bo is actually orthonormal (modulo the zeroed col).
    MatrixXd I = MatrixXd::Identity(m, m);
    I(BAD_COL, BAD_COL) = 0;
    ASSERT_TRUE((Bo.transpose() * Bo).isApprox(I, TOL));

    // Kernel output
    MatrixXd Bx(n, p);
    MatrixXd T(m, p);
    Bx.setZero();
    T.setZero();
    mars::orthonormalize(
        n, m, p,
        B.data(),    (int)B.outerStride(),
        x.data(),
        mask.data(),
        Bo.data(),   (int)Bo.outerStride(),
        Bx.data(),   (int)Bx.outerStride(),
        T.data(),    (int)T.outerStride(),
        TOL);

    // Bx is orthogonal to Bo
    ASSERT_TRUE((Bo.transpose() * Bx).isZero(1e-12));

    // Columns of Bx are unit-norm
    ASSERT_TRUE(Bx.colwise().norm().isOnes(1e-12));

    // Matches a straight Eigen implementation of the same math
    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo, TOL);
    ASSERT_TRUE(Bx.isApprox(Bx_ref, 1e-12));
}

// ---------------------------------------------------------------------------
// Stride-tolerance: ldBo / ldBx / ldT can be larger than the logical sizes
// (i.e. the matrices live inside larger backing storage). This mirrors what
// happens in MarsAlgo where scratch buffers are sized for max_terms but we
// only use the leading m×p block.
// ---------------------------------------------------------------------------
TEST(KernelsTest, OrthonormalizeRespectsLeadingDims)
{
    const int n = 64;
    const int m = 8;
    const int p = 5;
    const int max_terms = 16;  // > m, > p — exercises non-trivial strides
    constexpr double TOL = 1e-14;

    std::mt19937 rng(42);
    MatrixXdC Bo_full(n, max_terms);
    Bo_full.setRandom();
    {  // orthonormalize the leading m cols of Bo_full in place
        Bo_full.col(0) *= invnorm(Bo_full.col(0));
        for (int j = 1; j < m; ++j) {
            Bo_full.col(j) *= invnorm(Bo_full.col(j));
            for (int k = 0; k < j; ++k) {
                Bo_full.col(j) -= (Bo_full.col(k).transpose() * Bo_full.col(j)) * Bo_full.col(k);
                Bo_full.col(j) *= invnorm(Bo_full.col(j));
            }
        }
    }

    MatrixXf B(n, m);
    B.setRandom();
    ArrayXf  x = ArrayXf::Random(n);
    ArrayXi  mask = random_mask(m, p, rng);

    // Workspaces with extra padding to confirm strides are honored.
    MatrixXd Bx_full(n, max_terms);
    MatrixXd T_full (max_terms, max_terms);
    Bx_full.setZero();
    T_full.setZero();

    mars::orthonormalize(
        n, m, p,
        B.data(),       (int)B.outerStride(),
        x.data(),
        mask.data(),
        Bo_full.data(), (int)Bo_full.outerStride(),    // = max_terms (row-major)
        Bx_full.data(), (int)Bx_full.outerStride(),    // = n         (col-major)
        T_full.data(),  (int)T_full.outerStride(),     // = max_terms (col-major)
        TOL);

    MatrixXd Bx = Bx_full.leftCols(p);
    MatrixXdC Bo_used = Bo_full.leftCols(m);

    ASSERT_TRUE((Bo_used.transpose() * Bx).isZero(1e-12));
    ASSERT_TRUE(Bx.colwise().norm().isOnes(1e-12));

    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo_used, TOL);
    ASSERT_TRUE(Bx.isApprox(Bx_ref, 1e-12));

    // The unused padding columns of Bx_full must not have been touched.
    ASSERT_TRUE(Bx_full.rightCols(max_terms - p).isZero(0.0));
}
