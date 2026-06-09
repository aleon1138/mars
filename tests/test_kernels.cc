#include "../kernels.h"
#include <Eigen/Dense>
#include <atomic>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdC;
typedef Matrix<float,  Dynamic, Dynamic, RowMajor> MatrixXfC;

namespace {

double invnorm(VectorXd x)
{
    const double s = x.norm();
    return s > 1e-14 ? 1.0 / s : 0.0;
}

/*
 *  Build a strictly orthonormal Bo from a random matrix using modified
 *  Gram-Schmidt. Column `bad_col` is forced to be collinear with the
 *  previous column so it gets zeroed out -- exercises the degenerate path.
 *
 *  Internally builds in f64 for clean orthonormality, then casts to f32 --
 *  matching what MarsData does after `append()` Gram-Schmidt (f64 arith,
 *  f32 store).
 */
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

/*
 *  Eigen reference of what mars::orthonormalize computes -- used as ground
 *  truth in the test. Bo is f32 (matches the kernel input); the reference
 *  matmul casts to f64 lazily so the projection arithmetic is f64.
 */
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

/*
 *  Eigen MGS reference mirroring mars::orthonormalize_col (and MarsAlgo::append):
 *  modified Gram-Schmidt against the f32 Bo (upcast to f64), a single DGKS retry,
 *  then v/||v|| (no +tol) when above the degeneracy floor. Reports the
 *  pre-normalization length in w_out.
 */
VectorXd orthonormalize_col_reference(VectorXd v, const MatrixXfC &Bo, int m,
                                      double tol, double &w_out)
{
    const MatrixXd Bod = Bo.leftCols(m).cast<double>();
    double proj_norm2 = 0.0;
    for (int j = 0; j < m; ++j) {
        const double c = Bod.col(j).dot(v);
        v -= c * Bod.col(j);
        proj_norm2 += c * c;
    }
    double v_norm2 = v.squaredNorm();
    if (v_norm2 > tol && v_norm2 * mars::DGKS_GATE_RATIO_SQ < proj_norm2) {
        for (int j = 0; j < m; ++j) {
            const double c = Bod.col(j).dot(v);
            v -= c * Bod.col(j);
        }
        v_norm2 = v.squaredNorm();
    }
    const double w = std::sqrt(v_norm2);
    if (w * w > tol) v /= w;
    w_out = w;
    return v;
}

} // namespace

/*
 *  Math correctness: the kernel produces a Bx orthogonal to Bo, with unit
 *  column norms (or zero for degenerate columns), and matches an Eigen
 *  reference implementation within FP tolerance.
 */
TEST(KernelsTest, OrthonormalizeMatchesEigen)
{
    const int n = 89;
    const int m = 13;
    const int p = 7;
    constexpr int BAD_COL = 2;
    constexpr double TOL = 1e-14;
    /* f32 storage of Bx: orth/norm/vs-ref floor is ~eps_f32 ~ 1.2e-7. */
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(0xC0FFEE);
    MatrixXfC Bo = make_orthonormal_basis(n, m, BAD_COL, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    B.col(BAD_COL) = B.col(BAD_COL - 1);  // also collinear in B
    ArrayXf   x  = ArrayXf::Random(n) * 10;
    ArrayXi   mask = random_mask(m, p, rng);

    /*
     *  Sanity: the manually built Bo is orthonormal modulo the zeroed col and
     *  f32 storage round (relaxed from f64-tight TOL since Bo is now f32).
     */
    MatrixXd I = MatrixXd::Identity(m, m);
    I(BAD_COL, BAD_COL) = 0;
    MatrixXd Bod = Bo.cast<double>();
    ASSERT_TRUE((Bod.transpose() * Bod).isApprox(I, F32_TOL));

    /* Kernel output (Bx stored as f32) */
    MatrixXf Bx(n, p);
    MatrixXd T(m, p);
    VectorXd s(p);
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
        s.data(),
        TOL);

    MatrixXd Bxd = Bx.cast<double>();

    /* Bx is orthogonal to Bo */
    ASSERT_TRUE((Bod.transpose() * Bxd).isZero(F32_TOL));

    /* Columns of Bx are unit-norm */
    ASSERT_TRUE(Bxd.colwise().norm().isOnes(F32_TOL));

    /* Matches a straight Eigen implementation of the same math */
    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo, TOL);
    ASSERT_TRUE(Bxd.isApprox(Bx_ref, F32_TOL));
}

/*
 *  Stride-tolerance: ldBo / ldBx / ldT can be larger than the logical sizes
 *  (i.e. the matrices live inside larger backing storage). This mirrors what
 *  happens in MarsAlgo where scratch buffers are sized for max_terms but we
 *  only use the leading m×p block.
 */
TEST(KernelsTest, OrthonormalizeRespectsLeadingDims)
{
    const int n = 64;
    const int m = 8;
    const int p = 5;
    const int max_terms = 16;  // > m, > p — exercises non-trivial strides
    constexpr double TOL = 1e-14;
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(42);
    MatrixXdC Bo_full_d(n, max_terms);
    Bo_full_d.setRandom();
    {  // orthonormalize the leading m cols of Bo_full in place (f64 for
       // construction precision, then cast to f32 for the kernel call).
        Bo_full_d.col(0) *= invnorm(Bo_full_d.col(0));
        for (int j = 1; j < m; ++j) {
            Bo_full_d.col(j) *= invnorm(Bo_full_d.col(j));
            for (int k = 0; k < j; ++k) {
                Bo_full_d.col(j) -= (Bo_full_d.col(k).transpose() * Bo_full_d.col(j)) * Bo_full_d.col(k);
                Bo_full_d.col(j) *= invnorm(Bo_full_d.col(j));
            }
        }
    }
    MatrixXfC Bo_full = Bo_full_d.cast<float>();

    MatrixXf B(n, m);
    B.setRandom();
    ArrayXf  x = ArrayXf::Random(n);
    ArrayXi  mask = random_mask(m, p, rng);

    /* Workspaces with extra padding to confirm strides are honored. */
    MatrixXf Bx_full(n, max_terms);
    MatrixXd T_full (max_terms, max_terms);
    VectorXd s(max_terms);
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
        s.data(),
        TOL);

    MatrixXd Bx = Bx_full.leftCols(p).cast<double>();
    MatrixXfC Bo_used = Bo_full.leftCols(m);
    MatrixXd  Bo_used_d = Bo_used.cast<double>();

    ASSERT_TRUE((Bo_used_d.transpose() * Bx).isZero(F32_TOL));
    ASSERT_TRUE(Bx.colwise().norm().isOnes(F32_TOL));

    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo_used, TOL);
    ASSERT_TRUE(Bx.isApprox(Bx_ref, F32_TOL));

    /* The unused padding columns of Bx_full must not have been touched. */
    ASSERT_TRUE(Bx_full.rightCols(max_terms - p).isZero(0.0f));
}

/*
 *  DGKS gate: when the candidate column has most of its energy in span(Bo)
 *  (residual energy < 11% of projection energy), the kernel re-orthogonalizes
 *  once and increments the counter. The retry must produce a result still
 *  orthogonal to Bo and unit-norm.
 *
 *  Construction: blend Bo[:,0] with a unit vector orthogonal to span(Bo) at
 *  a 95%/5% energy split. That sits well past the s*9 < t_norm2 trigger
 *  (0.05*9=0.45 << 0.95) but leaves enough residual that normalization stays
 *  well above the degeneracy floor (tol = 1e-14).
 */
TEST(KernelsTest, OrthonormalizeFiresDgksOnSevereCancellation)
{
    const int n = 200;
    const int m = 4;
    const int p = 1;
    constexpr double TOL = 1e-14;
    constexpr double F32_TOL = 1e-5;

    std::mt19937 rng(0xBADCAFE);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    /*
     *  Build a unit vector orthogonal to span(Bo) by projecting random noise
     *  out of Bo. This becomes the 5% "true residual" of the candidate column.
     */
    VectorXd v_perp = VectorXd::Random(n);
    for (int k = 0; k < m; ++k) {
        v_perp -= (Bod.col(k).dot(v_perp)) * Bod.col(k);
    }
    v_perp.normalize();

    /* 95%/5% energy split -- well inside the DGKS trigger region. */
    MatrixXf B(n, m);
    B.setZero();
    B.col(0) = (std::sqrt(0.95) * Bod.col(0) + std::sqrt(0.05) * v_perp).cast<float>();

    ArrayXf x = ArrayXf::Ones(n);                // x=1 so B*x just selects col 0
    ArrayXi mask(p); mask[0] = 0;

    MatrixXf Bx(n, p);
    MatrixXd T(m, p);
    VectorXd s(p);
    Bx.setZero();
    T.setZero();

    std::atomic<long> counter{0};
    mars::orthonormalize(
        n, m, p,
        B.data(),    (int)B.outerStride(),
        x.data(),
        mask.data(),
        Bo.data(),   (int)Bo.outerStride(),
        Bx.data(),   (int)Bx.outerStride(),
        T.data(),    (int)T.outerStride(),
        s.data(),
        TOL,
        &counter);

    MatrixXd Bxd = Bx.cast<double>();

    /* The DGKS branch must have fired exactly once (single column). */
    ASSERT_EQ(counter.load(), 1);

    /* Post-DGKS Bx is still orthogonal to Bo and unit-norm. */
    ASSERT_TRUE((Bod.transpose() * Bxd).isZero(F32_TOL));
    ASSERT_TRUE(Bxd.colwise().norm().isOnes(F32_TOL));
}

/*
 *  Counter is left untouched when the input is well-conditioned (column has
 *  most of its energy orthogonal to Bo).
 */
TEST(KernelsTest, OrthonormalizeDoesNotFireDgksOnWellConditioned)
{
    const int n = 89;
    const int m = 4;
    const int p = 3;
    constexpr double TOL = 1e-14;

    std::mt19937 rng(0xFEED);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXf  B  = MatrixXf::Random(n, m);
    ArrayXf   x  = ArrayXf::Random(n);
    ArrayXi   mask = random_mask(m, p, rng);

    MatrixXf Bx(n, p);
    MatrixXd T(m, p);
    VectorXd s(p);
    Bx.setZero();
    T.setZero();

    std::atomic<long> counter{0};
    mars::orthonormalize(
        n, m, p,
        B.data(),    (int)B.outerStride(),
        x.data(),
        mask.data(),
        Bo.data(),   (int)Bo.outerStride(),
        Bx.data(),   (int)Bx.outerStride(),
        T.data(),    (int)T.outerStride(),
        s.data(),
        TOL,
        &counter);

    ASSERT_EQ(counter.load(), 0);
}

/*
 *  Precision regression. Locks in the floor of orthogonality and unit-norm
 *  accuracy under a stress configuration that mixes:
 *    - well-conditioned columns (typical case)
 *    - a DGKS-triggering near-collinear column (exercises the retry path)
 *    - an exactly-degenerate column (must zero out via the s>tol gate)
 *    - mixed-magnitude regressors in B (cancellation pressure in projection)
 *
 *  Why this exists: the precision floor of the f32 Bo + f32 Bx kernel is set
 *  by the per-row store rounding plus the f32-rounding of Bo itself. This
 *  test locks in a measurable baseline so future narrowing/widening changes
 *  (e.g. further reduction of d, or hoisting tc into f32) are detectable
 *  rather than eyeballed.
 */
TEST(KernelsTest, OrthonormalizePrecisionStressBaseline)
{
    const int n = 8192;
    const int m = 128;
    const int p = 32;
    constexpr double KERNEL_TOL = 1e-14;

    /*
     *  ---- tolerances for f32 Bx storage with f64 arithmetic ----
     *
     *  Pre-narrowing baseline (f64 storage) was:
     *    orth   ~ 2e-16,   norm ~ 1e-15 typical / 3.4e-13 on DGKS col,
     *    vs-ref ~ 0 (bit-identical to one-pass CGS).
     *
     *  After narrowing Bx -> f32 (arithmetic stays f64; only storage
     *  narrows), the floor is dominated by the f32 store-round at each
     *  row of Bx. For unit-norm columns:
     *    ||Bo^T * Bx_f32||_max ~ eps_f32 ~ 1.2e-7
     *    ||Bx_f32[:,j]||^2 - 1 ~ 2 * eps_f32 ~ 2.4e-7
     *    Bx_f32 - Bx_eigen_ref (one-pass CGS, f64 reference) ~ eps_f32
     *
     *  The bounds below sit ~10x above those floors -- catch a 10x
     *  regression while tolerating per-column FMA noise.
     */
    constexpr double ORTH_TOL    = 1e-6;   // ||Bo^T * Bx||_max
    constexpr double NORM_TOL    = 1e-5;   // ||Bx[:,j]||^2 deviates from 1
    constexpr double VS_REF_TOL  = 1e-5;   // vs Eigen one-pass GS, well-conditioned cols only

    std::mt19937 rng(0xBAD5EED);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    /*
     *  B with mixed magnitudes (factor up to 16 across columns) to apply
     *  cancellation pressure when columns of different scales are projected.
     */
    MatrixXf B = MatrixXf::Random(n, m);
    for (int j = 0; j < m; ++j) {
        B.col(j) *= std::pow(2.0f, j % 5);
    }

    /* Identify two "stress" columns that we will retarget for specific paths. */
    const int DGKS_COL  = 3;
    const int DEGEN_COL = 7;

    /*
     *  Near-collinear column: B[:,DGKS_COL] * 1 lands ~97% in span(Bo) and ~3%
     *  outside. With KERNEL_TOL = 1e-14 the column normalizes to a unit vector,
     *  so the DGKS gate (ratio 9) fires on the projection energy split.
     */
    VectorXd v_perp = VectorXd::Random(n);
    for (int k = 0; k < m; ++k) {
        v_perp -= (Bod.col(k).dot(v_perp)) * Bod.col(k);
    }
    v_perp.normalize();
    B.col(DGKS_COL) = (std::sqrt(0.97) * Bod.col(0)
                       + std::sqrt(0.03) * v_perp).cast<float>();

    /* Exactly-degenerate column: B[:,DEGEN_COL] * 1 sits entirely in span(Bo). */
    B.col(DEGEN_COL) = Bo.col(1);

    ArrayXf x = ArrayXf::Ones(n);  // x = 1 so B*x is just the column.

    /*
     *  Build mask: ensure both stress columns are present, fill the rest at
     *  random from the remaining indices.
     */
    std::vector<int> idx{DGKS_COL, DEGEN_COL};
    {
        std::vector<int> rest;
        for (int j = 0; j < m; ++j) {
            if (j != DGKS_COL && j != DEGEN_COL) rest.push_back(j);
        }
        std::shuffle(rest.begin(), rest.end(), rng);
        idx.insert(idx.end(), rest.begin(), rest.begin() + (p - 2));
    }
    std::sort(idx.begin(), idx.end());
    ArrayXi mask = Map<ArrayXi>(idx.data(), p);
    int dgks_j = -1, degen_j = -1;
    for (int j = 0; j < p; ++j) {
        if (mask[j] == DGKS_COL)  dgks_j  = j;
        if (mask[j] == DEGEN_COL) degen_j = j;
    }
    ASSERT_GE(dgks_j,  0);
    ASSERT_GE(degen_j, 0);

    /* Run the kernel. */
    MatrixXf Bx(n, p); Bx.setZero();
    MatrixXd T(m, p);  T.setZero();
    VectorXd s(p);
    std::atomic<long> dgks_counter{0};
    mars::orthonormalize(
        n, m, p,
        B.data(),  (int)B.outerStride(),
        x.data(),
        mask.data(),
        Bo.data(), (int)Bo.outerStride(),
        Bx.data(), (int)Bx.outerStride(),
        T.data(),  (int)T.outerStride(),
        s.data(),
        KERNEL_TOL,
        &dgks_counter);

    MatrixXd Bxd = Bx.cast<double>();

    /* Confirm the construction actually exercised DGKS at least once. */
    ASSERT_GE(dgks_counter.load(), 1) << "DGKS retry did not fire on stress column";

    /*
     *  (1) Orthogonality of Bx against Bo holds across all columns including
     *      the DGKS-corrected one. Degenerate column is zeroed and therefore
     *      trivially orthogonal, so it counts here too.
     */
    const double orth = (Bod.transpose() * Bxd).cwiseAbs().maxCoeff();
    EXPECT_LT(orth, ORTH_TOL) << "orth=" << orth;

    /* (2) Non-degenerate columns are unit-norm. */
    for (int j = 0; j < p; ++j) {
        if (j == degen_j) continue;
        const double norm_err = std::abs(Bxd.col(j).squaredNorm() - 1.0);
        EXPECT_LT(norm_err, NORM_TOL)
            << "col " << j << " (mask=" << mask[j] << ") norm_err=" << norm_err;
    }

    /*
     *  (3) Degenerate column is exactly zero. This is the contract for columns
     *      whose squared norm falls below tol — the kernel uses scale=0.
     */
    EXPECT_EQ(Bx.col(degen_j).squaredNorm(), 0.0f)
        << "degenerate column was not zeroed";

    /*
     *  (4) Well-conditioned columns agree with a one-pass Eigen reference. We
     *      skip the DGKS column (the reference does not retry, so it diverges
     *      by O(residual / sqrt(perp_fraction)) on that column) and the
     *      degenerate one (the reference normalizes a near-zero vector).
     */
    MatrixXd Bx_ref = eigen_reference(n, m, p, B, x, mask, Bo, KERNEL_TOL);
    for (int j = 0; j < p; ++j) {
        if (j == dgks_j || j == degen_j) continue;
        const double diff = (Bxd.col(j) - Bx_ref.col(j)).cwiseAbs().maxCoeff();
        EXPECT_LT(diff, VS_REF_TOL)
            << "col " << j << " (mask=" << mask[j] << ") vs-ref=" << diff;
    }
}

/*
 *  dot_widen: f32 inputs, f64 accumulation. The result must match an f64
 *  reference (the same f32 values promoted to double) to ~machine precision --
 *  far past the ~1e-3 relative error a naive f32 accumulation would hit on a
 *  long vector. This is the regression guard for the "dot products accumulate
 *  in f64" contract that backs narrowing y / Bx / Bo to f32 storage. Sizes
 *  straddle the 8-wide AVX block to exercise the scalar tail and the n==0 edge.
 */
TEST(KernelsTest, DotWidenAccumulatesInF64)
{
    std::mt19937 rng(0xD07);
    /*
     *  Positive range -> the sum is well-conditioned (cond ~ 1), so a tight
     *  tolerance still cleanly separates true f64 accumulation from f32.
     */
    std::uniform_real_distribution<float> uni(0.25f, 1.25f);

    for (int n : {0, 1, 7, 8, 9, 31, 1024, 50000}) {
        ArrayXf a(n), b(n);
        for (int i = 0; i < n; ++i) { a[i] = uni(rng); b[i] = uni(rng); }

        const double got = mars::dot_widen(a.data(), b.data(), n);

        /* Ground truth: a genuine f64 dot of the promoted f32 values. */
        const VectorXd ad = a.cast<double>();
        const VectorXd bd = b.cast<double>();
        const double   ref = ad.dot(bd);

        /*
         *  Only the summation order differs from the reference (both f64), so
         *  agreement is at the f64 floor. An f32 accumulation would miss by
         *  ~n*eps_f32 (orders of magnitude past this bound) at n=50000.
         */
        const double tol = 1e-9 * (1.0 + std::abs(ref));
        EXPECT_NEAR(got, ref, tol) << "n=" << n;
    }
}

/*
 *  orthonormalize_col: the single-column GS used by append(). A well-conditioned
 *  column must match the Eigen MGS reference, be orthogonal to Bo and unit-norm,
 *  and must NOT trip the DGKS retry. v is f64; Bo is f32, so orthogonality is
 *  floored at ~eps_f32 while the vs-reference agreement is at the f64 floor.
 */
TEST(KernelsTest, OrthonormalizeColMatchesEigen)
{
    const int n = 89;
    const int m = 13;
    constexpr double TOL      = 1e-14;
    constexpr double ORTH_TOL = 1e-6;   // limited by Bo's f32 storage
    constexpr double REF_TOL  = 1e-9;   // f64 vs f64, only summation order differs

    std::mt19937 rng(0x0C0FFEE);
    MatrixXfC Bo  = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    VectorXd v0 = VectorXd::Random(n) * 3.0;  // mostly outside span(Bo)

    VectorXd v = v0;
    std::atomic<long> counter{0};
    const double w = mars::orthonormalize_col(
        n, m, v.data(), Bo.data(), (int)Bo.outerStride(), TOL, &counter);

    double w_ref = 0.0;
    VectorXd v_ref = orthonormalize_col_reference(v0, Bo, m, TOL, w_ref);

    EXPECT_NEAR(w, w_ref, REF_TOL * (1.0 + std::abs(w_ref)));
    EXPECT_TRUE(v.isApprox(v_ref, REF_TOL));
    EXPECT_LT((Bod.transpose() * v).cwiseAbs().maxCoeff(), ORTH_TOL);
    EXPECT_NEAR(v.norm(), 1.0, REF_TOL);
    EXPECT_EQ(counter.load(), 0);  // well-conditioned: no retry
}

/*
 *  DGKS retry: a column with 95% of its energy in span(Bo) trips the gate once;
 *  the result is still orthogonal to Bo and unit-norm.
 */
TEST(KernelsTest, OrthonormalizeColFiresDgksOnSevereCancellation)
{
    const int n = 200;
    const int m = 4;
    constexpr double TOL      = 1e-14;
    constexpr double ORTH_TOL = 1e-6;

    std::mt19937 rng(0xBADF00D);
    MatrixXfC Bo  = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);
    MatrixXd  Bod = Bo.cast<double>();

    VectorXd v_perp = VectorXd::Random(n);
    for (int k = 0; k < m; ++k) v_perp -= Bod.col(k).dot(v_perp) * Bod.col(k);
    v_perp.normalize();

    /* 95% in span(Bo), 5% outside -> residual*9 = 0.45 < projected 0.95. */
    VectorXd v = std::sqrt(0.95) * Bod.col(0) + std::sqrt(0.05) * v_perp;

    std::atomic<long> counter{0};
    const double w = mars::orthonormalize_col(
        n, m, v.data(), Bo.data(), (int)Bo.outerStride(), TOL, &counter);

    EXPECT_EQ(counter.load(), 1);
    EXPECT_GT(w, 0.0);
    EXPECT_LT((Bod.transpose() * v).cwiseAbs().maxCoeff(), ORTH_TOL);
    EXPECT_NEAR(v.norm(), 1.0, ORTH_TOL);
}

/*
 *  Degenerate column (lies in span(Bo)): the residual collapses below the
 *  degeneracy floor, so the kernel returns a ~0 length and leaves v
 *  un-normalized (the caller rejects on w*w <= tol).
 */
TEST(KernelsTest, OrthonormalizeColLeavesDegenerateUnnormalized)
{
    const int n = 64;
    const int m = 5;
    constexpr double TOL = 1e-6;  // well above the f32 residual floor

    std::mt19937 rng(0xDEADBEEF);
    MatrixXfC Bo = make_orthonormal_basis(n, m, /*bad_col=*/-1, rng);

    VectorXd v = Bo.col(1).cast<double>();  // entirely in span(Bo)

    std::atomic<long> counter{0};
    const double w = mars::orthonormalize_col(
        n, m, v.data(), Bo.data(), (int)Bo.outerStride(), TOL, &counter);

    EXPECT_LE(w * w, TOL);        // below the degeneracy floor
    EXPECT_LT(v.norm(), 1e-3);    // left un-normalized (still the tiny residual)
    EXPECT_EQ(counter.load(), 0); // gate needs v_norm2 > tol, so no retry
}
