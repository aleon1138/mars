#include "../marsalgo.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <thread>
#include <cmath>
constexpr double EPS = 1e-14;

// Hinge-sweep delta-SSE goes through covariates_impl(). With AVX the f/g
// reduction accumulates 4-wide via FMA; without it (e.g. arm64) the scalar
// fallback sums in a different order, drifting the ill-conditioned
// (small-denominator) cuts by a few e-6 on dsse/N -- reduction order, not a
// real accuracy loss. Hold the AVX path to the tight f32-storage floor; relax
// only the scalar path. Gated on the same macro covariates_impl() selects on.
#if defined(__AVX__)
constexpr double HINGE_DSSE_TOL = 1e-6;
#else
constexpr double HINGE_DSSE_TOL = 1e-5;
#endif


using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXdC;
typedef Matrix<float, Dynamic,Dynamic,RowMajor> MatrixXfC;
typedef Array<int32_t,Dynamic,1> ArrayXi32;
typedef Array<int64_t,Dynamic,1> ArrayXi64;
typedef Array<bool,Dynamic,1> ArrayXb;

struct cov_t {
    double ff;
    double fy;
};

void argsort(int32_t *idx, const float *v, int n);
int nonzero(int *out, const bool *mask, int n);
cov_t covariates(double *f, double *g, const float *x, const double *y,
                 double xm, double ym, double k0, float k1, int m);


int argmax(const ArrayXd &x)
{
    int i;
    x.maxCoeff(&i);
    return i;
}

//
// Return the delta sum of squared errors (SSE), discarding the Y variance.
//   sse  = y' * y - beta' * (X' * y)
//
double slow_dsse(Ref<const MatrixXd> X, VectorXd y)
{
    VectorXd xy = X.transpose() * y;
    VectorXd b  = X.fullPivHouseholderQr().solve(y);
    double dsse = b.transpose() * xy;
    return dsse / y.squaredNorm();
}

double slow_mse(MatrixXd X, VectorXd y)
{
    VectorXd b = X.fullPivHouseholderQr().solve(y);
    VectorXd e = X * b - y;
    double mse = e.squaredNorm()/e.rows();
    return mse / y.squaredNorm();
}

// Closed-form WLS: minimizes sum(w * (y - X b)²); returns sum(w*e²)/n / sum(w*y²).
double slow_mse_w(MatrixXd X, VectorXd y, VectorXd w)
{
    ArrayXd  sw = w.array().sqrt();
    MatrixXd Xw = (X.array().colwise() * sw).matrix();
    VectorXd yw = (y.array() * sw).matrix();
    VectorXd b  = Xw.fullPivHouseholderQr().solve(yw);
    VectorXd e  = X * b - y;
    double wsse = (w.array() * e.array().square()).sum();
    double wyy  = (w.array() * y.array().square()).sum();
    return (wsse / e.rows()) / wyy;
}

struct Result {
    Result(MarsAlgo &algo, int xcol, const ArrayXb &mask, bool linear, int min_span = 1)
    {
        linear_dsse = ArrayXd(mask.rows());
        hinge_dsse  = ArrayXd(mask.rows());
        hinge_cut   = ArrayXd(mask.rows());
        base_dsse = algo.dsse();
        algo.eval(linear_dsse.data(), hinge_dsse.data(),
                  hinge_cut.data(), xcol, mask.data(), min_span, 0, linear);
    }

    double  base_dsse;
    ArrayXd linear_dsse;
    ArrayXd hinge_dsse;
    ArrayXd hinge_cut;
};

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, ArgSort)
{
    int n = 23;
    ArrayXf x(ArrayXf::Random(n));
    ArrayXf y = x;
    ArrayXi32 k(n);

    std::sort(y.data(), y.data()+n, [](float a, float b) {
        return a > b;
    });
    argsort(k.data(), x.data(), n);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(x[k[i]],y[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////

void sort_columns(Ref<MatrixXd> X, const ArrayXi32 &k)
{
    VectorXd tmp;
    for (int j = 0; j < X.cols(); ++j) {
        tmp = X.col(j); // make a copy
        Ref<VectorXd> x = X.col(j); // keep a reference
        for (int i = 0; i < x.rows(); ++i) {
            tmp[i] = x[k[i]];
        }
        X.col(j) = tmp;
    }
}

TEST(MarsTest, SortColumns)
{
    std::random_device rd;
    std::mt19937 g(rd());

    const int n = 89;  // number of rows
    const int m = 13;  // number of basis
    MatrixXd X(MatrixXd::Random(n,m));
    MatrixXd Y(X);

    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), g);
    ArrayXi32 k = Map<ArrayXi32>(idx.data(),idx.size());

    sort_columns(X,k);
    for (int i = 0; i < X.rows(); ++i) {
        ASSERT_TRUE(X.row(i).isApprox(Y.row(k[i])));
    }
}

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, NonZero)
{
    bool mask[5] = {false, true, true, false, true};
    int idx[5];
    int n = nonzero(idx, mask, 5);
    ASSERT_EQ(n,3);
    ASSERT_EQ(idx[0],1);
    ASSERT_EQ(idx[1],2);
    ASSERT_EQ(idx[2],4);
}

///////////////////////////////////////////////////////////////////////////////

cov_t covariates_slow(ArrayXd &f, ArrayXd &g, const Ref<VectorXf> &x,
                      const ArrayXd &y, double xm, double ym, double k0, double k1)
{
    int m = x.rows();
    cov_t o = {0};
    for (int i = 0; i < m; ++i) {
        f[i] += k0*g[i];
        g[i] += k1*double(x[i]);
        o.ff += f[i]*f[i];
        o.fy += f[i]*y[i];
    }
    f[m] += k0*g[m];
    g[m] += k1*xm;
    o.ff += f[m]*f[m];
    o.fy += f[m]*ym;
    return o;
}

TEST(MarsTest, Covariates)
{
    int n = 809;
    int m = 131;

    MatrixXfC X  = MatrixXfC::Random(n,m);
    ArrayXd   f0 = ArrayXd::Zero(m+1);
    ArrayXd   g0 = ArrayXd::Zero(m+1);
    ArrayXd   f1 = ArrayXd::Zero(m+1);
    ArrayXd   g1 = ArrayXd::Zero(m+1);
    VectorXd  y  = VectorXd::Random(m);
    MatrixXf  k  = MatrixXf::Random(n,4);

    for (int i = 0; i < X.rows(); ++i) {
        cov_t o0 = covariates(f0.data(), g0.data(), X.row(i).data(), y.data(), k(i,0), k(i,1), k(i,2), k(i,3), m);
        cov_t o1 = covariates_slow(f1, g1, X.row(i), y, k(i,0), k(i,1), k(i,2), k(i,3));

        ASSERT_NEAR((f0-f1).matrix().norm(), 0, 1e-9);
        ASSERT_NEAR((g0-g1).matrix().norm(), 0, 1e-9);
        ASSERT_NEAR(o1.ff > 0? o0.ff/o1.ff : 1, 1, 1e-6);
        ASSERT_NEAR(o1.fy > 0? o0.fy/o1.fy : 1, 1, 1e-6);
    }
}

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, DeltaSSE)
{
    srand(0);

    const int N = 5891;  // number of rows
    const int M = 13;    // number of basis
    const double CUT = 0.25;

    MatrixXd X(MatrixXd::Random(N,M));
    ArrayXd x3 = X.col(3).array();
    ArrayXd x7 = X.col(7).array();
    x7[100] = CUT;
    ArrayXd x9 = X.col(9).array();
    X.col(10) = VectorXd::Zero(N);

    VectorXd y = (x3*.3 - x9*.2 + x3*x9*.25
                  -5*(x7-CUT).cwiseMax(0) + 2*(CUT-x7).cwiseMax(0)).matrix();
    y += VectorXd::Random(y.rows()); // add noise

    //-------------------------------------------------------------------------

    MatrixXf X32 = X.cast<float>();
    VectorXf y32 = y.cast<float>();
    ArrayXf  w32 = ArrayXf::Ones(N);

    double dsse1, dsse2;
    double mse1, mse2;

    ArrayXb mask = ArrayXb::Zero(M);
    mask[0] = true;
    MarsAlgo algo(X32.data(), y32.data(), w32.data(), X.rows(), X.cols(), X.cols()/2, X.rows());

    MatrixXd ALL_B_full = MatrixXd::Zero(N,16);
    auto     ALL_B      = ALL_B_full.leftCols(M);

    ALL_B.col(0).array() = 1;
    int b_cols = 1; // number of valid columns in B

    //-------------------------------------------------------------------------
    // Pick the first linear basis
    //-------------------------------------------------------------------------
    {
        const int xcol = 3; // just pick one
        Result res(algo, xcol, mask.head(b_cols), 1);

        double dsse1 = res.base_dsse + res.linear_dsse[0];
        ALL_B.col(b_cols) = x3;
        double dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
        ASSERT_NEAR(dsse1, dsse2, 1e-7);

        mse1 = algo.append('l', xcol, 0, 0);
        mask[b_cols++] = true;
        mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
        ASSERT_NEAR(mse1, mse2, 1e-8);
    }

    //-------------------------------------------------------------------------
    // Try adding another linear basis
    //-------------------------------------------------------------------------
    {
        const int xcol = 9; // pick another column
        Result res(algo, xcol, mask.head(b_cols), 1);

        dsse1 = res.base_dsse + res.linear_dsse[0];
        ALL_B.col(b_cols) = x9;
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
        ASSERT_NEAR(dsse1, dsse2, 1e-7);

        mse1 = algo.append('l', xcol, 0, 0);
        mask[b_cols++] = true;
        mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
        ASSERT_NEAR(mse1, mse2, 1e-8);
    }

    //-------------------------------------------------------------------------
    // Ok, now add the interaction
    //-------------------------------------------------------------------------
    {
        const int xcol = 9;
        const int bcol = 1; // use x3 as interaction
        Result res(algo, xcol, mask.head(b_cols), 1);

        ASSERT_EQ(bcol, argmax(res.linear_dsse));
        dsse1 = res.base_dsse + res.linear_dsse[bcol];
        ALL_B.col(b_cols) = x3*x9;
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
        ASSERT_NEAR(dsse1, dsse2, 1e-7);

        mse1 = algo.append('l', xcol, bcol, 0);
        mask[b_cols++] = true;
        mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
        ASSERT_NEAR(mse1, mse2, 1e-8);
    }

    //-------------------------------------------------------------------------
    // Try adding the hinge at x7
    //-------------------------------------------------------------------------
    double last_mse;
    {
        const int xcol = 7;
        const int bcol = 0;
        Result res(algo, xcol, mask.head(b_cols), 0);

        ASSERT_EQ(bcol, argmax(res.hinge_dsse));
        ASSERT_TRUE((res.hinge_dsse > res.linear_dsse).all());
        dsse1 = res.base_dsse + res.hinge_dsse[bcol];

        ALL_B.col(b_cols  ) = (x7 - CUT).cwiseMax(0);
        ALL_B.col(b_cols+1) = (CUT - x7).cwiseMax(0);
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        // Tolerances relaxed from 1e-7..1e-8 to 1e-6 after narrowing Bx to f32
        // storage; hinge SSE noise floor is dominated by the per-step f32
        // round in the inner sweep accumulators (sqrt(N)*eps_f32 on dsse/N).
        // HINGE_DSSE_TOL relaxes further on the non-AVX scalar path (see top).
        ASSERT_NEAR(dsse1/N, dsse2/N, HINGE_DSSE_TOL);

        //---------------------------------------------------------------------
        // Test all permutations
        //---------------------------------------------------------------------
        ALL_B.col(b_cols  ) = ALL_B.col(0).array() * (x7 - res.hinge_cut[0]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(0).array() * (res.hinge_cut[0] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[0];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, HINGE_DSSE_TOL);

        ALL_B.col(b_cols  ) = ALL_B.col(1).array() * (x7 - res.hinge_cut[1]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(1).array() * (res.hinge_cut[1] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[1];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, HINGE_DSSE_TOL);

        ALL_B.col(b_cols  ) = ALL_B.col(2).array() * (x7 - res.hinge_cut[2]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(2).array() * (res.hinge_cut[2] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[2];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, HINGE_DSSE_TOL);

        ALL_B.col(b_cols  ) = ALL_B.col(3).array() * (x7 - res.hinge_cut[3]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(3).array() * (res.hinge_cut[3] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[3];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, HINGE_DSSE_TOL);

        //---------------------------------------------------------------------
        // Append the hinges
        //---------------------------------------------------------------------
        mse1 = algo.append('+', xcol, bcol, CUT);
        ALL_B.col(b_cols) = (x7 - CUT).cwiseMax(0);
        mask[b_cols++] = true;
        mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
        ASSERT_NEAR(mse1, mse2, 2e-8);

        mse1 = algo.append('-', xcol, bcol, CUT);
        ALL_B.col(b_cols) = (CUT - x7).cwiseMax(0);
        mask[b_cols++] = true;
        mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
        ASSERT_NEAR(mse1, mse2, 1e-7);

        last_mse = mse1; // save for later
    }

    //-------------------------------------------------------------------------
    // Try adding an empty data column
    //-------------------------------------------------------------------------
    {
        const int xcol = 10;
        Result res(algo, xcol, mask.head(b_cols), 0);

        ASSERT_NEAR(res.base_dsse, 1-last_mse*N, 1e-8);
        ASSERT_TRUE(res.linear_dsse.head(b_cols).isConstant(0));
        ASSERT_TRUE(res.hinge_dsse. head(b_cols).isConstant(0));
    }

    //-------------------------------------------------------------------------
    // Try adding a mask
    //-------------------------------------------------------------------------
    {
        const int xcol = 2;
        const int bcol = 3;
        mask[bcol] = false;
        Result res(algo, xcol, mask.head(b_cols), 0);

        ASSERT_TRUE(std::isnan(res.hinge_cut[bcol]));
        res.hinge_cut[bcol] = 42; // replace NAN value

        ArrayXd linear_sse_2(b_cols);
        ArrayXd hinge_sse_2 (b_cols);
        ArrayXd hinge_cut_2 (b_cols);

        mask[bcol] = true;
        Result foo(algo, xcol, mask.head(b_cols), 0);

        foo.linear_dsse[bcol] = 0;
        foo.hinge_dsse [bcol] = 0;
        foo.hinge_cut  [bcol] = 42; // make match with NAN value

        ASSERT_TRUE(res.linear_dsse.isApprox(foo.linear_dsse));
        ASSERT_TRUE(res.hinge_dsse.isApprox(foo.hinge_dsse));
        ASSERT_TRUE(res.hinge_cut.isApprox(foo.hinge_cut));
    }
}

///////////////////////////////////////////////////////////////////////////////

// Verify that MarsAlgo reproduces the closed-form WLS solution under
// non-uniform observation weights. This exercises the sqrt(w) scaling
// in the constructor.
TEST(MarsTest, WeightedLS)
{
    srand(0);

    const int N = 1000;
    const int M = 4;

    MatrixXd X(MatrixXd::Random(N, M));
    ArrayXd  x1 = X.col(1).array();
    ArrayXd  x2 = X.col(2).array();

    VectorXd y = (x1 * 0.5 + x2 * 0.3).matrix();
    y += VectorXd::Random(N) * 0.1;

    // Non-uniform weights spanning ~3 orders of magnitude
    VectorXd w(N);
    for (int i = 0; i < N; ++i) {
        w[i] = 0.01 + 10.0 * std::abs(X(i, 0));
    }

    MatrixXf X32 = X.cast<float>();
    VectorXf y32 = y.cast<float>();
    ArrayXf  w32 = w.cast<float>();

    MarsAlgo algo(X32.data(), y32.data(), w32.data(), N, M, M, N);

    MatrixXd ALL_B(N, 3);
    ALL_B.col(0) = VectorXd::Ones(N);

    // Add x1 as a linear basis and compare to closed-form WLS
    double mse1 = algo.append('l', 1, 0, 0);
    ALL_B.col(1) = x1.matrix();
    double mse2 = slow_mse_w(ALL_B.leftCols(2), y, w);
    ASSERT_NEAR(mse1, mse2, 1e-7);

    // Add x2 and re-check
    mse1 = algo.append('l', 2, 0, 0);
    ALL_B.col(2) = x2.matrix();
    mse2 = slow_mse_w(ALL_B.leftCols(3), y, w);
    ASSERT_NEAR(mse1, mse2, 1e-7);
}

///////////////////////////////////////////////////////////////////////////////

// Verify the min_span gating: with min_span=S the inner loop only checks SSE
// at every S-th sorted position. The f/g/k0/k1/w/b2/vb/bd accumulators must
// still update every iteration, so the hinge_dsse at any position evaluated
// with min_span=S must equal what the same position would yield with
// min_span=1 (i.e., min_span only restricts which cuts are *considered*, not
// the math at the considered cuts).
TEST(MarsTest, MinSpan)
{
    srand(0);

    const int    N   = 2000;
    const int    M   = 4;
    const double CUT = 0.25;

    MatrixXd X(MatrixXd::Random(N, M));
    ArrayXd  x0 = X.col(0).array();
    VectorXd y  = (-5*(x0-CUT).cwiseMax(0) + 2*(CUT-x0).cwiseMax(0)).matrix();
    y += VectorXd::Random(N) * 0.05;

    MatrixXf X32 = X.cast<float>();
    VectorXf y32 = y.cast<float>();
    ArrayXf  w32 = ArrayXf::Ones(N);

    MarsAlgo algo(X32.data(), y32.data(), w32.data(), N, M, M*2, N);
    ArrayXb  mask = ArrayXb::Ones(1); // just the intercept basis

    Result r1 (algo, 0, mask, /*linear_only=*/0, /*min_span=*/1);
    Result r2 (algo, 0, mask, /*linear_only=*/0, /*min_span=*/2);
    Result r10(algo, 0, mask, /*linear_only=*/0, /*min_span=*/10);

    // The strided runs evaluate a strict subset of the cuts the min_span=1
    // run does, so their best hinge_dsse cannot exceed it.
    EXPECT_GE(r1.hinge_dsse[0], r2 .hinge_dsse[0]);
    EXPECT_GE(r2.hinge_dsse[0], r10.hinge_dsse[0]);

    // linear_dsse is independent of cut placement; min_span must not affect it.
    ASSERT_NEAR(r1.linear_dsse[0], r2 .linear_dsse[0], 1e-12);
    ASSERT_NEAR(r1.linear_dsse[0], r10.linear_dsse[0], 1e-12);

    // The chosen cut should land near the true CUT for all min_span values.
    // With N=2000 samples on [-1,1] the average sorted gap is ~1e-3, so even
    // with min_span=10 the nearest grid point is well within 0.05.
    EXPECT_NEAR(r1 .hinge_cut[0], CUT, 0.05);
    EXPECT_NEAR(r2 .hinge_cut[0], CUT, 0.05);
    EXPECT_NEAR(r10.hinge_cut[0], CUT, 0.05);
}

// eval() is called concurrently over X columns from the OpenMP region in the
// bindings; production never exercised that path from the test suite. This
// grows the basis through append() (so Bo is the tight-stride, grown matrix),
// then checks that splitting the per-column eval() across std::threads -- which
// forces the function-local thread_local scratch to be constructed on non-main
// threads and has many readers hit the shared Bo at once -- reproduces the
// single-threaded result bit-for-bit. std::thread keeps this independent of the
// OpenMP runtime.
TEST(MarsTest, ConcurrentEval)
{
    srand(0);
    const int N = 4096; // rows
    const int M = 24;   // X columns

    MatrixXd X(MatrixXd::Random(N, M));
    VectorXd y = (X.col(0).array() * 0.4
                  - 0.7 * (X.col(1).array() - 0.1).cwiseMax(0)
                  + 0.5 * X.col(2).array() * X.col(3).array()).matrix();
    y += 0.3 * VectorXd::Random(N);

    MatrixXf X32 = X.cast<float>();
    VectorXf y32 = y.cast<float>();
    ArrayXf  w32 = ArrayXf::Ones(N);

    MarsAlgo algo(X32.data(), y32.data(), w32.data(), N, M, /*max_terms=*/40, N);

    // Grow the basis via append(): linears, a mirror hinge pair, an interaction.
    algo.append('l', 0, 0, 0);
    algo.append('l', 1, 0, 0);
    algo.append('+', 2, 0, 0.0f);
    algo.append('-', 2, 0, 0.0f);
    algo.append('l', 3, 1, 0);
    const int m = algo.nbasis();
    ASSERT_GE(m, 3);

    ArrayXb   mask     = ArrayXb::Constant(m, true);
    const int min_span = 1, end_span = 0;

    auto run = [&](std::vector<ArrayXd> &lin, std::vector<ArrayXd> &hin,
                   std::vector<ArrayXd> &cut, int t0, int t1) {
        for (int xc = t0; xc < t1; ++xc) {
            algo.eval(lin[xc].data(), hin[xc].data(), cut[xc].data(),
                      xc, mask.data(), min_span, end_span, /*linear_only=*/false);
        }
    };

    auto make = [&] { return std::vector<ArrayXd>(M, ArrayXd(m)); };
    std::vector<ArrayXd> lin_ref = make(), hin_ref = make(), cut_ref = make();
    run(lin_ref, hin_ref, cut_ref, 0, M); // single-threaded reference

    std::vector<ArrayXd> lin = make(), hin = make(), cut = make();
    const int nt = 4;
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) {
        pool.emplace_back(run, std::ref(lin), std::ref(hin), std::ref(cut),
                          (M * t) / nt, (M * (t + 1)) / nt);
    }
    for (auto &th : pool) th.join();

    for (int xc = 0; xc < M; ++xc) {
        for (int j = 0; j < m; ++j) {
            ASSERT_EQ(lin[xc][j], lin_ref[xc][j]) << "linear_dsse col " << xc << " j " << j;
            ASSERT_EQ(hin[xc][j], hin_ref[xc][j]) << "hinge_dsse col "  << xc << " j " << j;
            const double c = cut[xc][j], cr = cut_ref[xc][j];
            ASSERT_TRUE((std::isnan(c) && std::isnan(cr)) || c == cr)
                << "hinge_cut col " << xc << " j " << j;
        }
    }
}
