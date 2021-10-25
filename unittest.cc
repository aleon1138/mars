#include "marsalgo.h"
#include <gtest/gtest.h>
#include <vector>
constexpr double EPS = 1e-14;

double invnorm(VectorXd x)
{
    const double s = x.norm();
    return s > EPS? 1.0/s : 0.0;
}

int argmax(const ArrayXd &x)
{
    int i;
    x.maxCoeff(&i);
    return i;
}

ArrayXi create_mask(int m, int p)
{
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_shuffle(idx.begin(), idx.end());
    std::sort(idx.begin(), idx.begin()+p);
    return Map<ArrayXi>(idx.data(),p);
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

struct Result {
    Result(MarsAlgo &algo, int xcol, const ArrayXb &mask, bool linear)
    {
        linear_dsse = ArrayXd(mask.rows());
        hinge_dsse  = ArrayXd(mask.rows());
        hinge_cut   = ArrayXd(mask.rows());
        base_dsse = algo.dsse();
        algo.eval(linear_dsse.data(), hinge_dsse.data(),
                  hinge_cut.data(), xcol, mask.data(), 0, linear);
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

TEST(MarsTest, Orthonormalize)
{
    const int n = 89;  // number of rows
    const int m = 13;  // number of basis
    const int p = 7;   // basis mask
    constexpr int BAD_COL = 2;

    MatrixXdC Bo(n,m);
    MatrixXf  B(MatrixXf::Random(n,m));
    ArrayXf   x(ArrayXf::Random(n)*10);
    ArrayXd   y(ArrayXd::Random(n)+1);
    ArrayXi   mask = create_mask(m,p);
    B.col(BAD_COL) = B.col(1); // add a co-linear column

    //-------------------------------------------------------------------------
    // Initialize 'Bo' as the ortho-normal projection of 'B'
    //-------------------------------------------------------------------------
    Bo = B.cast<double>();
    Bo.col(0) *= invnorm(Bo.col(0));
    for (int j = 1; j < Bo.cols(); ++j) {
        Bo.col(j) *= invnorm(Bo.col(j));
        for (int k = 0; k < j; ++k) {
            Bo.col(j) -= (Bo.col(k).transpose()*Bo.col(j)) * Bo.col(k);
            Bo.col(j) *= invnorm(Bo.col(j));
        }
    }

    //-------------------------------------------------------------------------
    // Ensure Bo is indeed ortho-normal
    //-------------------------------------------------------------------------
    MatrixXd BoBo = MatrixXd::Identity(m,m);
    BoBo(BAD_COL,BAD_COL) = 0;
    ASSERT_TRUE((Bo.transpose()*Bo).isApprox(BoBo,EPS));

    //-------------------------------------------------------------------------
    // Test out ortho-normalize utility.
    //-------------------------------------------------------------------------
    MatrixXd Bx(n,p);
    orthonormalize(Bx, B, Bo, x, mask, EPS);

    ASSERT_TRUE((Bo.transpose()*Bx).isZero(EPS));
    ASSERT_TRUE(Bx.colwise().norm().isOnes(EPS));
}

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, SortColumns)
{
    const int n = 89;  // number of rows
    const int m = 13;  // number of basis
    MatrixXd X(MatrixXd::Random(n,m));
    MatrixXd Y(X);

    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_shuffle(idx.begin(), idx.end());
    ArrayXi32 k = Map<ArrayXi32>(idx.data(),idx.size());

    sort_columns(X,k);
    for (int i = 0; i < X.rows(); ++i) {
        ASSERT_TRUE(X.row(i).isApprox(Y.row(k[i])));
    }
}

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, NonZero)
{
    ArrayXb mask = ArrayXb::Ones(5);
    mask[0] = false;
    mask[3] = false;

    ArrayXi idx = nonzero(mask);
    ASSERT_EQ(idx.rows(),3);
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
        cov_t o0 = covariates(f0, g0, X.row(i).data(), y.data(), k(i,0), k(i,1), k(i,2), k(i,3), m);
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
    MatrixXd ALL_B(MatrixXd::Zero(N,M));
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
        ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

        //---------------------------------------------------------------------
        // Test all permutations
        //---------------------------------------------------------------------
        ALL_B.col(b_cols  ) = ALL_B.col(0).array() * (x7 - res.hinge_cut[0]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(0).array() * (res.hinge_cut[0] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[0];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

        ALL_B.col(b_cols  ) = ALL_B.col(1).array() * (x7 - res.hinge_cut[1]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(1).array() * (res.hinge_cut[1] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[1];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

        ALL_B.col(b_cols  ) = ALL_B.col(2).array() * (x7 - res.hinge_cut[2]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(2).array() * (res.hinge_cut[2] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[2];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

        ALL_B.col(b_cols  ) = ALL_B.col(3).array() * (x7 - res.hinge_cut[3]).cwiseMax(0);
        ALL_B.col(b_cols+1) = ALL_B.col(3).array() * (res.hinge_cut[3] - x7).cwiseMax(0);
        dsse1 = res.base_dsse+res.hinge_dsse[3];
        dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
        ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

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
        ASSERT_NEAR(mse1, mse2, 2e-8);

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

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
