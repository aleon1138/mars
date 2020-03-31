#include "marsalgo.h"
#include <gtest/gtest.h>
#include <vector>
constexpr double EPS = 1e-14;

double invnorm(VectorXd x) {
    const double s = x.norm();
    return s > EPS? 1.0/s : 0.0;
}

int argmax(const ArrayXd &x) {
    int i;
    x.maxCoeff(&i);
    return i;
}

ArrayXi64 create_mask(int m, int p) {
    std::vector<int64_t> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_shuffle(idx.begin(), idx.end());
    std::sort(idx.begin(), idx.begin()+p);
    return Map<ArrayXi64>(idx.data(),p);
}

double slow_dsse(Ref<const MatrixXd> X, VectorXd y) {
    VectorXd xy = X.transpose() * y;
    VectorXd b  = X.fullPivHouseholderQr().solve(y);
    return b.transpose() * xy;
}

double slow_mse(MatrixXd X, VectorXd y) {
    VectorXd b = X.fullPivHouseholderQr().solve(y);
    VectorXd e = X * b - y;
    return e.squaredNorm()/e.rows();
}

///////////////////////////////////////////////////////////////////////////////

TEST(MarsTest, ArgSort)
{
    int n = 23;
    ArrayXf x(ArrayXf::Random(n));
    ArrayXf y = x;
    ArrayXi32 k(n);

    std::sort(y.data(), y.data()+n, [](float a, float b) { return a > b; });
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
    ArrayXi64 mask = create_mask(m,p);
    B.col(BAD_COL) = B.col(1); // add a co-linear column

    //-------------------------------------------------------------------------
    // Initialize 'Bo' as the orthonomal projection of 'B'
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
    // Test out orthonormalize utility.
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

TEST(MarsTest, DeltaSSE)
{
    const int N = 5891;  // number of rows
    const int m = 13;    // number of basis
    const double CUT = 0.25;

    MatrixXd X(MatrixXd::Random(N,m));
    ArrayXd  x3 = X.col(3).array();
    ArrayXd  x7 = X.col(7).array(); x7[100] = CUT;
    ArrayXd  x9 = X.col(9).array();
    VectorXd y  = (x3*.3 - x9*.2 + x3*x9*.25
                   -5*(x7-CUT).cwiseMax(0) + 2*(CUT-x7).cwiseMax(0)).matrix();

    y += VectorXd::Random(y.rows()); // add noise
    y /= y.norm();

    MatrixXf X32 = X.cast<float>();
    VectorXf y32 = y.cast<float>();
    ArrayXf  w32 = ArrayXf::Ones(N);

    double dsse1, dsse2;
    double mse1, mse2;
    int xcol, bcol;

    std::vector<int64_t> mask = {0};
    MarsAlgo algo(X32.data(), y32.data(), w32.data(), X.rows(), X.cols(), X.cols()/2, X.rows());
    ArrayXd linear_sse(ArrayXd::Zero(m));
    ArrayXd hinge_sse (ArrayXd::Zero(m));
    ArrayXd hinge_cut (ArrayXd::Zero(m));
    MatrixXd ALL_B(MatrixXd::Zero(N,m));
    ALL_B.col(0).array() = 1;
    int b_cols = 1; // number of valid columns in B

    //-------------------------------------------------------------------------
    // Pick the first linear basis
    //-------------------------------------------------------------------------
    xcol = 3; // just pick one
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 1);
    dsse1 = linear_sse[0];
    ALL_B.col(b_cols) = x3;
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
    ASSERT_NEAR(dsse1, dsse2, 1e-7);

    mse1 = algo.append('l', xcol, 0, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
    ASSERT_NEAR(mse1, mse2, 1e-8);

    //-------------------------------------------------------------------------
    // Try adding another linear basis
    //-------------------------------------------------------------------------
    xcol = 9; // pick another column
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 1);
    dsse1 = linear_sse[0];
    ALL_B.col(b_cols) = x9;
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
    ASSERT_NEAR(dsse1, dsse2, 1e-7);

    mse1 = algo.append('l', xcol, 0, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
    ASSERT_NEAR(mse1, mse2, 1e-8);

    //-------------------------------------------------------------------------
    // Ok, now add the interaction
    //-------------------------------------------------------------------------
    xcol = 9;
    bcol = 1; // use x3 as interaction
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 1);
    ASSERT_EQ(bcol, argmax(linear_sse));
    dsse1 = linear_sse[bcol];
    ALL_B.col(b_cols) = x3*x9;
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y);
    ASSERT_NEAR(dsse1, dsse2, 1e-7);

    mse1 = algo.append('l', xcol, bcol, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y);
    ASSERT_NEAR(mse1, mse2, 1e-8);

    //-------------------------------------------------------------------------
    // Try adding the hinge at x7
    //-------------------------------------------------------------------------
    xcol = 7;
    bcol = 0;
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 0);
    ASSERT_EQ(bcol, argmax(hinge_sse));
    ASSERT_GT(hinge_sse.maxCoeff(),linear_sse.maxCoeff());
    dsse1 = hinge_sse[bcol];

    ALL_B.col(b_cols  ) = (x7 - CUT).cwiseMax(0);
    ALL_B.col(b_cols+1) = (CUT - x7).cwiseMax(0);
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
    ASSERT_NEAR(dsse1/N, dsse2/N, 1e-8);

    //-------------------------------------------------------------------------
    // Test all permutations
    //-------------------------------------------------------------------------
    ALL_B.col(b_cols  ) = ALL_B.col(0).array() * (x7 - hinge_cut[0]).cwiseMax(0);
    ALL_B.col(b_cols+1) = ALL_B.col(0).array() * (hinge_cut[0] - x7).cwiseMax(0);
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
    ASSERT_NEAR(hinge_sse[0]/N, dsse2/N, 1e-8);

    ALL_B.col(b_cols  ) = ALL_B.col(1).array() * (x7 - hinge_cut[1]).cwiseMax(0);
    ALL_B.col(b_cols+1) = ALL_B.col(1).array() * (hinge_cut[1] - x7).cwiseMax(0);
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
    ASSERT_NEAR(hinge_sse[1]/N, dsse2/N, 1e-8);

    ALL_B.col(b_cols  ) = ALL_B.col(2).array() * (x7 - hinge_cut[2]).cwiseMax(0);
    ALL_B.col(b_cols+1) = ALL_B.col(2).array() * (hinge_cut[2] - x7).cwiseMax(0);
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
    ASSERT_NEAR(hinge_sse[2]/N, dsse2/N, 1e-8);

    ALL_B.col(b_cols  ) = ALL_B.col(3).array() * (x7 - hinge_cut[3]).cwiseMax(0);
    ALL_B.col(b_cols+1) = ALL_B.col(3).array() * (hinge_cut[3] - x7).cwiseMax(0);
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+2), y);
    ASSERT_NEAR(hinge_sse[3]/N, dsse2/N, 1e-8);
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
