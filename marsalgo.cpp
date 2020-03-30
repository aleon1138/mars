#include <Eigen/Dense>
#include <numeric> // for std::iota
#include <cfloat>  // for DBL_EPSILON

using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXdC;
typedef Array<int32_t,Dynamic,1> ArrayXi32;
typedef Array<int64_t,Dynamic,1> ArrayXi64;

///////////////////////////////////////////////////////////////////////////////
///  Return indices in REVERSED order
///////////////////////////////////////////////////////////////////////////////
void argsort(int32_t *idx, const float *v, int n) {
    std::iota(idx, idx+n, 0);
    std::sort(idx, idx+n, [&v](size_t i, size_t j) { return v[i] > v[j]; });
}

///////////////////////////////////////////////////////////////////////////////
///  Ortho-normalize via inverse Cholesky method
///////////////////////////////////////////////////////////////////////////////
void orthonormalize(Ref<MatrixXd> Bx, const Ref<MatrixXf> &B, const Ref<MatrixXdC> &Bo,
                    const ArrayXf &x, const ArrayXi64 &mask, double tol)
{
    assert(B.cols()  == Bo.cols());
    assert(Bx.cols() == mask.rows());
    assert(Bx.rows() == B.rows());

    for (int j = 0; j < mask.rows(); ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }

    const MatrixXd h = Bo.transpose() * Bx;
    const ArrayXd  s = (Bx.colwise().squaredNorm() - h.colwise().squaredNorm()).array() + tol;

    Bx -= Bo * h;
    Bx *= (s > 0).select(1/s.sqrt(), 0).matrix().asDiagonal();
}

///////////////////////////////////////////////////////////////////////////////
///  Sort columns in-place
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

void covariates(double *ff_, double *fy_, ArrayXd &f, ArrayXd &g,
    const Ref<VectorXd> &x, const VectorXd &y, double k0, double k1)
{
    assert(x.cols()==1 && x.rows()>=x.cols());
    assert(x.rows()==y.rows());

    const int m = x.rows();
    double s0 = 0;
    double s1 = 0;

    static_assert(FP_FAST_FMA, "-mfma must be enabled");
    for (int i = 0; i < m; ++i) {
        f[i] = fma(k0,g[i],f[i]);
        g[i] = fma(k1,x[i],g[i]);
        s0   = fma(f[i],f[i],s0);
        s1   = fma(f[i],y[i],s1);
    }
    *ff_ = s0;
    *fy_ = s1;
}

///////////////////////////////////////////////////////////////////////////////

class MarsAlgo {

    // TODO - all large scratch buffers should be 32 bit

    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  _X;   // read-only view of the regressors
    ArrayXd     _y;         // target vector
    MatrixXf    _B;         // all basis
    MatrixXdC   _Bo;        // all basis, ortho-normalized
    MatrixXdC   _Bok;       // all basis, ortho-normalized and sorted (scratch buffer)
    MatrixXd    _Bx;        // B[k,mask]*x[k,None] (scratch buffer)
    ArrayXf     _s;         // scale of columns of 'X'
    int         _m    = 1;  // number of basis found
    double      _yvar = 0;  // variance of 'y'
    double      _tol  = 0;  // numerical error tolerance

public:
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
       : _X  (x,n,m,Stride<Dynamic,1>(ldx,1))
       , _B  (MatrixXf ::Zero(n,p))
       , _Bo (MatrixXdC::Zero(n,p))
       , _Bok(MatrixXdC::Zero(n,p))
       , _Bx (MatrixXdC::Zero(n,p))
       , _tol((n*0.02)*DBL_EPSILON) // rough guess
    {
        if (std::isfinite(NAN)) {
            throw std::runtime_error("NAN check is disabled, recompile without --fast-math");
        }
        ArrayXd vv;
        _y = Map<const ArrayXf>(y,n).cast<double>();
        vv = Map<const ArrayXf>(w,n).cast<double>();

        //---------------------------------------------------------------------
        // Filter out NAN's and apply weight to the target 'y'.
        //---------------------------------------------------------------------
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(y[i])) { _y[i] = vv[i] = 0; }
        }
        _y *= vv; // apply weight to target

        _y /= _y.matrix().norm();
        vv /= vv.matrix().norm();

        //---------------------------------------------------------------------
        // Initialize the first column of our basis with the intercept
        //---------------------------------------------------------------------
        _B .col(0) = vv.cast<float>();
        _Bo.col(0) = vv;

        //---------------------------------------------------------------------
        // Calculate the sample variance of the target 'y'.
        //---------------------------------------------------------------------
        _yvar = (_y - _y.mean()).square().mean();

        //---------------------------------------------------------------------
        // Calculate the column norm of 'X'.
        //---------------------------------------------------------------------
        _s = _X.colwise().norm();
        _s = (_s > 0.f).select(1.f/_s, 0.f);
    }

    ///////////////////////////////////////////////////////////////////////////

    void dsse(double *linear_sse, double *hinge_sse, double *hinge_cut,
              int xcol, const int64_t *mask, int p, int endspan, bool linear_only)
    {
        if (p > _m) {
            throw std::runtime_error("invalid mask array");
        }

        const int n = _X.rows();
        const int m = _m;
        ArrayXf   x = _X.col(xcol) * _s[xcol]; // normalize 'x' column
        Ref<MatrixXd>  Bx  = _Bx.leftCols(p);
        Ref<MatrixXdC> Bo  = _Bo.leftCols(m);
        Ref<MatrixXdC> Bok = _Bok.leftCols(m);

        //---------------------------------------------------------------------
        // Ortho-normalize via inverse Cholesky method
        //---------------------------------------------------------------------
        orthonormalize(Bx, _B.leftCols(m), Bo, x, Map<const ArrayXi64>(mask,p), _tol);

        //---------------------------------------------------------------------
        // Calculate the linear delta SSE
        //---------------------------------------------------------------------
        VectorXd yb  = (Bo.transpose() * _y.matrix());
        ArrayXd  yb2 = (Bx.transpose() * _y.matrix()).array();
        Map<ArrayXd>(linear_sse,p) = (yb.transpose()*yb) + yb2.square();

        //---------------------------------------------------------------------
        // Evaluate the delta SSE on all hinge locations
        //---------------------------------------------------------------------
        if (linear_only == false) {
            ArrayXi hinge_idx(p);
            for (int j = 0; j < p; ++j) {
                hinge_idx[j] = -1;
                hinge_sse[j] =  0;
            }

            //-----------------------------------------------------------------
            // Get sort indices
            //-----------------------------------------------------------------
            ArrayXi32 k(n);
            argsort(k.data(), _X.col(xcol).data(), n);

            //-----------------------------------------------------------------
            // Sort all rows of y, Bo, Bx and take the deltas of x
            //-----------------------------------------------------------------
            VectorXd y(n);
            for (int i = 0; i < n; ++i) {
                y[i] = _y(k[i]);
                Bok.row(i) = Bo.row(k[i]);
            }
            sort_columns(Bx, k);

            ArrayXd _d(n-1);
            double *d = _d.data()-1; // note minus-one hack
            for (int i = 1; i < n; ++i) {
                d[i] = x[k[i-1]] - x[k[i]];
            }

            const int head = endspan;
            const int tail = n-endspan;
            static_assert(FP_FAST_FMA, "-mfma must be enabled");

            for (int j = 0; j < p; ++j) {
                const Ref<VectorXf> b  = _B.col(mask[j]);
                const Ref<VectorXd> bx = Bx.col(j);

                double b_i = b[k[0]]; // sort and upscale to double
                double k0 = 0;
                double k1 = 0;
                double w  = 0;
                double bd = 0;
                double vb = b_i*y[0];
                double b2 = b_i*b_i;
                ArrayXd f = ArrayXd::Zero(m+1);
                ArrayXd g = ArrayXd::Zero(m+1);
                double ff = 0;
                double fy = 0;

                covariates(&ff,&fy,f,g,Bo.row(0),yb,0,b_i);
                g[m] += b_i*bx[0];

                for (int i = 1; i < tail; ++i) {
                    b_i = b[k[i]]; // sort and upcast to double
                    covariates(&ff,&fy,f,g,Bo.row(i),yb,d[i],b_i);
                    f[m] = fma(d[i],g[m],f[m]);
                    g[m] = fma(b_i,bx[i],g[m]);
                    ff   = fma(f[m],f[m],ff);
                    fy   = fma(f[m],yb2[j],fy);

                    k0 = fma(d[i]*d[i],b2,k0);
                    k1 = fma(d[i]*2,bd,k1);
                    w  = fma(d[i],vb,w);
                    bd = fma(d[i],b2,bd);
                    b2 = fma(b_i, b_i,b2);
                    vb = fma(y[i],b_i,vb);

                    const double uw  = fy - w;
                    const double den = (k0+k1) - ff;
                    const double sse = den > _tol? (uw*uw)/(den+_tol) : 0;
                    if ((i > head) and (sse > hinge_sse[j])) {
                        hinge_sse[j] = sse;
                        hinge_idx[j] = i;
                    }
                }
            }

            for (int j = 0; j < p; ++j) {
                hinge_sse[j] += linear_sse[j];
                hinge_cut[j] = hinge_idx[j]>=0? x[k[hinge_idx[j]]]/_s[xcol] : NAN;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    double append(char type, int xcol, int bcol, float h) {
        if (_m >= _B.cols()) {
            throw std::runtime_error("basis matrix is full");
        }

        const float  s = _s[xcol];
        Ref<ArrayXf> b = _B.col(bcol).array();
        Ref<const ArrayXf> x = _X.col(xcol).array();

        switch(type) {
            case 'l':
                _B.col(_m) = (s*b*x);
                break;
            case '+':
                _B.col(_m) = (s*b*(x-h).cwiseMax(0));
                break;
            case '-':
                _B.col(_m) = (s*b*(h-x).cwiseMax(0));
                break;
            default:
                throw std::runtime_error("invalid basis type");
        }

        //---------------------------------------------------------------------
        // Use Gram-Schmidt orthogonalization.
        //---------------------------------------------------------------------
        VectorXd v = _B.col(_m).cast<double>(); // make a copy
        for (int j = 0; j < _m; ++j) {
            v -= (_Bo.col(j).transpose()*v) * _Bo.col(j);
        }

        const double w = v.norm();
        if (w*w > _tol) {
            _Bo.col(_m) = v/w;
            VectorXd yb = _Bo.leftCols(_m+1).transpose() * _y.matrix();
            const double mse = (1. - yb.squaredNorm()) / _X.rows();
            if (mse >= 0.) {
                ++_m;
                return mse;
            }
        }
        return -1.;
    }
};

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                                 UNIT TESTS                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifdef UNIT_TEST
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

double slow_mse(Ref<MatrixXd> X, VectorXd y) {
    VectorXd b = X.fullPivHouseholderQr().solve(y);
    VectorXd e = X * b - y;
    return e.squaredNorm()/X.rows();
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

    for (int i = 0; i < n; ++i){
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
    const int N = 3891;  // number of rows
    const int m = 13;    // number of basis

    MatrixXf X(MatrixXf::Random(N,m));
    ArrayXf  x3 = X.col(3).array();
    ArrayXf  x7 = X.col(7).array();
    ArrayXf  x9 = X.col(9).array();
    VectorXf y  = (x3*.3 - x9*.2 + x3*x9*.25 - 0.2*(x7-.4).cwiseMax(0)).matrix();
    ArrayXf  w(ArrayXf::Ones(N));

    y += VectorXf::Random(y.rows()); // add noise
    for (int j = 0; j < X.cols(); ++j) X.col(j) /= X.col(j).cast<double>().norm();
    y /= y.cast<double>().norm();

/*
    ok - for some reason this is not working when we dont normalize the Y??
    but it should work
    so dump the X and Y matrixes in numpy and see if you can duplicate your work
    it may be that your "slow" versions are not workin
*/

    double dsse1, dsse2;
    double mse1, mse2;
    int xcol, bcol;

    std::vector<int64_t> mask = {0};
    MarsAlgo algo(X.data(), y.data(), w.data(), X.rows(), X.cols(), X.cols()/2, X.rows());
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
    ALL_B.col(b_cols) = x3.cast<double>();
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y.cast<double>());
    ASSERT_NEAR(dsse1, dsse2, 1e-8);

    mse1 = algo.append('l', xcol, 0, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y.cast<double>());
    ASSERT_NEAR(mse1, mse2, 1e-8);

    //-------------------------------------------------------------------------
    // Try adding another linear basis
    //-------------------------------------------------------------------------
    xcol = 9; // pick another column
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 1);
    dsse1 = linear_sse[0];
    ALL_B.col(b_cols) = x9.cast<double>();
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y.cast<double>());
    ASSERT_NEAR(dsse1, dsse2, 1e-8);

    mse1 = algo.append('l', xcol, 0, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y.cast<double>());
    ASSERT_NEAR(mse1, mse2, 1e-8);

    //-------------------------------------------------------------------------
    // Ok, now add the interaction
    //-------------------------------------------------------------------------
    xcol = 9;
    bcol = 1; // use x3 as interaction
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(), xcol, mask.data(), mask.size(), 0, 1);
    ASSERT_EQ(bcol, argmax(linear_sse));
    dsse1 = linear_sse[bcol];
    ALL_B.col(b_cols) = (x3*x9).cast<double>();
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y.cast<double>());
    ASSERT_NEAR(dsse1, dsse2, 2e-8);

    mse1 = algo.append('l', xcol, bcol, 0);
    mask.push_back(b_cols);
    b_cols++;
    mse2 = slow_mse(ALL_B.leftCols(b_cols), y.cast<double>());
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

    ALL_B.col(b_cols) = (x7-.4).cwiseMax(0).cast<double>();
    dsse2 = slow_dsse(ALL_B.leftCols(b_cols+1), y.cast<double>());
    //ASSERT_NEAR(dsse1, dsse2, 1e-8);
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}

#endif
