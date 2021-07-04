#include <Eigen/Dense>
#include <numeric>      // for std::iota
#include <cfloat>       // for DBL_EPSILON
#include <xmmintrin.h>  // for _mm_getcsr

using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXdC;
typedef Matrix<float, Dynamic,Dynamic,RowMajor> MatrixXfC;
typedef Array<int32_t,Dynamic,1> ArrayXi32;
typedef Array<int64_t,Dynamic,1> ArrayXi64;
typedef Array<bool,Dynamic,1> ArrayXb;

///////////////////////////////////////////////////////////////////////////////
///  Return indexes in REVERSED order
///////////////////////////////////////////////////////////////////////////////
void argsort(int32_t *idx, const float *v, int n)
{
    std::iota(idx, idx+n, 0);
    std::sort(idx, idx+n, [&v](size_t i, size_t j) {
        return v[i] > v[j];
    });
}

///////////////////////////////////////////////////////////////////////////////
///  Return the indexes of all non-zero values
///////////////////////////////////////////////////////////////////////////////
ArrayXi nonzero(const ArrayXb &x)
{
    ArrayXi y(x.size());
    int n = 0;
    for (int i = 0; i < x.rows(); ++i) {
        if (x[i]) {
            y[n++] = i;
        }
    }
    return y.head(n);
}

///////////////////////////////////////////////////////////////////////////////
///  Orthonormalize via inverse Cholesky method
///////////////////////////////////////////////////////////////////////////////
void orthonormalize(Ref<MatrixXd> Bx, const Ref<MatrixXf> &B, const Ref<MatrixXdC> &Bo,
                    const ArrayXf &x, const ArrayXi &mask, double tol)
{
    assert(B.cols()  == Bo.cols() && Bx.rows() == B.rows());
    assert(Bx.cols() == mask.rows());

    for (int j = 0; j < mask.rows(); ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }

    const MatrixXd h = Bo.transpose() * Bx;
    const ArrayXd  s = (Bx.colwise().squaredNorm() - h.colwise().squaredNorm()).array();

    Bx -= Bo * h;
    Bx *= (s > tol).select(1/(s+tol).sqrt(), 0).matrix().asDiagonal();
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

///////////////////////////////////////////////////////////////////////////////

struct cov_t {
    double ff;
    double fy;
};

///////////////////////////////////////////////////////////////////////////////
//  f :
//  g :
//  x : float[]
//      a row of ortho-normalized and pre-sorted basis that have already been
//      chosen in the model. This is labelled as `Bok` elsewhere in the code.
//  y : double[]
//      This is the result of `dot(Bo.T,y)`. It has the same length as `x`.
//  k0 :
//  k1 :
///////////////////////////////////////////////////////////////////////////////
cov_t covariates(ArrayXd &f_, ArrayXd &g_, const Ref<VectorXf> &x_, const ArrayXd &y_,
                 double xm, double k0, float k1)
{
    //-------------------------------------------------------------------------
    // Careful - without -mfma, you may get worse performance. By using FMA
    // we incur only half the error of doing the add and multiply separately.
    //-------------------------------------------------------------------------
    static_assert(FP_FAST_FMA, "-mfma must be enabled");

    const int m = x_.rows();
    assert(f_.rows()==m+1);
    assert(g_.rows()==m+1);
    assert(y_.rows()==m);

    //-------------------------------------------------------------------------
    // We cast Eigen to raw pointers, as the profiler showed that accessing the
    // values via the [] operator is not zero cost as of yet.
    //-------------------------------------------------------------------------
    double *f = f_.data();
    double *g = g_.data();
    const float  *x = x_.data();
    const double *y = y_.data();

#if 0
    int m0 = 0;
    cov_t o = {0};
#else
    __m256d K0 = _mm256_set1_pd(k0);
    __m256d K1 = _mm256_set1_pd(k1);
    __m256d S0 = _mm256_setzero_pd();
    __m256d S1 = _mm256_setzero_pd();

    int m0 = m - m%4;
    for (int i = 0; i < m0; i+=4) {
        __m256d f0 = _mm256_load_pd(f+i);
        __m256d g0 = _mm256_load_pd(g+i);
        __m256d y0 = _mm256_load_pd(y+i);

        f0 = _mm256_fmadd_pd(K0,g0,f0);
        S0 = _mm256_fmadd_pd(f0,f0,S0);
        S1 = _mm256_fmadd_pd(f0,y0,S1);
        _mm256_store_pd(f+i, f0);
    }

    for (int i = 0; i < m0; i+=4) {
        __m256d g0 = _mm256_load_pd(g+i);
        __m256d x0 = _mm256_cvtps_pd(_mm_load_ps(x+i));

        g0 = _mm256_fmadd_pd(K1,x0,g0);
        _mm256_store_pd(g+i, g0);
    }

    cov_t o = {
        ff: (S0[0]+S0[1])+(S0[2]+S0[3]),
        fy: (S1[0]+S1[1])+(S1[2]+S1[3]),
    };
#endif

    for (int i = m0; i < m; ++i) {
        f[i] = fma(k0,g[i],f[i]);
        g[i] = fma(k1,x[i],g[i]);
        o.ff = fma(f[i],f[i],o.ff);
        o.fy = fma(f[i],y[i],o.fy);
    }
    f[m] = fma(k0,g[m],f[m]);
    g[m] = fma(k1,xm,  g[m]);

    return o;
}

///////////////////////////////////////////////////////////////////////////////

class MarsAlgo {

    // TODO - all large scratch buffers should be 32 bit

    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  _X;   // read-only view of the regressors
    ArrayXd     _y;         // target vector
    MatrixXf    _B;         // all basis
    MatrixXdC   _Bo;        // all basis, ortho-normalized
    MatrixXfC   _Bok;       // all basis, ortho-normalized and sorted (scratch buffer)
    MatrixXd    _Bx;        // B[k,mask]*x[k,None] (scratch buffer)
    VectorXd    _By;        // dot product of basis Bo with Y target
    ArrayXf     _s;         // scale of columns of 'X'
    int         _m    = 1;  // number of basis found
    double      _yvar = 0;  // variance of 'y'
    double      _tol  = 0;  // numerical error tolerance

public:
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
        : _X  (x,n,m,Stride<Dynamic,1>(ldx,1))
        , _B  (MatrixXf ::Zero(n,p))
        , _Bo (MatrixXdC::Zero(n,p))
        , _Bok(MatrixXfC::Zero(n,p))
        , _Bx (MatrixXdC::Zero(n,p))
        , _tol((n*0.02)*DBL_EPSILON) // rough guess
    {
        if (std::isfinite(NAN)) {
            throw std::runtime_error("NAN check is disabled, recompile without --fast-math");
        }
        ArrayXd vv;
        _y = Map<const ArrayXf>(y,n).cast<double>();  // copy and upcast
        vv = Map<const ArrayXf>(w,n).cast<double>();  // copy and upcast

        //---------------------------------------------------------------------
        // Filter out NAN's and apply weight to the target 'y'.
        //---------------------------------------------------------------------
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(y[i])) {
                _y[i] = vv[i] = 0;
            }
        }
        _y *= vv; // apply weight to target

        _y /= _y.matrix().norm();
        vv /= vv.matrix().norm();

        //---------------------------------------------------------------------
        // Initialize the first column of our basis with the intercept
        //---------------------------------------------------------------------
        _B .col(0) = vv.cast<float>();
        _Bo.col(0) = vv;
        _By = _Bo.leftCols(1).transpose() * _y.matrix();

        //---------------------------------------------------------------------
        // Calculate the sample variance of the target 'y'.
        //---------------------------------------------------------------------
        _yvar = (_y - _y.mean()).square().mean();

        //---------------------------------------------------------------------
        // Calculate the column norm of 'X'.
        //---------------------------------------------------------------------
        _s = (_X.colwise().squaredNorm()/_X.rows()).cwiseSqrt();
        _s = (_s > 0.f).select(1.f/_s, 1.f);
    }

    int nbasis() const
    {
        return _m;
    }

    double dsse() const
    {
        return _By.squaredNorm();
    }

    double yvar() const
    {
        return _yvar;
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Returns the delta SSE (sum of squared errors).
    //
    //  linear_dsse : [out]
    //
    //  hinge_dsse : [out]
    //
    //  hinge_cuts : [out]
    //  xol : int
    //      which column of the training data to use.
    //  bmask : bool[]
    //      a boolean mask to filter out which bases to use.
    //  endspan : int
    //      how many samples to ignore from both ends of the training data.
    //  linear_only : bool
    //      do not find the optimal hinge cuts, only build a linear model.
    //      You can set "hinge_sse" and "hinge_cut" as NULL.
    ///////////////////////////////////////////////////////////////////////////
    void eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
              int xcol, const bool *bmask, int endspan, bool linear_only)
    {
        if (xcol < 0 || xcol >= _X.cols()) {
            throw std::runtime_error("invalid X column index");
        }

        //---------------------------------------------------------------------
        // Enable Flush-to-Zero (FTZ) and Denorms-as-Zero (DAZ)
        //---------------------------------------------------------------------
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040); // FTZ and DAZ

        ArrayXi Bcols = nonzero(Map<const ArrayXb>(bmask,_m));
        const int n = _X.rows();
        const int m = _m;
        const int p = Bcols.rows();

        ArrayXf        x   = _X.col(xcol) * _s[xcol]; // copy and normalize 'X' column
        Ref<MatrixXdC> Bo  = _Bo.leftCols(m);
        Ref<MatrixXfC> Bok = _Bok.leftCols(m);
        Ref<MatrixXd>  Bx  = _Bx.leftCols(p);

        //---------------------------------------------------------------------
        // Orthonormalize via inverse Cholesky method
        //---------------------------------------------------------------------
        orthonormalize(Bx, _B.leftCols(m), Bo, x, Bcols, _tol);

        //---------------------------------------------------------------------
        // Calculate the linear delta SSE
        //---------------------------------------------------------------------
        const VectorXd &yb = _By; // Bo.transpose() * _y.matrix();
        const VectorXd yb2 = Bx.transpose() * _y.matrix();
        const ArrayXd  linear_sse = yb2.array().square();

        //---------------------------------------------------------------------
        // Map the results to the output buffers
        //---------------------------------------------------------------------
        Map<ArrayXd>(linear_dsse,_m) = ArrayXd::Zero(_m);
        for (int j = 0; j < p; ++j) {
            linear_dsse[Bcols[j]] = linear_sse[j];
        }

        //---------------------------------------------------------------------
        // Evaluate the delta SSE on all hinge locations
        //---------------------------------------------------------------------
        if (linear_only == false) {
            ArrayXi hinge_idx = ArrayXi::Constant(p,-1);
            ArrayXd hinge_sse = ArrayXd::Constant(p,0);
            ArrayXd hinge_cut = ArrayXd::Constant(p,NAN);

            //-----------------------------------------------------------------
            // Get sort indexes
            // TODO - this needs to be cached ?
            //-----------------------------------------------------------------
            ArrayXi32 k(n);
            argsort(k.data(), _X.col(xcol).data(), n);

            //-----------------------------------------------------------------
            // Sort all rows of y, Bo, Bx and take the deltas of x
            //-----------------------------------------------------------------
            VectorXd yk(n);
            for (int i = 0; i < n; ++i) {
                yk[i] = _y[k[i]];
                Bok.row(i) = Bo.row(k[i]).cast<float>();
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
                const Ref<VectorXf> b  = _B.col(Bcols[j]);
                const Ref<VectorXd> bx = Bx.col(j);

                double b_i = b[k[0]]; // sort and upcast to double
                double k0 = 0;
                double k1 = 0;
                double w  = 0;
                double bd = 0;
                double vb = b_i*yk[0];
                double b2 = b_i*b_i;
                ArrayXd f = ArrayXd::Zero(m+1);
                ArrayXd g = ArrayXd::Zero(m+1);
                covariates(f,g,Bok.row(0),yb,bx[0],0,b_i);

                for (int i = 1; i < tail; ++i) {
                    b_i = b[k[i]]; // sort and upcast to double
                    cov_t o = covariates(f,g,Bok.row(i),yb,bx[i],d[i],b_i);
                    o.ff = fma(f[m],f[m],o.ff);
                    o.fy = fma(f[m],yb2[j],o.fy);

                    k0 = fma(d[i]*d[i],b2,k0);
                    k1 = fma(d[i]*2,bd,k1);
                    w  = fma(d[i],vb,w);
                    bd = fma(d[i],b2,bd);
                    b2 = fma(b_i, b_i,b2);
                    vb = fma(yk[i],b_i,vb);

                    const double uw  = o.fy - w;
                    const double den = (k0+k1) - o.ff;
                    const double sse = den > _tol? (uw*uw)/(den+_tol) : 0;
                    if ((i > head) and (sse > hinge_sse[j])) {
                        hinge_sse[j] = sse;
                        hinge_idx[j] = i;
                    }
                }
            }

            //-----------------------------------------------------------------
            // Map the results to the output arrays
            //-----------------------------------------------------------------
            Map<ArrayXd>(hinge_dsse,_m) = ArrayXd::Zero(_m);
            Map<ArrayXd>(hinge_cuts,_m) = ArrayXd::Constant(_m,NAN);
            for (int j = 0; j < p; ++j) {
                if (hinge_idx[j] >= 0) {
                    hinge_dsse[Bcols[j]] = linear_sse[j] + hinge_sse[j];
                    hinge_cuts[Bcols[j]] = _X(k[hinge_idx[j]],xcol);
                }
            }
        }
        _mm_setcsr(csr); // revert
    }

    ///////////////////////////////////////////////////////////////////////////

    double append(char type, int xcol, int bcol, float h)
    {
        if (_m >= _B.cols()) {
            throw std::runtime_error("basis matrix is full");
        }
        if (bcol < 0 || bcol >= _m) {
            char msg[80];
            sprintf(msg, "invalid basis column number: %d", bcol);
            throw std::runtime_error(msg);
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
        _B.col(_m) /= v.norm();

        for (int j = 0; j < _m; ++j) {
            v -= (_Bo.col(j).transpose()*v) * _Bo.col(j);
        }

        const double w = v.norm();
        if (w*w > _tol) {
            _Bo.col(_m) = v/w;

            // This assumes the norm of '_y' == 1
            VectorXd yb = _Bo.leftCols(_m+1).transpose() * _y.matrix();
            const double mse = (1. - yb.squaredNorm()) / _X.rows();
            if (mse >= -_tol) { // gracefully handle values close to zero
                _m += 1;
                _By = yb; // save for next iteration
                return std::max(mse,0.0);
            }
        }
        return -1.;
    }
};
