#include <Eigen/Dense>
#include <numeric>      // for std::iota
#include <cfloat>       // for DBL_EPSILON
#ifdef __SSE__
#   include <immintrin.h>  // for _mm_getcsr
#endif

/*
 *  The main benefit of using FMA is that you only incur half the error of doing
 *  the add and multiply separately. If the below macro is not available, then
 *  std::fma() is much slower than the underlying implementation. So disable it.
 */
#ifndef FP_FAST_FMA
#   undef fma
    inline double fma(double x, double y, double z)
    {
        return x * y + z;
    }
#endif

using namespace Eigen;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixXdC;
typedef Matrix<float, Dynamic,Dynamic,RowMajor> MatrixXfC;
typedef Array<int32_t,Dynamic,1> ArrayXi32;
typedef Array<int64_t,Dynamic,1> ArrayXi64;
typedef Array<bool,Dynamic,1> ArrayXb;

/*
 *  Return sort indexes in descending order.
 */
void argsort(int32_t *idx, const float *v, int n)
{
    std::iota(idx, idx+n, 0);
    std::sort(idx, idx+n, [&v](size_t i, size_t j) {
        return v[i] > v[j];
    });
}

/*
 *  Return the indexes of non-zero values.
 */
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

/*
 *  Interact a candidate regressor `x` with all existing basis `B`, and then
 *  ortho-normalize via inverse Cholesky method against the previous set
 *  of normalized basis `Bo`.
 *
 *  Bx : double(n,p) [out]
 *      returns `B * x`, ortho-normalized against `Bo`
 *
 *  B : float(n,m)
 *      the existing set of basis.
 *
 *  Bo : double(n,m)
 *      the existing set of ortho-normalized basis.
 *
 *  x : float(n)
 *      the candidate regressor.
 *
 *  mask : int(p)
 *      which columns of `B` to use.
 *
 *  tol : double
 *      a small epsilpon used to truncate small values to zero.
 */
void orthonormalize(Ref<MatrixXd> Bx, const Ref<MatrixXf> &B, const Ref<MatrixXdC> &Bo,
                    const ArrayXf &x, const ArrayXi &mask, double tol)
{
    for (int j = 0; j < mask.rows(); ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }

    const MatrixXd h = Bo.transpose() * Bx;
    const ArrayXd  s = (Bx.colwise().squaredNorm() - h.colwise().squaredNorm()).array();

    Bx -= Bo * h;
    Bx *= (s > tol).select(1/(s+tol).sqrt(), 0).matrix().asDiagonal();
}

struct cov_t {
    double ff;
    double fy;
};

/*
 *  f_ : double(m+1)
 *
 *  g_ : double(m+1)
 *
 *  x : float(m)
 *      a row from the matrix of ortho-normalized and pre-sorted existing basis,
 *      elsewhere in the code this is referred to as `Bo[k]`.
 *
 *  y : double(m)
 *      the result of `dot(Bo.T,y)`
 *
 *  xm: float
 *      the value at `x[m]` which holds the new candidate regressor.
 *
 *  ym: double
 *      the value at `y[m]` which holds the new candidate regressor.
 *
 *  k0 :
 *
 *  k1 :
 */
cov_t covariates(ArrayXd &f_, ArrayXd &g_, const double *x, const double *y,
                 double xm, double ym, double k0, float k1, int m)
{
    assert(f_.rows()==m+1);
    assert(g_.rows()==m+1);

    // Cast to raw pointers, as the the `[]` operator is surprisingly expensive!
    double *f = f_.data();
    double *g = g_.data();

#ifndef __AVX__
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
        __m256d x0 = _mm256_load_pd(x+i);

        f0 = _mm256_fmadd_pd(K0,g0,f0);
        g0 = _mm256_fmadd_pd(K1,x0,g0);
        S0 = _mm256_fmadd_pd(f0,f0,S0);
        S1 = _mm256_fmadd_pd(f0,y0,S1);

        _mm256_store_pd(f+i, f0);
        _mm256_store_pd(g+i, g0);
    }

    cov_t o = {
        .ff = (S0[0]+S0[1])+(S0[2]+S0[3]),
        .fy = (S1[0]+S1[1])+(S1[2]+S1[3]),
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
    o.ff = fma(f[m],f[m],o.ff);
    o.fy = fma(f[m],ym,  o.fy);
    return o;
}

///////////////////////////////////////////////////////////////////////////////

class MarsAlgo {

    // TODO - all large scratch buffers should be 32 bit

    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  _X;   // read-only view of the regressors
    ArrayXd     _y;         // target vector
    MatrixXf    _B;         // all basis
    MatrixXdC   _Bo;        // all basis, ortho-normalized
    VectorXd    _ybo;       // dot product of basis Bo with Y target
    ArrayXf     _s;         // normalization constant for columns of 'X'
    int         _m    = 1;  // number of basis found
    double      _yvar = 0;  // variance of 'y'
    double      _tol  = 0;  // numerical error tolerance

public:
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
        : _X  (x,n,m,Stride<Dynamic,1>(ldx,1))
        , _B  (MatrixXf ::Zero(n,p))
        , _Bo (MatrixXdC::Zero(n,p))
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
        _ybo = _Bo.leftCols(1).transpose() * _y.matrix();

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
        return _ybo.squaredNorm();
    }

    double yvar() const
    {
        return _yvar;
    }

    /*
     *  Returns the delta SSE (sum of squared errors) given the existing basis
     *  set and a candidate column of `X` to evaluate.
     *
     *  linear_dsse : [out]
     *
     *  hinge_dsse : [out]
     *
     *  hinge_cuts : [out]
     *
     *  xol : int
     *      which column of the training `X` data to use.
     *
     *  bmask : bool(m)
     *      a boolean mask to filter out which bases to use.
     *
     *  endspan : int
     *      how many samples to ignore from both the extreme ends of the
     *      training data.
     *
     *  linear_only : bool
     *      do not attempt to find any hinge cuts, only build a linear model.
     *      This will ignore the input values of `hinge_sse` and `hinge_cut`
     */
    void eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
              int xcol, const bool *bmask, int endspan, bool linear_only)
    {
        if (xcol < 0 || xcol >= _X.cols()) {
            throw std::runtime_error("invalid X column index");
        }

        ArrayXi bcols = nonzero(Map<const ArrayXb>(bmask,_m));
        if (bcols.rows() == 0) {
            return;
        }

#       ifdef __SSE__
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040); // enable FTZ and DAZ
#       endif

        const int n = _X.rows();
        const int m = _m;           // number of all currently existing basis
        const int p = bcols.rows(); // number of non-ignored basis

        const ArrayXf        x  = _X.col(xcol) * _s[xcol]; // copy and normalize 'X' column
        const Ref<MatrixXdC> Bo = _Bo.leftCols(m);

        // Evaluate `B[:,bcols] * x` and ortho-normalize against `Bo`
        MatrixXd Bx(n, p); // TODO - put in thread-local storage
        orthonormalize(Bx, _B.leftCols(m), Bo, x, bcols, _tol);

        // Calculate the linear delta SSE and map to the output buffer
        const VectorXd ybx = Bx.transpose() * _y.matrix();
        Map<ArrayXd>(linear_dsse,m) = ArrayXd::Zero(m);
        for (int j = 0; j < p; ++j) {
            linear_dsse[bcols[j]] = ybx[j]*ybx[j];
        }

        // Evaluate the delta SSE on all hinge locations
        if (linear_only == false) {
            const double *ybo = _ybo.data(); // dot(Bo.T,_y);
            ArrayXi hinge_idx = ArrayXi::Constant(p,-1);
            ArrayXd hinge_sse = ArrayXd::Constant(p,0);
            ArrayXd hinge_cut = ArrayXd::Constant(p,NAN);

            // Get sort indexes
            // TODO - we should keep a LRU cache as we usually pick from the
            //        same pool of regressors in Fast-MARS.
            ArrayXi32 k(n);
            argsort(k.data(), _X.col(xcol).data(), n);

            ArrayXd _d(n-1);
            double *d = _d.data()-1; // note minus-one hack
            for (int i = 1; i < n; ++i) {
                d[i] = x[k[i-1]] - x[k[i]];
            }

            const int head = endspan;
            const int tail = n-endspan;

            for (int j = 0; j < p; ++j) {
                ArrayXd f = ArrayXd::Zero(m+1);
                ArrayXd g = ArrayXd::Zero(m+1);
                const float  *b  = _B.col(bcols[j]).data();
                const double *bx = Bx.col(j).data();

                double b_k  = b [k[0]]; // sort and upcast to double
                double bx_k = bx[k[0]];
                double y_k  = _y[k[0]];
                covariates(f,g,Bo.row(k[0]).data(),ybo,bx_k,ybx[0],0,b_k,m);

                double k0 = 0;
                double k1 = 0;
                double w  = 0;
                double bd = 0;
                double vb = b_k*y_k;
                double b2 = b_k*b_k;

                for (int i = 1; i < tail; ++i) {
                    __builtin_prefetch(Bo.row(k[i+1]).data());
                    b_k  = b [k[i]]; // sort and upcast to double
                    bx_k = bx[k[i]];
                    y_k  = _y[k[i]];
                    cov_t o = covariates(f,g,Bo.row(k[i]).data(),ybo,bx_k,ybx[j],d[i],b_k,m);

                    k0 = fma(d[i]*d[i],b2,k0);
                    k1 = fma(d[i]*2,bd,k1);
                    w  = fma(d[i],vb,w);
                    bd = fma(d[i],b2,bd);
                    b2 = fma(b_k,b_k,b2);
                    vb = fma(y_k,b_k,vb);

                    const double uw  = o.fy - w;
                    const double den = (k0+k1) - o.ff;
                    const double sse = den > _tol? (uw*uw)/(den+_tol) : 0;
                    if ((i > head) and (sse > hinge_sse[j])) {
                        hinge_sse[j] = sse;
                        hinge_idx[j] = i;
                    }
                }
            }

            // Map the results to the output arrays
            Map<ArrayXd>(hinge_dsse,m) = ArrayXd::Zero(m);
            Map<ArrayXd>(hinge_cuts,m) = ArrayXd::Constant(m,NAN);
            for (int j = 0; j < p; ++j) {
                if (hinge_idx[j] >= 0) {
                    hinge_dsse[bcols[j]] = linear_dsse[bcols[j]] + hinge_sse[j];
                    hinge_cuts[bcols[j]] = _X(k[hinge_idx[j]],xcol);
                }
            }
        }

#       ifdef __SSE__
        _mm_setcsr(csr); // revert
#       endif
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
            VectorXd ybo = _Bo.leftCols(_m+1).transpose() * _y.matrix();
            const double mse = (1. - ybo.squaredNorm()) / _X.rows();
            if (mse >= -_tol) { // gracefully handle values close to zero
                _m += 1;
                _ybo = ybo; // save for next iteration
                return std::max(mse,0.0);
            }
        }
        return -1.;
    }
};
