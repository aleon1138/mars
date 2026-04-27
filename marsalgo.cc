#include "marsalgo.h"
#include <Eigen/Dense>
#include <numeric>          // for std::iota
#include <cfloat>           // for DBL_EPSILON
#include <cassert>
#ifdef __SSE__
#   include <immintrin.h>   // for _mm_getcsr
#endif

/*
 *  Fused Multiply-Add (FMA) incurs only half the error of computing them
 *  separately. If the below macro is not available, then the implementation
 *  provided by the standard library will be much slower than the macro below.
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
 *  Return sort indexes in  stable descending order. Tied X values keep their
 *  original row order, so the gather of Bo rows downstream is invariant to
 *  input row permutations.
 */
void argsort(int32_t *idx, const float *v, int n)
{
    std::iota(idx, idx+n, 0);
    std::stable_sort(idx, idx+n, [&v](size_t i, size_t j) {
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
 *  orthonormalize via inverse Cholesky method against the previous set
 *  of normalized basis `Bo`.
 *
 *  Bx : double(n,p) [out]
 *      returns `B[:,mask] .* x` (element-wise), orthonormalized against `Bo`
 *
 *  B : float(n,m)
 *      the existing set of basis.
 *
 *  Bo : double(n,m)
 *      the existing set of orthonormalized basis.
 *
 *  x : float(n)
 *      the candidate regressor.
 *
 *  mask : int(p)
 *      which columns of `B` to use.
 *
 *  tol : double
 *      a small epsilon used to truncate small values to zero.
 */
void orthonormalize(Ref<MatrixXd> Bx, const Ref<MatrixXf> &B, const Ref<MatrixXdC> &Bo,
                    const ArrayXf &x, const ArrayXi &mask, double tol)
{
    for (int j = 0; j < mask.rows(); ++j) {
        Bx.col(j) = (B.col(mask[j]).array() * x).cast<double>();
    }

    Bx -= Bo * (Bo.transpose() * Bx);
    const ArrayXd s = Bx.colwise().squaredNorm().array();
    Bx *= (s > tol).select(1/(s+tol).sqrt(), 0).matrix().asDiagonal();
}

struct cov_t {
    double ff;
    double fy;
};

/*
 *  Incrementally updates running accumulators `f_` and `g_` and returns their
 *  inner products. Called once per sorted data point while sweeping through
 *  candidate hinge cut locations.
 *
 *  f_ : double(m+1) [in/out]
 *      running accumulator for the hinge projection; f[i] += k0*g[i]
 *
 *  g_ : double(m+1) [in/out]
 *      running accumulator for the basis-weighted ortho-basis; g[i] += k1*x[i]
 *
 *  x : float(m)
 *      a row from the orthonormalized and pre-sorted existing basis matrix
 *      (referred to as `Bok` in the calling code).
 *
 *  y : double(m)
 *      the result of `dot(Bo.T, y)`, i.e. projection of existing basis onto target.
 *
 *  xm : float
 *      the orthonormalized candidate basis value at this sorted position (index m).
 *
 *  ym : double
 *      the projection of the candidate basis column onto the target (index m).
 *
 *  k0 : double
 *      spacing between the previous and current sorted x values: `x[k[i-1]] - x[k[i]]`.
 *
 *  k1 : float
 *      basis value at the current sorted position: `B[k[i], bcol]`.
 */
template <bool need_sse>
cov_t covariates_impl(ArrayXd &f_, ArrayXd &g_, const float *x, const double *y,
                      double xm, double ym, double k0, float k1, int m)
{
    assert(f_.rows()==m+1);
    assert(g_.rows()==m+1);

    // Cast to raw pointers, as the `[]` operator is surprisingly expensive!
    double *f = f_.data();
    double *g = g_.data();

    assert(reinterpret_cast<uintptr_t>(f) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(g) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(y) % 16 == 0);

    cov_t o = {0,0};

#ifndef __AVX__
    int m0 = 0;
#else
    __m256d K0 = _mm256_set1_pd(k0);
    __m256d K1 = _mm256_set1_pd(k1);
    __m256d S0 = _mm256_setzero_pd();
    __m256d S1 = _mm256_setzero_pd();

    int m0 = m - m%4;
    for (int i = 0; i < m0; i+=4) {
        __m256d f0 = _mm256_load_pd(f+i);
        __m256d g0 = _mm256_load_pd(g+i);
        __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(x+i)); // `x` might be unaligned!

        f0 = _mm256_fmadd_pd(K0,g0,f0);
        g0 = _mm256_fmadd_pd(K1,x0,g0);

        if constexpr (need_sse) {
            __m256d y0 = _mm256_load_pd(y+i);
            S0 = _mm256_fmadd_pd(f0,f0,S0);
            S1 = _mm256_fmadd_pd(f0,y0,S1);
        }

        _mm256_store_pd(f+i, f0);
        _mm256_store_pd(g+i, g0);
    }

    if constexpr (need_sse) {
        o.ff = (S0[0]+S0[1])+(S0[2]+S0[3]);
        o.fy = (S1[0]+S1[1])+(S1[2]+S1[3]);
    }
#endif

    for (int i = m0; i < m; ++i) {
        f[i] = fma(k0,g[i],f[i]);
        g[i] = fma(k1,x[i],g[i]);
        if constexpr (need_sse) {
            o.ff = fma(f[i],f[i],o.ff);
            o.fy = fma(f[i],y[i],o.fy);
        }
    }
    f[m] = fma(k0,g[m],f[m]);
    g[m] = fma(k1,xm,  g[m]);
    if constexpr (need_sse) {
        o.ff = fma(f[m],f[m],o.ff);
        o.fy = fma(f[m],ym,  o.fy);
    }
    return o;
}

/*
 *  Non-template wrapper preserving the original symbol used by the unit test
 *  link. Internal callers should use `covariates_impl<...>` directly so the
 *  SSE-reduction path can be elided.
 */
cov_t covariates(ArrayXd &f_, ArrayXd &g_, const float *x, const double *y,
                 double xm, double ym, double k0, float k1, int m)
{
    return covariates_impl<true>(f_, g_, x, y, xm, ym, k0, k1, m);
}

///////////////////////////////////////////////////////////////////////////////

struct MarsData {
    MarsData(const float *x, int n, int m, int p, int ldx)
        : X (x,n,m,Stride<Dynamic,1>(ldx,1))
        , B (MatrixXf ::Zero(n,p))
        , Bo(MatrixXdC::Zero(n,p)) {}

    // TODO - all large scratch buffers should be 32 bit
    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  X;   // read-only view of the regressors
    ArrayXd     y;      // target vector
    MatrixXf    B;      // all basis
    MatrixXdC   Bo;     // all basis, orthonormalized
    VectorXd    ybo;    // dot product of basis Bo with Y target
    ArrayXf     s;      // normalization constant for columns of 'X'
};

MarsAlgo::MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
    : _data(new MarsData(x, n, m, p, ldx))
    , _tol((n*0.02)*DBL_EPSILON) // rough guess
{
    verify(!std::isfinite(NAN), "NAN check is disabled, recompile without --fast-math");

    // Copy and upcast target
    _data->y = Map<const ArrayXf>(y,n).cast<double>();

    // For WLS we scale rows by sqrt(w), so that the OLS objective on the
    // transformed problem equals the weighted RSS on the original.
    ArrayXd sqrt_w = Map<const ArrayXf>(w,n).cast<double>().sqrt();

    // Filter out NAN's and apply weight to the target 'y'.
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) {
            _data->y[i] = sqrt_w[i] = 0;
        }
    }
    _data->y *= sqrt_w; // apply sqrt(w) to target

    // TODO - these row-order reductions (y_norm, w_norm, _yvar below, and
    // the column norms in _data->s) are sensitive to the input row order:
    // structured inputs (sorted, time-correlated, grouped) accumulate biased
    // running sums and lose precision vs. shuffled inputs. The downstream
    // greedy search amplifies these tiny perturbations into different basis
    // selections. Switching to compensated/pairwise summation here would make
    // the algorithm row-order-invariant; see tests/repro_shuffle.py.
    double y_norm = _data->y.matrix().norm();
    double w_norm = sqrt_w.matrix().norm();
    verify(y_norm > 0.0 && w_norm > 0, "target Y is all zero or NANs");

    _data->y /= y_norm;
    sqrt_w   /= w_norm;

    // Initialize the first column of our basis with the intercept
    _data->B .col(0) = sqrt_w.cast<float>();
    _data->Bo.col(0) = sqrt_w;
    _data->ybo = _data->Bo.leftCols(1).transpose() * _data->y.matrix();

    // Calculate the sample variance of the target 'y'.
    _yvar = (_data->y - _data->y.mean()).square().mean();

    // Calculate the column norm of 'X'.
    _data->s = (_data->X.colwise().squaredNorm()/_data->X.rows()).cwiseSqrt();
    verify(_data->s.isFinite().all(), "not all columns in X are finite");
    _data->s = (_data->s > 0.f).select(1.f/_data->s, 1.f);
}

MarsAlgo::~MarsAlgo()
{
    delete _data;
}
int MarsAlgo::nbasis() const
{
    return _m;
}
double MarsAlgo::dsse() const
{
    return _data->ybo.squaredNorm();
}
double MarsAlgo::yvar() const
{
    return _yvar;
}

void MarsAlgo::eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
                    int xcol, const bool *bmask, int min_span, int end_span, bool linear_only)
{
    verify(xcol >= 0 && xcol < _data->X.cols(), "invalid X column index");
    verify(min_span >= 1, "min_span must be >= 1");

    Map<ArrayXd>(linear_dsse,_m) = ArrayXd::Zero(_m);
    Map<ArrayXd>(hinge_dsse, _m) = ArrayXd::Zero(_m);
    Map<ArrayXd>(hinge_cuts, _m) = ArrayXd::Constant(_m,NAN);

    ArrayXi bcols = nonzero(Map<const ArrayXb>(bmask,_m));
    if (bcols.rows() == 0) {
        return;
    }

#ifdef __SSE__
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040); // enable FTZ and DAZ
#endif

    const int n = _data->X.rows();
    const int m = _m;           // number of all currently existing basis
    const int p = bcols.rows(); // number of non-ignored basis

    const ArrayXf        x  = _data->X.col(xcol) * _data->s[xcol]; // copy and normalize 'X' column
    const Ref<MatrixXdC> Bo = _data->Bo.leftCols(m);

    // Evaluate `B[:,bcols] * x` and ortho-normalize against `Bo`
    MatrixXd Bx(n, p); // TODO - put in thread-local storage
    orthonormalize(Bx, _data->B.leftCols(m), Bo, x, bcols, _tol);

    // Calculate the linear delta SSE and map to the output buffer
    const VectorXd ybx = Bx.transpose() * _data->y.matrix();
    for (int j = 0; j < p; ++j) {
        linear_dsse[bcols[j]] = ybx[j]*ybx[j];
    }

    // Evaluate the delta SSE on all hinge locations
    if (linear_only == false) {
        const double *ybo = _data->ybo.data(); // dot(Bo.T,_data->y);
        ArrayXi hinge_idx = ArrayXi::Constant(p,-1);
        ArrayXd hinge_sse = ArrayXd::Constant(p,0);
        ArrayXd hinge_cut = ArrayXd::Constant(p,NAN);

        // Get sort indexes
        // TODO - we should keep a LRU cache as we usually pick from the
        //        same pool of regressors in Fast-MARS.
        ArrayXi32 k(n);
        argsort(k.data(), _data->X.col(xcol).data(), n);

        // Sort the rows of `Bo`
        MatrixXfC Bok(n,m); // TODO - put in thread-local storage
        for (int i = 0; i < n; ++i) {
            Bok.row(i) = Bo.row(k[i]).cast<float>();
        }

        // Take the deltas of `x`
        ArrayXd _d(n-1);
        double *d = _d.data()-1; // note minus-one hack
        for (int i = 1; i < n; ++i) {
            d[i] = x[k[i-1]] - x[k[i]];
        }

        const int head = end_span;
        const int tail = n-end_span;
        const MatrixXf &B = _data->B;
        const ArrayXd  &y = _data->y;

        /*
         *  For each parent basis column b = B[:,bcols[j]], sweep potential hinge
         *  cut locations from largest to smallest x (descending sort order). At
         *  each cut h = x[k[i]] we evaluate the positive hinge h_plus = b*max(x-h,0),
         *  which is nonzero only for the i samples above the cut.
         *
         *  `covariates()` maintains f/g to track the projection of h_plus onto the
         *  existing ortho-basis (Bo) AND the linear candidate (Bx[:,j]). The local
         *  accumulators below build up the remaining terms needed for the delta-SSE:
         *
         *    b2   = sum_{j<i} b[k[j]]^2                  (squared norm of b above cut)
         *    vb   = sum_{j<i} b[k[j]] * y[k[j]]          (dot product <b,y> above cut)
         *    bd   = sum_{j<i} b[k[j]]^2 * (x[k[j]] - h)  (helper for k0/k1 update)
         *    k0+k1 = ||h_plus||^2                        (squared norm of positive hinge)
         *    w    = h_plus^T * y                         (dot product of hinge with target)
         *
         *  Note: b2, vb, bd are updated AFTER computing the SSE for cut i, so they
         *  always reflect the i samples above the current cut (0..i-1), not 0..i.
         *
         *  Final SSE formula at each cut:
         *    den = ||h_plus||^2 - ||proj_{Bo,Bx}(h_plus)||^2  = ||h_plus_perp||^2
         *    uw  = h_plus^T * proj_{Bo,Bx}(y) - h_plus^T * y  = -(h_plus_perp^T * y_perp)
         *    sse = uw^2 / den  (extra gain from the hinge beyond the linear candidate)
         *
         *  The final hinge_dsse = linear_dsse + hinge_sse captures the combined gain
         *  from the full hinge pair (h_plus and h_minus together).
         */
        for (int j = 0; j < p; ++j) {
            ArrayXd f = ArrayXd::Zero(m+1);
            ArrayXd g = ArrayXd::Zero(m+1);
            const float  *b  = B.col(bcols[j]).data();
            const double *bx = Bx.col(j).data();

            double b_k  = b [k[0]]; // sort and upcast to double
            double bx_k = bx[k[0]];
            double y_k  = y [k[0]];
            covariates_impl<true>(f,g,Bok.row(0).data(),ybo,bx_k,ybx[0],0,b_k,m);

            double k0 = 0;
            double k1 = 0;
            double w  = 0;
            double bd = 0;
            double vb = b_k*y_k;
            double b2 = b_k*b_k;

            // Cuts are evaluated on a grid spaced by `min_span` along the sorted
            // index, anchored at `head+1` (the first eligible position). The
            // f/g/k0/k1/w/b2/vb/bd accumulators are running sums that depend on
            // every sample, so they must update every iteration regardless of
            // whether this `i` is on the grid; only the SSE reduction (o.ff,
            // o.fy) is gated, since it is only consumed at on-grid positions.
            for (int i = 1; i < tail; ++i) {
                b_k  = b [k[i]]; // sort and upcast to double
                bx_k = bx[k[i]];
                y_k  = y [k[i]];

                const bool on_grid = (i > head) && ((i - head - 1) % min_span == 0);
                cov_t o = on_grid
                    ? covariates_impl<true >(f,g,Bok.row(i).data(),ybo,bx_k,ybx[j],d[i],b_k,m)
                    : covariates_impl<false>(f,g,Bok.row(i).data(),ybo,bx_k,ybx[j],d[i],b_k,m);

                k0 = fma(d[i]*d[i],b2,k0);  // build up ||h_plus||^2 incrementally
                k1 = fma(d[i]*2,bd,k1);
                w  = fma(d[i],vb,w);        // w = h_plus^T * y
                bd = fma(d[i],b2,bd);
                b2 = fma(b_k,b_k,b2);
                vb = fma(y_k,b_k,vb);

                if (on_grid) {
                    const double uw  = o.fy - w;
                    const double den = (k0+k1) - o.ff;
                    const double sse = den > _tol? (uw*uw)/(den+_tol) : 0;
                    if (sse > hinge_sse[j]) {
                        hinge_sse[j] = sse;
                        hinge_idx[j] = i;
                    }
                }
            }
        }

        // Map the results to the output arrays
        for (int j = 0; j < p; ++j) {
            if (hinge_idx[j] >= 0) {
                hinge_dsse[bcols[j]] = linear_dsse[bcols[j]] + hinge_sse[j];
                hinge_cuts[bcols[j]] = _data->X(k[hinge_idx[j]],xcol);
            }
        }
    }

#ifdef __SSE__
    _mm_setcsr(csr); // revert
#endif
}

///////////////////////////////////////////////////////////////////////////////

double MarsAlgo::append(char type, int xcol, int bcol, float h)
{
    if (_m >= _data->B.cols()) {
        throw std::runtime_error("basis matrix is full");
    }
    if (bcol < 0 || bcol >= _m) {
        char msg[80];
        snprintf(msg, sizeof(msg), "invalid basis column number: %d", bcol);
        throw std::runtime_error(msg);
    }

    const float        s = _data->s[xcol];
    Ref<ArrayXf>       b = _data->B.col(bcol).array();
    Ref<const ArrayXf> x = _data->X.col(xcol).array();

    switch(type) {
        case 'l':
            _data->B.col(_m) = (s*b*x);
            break;
        case '+':
            _data->B.col(_m) = (s*b*(x-h).cwiseMax(0));
            break;
        case '-':
            _data->B.col(_m) = (s*b*(h-x).cwiseMax(0));
            break;
        default:
            throw std::runtime_error("invalid basis type");
    }

    // Use Gram-Schmidt orthogonalization.
    VectorXd v = _data->B.col(_m).cast<double>(); // make a copy
    _data->B.col(_m) /= v.norm();

    for (int j = 0; j < _m; ++j) {
        v -= (_data->Bo.col(j).transpose()*v) * _data->Bo.col(j);
    }

    const double w = v.norm();
    if (w*w > _tol) {
        _data->Bo.col(_m) = v/w;

        // This assumes the norm of '_data->y' == 1
        VectorXd ybo = _data->Bo.leftCols(_m+1).transpose() * _data->y.matrix();
        const double mse = (1. - ybo.squaredNorm()) / _data->X.rows();
        if (mse >= -_tol) { // gracefully handle values close to zero
            _m += 1;
            _data->ybo = ybo; // save for next iteration
            return std::max(mse,0.0);
        }
    }
    return -1.;
}
