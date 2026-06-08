#include "marsalgo.h"
#include "kernels.h"
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
 *  Fill `out` with the indexes of `true` entries in `mask` (length `n`) and
 *  return the count. `out` must have capacity for at least `n` entries.
 */
int nonzero(int *out, const bool *mask, int n)
{
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (mask[i]) {
            out[count++] = i;
        }
    }
    return count;
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
cov_t covariates_impl(Ref<ArrayXd> f_, Ref<ArrayXd> g_, const float *x, const double *y,
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
cov_t covariates(Ref<ArrayXd> f_, Ref<ArrayXd> g_, const float *x, const double *y,
                 double xm, double ym, double k0, float k1, int m)
{
    return covariates_impl<true>(f_, g_, x, y, xm, ym, k0, k1, m);
}

///////////////////////////////////////////////////////////////////////////////

struct MarsData {
    // B/Bo start at one column (the intercept) and grow one column per append()
    // so Bo's row stride tracks the live basis count; see the rationale there.
    MarsData(const float *x, int n, int m, int ldx)
        : X (x,n,m,Stride<Dynamic,1>(ldx,1))
        , B (MatrixXf ::Zero(n,1))
        , Bo(MatrixXfC::Zero(n,1)) {}

    Map<const MatrixXf,Aligned,Stride<Dynamic,1>>  X;   // read-only view of the regressors
    ArrayXf     y;      // target vector (f32 storage; dot products upcast to f64)
    MatrixXf    B;      // all basis
    MatrixXfC   Bo;     // all basis, orthonormalized (f32 storage; f64 arithmetic)
    VectorXd    ybo;    // dot product of basis Bo with Y target
    ArrayXf     s;      // normalization constant for columns of 'X'
};

/*
 *  Per-thread eval() working memory. Held in a function-local thread_local
 *  (see MarsAlgo::eval), so each OpenMP worker owns one for the lifetime of the
 *  thread and ensure() reallocates only when the problem grows. Basis-sized
 *  buffers grow geometrically, so a whole forward pass triggers O(log m)
 *  reallocations per thread -- that keeps the large-allocation mmap traffic off
 *  the hot path without the explicit pre-allocated scratch pool we used to
 *  thread through reserve_scratches()/scratch(tid).
 *
 *  All basis-dimensioned buffers are over-allocated to `cap` (>= live m); the
 *  hot loops slice leading sub-blocks via .head()/.leftCols(), so the slack is
 *  invisible. Bx (and Bo, in MarsData) are stored as f32 to halve the largest
 *  buffers; all Gram-Schmidt arithmetic in orthonormalize()/append() stays f64
 *  -- only the storage narrows. The DGKS retry in kernels.cc and append() keeps
 *  orthogonality bounded by O(eps_f32) ~ 1e-7 against Bo. Downstream:
 *    - linear_dsse: ybx = Bx^T * y accumulates in f64 so the dot products do
 *      not lose precision against the f32 inputs.
 *    - hinge sweep: bx_k = (double)Bx[k[i], j] upcasts at the load; Bo rows
 *      are gathered on the fly via Bo_data + k[i]*ldBo (ldBo == live m now) with
 *      a software prefetch a few iterations ahead.
 *
 *  TODO - narrow `d` from f64 to f32. Enters a long FMA chain in
 *  covariates_impl() over the full length of n on every hinge sweep;
 *  narrowing the input compounds roundoff across ~6M FMAs per call.
 *  Worth measuring once a hinge-heavy workload is profiled.
 */
namespace {
struct EvalScratch {
    int       n   = 0;     // row count the n-sized buffers are sized for
    int       cap = 0;     // basis capacity the m-sized buffers are sized for
    ArrayXf   x;           // normalized candidate column
    ArrayXi32 k;           // sort permutation of x
    ArrayXf   d;           // adjacent deltas of x along sort order (d[0] unused). f32 storage -- the subtraction is f32-f32 so no precision is lost vs f64; upcast to f64 at the load before the FMA chain.
    MatrixXf  Bx;          // (n, cap) column-major — basis interacted with x, ortho-normalized (f32 storage; f64 arith)
    ArrayXd   f;           // (cap+1) hinge projection accumulator
    ArrayXd   g;           // (cap+1) basis-weighted ortho accumulator
    ArrayXi   bcols;       // (cap) indexes of non-ignored basis (output of nonzero)
    VectorXd  ybx;         // (cap) Bx^T * y, leading p entries used
    ArrayXi   hinge_idx;   // (cap) best hinge sort position per j (leading p)
    ArrayXd   hinge_sse;   // (cap) best hinge delta-SSE per j (leading p)
    MatrixXd  BoTBx;       // (cap, cap) workspace for Bo^T*Bx in orthonormalize()

    // Grow (never shrink) to hold n rows and at least m basis columns. The
    // n-sized buffers reallocate only when the row count changes (a different
    // MarsAlgo reused this thread's scratch); the basis-sized buffers grow
    // geometrically so a forward pass triggers O(log m) reallocations.
    void ensure(int n_, int m)
    {
        if (n_ != n) {
            n = n_;
            x.resize(n);
            k.resize(n);
            d.resize(n);
            cap = 0;       // force the basis-sized buffers below to reallocate
        }
        if (m > cap) {
            cap = m > 2 * cap ? m : 2 * cap;
            Bx.resize(n, cap);
            f.resize(cap + 1);
            g.resize(cap + 1);
            bcols.resize(cap);
            ybx.resize(cap);
            hinge_idx.resize(cap);
            hinge_sse.resize(cap);
            BoTBx.resize(cap, cap);
        }
    }
};
} // namespace

MarsAlgo::MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
    : _data(new MarsData(x, n, m, ldx))
    , _tol((n*0.02)*DBL_EPSILON) // rough guess
{
    _max_terms = p;
    verify(!std::isfinite(NAN), "NAN check is disabled, recompile without --fast-math");

    // Build the weighted, normalized target in f64, then store it as f32 (see
    // the store below). The per-row weighting and the norm/variance reductions
    // stay in f64 for precision; only the stored target narrows. Every dot
    // product against _data->y downstream upcasts on the load, so it still
    // accumulates in f64.
    ArrayXd yd = Map<const ArrayXf>(y,n).cast<double>();

    // For WLS we scale rows by sqrt(w), so that the OLS objective on the
    // transformed problem equals the weighted RSS on the original.
    ArrayXd sqrt_w = Map<const ArrayXf>(w,n).cast<double>().sqrt();

    // Filter out NAN's and apply weight to the target 'y'.
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) {
            yd[i] = sqrt_w[i] = 0;
        }
    }
    yd *= sqrt_w; // apply sqrt(w) to target

    // TODO - these row-order reductions (y_norm, w_norm, _yvar below, and
    // the column norms in _data->s) are sensitive to the input row order:
    // structured inputs (sorted, time-correlated, grouped) accumulate biased
    // running sums and lose precision vs. shuffled inputs. The downstream
    // greedy search amplifies these tiny perturbations into different basis
    // selections. Switching to compensated/pairwise summation here would make
    // the algorithm row-order-invariant; see tests/repro_shuffle.py.
    double y_norm = yd.matrix().norm();
    double w_norm = sqrt_w.matrix().norm();
    verify(y_norm > 0.0 && w_norm > 0, "target Y is all zero or NANs");

    yd     /= y_norm;
    sqrt_w /= w_norm;

    // Store the normalized target as f32. This halves the footprint of the
    // random y[k[i]] gather in the eval() hinge sweep; every dot product
    // against it upcasts on the load so the accumulation stays f64.
    _data->y = yd.cast<float>();

    // Initialize the first column of our basis with the intercept.
    // Bo storage is f32 (sqrt_w computed in f64, downcast on store); ybo
    // is computed in f64 with both factors lazily upcast.
    _data->B .col(0) = sqrt_w.cast<float>();
    _data->Bo.col(0) = sqrt_w.cast<float>();
    _data->ybo = _data->Bo.leftCols(1).cast<double>().transpose() * _data->y.matrix().cast<double>();

    // Calculate the sample variance of the target 'y' (f64 reduction).
    _yvar = (yd - yd.mean()).square().mean();

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
int MarsAlgo::nrows() const
{
    return _data->X.rows();
}
double MarsAlgo::dsse() const
{
    return _data->ybo.squaredNorm();
}
double MarsAlgo::yvar() const
{
    return _yvar;
}

///////////////////////////////////////////////////////////////////////////////

void MarsAlgo::eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
                    int xcol, const bool *bmask, int min_span, int end_span, bool linear_only)
{
    verify(xcol >= 0 && xcol < _data->X.cols(), "invalid X column index");
    verify(min_span >= 1, "min_span must be >= 1");

    std::fill_n(linear_dsse, _m, 0.0);
    std::fill_n(hinge_dsse,  _m, 0.0);
    std::fill_n(hinge_cuts,  _m, NAN);

    // Per-thread working memory, grown on demand and reused across calls. The
    // thread_local is a *pointer* with a constant initializer (no lazy-init
    // guard, no thread-exit destructor registration); the buffer is heap-
    // allocated on first touch. A non-trivial thread_local object would instead
    // be constructed the first time a thread reaches this line, and that
    // construction crashes on an OpenMP worker thread under the statically
    // linked libomp on macOS. The per-thread EvalScratch is intentionally
    // leaked -- worker threads live for the process and there are <= `threads`.
    thread_local EvalScratch *Sp = nullptr;
    if (!Sp) {
        Sp = new EvalScratch();
    }
    EvalScratch &S = *Sp;
    S.ensure(_data->X.rows(), _m);
    const int p = nonzero(S.bcols.data(), bmask, _m);
    if (p == 0) {
        return;
    }
    Ref<ArrayXi> bcols = S.bcols.head(p);

#ifdef __SSE__
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040);   // enable FTZ and DAZ
#endif

    const int n = _data->X.rows();
    const int m = _m;           // number of all currently existing basis

    S.x = _data->X.col(xcol).array() * _data->s[xcol]; // in-place into pre-allocated scratch
    const Ref<const ArrayXf> x  = S.x;
    const Ref<MatrixXfC>     Bo = _data->Bo.leftCols(m);

    /*
     *  Evaluate `B[:,bcols] * x` and ortho-normalize against `Bo`. BoTBx is
     *  the (m, p) workspace for the Bo^T*Bx intermediate; sized at
     *  (max_terms, max_terms) so the leading m×p block is what the kernel uses.
     */
    Ref<MatrixXf> Bx = S.Bx.leftCols(p);
    // `S.ybx` is reused as the per-column squared-norm scratch for
    // orthonormalize(); it is overwritten immediately below with Bx^T * y.
    Ref<VectorXd> ybx = S.ybx.head(p);
    mars::orthonormalize(
        n, m, p,
        _data->B.data(),  (int)_data->B.outerStride(),
        x.data(),
        bcols.data(),
        _data->Bo.data(), (int)_data->Bo.outerStride(),
        Bx.data(),        (int)Bx.outerStride(),
        S.BoTBx.data(),   (int)S.BoTBx.outerStride(),
        ybx.data(),
        _tol);

    // Calculate the linear delta SSE and map to the output buffer. Bx.col(j)
    // and y are both contiguous f32; mars::dot_widen upcasts each at the load
    // and accumulates in f64 (a vectorized f32->f64 dot that Eigen's
    // cast-then-redux would otherwise scalarize). See kernels.h.
    for (int j = 0; j < p; ++j) {
        ybx[j] = mars::dot_widen(Bx.col(j).data(), _data->y.data(), n);
        linear_dsse[bcols[j]] = ybx[j]*ybx[j];
    }

    // Evaluate the delta SSE on all hinge locations
    if (linear_only == false) {
        const double *ybo = _data->ybo.data(); // dot(Bo.T,_data->y);
        Ref<ArrayXi> hinge_idx = S.hinge_idx.head(p);
        hinge_idx.setConstant(-1);
        Ref<ArrayXd> hinge_sse = S.hinge_sse.head(p);
        hinge_sse.setZero();

        // Get sort indexes (into scratch)
        // TODO - we should keep a LRU cache as we usually pick from the
        //        same pool of regressors in Fast-MARS.
        int32_t *k = S.k.data();
        argsort(k, _data->X.col(xcol).data(), n);

        // Take the deltas of `x` (into scratch). Stored as f32: x is f32, so
        // the subtraction is f32-f32 anyway; the f64 store was just a wider
        // copy. Upcast happens at every read site below.
        float *d = S.d.data(); // d[0] unused; valid indices are 1..n-1
        for (int i = 1; i < n; ++i) {
            d[i] = x[k[i-1]] - x[k[i]];
        }

        const int head = end_span;
        const int tail = n-end_span;
        const MatrixXf &B = _data->B;
        const ArrayXf  &y = _data->y; // f32 storage; each y[k[i]] upcasts into double y_k below

        // Bo rows are now gathered on the fly inside the inner sweep
        // (no Bok scratch). Each iteration reads Bo.row(k[i]) at a random
        // offset; the access pattern forfeits HW prefetch, so we issue
        // software prefetch a few iterations ahead.
        const float *Bo_data = Bo.data();
        const int    ldBo    = (int)Bo.outerStride();
        constexpr int PREFETCH_DIST = 4;

        /*
         *  For each parent basis column b = B[:,bcols[j]], sweep potential
         *  hinge cut locations from largest to smallest x (descending sort
         *  order). At each cut h = x[k[i]] we evaluate the positive hinge
         *  h_plus = b*max(x-h,0), which is nonzero only for the i samples
         *  above the cut.
         *
         *  `covariates()` maintains f/g to track the projection of h_plus onto
         *  the existing ortho-basis (Bo) AND the linear candidate (Bx
         *  [:,j]). The local accumulators below build up the remaining terms
         *  needed for the delta-SSE:
         *
         *    b2   = sum_{j<i} b[k[j]]^2                  (squared norm of b above cut)
         *    vb   = sum_{j<i} b[k[j]] * y[k[j]]          (dot product <b,y> above cut)
         *    bd   = sum_{j<i} b[k[j]]^2 * (x[k[j]] - h)  (helper for k0/k1 update)
         *    k0+k1 = ||h_plus||^2                        (squared norm of positive hinge)
         *    w    = h_plus^T * y                         (dot product of hinge with target)
         *
         *  Note: b2, vb, bd are updated AFTER computing the SSE for cut i, so
         *  they always reflect the i samples above the current cut
         *  (0..i-1), not 0..i.
         *
         *  Final SSE formula at each cut:
         *    den = ||h_plus||^2 - ||proj_{Bo,Bx}(h_plus)||^2  = ||h_plus_perp||^2
         *    uw  = h_plus^T * proj_{Bo,Bx}(y) - h_plus^T * y  = -(h_plus_perp^T * y_perp)
         *    sse = uw^2 / den  (extra gain from the hinge beyond the linear candidate)
         *
         *  The final hinge_dsse = linear_dsse + hinge_sse captures the combined
         *  gain from the full hinge pair (h_plus and h_minus together).
         */
        for (int j = 0; j < p; ++j) {
            Ref<ArrayXd> f = S.f.head(m+1);
            f.setZero();
            Ref<ArrayXd> g = S.g.head(m+1);
            g.setZero();
            const float  *b  = B.col(bcols[j]).data();
            const float  *bx = Bx.col(j).data();        // f32 storage; upcast at the load

            double b_k  = b [k[0]]; // sort and upcast to double
            double bx_k = bx[k[0]];
            double y_k  = y [k[0]];
            covariates_impl<false>(f,g,Bo_data + k[0]*ldBo,ybo,bx_k,ybx[j],0,b_k,m);

            double k0 = 0;
            double k1 = 0;
            double w  = 0;
            double bd = 0;
            double vb = b_k*y_k;
            double b2 = b_k*b_k;

            /*
             *  Cuts are evaluated on a grid spaced by `min_span` along the
             *  sorted index, anchored at `head+1` (the first eligible
             *  position). The f/g/k0/k1/w/b2/vb/bd accumulators are running
             *  sums that depend on every sample, so they must update every
             *  iteration regardless of whether this `i` is on the grid; only
             *  the SSE reduction (o.ff, o.fy) is gated, since it is only
             *  consumed at on-grid positions.
             */
            for (int i = 1; i < tail; ++i) {
                // Prefetch the Bo row we'll need PREFETCH_DIST iterations
                // ahead; the HW prefetcher fills the rest of the row once
                // we touch the first cache line.
                if (i + PREFETCH_DIST < n) {
                    // __builtin_prefetch(addr, rw=0 read, locality=3 high) lowers
                    // to prefetcht0 on x86 (identical to _mm_prefetch + _MM_HINT_T0)
                    // and is portable to non-x86 (arm64) where xmmintrin.h is absent.
                    __builtin_prefetch(Bo_data + k[i + PREFETCH_DIST] * ldBo, 0, 3);
                }

                b_k  = b [k[i]]; // sort and upcast to double
                bx_k = bx[k[i]];
                y_k  = y [k[i]];
                const double di = d[i]; // upcast f32 delta once for the FMA chain

                const bool on_grid = (i > head) && ((i - head - 1) % min_span == 0);
                const float *bo_row = Bo_data + k[i]*ldBo;
                cov_t o = on_grid
                          ? covariates_impl<true >(f,g,bo_row,ybo,bx_k,ybx[j],di,b_k,m)
                          : covariates_impl<false>(f,g,bo_row,ybo,bx_k,ybx[j],di,b_k,m);

                k0 = fma(di*di,b2,k0);  // build up ||h_plus||^2 incrementally
                k1 = fma(di*2,bd,k1);
                w  = fma(di,vb,w);      // w = h_plus^T * y
                bd = fma(di,b2,bd);
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
    if (_m >= _max_terms) {
        throw std::runtime_error("basis matrix is full");
    }
    if (bcol < 0 || bcol >= _m) {
        char msg[80];
        snprintf(msg, sizeof(msg), "invalid basis column number: %d", bcol);
        throw std::runtime_error(msg);
    }

    // Grow the basis storage by exactly one column so Bo's row stride stays
    // equal to the live basis count. The eval() hinge sweep gathers Bo rows by
    // that stride (ldBo == cols), so a tight stride keeps each randomly-gathered
    // row to the minimum number of cache lines. The old allocate-once-at-
    // max_terms storage left the stride at max_terms, which made a larger
    // max_terms slower at equal basis count -- a TLB/cache sink, not just extra
    // memory. conservativeResize preserves the existing columns; the resize must
    // precede the `b` reference below, which would otherwise dangle.
    if (_data->B.cols() < _m + 1) {
        _data->B .conservativeResize(Eigen::NoChange, _m + 1);
        _data->Bo.conservativeResize(Eigen::NoChange, _m + 1);
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

    /*
     *  Gram-Schmidt with a DGKS retry. The eval() side assumes Bo^T Bo = I, so
     *  any orthogonality drift accumulates across the whole fit. See
     *  mars::DGKS_GATE_RATIO_SQ in kernels.h for the trigger rationale.
     *
     *  Bo is f32 storage; v and the dot products stay f64. Each Bo column
     *  is lazily cast to f64 on the load (no temp matrix). The final v/w
     *  is downcast to f32 on the store into Bo.col(_m).
     */
    VectorXd v = _data->B.col(_m).cast<double>(); // make a copy
    _data->B.col(_m) /= v.norm();

    double proj_norm2 = 0.0;
    for (int j = 0; j < _m; ++j) {
        const auto   bj = _data->Bo.col(j).cast<double>();   // lazy expression
        const double c  = bj.dot(v);
        v.noalias() -= c * bj;
        proj_norm2 += c * c;
    }
    const double v_norm2_post = v.squaredNorm();
    if (v_norm2_post > _tol && v_norm2_post * mars::DGKS_GATE_RATIO_SQ < proj_norm2) {
        for (int j = 0; j < _m; ++j) {
            const auto   bj = _data->Bo.col(j).cast<double>();
            const double c  = bj.dot(v);
            v.noalias() -= c * bj;
        }
    }

    const double w = v.norm();
    if (w*w > _tol) {
        _data->Bo.col(_m) = (v/w).cast<float>();

        // Extend the cached ybo with one new entry; relies on ||y|| == 1 so
        // mse = (||y||^2 - ||ybo||^2) / n collapses to (1 - ||ybo||^2) / n.
        VectorXd ybo(_m + 1);
        ybo.head(_m) = _data->ybo;
        ybo[_m] = _data->Bo.col(_m).cast<double>().dot(_data->y.matrix().cast<double>());
        const double mse = (1. - ybo.squaredNorm()) / _data->X.rows();
        if (mse >= -_tol) { // gracefully handle values close to zero
            _m += 1;
            _data->ybo = ybo; // save for next iteration
            return std::max(mse,0.0);
        }
    }
    return -1.;
}
