#include "marsalgo.h"
#include "kernels.h"
#include <numeric>          // for std::iota
#include <cfloat>           // for DBL_EPSILON
#include <cmath>            // for std::sqrt, std::isfinite
#include <vector>
#include <algorithm>        // for std::fill_n, std::max, std::stable_sort
#include <cstdio>           // for snprintf
#include <cstddef>          // for size_t
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
cov_t covariates_impl(double *f, double *g, const float *x, const double *y,
                      double xm, double ym, double k0, float k1, int m)
{
    /*
     *  f/g/y use unaligned loads/stores below, so no alignment is required of
     *  the caller. On Haswell+ loadu/storeu on aligned data is as fast as the
     *  aligned forms, and dropping the requirement lets callers hand us plain
     *  (16-byte) std::vector storage instead of over-aligned buffers.
     */
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
        __m256d f0 = _mm256_loadu_pd(f+i);
        __m256d g0 = _mm256_loadu_pd(g+i);
        __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(x+i));

        f0 = _mm256_fmadd_pd(K0,g0,f0);
        g0 = _mm256_fmadd_pd(K1,x0,g0);

        if constexpr (need_sse) {
            __m256d y0 = _mm256_loadu_pd(y+i);
            S0 = _mm256_fmadd_pd(f0,f0,S0);
            S1 = _mm256_fmadd_pd(f0,y0,S1);
        }

        _mm256_storeu_pd(f+i, f0);
        _mm256_storeu_pd(g+i, g0);
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
cov_t covariates(double *f, double *g, const float *x, const double *y,
                 double xm, double ym, double k0, float k1, int m)
{
    return covariates_impl<true>(f, g, x, y, xm, ym, k0, k1, m);
}

///////////////////////////////////////////////////////////////////////////////

/*
 *  Grow a tightly-packed row-major (n x cols) f32 matrix in place to cols+1
 *  columns, within a buffer of capacity >= n*(cols+1). As the row stride grows
 *  the rows spread apart, so repack back-to-front (high row first, high column
 *  first within a row); the new last column is left for the caller to fill.
 *  This replicates Eigen conservativeResize on the row-major Bo, keeping the
 *  tight stride the eval() sweep relies on. Cold path (once per append).
 */
static void bo_grow_one_column(float *Bo, int n, int cols)
{
    for (int i = n - 1; i >= 0; --i) {
        const float *src = Bo + (size_t)i * cols;
        float       *dst = Bo + (size_t)i * (cols + 1);
        for (int j = cols - 1; j >= 0; --j) dst[j] = src[j];
    }
}

struct MarsData {
    // B/Bo start at one column (the intercept) and grow one column per append()
    // so Bo's row stride tracks the live basis count; see the rationale there.
    MarsData(const float *x, int n, int m, int ldx, int max_terms)
        : n(n), p(m), ldX(ldx), X(x)
        , B ((size_t)n*max_terms, 0.0f)
        , Bo((size_t)n*max_terms, 0.0f) {}

    const int    n;     // rows in X / length of y / rows of B, Bo
    const int    p;     // columns in X (candidate regressors)
    const int    ldX;   // X column stride (column j starts at X + j*ldX)
    const float *X;     // read-only column-major view of the regressors
    std::vector<float>  y;   // target vector (f32 storage; dot products upcast to f64)
    std::vector<float>  B;   // all basis (col-major, n x max_terms preallocated; col stride n)
    std::vector<float>  Bo;  // orthonormalized basis (ROW-major; row stride == bo_cols)
    int         bo_cols = 1; // live columns of Bo == its row stride (starts at the intercept)
    std::vector<double> ybo; // dot product of basis Bo with Y target
    std::vector<float> s; // normalization constant for columns of 'X' (1/rms)
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
    std::vector<float>   x;        // normalized candidate column (n)
    std::vector<int32_t> k;        // sort permutation of x (n)
    std::vector<float>   d;        // adjacent deltas of x along sort order (d[0] unused, n). f32 storage -- the subtraction is f32-f32 so no precision is lost vs f64; upcast to f64 at the load before the FMA chain.
    std::vector<float>   Bx;       // (n, cap) column-major (col stride n) -- basis interacted with x, ortho-normalized (f32 storage; f64 arith)
    std::vector<double>  f;        // (cap+1) hinge projection accumulator
    std::vector<double>  g;        // (cap+1) basis-weighted ortho accumulator
    std::vector<int>     bcols;    // (cap) indexes of non-ignored basis (output of nonzero)
    std::vector<double>  ybx;      // (cap) Bx^T * y, leading p entries used
    std::vector<int>     hinge_idx;// (cap) best hinge sort position per j (leading p)
    std::vector<double>  hinge_sse;// (cap) best hinge delta-SSE per j (leading p)
    std::vector<double>  BoTBx;    // (cap, cap) column-major (col stride cap) -- Bo^T*Bx workspace

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
            Bx.resize((size_t)n * cap);
            f.resize(cap + 1);
            g.resize(cap + 1);
            bcols.resize(cap);
            ybx.resize(cap);
            hinge_idx.resize(cap);
            hinge_sse.resize(cap);
            BoTBx.resize((size_t)cap * cap);
        }
    }
};
} // namespace

MarsAlgo::MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx)
    : _data(new MarsData(x, n, m, ldx, p))
    , _tol((n*0.02)*DBL_EPSILON) // rough guess
{
    _max_terms = p;
    verify(!std::isfinite(NAN), "NAN check is disabled, recompile without --fast-math");

    // Build the weighted, normalized target in f64, then store it as f32 (see
    // the store below). The per-row weighting and the norm/variance reductions
    // stay in f64 for precision; only the stored target narrows. Every dot
    // product against _data->y downstream upcasts on the load, so it still
    // accumulates in f64.
    // Build the weighted target in f64. For WLS we scale each row by sqrt(w)
    // so the OLS objective on the transformed problem equals the weighted RSS
    // on the original. A non-finite target zeros both factors for that row,
    // before the weighting multiply (matches the prior filter-then-multiply).
    std::vector<double> yd(n), sqrt_w(n);
    for (int i = 0; i < n; ++i) {
        double yi  = (double)y[i];
        double swi = std::sqrt((double)w[i]);
        if (!std::isfinite(y[i])) {
            yi = swi = 0.0;
        }
        sqrt_w[i] = swi;
        yd[i]     = yi * swi; // apply sqrt(w) to target
    }

    // TODO - these row-order reductions (y_norm, w_norm, _yvar below, and
    // the column norms in _data->s) are sensitive to the input row order:
    // structured inputs (sorted, time-correlated, grouped) accumulate biased
    // running sums and lose precision vs. shuffled inputs. The downstream
    // greedy search amplifies these tiny perturbations into different basis
    // selections. Switching to compensated/pairwise summation here would make
    // the algorithm row-order-invariant; see tests/repro_shuffle.py.
    double y_sq = 0.0, w_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        y_sq += yd[i]*yd[i];
        w_sq += sqrt_w[i]*sqrt_w[i];
    }
    const double y_norm = std::sqrt(y_sq);
    const double w_norm = std::sqrt(w_sq);
    verify(y_norm > 0.0 && w_norm > 0.0, "target Y is all zero or NANs");

    for (int i = 0; i < n; ++i) {
        yd[i]     /= y_norm;
        sqrt_w[i] /= w_norm;
    }

    // Store the normalized target as f32. This halves the footprint of the
    // random y[k[i]] gather in the eval() hinge sweep; every dot product
    // against it upcasts on the load so the accumulation stays f64.
    _data->y.resize(n);
    float *yp = _data->y.data();
    for (int i = 0; i < n; ++i) yp[i] = (float)yd[i];

    // Initialize the first basis column (the intercept) with sqrt(w), in both
    // B and its ortho-normalized copy Bo (both n x 1 here, so each column is
    // contiguous). ybo is then the f64 dot of the *stored* f32 intercept with
    // the *stored* f32 target -- each upcast on the load, matching the prior
    // Bo.col(0).cast<double>() . y.cast<double>().
    float *b0  = _data->B.data();   // col 0 of col-major B is the base
    float *bo0 = _data->Bo.data();  // col 0 of Bo is contiguous at bo_cols==1
    for (int i = 0; i < n; ++i) b0[i] = bo0[i] = (float)sqrt_w[i];

    double ybo0 = 0.0;
    for (int i = 0; i < n; ++i) ybo0 += (double)bo0[i] * (double)yp[i];
    _data->ybo.resize(1);
    _data->ybo[0] = ybo0;

    // Sample variance of the (normalized) target 'y' (f64 reduction).
    double ymean_sum = 0.0;
    for (int i = 0; i < n; ++i) ymean_sum += yd[i];
    const double ymean = ymean_sum / n;
    double vsum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dv = yd[i] - ymean;
        vsum += dv*dv;
    }
    _yvar = vsum / n;

    // Per-column scale: 1/rms(X[:,j]), or 1 for a zero/constant column. The
    // sum of squares is accumulated in f64 then narrowed to f32 -- the prior
    // Eigen path (X.colwise().squaredNorm() on a float matrix) accumulated in
    // f32; widened here on purpose (de-Eigen decision 2026-06-08) since a
    // scalar f32 sum would lose accuracy on large n. Each X column is
    // contiguous (Map inner stride 1), so x + j*ldX walks it directly.
    _data->s.resize(m);
    for (int j = 0; j < m; ++j) {
        const float *xj = x + (size_t)j*ldx;
        double sq = 0.0;
        for (int i = 0; i < n; ++i) sq += (double)xj[i] * (double)xj[i];
        const double rms = std::sqrt(sq / n);
        verify(std::isfinite(rms), "not all columns in X are finite");
        _data->s[j] = rms > 0.0 ? (float)(1.0 / rms) : 1.0f;
    }
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
    return _data->n;
}
double MarsAlgo::dsse() const
{
    double s = 0.0;
    for (double v : _data->ybo) s += v * v;
    return s;
}
double MarsAlgo::yvar() const
{
    return _yvar;
}

///////////////////////////////////////////////////////////////////////////////

void MarsAlgo::eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
                    int xcol, const bool *bmask, int min_span, int end_span, bool linear_only)
{
    verify(xcol >= 0 && xcol < _data->p, "invalid X column index");
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
    S.ensure(_data->n, _m);
    const int p = nonzero(S.bcols.data(), bmask, _m);
    if (p == 0) {
        return;
    }
    int *bcols = S.bcols.data();

#ifdef __SSE__
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040);   // enable FTZ and DAZ
#endif

    const int n = _data->n;
    const int m = _m;           // number of all currently existing basis

    // Scale the candidate column by its normalization constant into scratch.
    // Bit-identical to the prior Eigen expression: elementwise f32 multiply,
    // no FMA, S.x is already sized to exactly n (EvalScratch::ensure).
    {
        const float *xc     = _data->X + (size_t)xcol*_data->ldX;
        const float  sx     = _data->s[xcol];
        float       *sx_out = S.x.data();
        for (int i = 0; i < n; ++i) sx_out[i] = xc[i] * sx;
    }
    const float *x = S.x.data();

    /*
     *  Evaluate `B[:,bcols] * x` and ortho-normalize against `Bo`. BoTBx is
     *  the (m, p) workspace for the Bo^T*Bx intermediate; sized at
     *  (max_terms, max_terms) so the leading m×p block is what the kernel uses.
     */
    float  *Bx  = S.Bx.data();   // (n, cap) col-major, col stride n
    // `S.ybx` is reused as the per-column squared-norm scratch for
    // orthonormalize(); it is overwritten immediately below with Bx^T * y.
    double *ybx = S.ybx.data();
    mars::orthonormalize(
        n, m, p,
        _data->B.data(),  _data->n,
        x,
        bcols,
        _data->Bo.data(), _data->bo_cols,
        Bx,               n,
        S.BoTBx.data(),   S.cap,
        ybx,
        _tol);

    // Calculate the linear delta SSE and map to the output buffer. Bx.col(j)
    // and y are both contiguous f32; mars::dot_widen upcasts each at the load
    // and accumulates in f64 (a vectorized f32->f64 dot that Eigen's
    // cast-then-redux would otherwise scalarize). See kernels.h.
    for (int j = 0; j < p; ++j) {
        ybx[j] = mars::dot_widen(Bx + (size_t)j*n, _data->y.data(), n);
        linear_dsse[bcols[j]] = ybx[j]*ybx[j];
    }

    // Evaluate the delta SSE on all hinge locations
    if (linear_only == false) {
        const double *ybo = _data->ybo.data(); // dot(Bo.T,_data->y);
        int    *hinge_idx = S.hinge_idx.data();
        std::fill_n(hinge_idx, p, -1);
        double *hinge_sse = S.hinge_sse.data();
        std::fill_n(hinge_sse, p, 0.0);

        // Get sort indexes (into scratch)
        // TODO - we should keep a LRU cache as we usually pick from the
        //        same pool of regressors in Fast-MARS.
        int32_t *k = S.k.data();
        argsort(k, _data->X + (size_t)xcol*_data->ldX, n);

        // Take the deltas of `x` (into scratch). Stored as f32: x is f32, so
        // the subtraction is f32-f32 anyway; the f64 store was just a wider
        // copy. Upcast happens at every read site below.
        float *d = S.d.data(); // d[0] unused; valid indices are 1..n-1
        for (int i = 1; i < n; ++i) {
            d[i] = x[k[i-1]] - x[k[i]];
        }

        const int head = end_span;
        const int tail = n-end_span;
        const float    *B = _data->B.data(); // col-major, col stride n
        const float    *y = _data->y.data(); // f32 storage; each y[k[i]] upcasts into double y_k below

        /*
         *  Bo rows are now gathered on the fly inside the inner sweep
         *  (no Bok scratch). Each iteration reads Bo.row(k[i]) at a random
         *  offset; the access pattern forfeits HW prefetch, so we issue
         *  software prefetch a few iterations ahead.
         */
        const float *Bo_data = _data->Bo.data();
        const int    ldBo    = _data->bo_cols;
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
            double *f = S.f.data();
            std::fill_n(f, m+1, 0.0);
            double *g = S.g.data();
            std::fill_n(g, m+1, 0.0);
            const float  *b  = B + (size_t)bcols[j]*n;
            const float  *bx = Bx + (size_t)j*n;        // f32 storage; upcast at the load

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
                hinge_cuts[bcols[j]] = _data->X[(size_t)xcol*_data->ldX + k[hinge_idx[j]]];
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

    /*
     *  Grow Bo by exactly one column so its row stride stays equal to the live
     *  basis count. The eval() hinge sweep gathers Bo rows by that stride
     *  (ldBo == cols), so a tight stride keeps each randomly-gathered row to the
     *  minimum number of cache lines; a fixed max_terms stride would be a
     *  TLB/cache sink at equal basis count. B is column-major (col stride n
     *  regardless of column count) and preallocated to max_terms, so it needs no
     *  resize. conservativeResize preserves Bo's existing columns.
     */
    if (_data->bo_cols < _m + 1) {
        bo_grow_one_column(_data->Bo.data(), _data->n, _data->bo_cols);
        _data->bo_cols = _m + 1;
    }

    const int    n  = _data->n;
    const float  s  = _data->s[xcol];
    const float *bp = _data->B.data() + (size_t)bcol*n;        // parent basis column
    const float *xp = _data->X        + (size_t)xcol*_data->ldX; // candidate X column
    float       *Bm = _data->B.data() + (size_t)_m*n;          // new basis column

    // B[:,_m] = s * B[:,bcol] * f(X[:,xcol]); grouped (s*bp[i])*... to match the
    // prior Eigen array expression bit-for-bit (elementwise f32, no FMA).
    switch(type) {
        case 'l':
            for (int i = 0; i < n; ++i) Bm[i] = s*bp[i]*xp[i];
            break;
        case '+':
            for (int i = 0; i < n; ++i) Bm[i] = s*bp[i]*std::max(xp[i]-h, 0.0f);
            break;
        case '-':
            for (int i = 0; i < n; ++i) Bm[i] = s*bp[i]*std::max(h-xp[i], 0.0f);
            break;
        default:
            throw std::runtime_error("invalid basis type");
    }

    /*
     *  Gram-Schmidt with a DGKS retry (mars::orthonormalize_col). The eval()
     *  side assumes Bo^T Bo = I, so any orthogonality drift accumulates across
     *  the whole fit. Bo is f32 storage; the residual v and the dot products
     *  stay f64. The stored f32 B column is separately normalized by its own
     *  pre-projection norm (matching the prior B.col(_m) /= v.norm()).
     */
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = (double)Bm[i];

    double v_raw_norm2 = 0.0;
    for (int i = 0; i < n; ++i) v_raw_norm2 += v[i] * v[i];
    const float v_raw_norm = (float)std::sqrt(v_raw_norm2);
    for (int i = 0; i < n; ++i) Bm[i] /= v_raw_norm;

    const double w = mars::orthonormalize_col(
        n, _m, v.data(), _data->Bo.data(), _data->bo_cols, _tol);

    if (w*w > _tol) {
        // orthonormalize_col left v as the unit residual; store it (f32) into
        // Bo's new column _m (row-major, stride bo_cols).
        float    *Bo   = _data->Bo.data();
        const int ldBo = _data->bo_cols;
        for (int i = 0; i < n; ++i) Bo[(size_t)i*ldBo + _m] = (float)v[i];

        /*
         *  Extend the cached ybo with one new entry; relies on ||y|| == 1 so
         *  mse = (||y||^2 - ||ybo||^2) / n collapses to (1 - ||ybo||^2) / n.
         *  The new entry is the f64 dot of the *stored* f32 Bo column with the
         *  *stored* f32 target (each upcast on the load).
         */
        const float *y = _data->y.data();
        double ybo_m = 0.0;
        for (int i = 0; i < n; ++i) ybo_m += (double)Bo[(size_t)i*ldBo + _m] * (double)y[i];
        double ybo_sq = ybo_m * ybo_m;
        for (double e : _data->ybo) ybo_sq += e * e;
        const double mse = (1. - ybo_sq) / n;
        if (mse >= -_tol) { // gracefully handle values close to zero
            _m += 1;
            _data->ybo.push_back(ybo_m); // extend the cache for next iteration
            return std::max(mse,0.0);
        }
    }
    return -1.;
}
