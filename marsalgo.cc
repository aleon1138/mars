#include "marsalgo.h"
#include "kernels.h"
#ifdef MARS_USE_CUDA
#   include "cuda/mars_cuda.h"
#endif
#include <cstdint>
#include <cstring>          // for std::memcpy
#include <cfloat>           // for DBL_EPSILON, FLT_EPSILON
#include <cmath>            // for std::sqrt, std::isfinite
#include <vector>
#include <algorithm>        // for std::fill_n, std::max, std::min
#include <cstdio>           // for snprintf
#include <cstddef>          // for size_t
#ifdef __SSE__
#   include <immintrin.h>   // for _mm_getcsr
#endif

/*
 *  The scalar accumulators below use fma() for its single rounding. Without a
 *  hardware FMA (FP_FAST_FMA undefined) std::fma is a slow library call, so
 *  fall back to the plain expression, which the compiler may still contract.
 */
#ifndef FP_FAST_FMA
#   undef fma
inline double fma(double x, double y, double z)
{
    return x * y + z;
}
#endif

/*
 *  Flush denormals to zero for the duration of eval(): the sweep accumulators
 *  never need them and they are slow. Restored on every exit path, including
 *  exceptions.
 */
struct FtzDazScope {
#ifdef __SSE__
    const unsigned csr;
    FtzDazScope() : csr(_mm_getcsr())
    {
        _mm_setcsr(csr | 0x8040);
    }
    ~FtzDazScope()
    {
        _mm_setcsr(csr);
    }
#endif
};

/*
 *  Return sort indexes in stable descending order (ties keep ascending row
 *  order), as a comparison sort on `v[i] > v[j]` would. Implemented as an LSD
 *  radix sort on the float bit pattern: three 11-bit passes over packed
 *  (key, index) words, O(n) with sequential access, versus an indirect
 *  std::stable_sort whose comparisons gather v at random. -0.0 is folded into
 *  +0.0 so the two compare equal as they do for floats. `v` must be free of
 *  NaN (the constructor verifies X is finite). Scratch is function-local and
 *  thread-local, so this is safe from the OpenMP eval() loop.
 */
void argsort(int32_t *idx, const float *v, size_t n)
{
    thread_local std::vector<uint64_t> a, b;
    a.resize(n);
    b.resize(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, v + i, sizeof u);
        if (u == 0x80000000u) {
            u = 0;
        }
        const uint32_t asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        a[i] = ((uint64_t)(~asc) << 32) | (uint32_t)i;
    }
    uint64_t *src = a.data(), *dst = b.data();
    for (int shift = 32; shift < 64; shift += 11) {
        size_t count[2048] = {};
        for (size_t i = 0; i < n; ++i) {
            ++count[(src[i] >> shift) & 2047];
        }
        size_t sum = 0;
        bool single = false;
        for (size_t &c : count) {
            single |= (c == n);
            const size_t t = c;
            c = sum;
            sum += t;
        }
        if (single) {
            continue;   // every key shares this digit: the pass is the identity
        }
        for (size_t i = 0; i < n; ++i) {
            dst[count[(src[i] >> shift) & 2047]++] = src[i];
        }
        std::swap(src, dst);
    }
    for (size_t i = 0; i < n; ++i) {
        idx[i] = (int32_t)(uint32_t)src[i];
    }
}

/*
 *  Return the indexes (and count) of true entries in `mask` (length `n`).
 *  `out` must have capacity for at least `n` entries.
 */
size_t nonzero(int *out, const bool *mask, size_t n)
{
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (mask[i]) {
            out[count++] = i;
        }
    }
    return count;
}

/*
 *  Exact count of nonzero entries in a stored basis column.
 */
int64_t count_nonzero(const float *b, size_t n)
{
    int64_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        cnt += (b[i] != 0.0f);
    }
    return cnt;
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
__attribute__((always_inline)) inline
cov_t covariates_impl(double *f, double *g, const float *x, const double *y,
                      double xm, double ym, double k0, float k1, size_t m)
{
    /*
     *  f/g/y use unaligned loads/stores below, so no alignment is required of
     *  the caller. On Haswell+ loadu/storeu on aligned data is as fast as the
     *  aligned forms, and dropping the requirement lets callers hand us plain
     *  (16-byte) std::vector storage instead of over-aligned buffers.
     */
    cov_t o = {0,0};

#ifndef __AVX__
    size_t m0 = 0;
#else
    __m256d K0 = _mm256_set1_pd(k0);
    __m256d K1 = _mm256_set1_pd(k1);
    __m256d S0 = _mm256_setzero_pd();
    __m256d S1 = _mm256_setzero_pd();

    size_t m0 = m - m%4;
    for (size_t i = 0; i < m0; i+=4) {
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

    for (size_t i = m0; i < m; ++i) {
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
 *  Non-template, out-of-line wrapper for the unit tests. The sweep calls the
 *  template directly (and forces it inline: it runs once per sorted row and
 *  basis column, ~25 vector iterations at m=100, so the 9-argument call and
 *  struct return are a measurable share).
 */
cov_t covariates(double *f, double *g, const float *x, const double *y,
                 double xm, double ym, double k0, float k1, size_t m)
{
    return covariates_impl<true>(f, g, x, y, xm, ym, k0, k1, m);
}

/*
 *  Re-stride a row-major (n x old_stride) matrix in place to a larger row
 *  stride `new_stride`, within a buffer of capacity >= n*new_stride. All
 *  old_stride existing columns of each row are preserved; the widened tail is
 *  left for the caller. As the stride grows the rows spread apart, so repack
 *  back-to-front (high row first, high column first within a row). The `i-- > 0`
 *  idiom walks a size_t counter down to 0 without an unsigned-wrap landmine and
 *  without a cast. Cold path: with geometric stride growth this runs
 *  O(log max_terms) times across a fit, not once per append.
 */
void restride_rows(float *X, size_t n, size_t old_stride, size_t new_stride)
{
    for (size_t i = n; i-- > 0; ) {
        const float *src = X + i * old_stride;
        float       *dst = X + i * new_stride;
        for (size_t j = old_stride; j-- > 0; ) {
            dst[j] = src[j];
        }
    }
}

struct MarsData {
    MarsData(const float *x, size_t n, size_t m, size_t ldx, size_t max_terms)
        : n(n), p(m), ldX(ldx), X(x)
        , B (n*max_terms, 0.0f)
        , Bo(n*max_terms, 0.0f)
        , nnz(max_terms, 0) {}

    const size_t n;             // rows in X / length of y / rows of B, Bo
    const size_t p;             // columns in X (candidate regressors)
    const size_t ldX;           // X column stride (column j starts at X + j*ldX)
    const float *X;             // read-only column-major view of the regressors
    size_t bo_stride = 1;       // Bo row stride (capacity >= live count _m); grows geometrically
    std::vector<float>  y;      // target vector (f32 storage; dot products upcast to f64)
    std::vector<float>  B;      // all basis (col-major, n x max_terms preallocated; col stride n)
    std::vector<float>  Bo;     // orthonormalized basis (ROW-major; row stride == bo_stride)
    std::vector<double> ybo;    // dot product of basis Bo with Y target
    std::vector<float>  s;      // normalization constant for columns of 'X' (1/rms)
    std::vector<int64_t> nnz;   // exact nonzero count of each stored B column (support size)
};

/*
 *  Basis columns swept per gathered Bo row in the hinge sweep (see eval()).
 *  Each column carries 2*(m+1) f64 accumulators, so JB=4 keeps the working set
 *  (~16 KB at m=250, plus the 1 KB Bo row) inside L1 while cutting the random
 *  Bo-row traffic by 4x.
 */
constexpr size_t JB = 4;

/*
 *  Relative floor on the hinge-sweep denominator den = ||h+||^2 - ||proj h+||^2
 *  (the energy of the candidate hinge outside span(Bo, Bx)). den is a
 *  difference of two O(||h+||^2) sums, so for a hinge that lies in the span it
 *  comes out as noise of either sign at ~||Bo^T Bo - I||*||h+||^2 -- about
 *  eps_f32/sqrt(n) of ||h+||^2 for the f32-stored Bo -- while the matching
 *  numerator uw = -<h+_perp, y> is noise of the same origin, so sse = uw^2/den
 *  can land anywhere up to ||y_perp||^2 (observed: spurious cuts at
 *  den/||h+||^2 ~ 1e-11 scoring 1e-2, above every legitimate cut). The
 *  absolute _tol cannot catch this (it scales with n, the noise floor shrinks
 *  with n). Require instead that at least this fraction of the hinge's energy
 *  be new: below it the candidate is a near-duplicate of terms already in the
 *  model whose coefficient would be O(1/sqrt(den)), and its gain cannot be
 *  resolved in f32 anyway. 8*eps_f32 ~ 1e-6 sits ~4 decades above the floor
 *  and well below any cut min_span would admit.
 */
constexpr double HINGE_DEN_REL = 8.0 * FLT_EPSILON;

inline double *align64(double *p)
{
    return reinterpret_cast<double *>((reinterpret_cast<uintptr_t>(p) + 63) & ~uintptr_t(63));
}

/*
 *  Per-thread working memory, function-local and thread-local inside `eval()`,
 *  so each thread owns one for its lifetime and the ensure*() calls reallocate
 *  only when the problem grows. Basis-sized buffers grow geometrically, so a
 *  full forward pass triggers O(log m) reallocations per thread.
 *
 *  Two capacities: `cap` (>= live m) sizes the small per-basis buffers; `pcap`
 *  (>= active columns p, <= cap) sizes the n-row Bx workspace and the Bo^T*Bx
 *  intermediate. Keeping Bx at p rather than m matters under Fast-MARS, where
 *  p (the active parents) is a small fraction of m and Bx is by far the largest
 *  per-thread buffer (n * pcap floats). The hot loops slice leading sub-blocks,
 *  so the slack is invisible.
 *
 *  Bx (and Bo, in MarsData) are stored as f32 but all Gram-Schmidt arithmetic
 *  in `orthonormalize()`/`append()` is f64:
 *    - DGKS retry (kernels.cc) and `append()`: orthogonality is bounded by
 *      eps(f32) (~1e-7) against Bo.
 *    - linear_dsse: ybx = Bx^T * y accumulates in f64.
 *    - hinge sweep: bx_k = (double)Bx[k[i], j] upcasts at the load; Bo rows
 *      are gathered on the fly via Bo_data + k[i]*ldBo with a software
 *      prefetch a few iterations ahead.
 */
struct Scratch {
    size_t n    = 0;               // row count the n-sized buffers are sized for
    size_t cap  = 0;               // basis capacity of the m-sized buffers
    size_t pcap = 0;               // column capacity of Bx / BoTBx (<= cap)
    size_t fg_stride = 0;          // row stride of f/g: cap+1 rounded up to 8 doubles (64 B)
    std::vector<float>   x;        // normalized candidate column (n)
    std::vector<int32_t> k;        // sort permutation of x (n)
    std::vector<float>   d;        // adjacent deltas of x along sort order (n), d[0] unused
    std::vector<float>   Bx;       // (n, pcap) column-major (col stride n), B*x, ortho-normalized
    std::vector<double>  f_buf;    // backing store of f (JB rows + alignment slack)
    std::vector<double>  g_buf;    // backing store of g
    double *f = nullptr;           // (JB, fg_stride) hinge projection accumulators, 64 B aligned
    double *g = nullptr;           // (JB, fg_stride) basis-weighted ortho-accumulators, 64 B aligned
    std::vector<int>     bcols;    // (cap) indexes of non-ignored basis (output of nonzero)
    std::vector<double>  ybx;      // (cap) Bx^T * y, leading p entries used
    std::vector<int>     hinge_idx;// (cap) best hinge sort position per j (leading p)
    std::vector<double>  hinge_sse;// (cap) best hinge delta-SSE per j (leading p)
    std::vector<double>  BoTBx;    // (cap, pcap) column-major (col stride cap) -- Bo^T*Bx workspace

    /*
     *  Grow the n-sized buffers to `n_` rows and the basis-sized buffers to at
     *  least `m` columns. A new row count invalidates everything.
     */
    void ensure(size_t n_, size_t m)
    {
        if (n_ != n) {
            n = n_;
            x.resize(n);
            k.resize(n);
            d.resize(n);
            cap = pcap = 0;
        }
        if (m > cap) {
            cap = std::max(m, 2 * cap);
            /*
             *  f/g rows are read and written as 256-bit vectors every sweep
             *  step; keep each row 64-byte aligned so none of those accesses
             *  straddle a cache line (~20% of the inner loop otherwise).
             */
            fg_stride = (cap + 1 + 7) & ~size_t(7);
            f_buf.resize(JB * fg_stride + 8);
            g_buf.resize(JB * fg_stride + 8);
            f = align64(f_buf.data());
            g = align64(g_buf.data());
            bcols.resize(cap);
            ybx.resize(cap);
            hinge_idx.resize(cap);
            hinge_sse.resize(cap);
            BoTBx.resize(cap * pcap);
        }
    }

    /*
     *  Grow Bx / BoTBx to hold `p` candidate columns (p <= cap). Called after
     *  the mask is counted, so the n-row workspace tracks the active columns.
     */
    void ensure_cols(size_t p)
    {
        if (p > pcap) {
            pcap = std::min(std::max(p, 2 * pcap), cap);
            Bx.resize(n * pcap);
            BoTBx.resize(cap * pcap);
        }
    }
};

/*
 *  Degeneracy threshold on squared norms. Every quantity it gates -- the
 *  residual energy of a candidate column in orthonormalize()/append() and the
 *  hinge-sweep denominator -- is O(1) by construction (unit-norm basis, unit-rms
 *  x, unit-norm y), so one absolute threshold serves all three. The n-scaled
 *  term covers f64 summation error over n rows; the floor covers the f32
 *  storage of B/Bo/Bx: an exactly degenerate candidate (its parent times an
 *  input already in the model) leaves a residual of pure f32 rounding noise,
 *  ~0.5*eps_f32^2 in energy, which must be zeroed rather than normalized into a
 *  unit "basis" of noise. Below n ~ 1000 the n-scaled term alone sits under
 *  that noise (observed: a duplicate input scored ~1/n at n=300), hence the
 *  floor; above n ~ 25k the n-scaled term dominates and nothing changes.
 */
static double degeneracy_tol(size_t n)
{
    const double f64_sum  = (n * 0.02) * DBL_EPSILON;
    const double f32_floor = 8.0 * FLT_EPSILON * FLT_EPSILON;
    return std::max(f64_sum, f32_floor);
}

MarsAlgo::MarsAlgo(const float *x, const float *y, const float *w, size_t n, size_t m, size_t p, size_t ldx)
    : _data(new MarsData(x, n, m, ldx, p))
    , _tol(degeneracy_tol(n))
{
    _max_terms = p;
    verify(!std::isfinite(NAN), "NAN check is disabled, recompile without --fast-math");

    /*
     *  Build the weighted target in f64. For WLS we scale each row by sqrt(w)
     *  so the OLS objective on the transformed problem equals the weighted
     *  RSS on the original.
     */
    std::vector<double> yd(n), sqrt_w(n);
    for (size_t i = 0; i < n; ++i) {
        double yi  = (double)y[i];
        double swi = std::sqrt((double)w[i]);
        if (!std::isfinite(yi)) {
            yi = swi = 0.0;
        }
        sqrt_w[i] = swi;
        yd[i]     = yi * swi;
    }

    /*
     *  The reductions here (y_norm, w_norm, _yvar, the column norms in s) are
     *  plain f64 sums; their row-order sensitivity is ~sqrt(n)*eps_f64 and not
     *  a source of fit variation. The row-order dependence that
     *  tests/repro_shuffle.py measures comes from the hinge sweep: ties in x
     *  are broken by row index (stable sort), which moves candidate cuts.
     */
    double y_sq = 0.0, w_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        y_sq += yd[i]*yd[i];
        w_sq += sqrt_w[i]*sqrt_w[i];
    }
    const double y_norm = std::sqrt(y_sq);
    const double w_norm = std::sqrt(w_sq);
    verify(y_norm > 0.0 && w_norm > 0.0, "target Y is all zero or NANs");

    for (size_t i = 0; i < n; ++i) {
        yd[i]     /= y_norm;
        sqrt_w[i] /= w_norm;
    }

    /*
     *  Store the normalized target as f32. This halves the footprint of the
     *  random y[k[i]] gather in the eval() hinge sweep; every dot product
     *  against it upcasts on the load so the accumulation stays f64.
     */
    _data->y.resize(n);
    float *yp = _data->y.data();
    for (size_t i = 0; i < n; ++i) {
        yp[i] = (float)yd[i];
    }

    /*
     *  Initialize the first basis column (the intercept) with sqrt(w), in both
     *  B and its ortho-normalized copy Bo (both n x 1 here, so each column is
     *  contiguous). ybo is the f64 dot of the *stored* f32 intercept with the
     *  *stored* f32 target, deliberately not of the f64 originals: dsse() and
     *  the linear delta-SSE in eval() are built from the stored values, so ybo
     *  has to be too or the two would disagree at the f32 floor.
     */
    float *b0  = _data->B.data();   // col 0 of col-major B is the base
    float *bo0 = _data->Bo.data();  // col 0 of Bo is contiguous at bo_stride==1
    for (size_t i = 0; i < n; ++i) {
        b0[i] = bo0[i] = (float)sqrt_w[i];
    }
    _data->ybo.assign(1, mars::dot_widen(bo0, yp, n));
    _data->nnz[0] = count_nonzero(b0, n);

    /*
     *  Compute the sample variance of the (normalized) target 'y'.
     */
    double ymean_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ymean_sum += yd[i];
    }
    const double ymean = ymean_sum / n;
    double vsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dv = yd[i] - ymean;
        vsum += dv*dv;
    }
    _yvar = vsum / n;

    /*
     *  Per-column scale: 1/rms(X[:,j]), or 1 for a zero/constant column. The
     *  sum of squares is accumulated in f64 then narrowed to f32. Each X
     *  column is contiguous, so x + j*ldX walks it directly.
     */
    _data->s.resize(m);
    for (size_t j = 0; j < m; ++j) {
        const float *xj = x + j*ldx;
        double sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sq += (double)xj[i] * (double)xj[i];
        }
        const double rms = std::sqrt(sq / n);
        verify(std::isfinite(rms), "not all columns in X are finite");
        _data->s[j] = rms > 0.0 ? (float)(1.0 / rms) : 1.0f;
    }
}

MarsAlgo::~MarsAlgo()
{
    delete _data;
#ifdef MARS_USE_CUDA
    if (_cuda) {
        mars::cuda::context_destroy(_cuda);
    }
#endif
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
    for (double v : _data->ybo) {
        s += v * v;
    }
    return s;
}
double MarsAlgo::yvar() const
{
    return _yvar;
}

void MarsAlgo::eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
                    int xcol, const bool *bmask, int min_span, int end_span,
                    bool linear_only, bool cuda)
{
    // xcol is a signed caller-supplied index; the >= 0 guard makes the
    // (size_t) compare against the column count well-defined.
    verify(xcol >= 0 && (size_t)xcol < _data->p, "invalid X column index");
    verify(min_span >= 1, "min_span must be >= 1");

    std::fill_n(linear_dsse, _m, 0.0);
    std::fill_n(hinge_dsse,  _m, 0.0);
    std::fill_n(hinge_cuts,  _m, NAN);

    thread_local Scratch *Sp = nullptr;
    if (!Sp) {
        Sp = new Scratch();
    }
    Scratch &S = *Sp;
    S.ensure(_data->n, _m);
    const size_t p = nonzero(S.bcols.data(), bmask, _m);
    if (p == 0) {
        return;
    }
    S.ensure_cols(p);
    int *bcols = S.bcols.data();
    const size_t n = _data->n;
    FtzDazScope ftz;

    /*
     *  Scale the candidate column by its normalization constant into scratch.
     */
    const float *xc = _data->X + xcol*_data->ldX;
    const float  sx = _data->s[xcol];
    for (size_t i = 0; i < n; ++i) {
        S.x[i] = xc[i] * sx;
    }
    const float *x = S.x.data();

    /*
     *  Evaluate `B[:,bcols] * x` and ortho-normalize against `Bo`. BoTBx is
     *  the (m, p) workspace for the Bo^T*Bx intermediate, col stride S.cap.
     */
    float *Bx  = S.Bx.data();   // (n, pcap) col-major, col stride n

    /*
     *  `S.ybx` is reused as the per-column squared-norm scratch for
     *  orthonormalize(); it is overwritten immediately below with Bxᵀ * y.
     */
    double *ybx = S.ybx.data();
    if (cuda) {
#ifdef MARS_USE_CUDA
        /*
         *  GPU orthonormalize. The resident context is created lazily (so
         *  CPU-only fits never touch the GPU) and synced by column count -- B/Bo
         *  are append-only, so this uploads only columns added since the last
         *  cuda eval. T and the per-column norms stay on the device; only Bx is
         *  copied back, and the linear ΔSSE below runs on the CPU as usual.
         */
        if (!_cuda) {
            _cuda = mars::cuda::context_create(n, _max_terms, _tol);
        }
        mars::cuda::context_set_target(_cuda, _data->y.data());
        mars::cuda::context_sync_basis(_cuda, _m,
                                       _data->B.data(),  _data->n,
                                       _data->Bo.data(), _data->bo_stride);
        /*
         *  ybx = Bxᵀ*y is computed on the GPU (only p doubles come back). Bx
         *  itself is downloaded only when the hinge sweep needs it (i.e. not
         *  linear_only) -- in linear_only the n×p matrix never leaves the device.
         */
        float *Bx_out = linear_only ? nullptr : Bx;
        mars::cuda::orthonormalize(_cuda, _m, p, x, bcols, Bx_out, n, ybx);
#else
        verify(false, "marslib was built without CUDA support (configure with -DUSE_CUDA=ON)");
#endif
    }
    else {
        mars::orthonormalize(
            n, _m, p,
            _data->B.data(),  _data->n,
            x,
            bcols,
            _data->Bo.data(), _data->bo_stride,
            Bx,               n,
            S.BoTBx.data(),   S.cap,
            ybx,
            _tol);
    }

    /*
     *  Calculate the linear delta SSE and map to the output buffer. Bx.col(j)
     *  and y are both contiguous f32; mars::dot_widen upcasts each at the load
     *  and accumulates in f64. See kernels.h.
     */
    for (size_t j = 0; j < p; ++j) {
        if (!cuda) {
            ybx[j] = mars::dot_widen(Bx + j*n, _data->y.data(), n);
        }  // else: ybx was filled on the GPU by orthonormalize()
        linear_dsse[bcols[j]] = ybx[j]*ybx[j];
    }

    /* Evaluate the delta SSE on all hinge locations */
    if (linear_only == false) {
        const double *ybo = _data->ybo.data();
        int    *hinge_idx = S.hinge_idx.data();
        double *hinge_sse = S.hinge_sse.data();
        std::fill_n(hinge_idx, p, -1);
        std::fill_n(hinge_sse, p, 0.0);

        /*
         *  Sort order of the raw X column (descending, ties by row index).
         *  Scaling by s > 0 is monotone, so it is also the sort order of x.
         */
        int32_t *k = S.k.data();
        argsort(k, _data->X + xcol*_data->ldX, n);

        /*
         *  Take the deltas of `x` (into scratch).
         */
        float *d = S.d.data(); // d[0] unused; valid indices are 1..n-1
        for (size_t i = 1; i < n; ++i) {
            d[i] = x[k[i-1]] - x[k[i]];
        }

        const size_t head = end_span;
        const size_t tail = n-end_span;
        const float *B = _data->B.data(); // col-major, col stride n
        const float *y = _data->y.data(); // f32 storage; each y[k[i]] upcasts into double y_k below

        /*
         *  The sweep walks the rows in sorted order, so Bo.row(k[i]), b[k[i]],
         *  bx[k[i]] and y[k[i]] are all random gathers the HW prefetcher cannot
         *  predict; each is software-prefetched a few rows ahead. One gathered
         *  Bo row (and y, d) serves a block of JB basis columns, dividing the
         *  random Bo traffic by JB. See TODO.md for what this loop still leaves
         *  on the table.
         */
        const float *Bo_data = _data->Bo.data();
        const size_t ldBo    = _data->bo_stride;
        constexpr size_t PREFETCH_DIST = 4;
        auto prefetch_ahead = [&](size_t i, const float *const *b, const float *const *bx, size_t jn) {
            if (i + PREFETCH_DIST < n) {
                const size_t kk  = (size_t)k[i + PREFETCH_DIST];
                const float *row = Bo_data + kk*ldBo;
                for (size_t off = 0; off < _m; off += 16) {
                    __builtin_prefetch(row + off, 0, 3);
                }
                __builtin_prefetch(y + kk, 0, 3);
                for (size_t jj = 0; jj < jn; ++jj) {
                    __builtin_prefetch(b[jj]  + kk, 0, 3);
                    __builtin_prefetch(bx[jj] + kk, 0, 3);
                }
            }
        };

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
         *    w    = h_plusᵀ * y                          (dot product of hinge with target)
         *
         *  Note: b2, vb, bd are updated AFTER computing the SSE for cut i, so
         *  they always reflect the i samples above the current cut
         *  (0..i-1), not 0..i.
         *
         *  Final SSE formula at each cut:
         *    den = ||h_plus||^2 - ||proj_{Bo,Bx}(h_plus)||^2  = ||h_plus_perp||^2
         *    uw  = h_plusᵀ * proj_{Bo,Bx}(y) - h_plusᵀ * y  = -(h_plus_perpᵀ * y_perp)
         *    sse = uw^2 / den  (extra gain from the hinge beyond the linear candidate)
         *
         *  The final hinge_dsse = linear_dsse + hinge_sse captures the combined
         *  gain from the full hinge pair (h_plus and h_minus together): h_minus
         *  = h_plus - b*(x-h) lies in span(Bo, Bx[:,j], h_plus), so the pair
         *  adds exactly the linear candidate plus h_plus.
         *
         *  Cuts are evaluated on a grid spaced by `min_span` along the sorted
         *  index, anchored at `head+1` (the first eligible position). The
         *  accumulators are running sums over every sample, so they update
         *  every iteration regardless of the grid; only the SSE reduction
         *  (o.ff, o.fy) is gated, since it is only consumed at on-grid
         *  positions. `next_grid` walks the grid points so the per-sample test
         *  is an equality, not a modulo.
         */
        /*
         *  Support tracking: once every nonzero row of the parent b lies above
         *  the cut, every later hinge equals b*x - h*b on the support and zero
         *  off it -- exactly in span(Bo, Bx[:,j]) -- so den is 0 in exact
         *  arithmetic and the remaining cuts are all degenerate. Stop sweeping
         *  that column there (`done`): it is both the exact answer and, for
         *  hinge parents with small support, most of the work.
         */
        for (size_t j0 = 0; j0 < p; j0 += JB) {
            const size_t jn = std::min(JB, p - j0);
            double       *f[JB], *g[JB];
            const float  *b[JB], *bx[JB];
            double  ybxj[JB], k0[JB], k1[JB], w[JB], bd[JB], vb[JB], b2[JB];
            int64_t nz[JB], nnz[JB];
            bool    done[JB];
            size_t  n_done = 0;

            const size_t kk0 = (size_t)k[0];
            const float *bo0 = Bo_data + kk0*ldBo;
            const double y0  = y[kk0];
            for (size_t jj = 0; jj < jn; ++jj) {
                const size_t j = j0 + jj;
                f[jj]  = S.f + jj*S.fg_stride;
                g[jj]  = S.g + jj*S.fg_stride;
                std::fill_n(f[jj], _m+1, 0.0);
                std::fill_n(g[jj], _m+1, 0.0);
                b[jj]    = B  + (size_t)bcols[j]*n;
                bx[jj]   = Bx + j*n;
                ybxj[jj] = ybx[j];
                nnz[jj]  = _data->nnz[bcols[j]];
                const double b_k  = b[jj][kk0];
                const double bx_k = bx[jj][kk0];
                covariates_impl<false>(f[jj],g[jj],bo0,ybo,bx_k,ybxj[jj],0,b_k,_m);
                k0[jj] = k1[jj] = w[jj] = bd[jj] = 0;
                vb[jj] = b_k*y0;
                b2[jj] = b_k*b_k;
                nz[jj]   = (b_k != 0.0);
                done[jj] = (nz[jj] == nnz[jj]);
                n_done  += done[jj];
            }

            size_t next_grid = head + 1;
            for (size_t i = 1; i < tail && n_done < jn; ++i) {
                prefetch_ahead(i, b, bx, jn);
                const size_t kk  = (size_t)k[i];
                const double y_k = y[kk];
                const double di  = d[i]; // upcast f32 delta once for the FMA chain
                const bool on_grid = (i == next_grid);
                if (on_grid) {
                    next_grid += min_span;
                }
                const float *bo_row = Bo_data + kk*ldBo;

                for (size_t jj = 0; jj < jn; ++jj) {
                    if (done[jj]) {
                        continue;
                    }
                    const double b_k  = b[jj][kk];
                    const double bx_k = bx[jj][kk];
                    cov_t o = on_grid
                              ? covariates_impl<true >(f[jj],g[jj],bo_row,ybo,bx_k,ybxj[jj],di,b_k,_m)
                              : covariates_impl<false>(f[jj],g[jj],bo_row,ybo,bx_k,ybxj[jj],di,b_k,_m);

                    k0[jj] = fma(di*di,b2[jj],k0[jj]);  // build up ||h_plus||^2 incrementally
                    k1[jj] = fma(di*2,bd[jj],k1[jj]);
                    w[jj]  = fma(di,vb[jj],w[jj]);      // w = h_plusᵀ * y
                    bd[jj] = fma(di,b2[jj],bd[jj]);
                    b2[jj] = fma(b_k,b_k,b2[jj]);
                    vb[jj] = fma(y_k,b_k,vb[jj]);

                    if (on_grid) {
                        const double uw  = o.fy - w[jj];
                        const double h2  = k0[jj] + k1[jj];
                        const double den = h2 - o.ff;
                        const double sse = den > std::max(_tol, HINGE_DEN_REL*h2)
                                           ? (uw*uw)/(den+_tol) : 0;
                        if (sse > hinge_sse[j0+jj]) {
                            hinge_sse[j0+jj] = sse;
                            hinge_idx[j0+jj] = i;
                        }
                    }

                    nz[jj] += (b_k != 0.0);
                    if (nz[jj] == nnz[jj]) {
                        done[jj] = true;
                        ++n_done;
                    }
                }
            }
        }

        for (size_t j = 0; j < p; ++j) {
            if (hinge_idx[j] >= 0) {
                hinge_dsse[bcols[j]] = linear_dsse[bcols[j]] + hinge_sse[j];
                hinge_cuts[bcols[j]] = _data->X[xcol*_data->ldX + k[hinge_idx[j]]];
            }
        }
    }

}

void MarsAlgo::eval_batch(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
                          const bool *mask, int n_xcols, int mask_row_stride)
{
    const size_t m = _m;

    // Outputs are (n_xcols, m) row-major contiguous; zero / NaN them.
    const size_t total = (size_t)n_xcols * m;
    std::fill_n(linear_dsse, total, 0.0);
    std::fill_n(hinge_dsse,  total, 0.0);
    std::fill_n(hinge_cuts,  total, NAN);

#ifdef MARS_USE_CUDA
    const size_t n = _data->n;
    if (!_cuda) {
        _cuda = mars::cuda::context_create(n, _max_terms, _tol);
    }
    mars::cuda::context_set_target(_cuda, _data->y.data());
    mars::cuda::context_sync_basis(_cuda, m,
                                   _data->B.data(),  _data->n,
                                   _data->Bo.data(), _data->bo_stride);

    // The active X-columns this epoch (rows with >=1 candidate). Make their
    // scaled candidates resident -- uploaded once over the whole fit, not per
    // epoch.
    std::vector<int> active;
    active.reserve(n_xcols);
    for (int i = 0; i < n_xcols; ++i) {
        const bool *row = mask + (size_t)i * mask_row_stride;
        for (size_t k = 0; k < m; ++k) {
            if (row[k]) {
                active.push_back(i);
                break;
            }
        }
    }
    mars::cuda::context_sync_xcols(_cuda, _data->p, _data->X, _data->ldX,
                                   _data->s.data(), active.data(), active.size());

    const size_t cap = mars::cuda::context_batch_capacity(_cuda);
    std::vector<int>    src_xcol(cap), src_basis(cap);
    std::vector<double> ybx(cap);

    // Block the active columns so each batched call's candidate count stays within
    // the GPU scratch capacity. src_xcol holds the GLOBAL X column (into the
    // resident scaled candidates). A single X-column always fits (active count <=
    // m <= max_terms <= cap).
    size_t a0 = 0;
    while (a0 < active.size()) {
        size_t P = 0, a1 = a0;
        while (a1 < active.size()) {
            const int i = active[a1];
            const bool *row = mask + (size_t)i * mask_row_stride;
            size_t p_i = 0;
            for (size_t k = 0; k < m; ++k) {
                p_i += row[k] ? 1 : 0;
            }
            if (P > 0 && P + p_i > cap) {
                break;
            }
            for (size_t k = 0; k < m; ++k) {
                if (row[k]) {
                    src_xcol[P]  = i;
                    src_basis[P] = (int)k;
                    ++P;
                }
            }
            ++a1;
        }

        if (P > 0) {
            mars::cuda::orthonormalize_batch(_cuda, m, P,
                                             src_xcol.data(), src_basis.data(),
                                             ybx.data());
            // Scatter ybx² back, re-deriving (xcol, basis) in build order.
            size_t j = 0;
            for (size_t a = a0; a < a1; ++a) {
                const int i = active[a];
                const bool *row = mask + (size_t)i * mask_row_stride;
                double *out = linear_dsse + (size_t)i * m;
                for (size_t k = 0; k < m; ++k) {
                    if (row[k]) {
                        out[k] = ybx[j] * ybx[j];
                        ++j;
                    }
                }
            }
        }
        a0 = a1;
    }
#else
    (void)mask;
    (void)mask_row_stride;
    verify(false, "marslib was built without CUDA support (configure with -DUSE_CUDA=ON)");
#endif
}

double MarsAlgo::append(char type, int xcol, int bcol, float h)
{
    if (_m >= _max_terms) {
        throw std::runtime_error("basis matrix is full");
    }
    // bcol is a signed caller-supplied index; the >= 0 guard makes the
    // (size_t) compare against the live basis count well-defined.
    if (bcol < 0 || (size_t)bcol >= _m) {
        char msg[80];
        snprintf(msg, sizeof(msg), "invalid basis column number: %d", bcol);
        throw std::runtime_error(msg);
    }

    /*
     *  Ensure `Bo` has room for the new column _m. The `eval()` hinge sweep
     *  gathers `Bo` rows by the row stride, so we keep the stride near the live
     *  basis count (a fixed max_terms stride would spread the random gather over
     *  more pages). Rather than repack on every append (O(n*m) each, O(n*m^2)
     *  over a fit), grow the stride geometrically -- doubling, capped at
     *  max_terms -- so the in-place repack runs only O(log max_terms) times
     *  (O(n*m) total), at the cost of a stride up to 2x the live count. `B` is
     *  column-major (col stride n) and preallocated to max_terms, so it needs no
     *  resize.
     */
    if (_data->bo_stride < _m + 1) {
        const size_t new_stride =
            std::min((size_t)_max_terms, std::max(_m + 1, 2 * _data->bo_stride));
        restride_rows(_data->Bo.data(), _data->n, _data->bo_stride, new_stride);
        _data->bo_stride = new_stride;
    }

    const size_t n  = _data->n;
    const float  s  = _data->s[xcol];
    const float *bp = _data->B.data() + bcol*n;             // parent basis column
    const float *xp = _data->X        + xcol*_data->ldX;    // candidate X column
    float       *Bm = _data->B.data() + _m*n;               // new basis column

    /*
     *  Form the raw column in f32 with the same operation order eval() uses
     *  (scale x first, then multiply by the parent), so the appended linear
     *  column is bit-identical to the candidate that was scored.
     */
    switch(type) {
        case 'l':
            for (size_t i = 0; i < n; ++i) {
                Bm[i] = bp[i]*(xp[i]*s);
            }
            break;
        case '+':
            for (size_t i = 0; i < n; ++i) {
                Bm[i] = bp[i]*(std::max(xp[i]-h, 0.0f)*s);
            }
            break;
        case '-':
            for (size_t i = 0; i < n; ++i) {
                Bm[i] = bp[i]*(std::max(h-xp[i], 0.0f)*s);
            }
            break;
        default:
            throw std::runtime_error("invalid basis type");
    }

    /*
     *  Gram-Schmidt with a DGKS retry (mars::orthonormalize_col). The eval()
     *  side assumes Bo^T Bo = I, so any orthogonality drift accumulates across
     *  the whole fit. Bo is f32 storage; the residual v and the dot products
     *  stay f64. The stored f32 B column is separately normalized by its own
     *  pre-projection norm. A column with no energy at all (e.g. a hinge whose
     *  cut lies outside the parent's support) is rejected before it can be
     *  divided by zero.
     */
    std::vector<double> v(n);
    double v_raw_norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        v[i] = (double)Bm[i];
        v_raw_norm2 += v[i] * v[i];
    }
    if (!(v_raw_norm2 > _tol)) {
        return -1.;
    }
    const float v_raw_norm = (float)std::sqrt(v_raw_norm2);
    for (size_t i = 0; i < n; ++i) {
        Bm[i] /= v_raw_norm;
    }

    const double w = mars::orthonormalize_col(
                         n, _m, v.data(), _data->Bo.data(), _data->bo_stride, _tol);

    if (w*w > _tol) {
        /*
         *  orthonormalize_col left v as the unit residual; store it (f32) into
         *  Bo's new column _m (row-major, stride bo_stride).
         */
        float       *Bo   = _data->Bo.data();
        const size_t ldBo = _data->bo_stride;
        for (size_t i = 0; i < n; ++i) {
            Bo[i*ldBo + _m] = (float)v[i];
        }

        /*
         *  Extend the cached ybo with one new entry; relies on ||y|| == 1 so
         *  mse = (||y||^2 - ||ybo||^2) / n collapses to (1 - ||ybo||^2) / n.
         *  The new entry is the f64 dot of the *stored* f32 Bo column with the
         *  *stored* f32 target (each upcast on the load), for the same reason
         *  as ybo[0] in the constructor.
         */
        const float *y = _data->y.data();
        double ybo_m = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ybo_m += (double)Bo[i*ldBo + _m] * (double)y[i];
        }
        double ybo_sq = ybo_m * ybo_m;
        for (double e : _data->ybo) {
            ybo_sq += e * e;
        }
        const double mse = (1. - ybo_sq) / n;
        if (mse >= -_tol) { // gracefully handle values close to zero
            _data->nnz[_m] = count_nonzero(Bm, n);
            _m += 1;
            _data->ybo.push_back(ybo_m); // extend the cache for next iteration
            return std::max(mse,0.0);
        }
    }
    return -1.;
}
