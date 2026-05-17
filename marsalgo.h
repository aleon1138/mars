#include <stdexcept>
#include <memory>
#include <vector>
#include <atomic>

inline void verify(bool check, const char *msg)
{
    if (!check) {
        throw std::runtime_error(msg);
    }
}

/*
 *  Per-thread scratch buffers for MarsAlgo::eval(). Production callers should
 *  use MarsAlgo::reserve_scratches() + MarsAlgo::scratch(tid), which pools
 *  these for the lifetime of the MarsAlgo so the OMP-parallel eval path never
 *  hits glibc's mmap-based large-allocation route -- 64 threads hammering
 *  mmap/munmap on the same process serializes on the kernel's mmap_lock and
 *  stalls threads in state D. Constructible directly for unit tests.
 */
class MarsScratch {
public:
    MarsScratch(int n, int max_terms);
    ~MarsScratch();
    MarsScratch(const MarsScratch &) = delete;
    MarsScratch &operator=(const MarsScratch &) = delete;
private:
    struct Impl;
    Impl *_impl;
    friend class MarsAlgo;
};

class MarsAlgo {
    struct MarsData *_data = nullptr;
    int     _m    = 1;  // number of basis found
    double  _yvar = 0;  // variance of 'y'
    double  _tol  = 0;  // numerical error tolerance
    std::vector<std::unique_ptr<MarsScratch>> _scratches;
    std::atomic<long> _dgks_count{0}; // DGKS re-orth triggers since last consume()

public:
    /*
     *  Construct a MarsAlgo over training data (x, y) with per-row weights w.
     *  Weights follow inverse-variance semantics: the fitted objective is
     *  Sum_i w_i * (y_i - f(x_i))^2. For heteroskedastic fitting pass:
     *
     *    w_i = 1/sigma_i^2.
     *
     *  Uniform weights (w=1) recover ordinary least squares.
     */
    MarsAlgo(const float *x, const float *y, const float *w, int n, int m, int p, int ldx);
    ~MarsAlgo();

    int nbasis() const;
    int nrows() const;
    int max_basis() const;
    double dsse() const;
    double yvar() const;

    /*
     *  Ensure at least `threads` per-thread scratch buffers exist. Idempotent;
     *  call from the main thread before entering a parallel region. Allocations
     *  happen serially here so they don't contend on the kernel's mmap_lock.
     */
    void reserve_scratches(int threads);

    /*
     *  Return the scratch buffer for thread `tid`. `reserve_scratches(n)` with
     *  n > tid must have been called previously.
     */
    MarsScratch &scratch(int tid);

    /*
     *  Returns the delta SSE (sum of squared errors) given the existing basis
     *  set and a candidate column of `X` to evaluate.
     *
     *  linear_dsse : double(m) [out]
     *      delta SSE from adding a linear term for each parent basis.
     *
     *  hinge_dsse : double(m) [out]
     *      delta SSE from adding a pair of mirror hinges for each parent basis.
     *
     *  hinge_cuts : double(m) [out]
     *      optimal hinge cut point for each parent basis (NaN if no hinge).
     *
     *  xcol : int
     *      which column of the training `X` data to use.
     *
     *  bmask : bool(m)
     *      a boolean mask to filter out which bases to use.
     *
     *  min_span : int
     *      minimum gap between candidate hinge cut locations along sorted X.
     *      Cuts are evaluated on a grid of every `min_span`-th sample in the
     *      sorted order. Pass 1 to evaluate every sample.
     *
     *  end_span : int
     *      how many samples to ignore from both the extreme ends of the
     *      training data.
     *
     *  linear_only : bool
     *      do not attempt to find any hinge cuts, only build a linear model.
     *      This will ignore the output values of `hinge_dsse` and `hinge_cuts`.
     */
    void eval(double *linear_dsse, double *hinge_dsse, double *hinge_cuts,
              int xcol, const bool *bmask, int min_span, int end_span, bool linear_only,
              MarsScratch &scratch);

    /*
     *  Append a new basis function and update the orthonormalized state.
     *  Returns the MSE after adding the basis, or -1 if the basis is
     *  numerically degenerate.
     *
     *  type : char
     *      'l' for linear, '+' for positive hinge, '-' for negative hinge.
     *
     *  xcol : int
     *      column index into the training `X` data.
     *
     *  bcol : int
     *      index of the parent basis to interact with.
     *
     *  h : float
     *      hinge cut point (ignored for linear basis).
     */
    double append(char type, int xcol, int bcol, float h);

    /*
     *  Return the number of DGKS re-orthogonalization triggers accumulated
     *  in this instance since the last call, and reset the counter to zero.
     *  Bumped by append() and (via &_dgks_count passed to orthonormalize())
     *  by eval(). Atomic so the eval() OMP workers may race-free increment.
     */
    long dgks_consume();
};
