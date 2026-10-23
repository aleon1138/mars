#include "marsalgo.h"
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

///////////////////////////////////////////////////////////////////////////////

MarsAlgo * new_algo(
    py::array_t<float> X_array,
    py::array_t<float> y_array,
    py::array_t<float> w_array,
    int max_terms)
{
    py::buffer_info X = X_array.request();
    py::buffer_info y = y_array.request();
    py::buffer_info w = w_array.request();

    verify(X.ndim == 2, "expected 2D array for X");
    verify(y.ndim == 1, "expected 1D array for y");
    verify(w.ndim == 1, "expected 1D array for w");
    verify(X.strides[0] == sizeof(float), "X must be column-major");
    verify(y.strides[0] == sizeof(float), "y must be column-major");
    verify(w.strides[0] == sizeof(float), "w must be column-major");
    verify(X.shape[0] == y.shape[0], "invalid dimension for y");
    verify(X.shape[0] == w.shape[0], "invalid dimension for w");

    ssize_t outer_stride = X.strides[1] / sizeof(float);
    return new MarsAlgo(static_cast<const float*>(X.ptr),
                        static_cast<const float*>(y.ptr),
                        static_cast<const float*>(w.ptr),
                        X.shape[0], X.shape[1],
                        max_terms, outer_stride);
}

///////////////////////////////////////////////////////////////////////////////

py::tuple eval(MarsAlgo &algo, py::array_t<bool> mask_array,
               int min_span, int endspan, bool linear, int threads, bool cuda)
{
    py::buffer_info mask_info = mask_array.request();
    verify(mask_info.ndim == 2, "expected 2D array for mask");
    verify(mask_info.strides[1] == sizeof(bool), "mask must be row-major");
    verify(mask_info.shape[1] == algo.nbasis(), "invalid dimension for mask");
    verify(min_span >= 1, "min_span must be >= 1");

    auto mask = mask_array.unchecked<2>();
    ssize_t mask_rows = mask.shape(0);
    ssize_t mask_cols = mask.shape(1);

    py::array_t<double> dsse1({mask_rows, mask_cols});
    py::array_t<double> dsse2({mask_rows, mask_cols});
    py::array_t<double> h_cut({mask_rows, mask_cols});
    auto dsse1_ptr = dsse1.mutable_unchecked<2>();
    auto dsse2_ptr = dsse2.mutable_unchecked<2>();
    auto h_cut_ptr = h_cut.mutable_unchecked<2>();

    int n_procs = omp_get_num_procs();
    threads = threads <= 0? std::min(64, n_procs) : std::min(threads, n_procs);
    threads = std::max<int>(1, std::min<int>(threads, mask_rows));

    /*
     *  The CUDA orthonormalize context is shared, single-stream and not
     *  thread-safe, so the GPU path runs from a single host thread (the GPU
     *  itself provides the parallelism within each eval()).
     */
    if (cuda) {
        threads = 1;
    }

    /*
     *  Batched GPU path for the linear_only regime: one set of blocked GEMMs for
     *  ALL X-columns instead of one eval() per column. Fills the whole output in
     *  a single call (no OpenMP loop). Runs outside the OMP region, so a CUDA
     *  error just propagates as a normal exception.
     */
    if (cuda && linear) {
        const bool *mask_base = static_cast<const bool *>(mask_info.ptr);
        const int   mask_rs   = (int)(mask_info.strides[0] / sizeof(bool));
        {
            py::gil_scoped_release gil_r;
            algo.eval_batch(dsse1.mutable_data(), dsse2.mutable_data(),
                            h_cut.mutable_data(), mask_base, (int)mask_rows, mask_rs);
        }
        return py::make_tuple(algo.dsse(), dsse1, dsse2, h_cut);
    }

    /*
     * `mask` is a (r,c) boolean matrix where `r` is the number of columns of
     * the design matrix `X` and `c` is the number of basis already chosen by
     * the algorithm.
     */
    std::atomic<bool> ok{true};
    std::string err;  // first C++ error message, if any (guarded by `critical`)
    {
        py::gil_scoped_release gil_r;
        #pragma omp parallel num_threads(threads)
        {
            char name[16]; // Give this thread a name so it shows up in `htop`
            snprintf(name, sizeof(name), "mars-%02d", omp_get_thread_num());
#ifdef __APPLE__
            pthread_setname_np(name);
#else
            pthread_setname_np(pthread_self(), name);
#endif

            /*
             *  eval() uses a function-local thread_local scratch, grown on
             *  demand and reused across calls -- no caller-supplied buffer.
             *  C++ exceptions (e.g. a CUDA error) must not escape the OpenMP
             *  region, so catch them here and surface via `ok`/`err`.
             */
            #pragma omp for schedule(static)
            for (int i = 0; i < mask_rows; ++i) {
                if (ok.load(std::memory_order_relaxed)) {
                    try {
                        algo.eval(&dsse1_ptr(i,0), &dsse2_ptr(i,0), &h_cut_ptr(i,0),
                                  i, &mask(i,0), min_span, endspan, linear, cuda);
                    }
                    catch (const std::exception &e) {
                        #pragma omp critical
                        { if (err.empty()) err = e.what(); }
                        ok.store(false, std::memory_order_relaxed);
                    }
                }

                if (i % 32 == 0) {
                    py::gil_scoped_acquire gil_a;
                    if (PyErr_CheckSignals() != 0) { // check for CRTL-C
                        ok.store(false, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    if (!ok) {
        if (!err.empty()) {
            throw std::runtime_error(err);
        }
        throw py::error_already_set();
    }

    return py::make_tuple(algo.dsse(), dsse1, dsse2, h_cut);
}

///////////////////////////////////////////////////////////////////////////////

/*
 *  One basis column over one row block: out = par * f(x), where f is the
 *  node's factor (identity, or one of the two hinges). `so` is the row stride
 *  of B and `sx` that of X, in elements. When both are 1 (column-major B and
 *  X) the loop is contiguous and vectorizes; otherwise it is a strided scalar
 *  loop (a C-ordered X; a C-ordered B is formed in a column-major tile and
 *  transposed out, see transpose_tile()).
 */
template <typename T, typename F>
static inline void expand_col(T *__restrict__ out, const T *__restrict__ par,
                              const T *__restrict__ x, py::ssize_t nb,
                              py::ssize_t so, py::ssize_t sx, F f)
{
    if (so == 1 && sx == 1) {
        for (py::ssize_t k = 0; k < nb; ++k) {
            out[k] = par[k] * f(x[k]);
        }
    }
    else {
        for (py::ssize_t k = 0; k < nb; ++k) {
            out[k*so] = par[k*so] * f(x[k*sx]);
        }
    }
}

/*
 *  Scatter a column-major (nb, m) tile (column stride ldt) into rows of a
 *  row-major B (row stride ldb). Done in 16x16 sub-blocks so the 16 source
 *  and 16 destination lines stay in L1 while each destination line is
 *  written in full; the inner loop stores contiguously.
 */
template <typename T>
static inline void transpose_tile(T *__restrict__ B, py::ssize_t ldb,
                                  const T *__restrict__ tile, py::ssize_t ldt,
                                  py::ssize_t nb, py::ssize_t m)
{
    constexpr py::ssize_t TB = 16;
    for (py::ssize_t k0 = 0; k0 < nb; k0 += TB) {
        const py::ssize_t k1 = std::min(k0 + TB, nb);
        for (py::ssize_t i0 = 0; i0 < m; i0 += TB) {
            const py::ssize_t i1 = std::min(i0 + TB, m);
            for (py::ssize_t k = k0; k < k1; ++k) {
                T *dst = B + k*ldb;
                for (py::ssize_t i = i0; i < i1; ++i) {
                    dst[i] = tile[k + i*ldt];
                }
            }
        }
    }
}

/*
 *  Fill B (n, m) with the basis functions of a model over X (n, p). Node i is
 *
 *      'i'  B[:,i] = 1
 *      'l'  B[:,i] = B[:,basis[i]] * X[:,input[i]]
 *      '+'  B[:,i] = B[:,basis[i]] * max(X[:,input[i]] - hinge[i], 0)
 *      '-'  B[:,i] = B[:,basis[i]] * max(hinge[i] - X[:,input[i]], 0)
 *
 *  with basis[i] < i, so one pass over the nodes in index order resolves every
 *  dependency. The work is split over ROW BLOCKS, not columns: a block walks
 *  all m nodes while the block of every parent column is still in L2, so a
 *  parent is never re-read from memory and the DRAM traffic is one read of the
 *  X columns the model uses plus one write of B. Blocks are independent, which
 *  is what the OpenMP loop parallelizes -- the interactions need no second
 *  pass and no synchronization. The block is sized so all m columns of it stay
 *  around 256 KB.
 *
 *  Strides are arbitrary (element units), so X may be C- or F-ordered and B is
 *  written in whichever order the caller allocated it. A column-major B is
 *  filled in place, a contiguous (vectorized) column segment at a time. A
 *  row-major B is computed the same way into a per-thread column-major tile
 *  and then transposed out (transpose_tile): computing straight into the
 *  strided rows was measured ~3x slower per thread, whether walked by column
 *  (a cache-line fill per element for the parent load and the store) or by
 *  row (scalar, ~4 cycles per element). The arithmetic is in the storage type
 *  T with the cut rounded to T first -- the same f32 operations append() uses
 *  to form the column that fit() scored (there the column is also scaled,
 *  which does not change the shape of the term).
 */
template <typename T>
static void expand_impl(const py::buffer_info &X, const py::buffer_info &B,
                        const uint8_t *type, const int *basis, const int *input,
                        const double *hinge, int threads)
{
    const py::ssize_t n   = X.shape[0];
    const py::ssize_t m   = B.shape[1];
    const py::ssize_t isz = (py::ssize_t)sizeof(T);
    verify(X.strides[0] % isz == 0 && X.strides[1] % isz == 0 &&
           B.strides[0] % isz == 0 && B.strides[1] % isz == 0,
           "X and B strides must be multiples of the item size");
    const py::ssize_t xs0 = X.strides[0] / isz, xs1 = X.strides[1] / isz;
    const py::ssize_t bs0 = B.strides[0] / isz, bs1 = B.strides[1] / isz;
    const T *Xp = static_cast<const T *>(X.ptr);
    T       *Bp = static_cast<T *>(B.ptr);

    /*
     *  Rows per block: all m columns of a block in ~256 KB, in whole 64-byte
     *  f32 lines, between 64 rows (large m) and 8192 (small m).
     */
    py::ssize_t NB = (py::ssize_t)262144 / (std::max<py::ssize_t>(m, 1) * isz);
    NB = std::clamp<py::ssize_t>(NB & ~py::ssize_t(15), 64, 8192);
    const py::ssize_t nblocks = (n + NB - 1) / NB;

    const int n_procs = omp_get_num_procs();
    threads = threads <= 0 ? std::min(64, n_procs) : std::min(threads, n_procs);
    threads = (int)std::max<py::ssize_t>(1, std::min<py::ssize_t>(threads, nblocks));

    std::vector<T> hT(m);   // cuts rounded to the storage type, once
    for (py::ssize_t i = 0; i < m; ++i) {
        hT[i] = (T)hinge[i];
    }

    std::atomic<bool> ok{true};
    {
        py::gil_scoped_release gil_r;
        #pragma omp parallel num_threads(threads)
        {
            char name[16]; // Give this thread a name so it shows up in `htop`
            snprintf(name, sizeof(name), "mars-%02d", omp_get_thread_num());
#ifdef __APPLE__
            pthread_setname_np(name);
#else
            pthread_setname_np(pthread_self(), name);
#endif
            /*
             *  Row-major B is computed into this column-major (NB, m) tile
             *  and transposed out per block; column-major B is computed in
             *  place. `so`/`s1` are the row/column strides of the target.
             */
            const bool via_tile = (bs1 == 1);
            std::vector<T> tile(via_tile ? NB * m : 0);
            const py::ssize_t so = via_tile ? 1  : bs0;
            const py::ssize_t s1 = via_tile ? NB : bs1;

            #pragma omp for schedule(static)
            for (py::ssize_t blk = 0; blk < nblocks; ++blk) {
                if (!ok.load(std::memory_order_relaxed)) {
                    continue;
                }
                const py::ssize_t r0 = blk * NB;
                const py::ssize_t nb = std::min(NB, n - r0);
                T *base = via_tile ? tile.data() : Bp + r0*bs0;
                for (py::ssize_t i = 0; i < m; ++i) {
                    T *out = base + i*s1;
                    if (type[i] == 'i') {
                        for (py::ssize_t k = 0; k < nb; ++k) {
                            out[k*so] = T(1);
                        }
                        continue;
                    }
                    const T *par = base + basis[i]*s1;
                    const T *x   = Xp + r0*xs0 + input[i]*xs1;
                    const T  h   = hT[i];
                    switch (type[i]) {
                        case 'l':
                            expand_col(out, par, x, nb, so, xs0,
                                       [](T v) { return v; });
                            break;
                        case '+':
                            expand_col(out, par, x, nb, so, xs0,
                                       [h](T v) { return std::max(v - h, T(0)); });
                            break;
                        default: // '-'
                            expand_col(out, par, x, nb, so, xs0,
                                       [h](T v) { return std::max(h - v, T(0)); });
                            break;
                    }
                }
                if (via_tile) {
                    transpose_tile(Bp + r0*bs0, bs0, tile.data(), NB, nb, m);
                }
                /*
                 *  Ctrl-C: only the main thread runs Python signal handlers,
                 *  and under the static schedule it owns the leading blocks,
                 *  so poll from there every 256 blocks.
                 */
                if (omp_get_thread_num() == 0 && (blk & 255) == 0) {
                    py::gil_scoped_acquire gil_a;
                    if (PyErr_CheckSignals() != 0) {
                        ok.store(false, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
    if (!ok) {
        throw py::error_already_set();
    }
}

/*
 *  Python entry point: B[:] = basis functions of the model over X. B is
 *  allocated by the caller in the desired order (C or F) and X's dtype (f32 or
 *  f64); type/basis/input/hinge are the model columns. See expand_impl().
 */
void expand(py::array X_array, py::array B_array,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> type,
            py::array_t<int,     py::array::c_style | py::array::forcecast> basis,
            py::array_t<int,     py::array::c_style | py::array::forcecast> input,
            py::array_t<double,  py::array::c_style | py::array::forcecast> hinge,
            int threads)
{
    py::buffer_info X = X_array.request();
    py::buffer_info B = B_array.request(true);
    verify(X.ndim == 2, "expected 2D array for X");
    verify(B.ndim == 2, "expected 2D array for B");
    verify(B.shape[0] == X.shape[0], "B must have one row per row of X");

    const py::ssize_t m = B.shape[1];
    const py::ssize_t p = X.shape[1];
    verify(type.ndim()  == 1 && type.shape(0)  == m &&
           basis.ndim() == 1 && basis.shape(0) == m &&
           input.ndim() == 1 && input.shape(0) == m &&
           hinge.ndim() == 1 && hinge.shape(0) == m,
           "model arrays must have one entry per column of B");

    /*
     *  Validate the model up front so the parallel region can index freely.
     */
    const uint8_t *tp = type.data();
    const int     *bp = basis.data();
    const int     *ip = input.data();
    for (py::ssize_t i = 0; i < m; ++i) {
        if (tp[i] == 'i') {
            continue;
        }
        verify(tp[i] == 'l' || tp[i] == '+' || tp[i] == '-', "invalid basis type in model");
        verify(bp[i] >= 0 && bp[i] < i, "model parent index must precede the node");
        verify(ip[i] >= 0 && ip[i] < p, "model input index out of range for X");
    }

    if (py::isinstance<py::array_t<float>>(X_array)) {
        verify(py::isinstance<py::array_t<float>>(B_array), "B must have the dtype of X");
        expand_impl<float>(X, B, tp, bp, ip, hinge.data(), threads);
    }
    else if (py::isinstance<py::array_t<double>>(X_array)) {
        verify(py::isinstance<py::array_t<double>>(B_array), "B must have the dtype of X");
        expand_impl<double>(X, B, tp, bp, ip, hinge.data(), threads);
    }
    else {
        verify(false, "X must be float32 or float64");
    }
}

///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(marslib, m)
{
    py::options options;
    options.disable_function_signatures();

    m.doc() = "Multivariate Adaptive Regression Splines";
    m.attr("__version__") = "0.13";

    py::class_<MarsAlgo>(m, "MarsAlgo")
    .def(py::init(&new_algo)
         , py::arg("X").noconvert()
         , py::arg("y").noconvert()
         , py::arg("w").noconvert()
         , py::arg("max_terms")
         , py::keep_alive<1, 2>()   // keep `X` in scope
        )
    .def("eval",&eval
         , py::arg("mask").noconvert()
         , py::arg("min_span")
         , py::arg("endspan")
         , py::arg("linear_only")
         , py::arg("threads")
         , py::arg("cuda") = false
        )
    .def("nbasis",       &MarsAlgo::nbasis)
    .def("yvar",         &MarsAlgo::yvar)
    .def("append",       &MarsAlgo::append)
    ;

    m.def("expand", &expand
          , py::arg("X").noconvert()
          , py::arg("B").noconvert()
          , py::arg("type")
          , py::arg("basis")
          , py::arg("input")
          , py::arg("hinge")
          , py::arg("threads") = -1
         );
}
