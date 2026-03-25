#include "marsalgo.h"
#include <omp.h>
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
               int endspan, bool linear, int threads)
{
    py::buffer_info mask_info = mask_array.request();
    verify(mask_info.ndim == 2, "expected 2D array for mask");
    verify(mask_info.strides[1] == sizeof(bool), "mask must be row-major");
    verify(mask_info.shape[1] == algo.nbasis(), "invalid dimension for mask");

    auto mask = mask_array.unchecked<2>();
    ssize_t mask_rows = mask.shape(0);
    ssize_t mask_cols = mask.shape(1);

    py::array_t<double> dsse1({mask_rows, mask_cols});
    py::array_t<double> dsse2({mask_rows, mask_cols});
    py::array_t<double> h_cut({mask_rows, mask_cols});
    auto dsse1_ptr = dsse1.mutable_unchecked<2>();
    auto dsse2_ptr = dsse2.mutable_unchecked<2>();
    auto h_cut_ptr = h_cut.mutable_unchecked<2>();

    threads = threads <= 0? omp_get_num_procs() : std::min(threads, omp_get_num_procs());
    threads = std::min<int>(threads, mask_rows);

    /*
     * `mask` is a (r,c) boolean matrix where `r` is the number of columns of
     * the design matrix `X` and `c` is the number of basis already chosen by
     * the algorithm.
     */
    std::atomic<bool> ok{true};
    {
        py::gil_scoped_release gil_r;
        #pragma omp parallel num_threads(threads)
        {
            char name[16]; // Give this thread a name so it shows up in `htop`
            snprintf(name, sizeof(name), "mars-%02d", omp_get_thread_num());
            pthread_setname_np(pthread_self(), name);

            #pragma omp for schedule(static)
            for (int i = 0; i < mask_rows; ++i) {
                if (ok.load(std::memory_order_relaxed)) {
                    algo.eval(&dsse1_ptr(i,0), &dsse2_ptr(i,0), &h_cut_ptr(i,0),
                              i, &mask(i,0), endspan, linear);
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
        throw py::error_already_set();
    }

    return py::make_tuple(algo.dsse(), dsse1, dsse2, h_cut);
}

///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(marslib, m)
{
    py::options options;
    options.disable_function_signatures();

    m.doc() = "Multivariate Adaptive Regression Splines";
    m.attr("__version__") = "0.3";

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
         , py::arg("endspan")
         , py::arg("linear_only")
         , py::arg("threads")
        )
    .def("nbasis", &MarsAlgo::nbasis)
    .def("yvar",   &MarsAlgo::yvar)
    .def("append", &MarsAlgo::append)
    ;
}
