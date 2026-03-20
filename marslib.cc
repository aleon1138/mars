#include "marsalgo.h"
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // somehow needed for bool matrices
#include <pybind11/numpy.h>
namespace py = pybind11;

void verify(bool check, const char *msg)
{
    if (!check) {
        throw std::runtime_error(msg);
    }
}

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

    auto mask = mask_array.unchecked<2>();
    ssize_t mask_rows = mask.shape(0);
    ssize_t mask_cols = mask.shape(1);

    typedef Array<double,Dynamic,Dynamic,RowMajor> matrix_t;
    typedef Array<double,1,Dynamic> vector_t;

    matrix_t dsse1 = matrix_t::Zero(mask_rows, mask_cols);
    matrix_t dsse2 = matrix_t::Zero(mask_rows, mask_cols);
    matrix_t h_cut = matrix_t::Zero(mask_rows, mask_cols);

    threads = threads <= 0? omp_get_num_procs() : std::min(threads, omp_get_num_procs());
    threads = std::min<int>(threads, mask_rows);

    // `mask` is a (r,c) boolean matrix where `r` is the number of columns of
    // the design matrix `X` and `c` is the number of basis already chosen by
    // the algorithm.

    bool ok = true;
    {
        py::gil_scoped_release gil_r;
        #pragma omp parallel num_threads(threads)
        {
            char name[16]; // Give this thread a name
            snprintf(name, sizeof(name), "mars-%02d", omp_get_thread_num());
            pthread_setname_np(pthread_self(), name);

            vector_t dsse1_row = vector_t::Zero(mask_cols);
            vector_t dsse2_row = vector_t::Zero(mask_cols);
            vector_t h_cut_row = vector_t::Zero(mask_cols);

            #pragma omp for schedule(static)
            for (int i = 0; i < mask_rows; ++i) {
                if (ok) {
                    algo.eval(dsse1_row.data(), dsse2_row.data(), h_cut_row.data(),
                              i, &mask(i,0), endspan, linear);
                }

                #pragma omp critical
                {
                    dsse1.row(i) = dsse1_row;
                    dsse2.row(i) = dsse2_row;
                    h_cut.row(i) = h_cut_row;

                    py::gil_scoped_acquire gil_a;
                    ok &= (PyErr_CheckSignals() == 0); // check for CRTL-C
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
         , py::keep_alive<1, 2>()   // X
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
