#include "marsalgo.h"
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
typedef Matrix<bool,Dynamic,Dynamic,RowMajor> MatrixXbC;

///////////////////////////////////////////////////////////////////////////////

MarsAlgo * new_algo(
    const Ref<const MatrixXf> &X,
    const Ref<const VectorXf> &y,
    const Ref<const VectorXf> &w,
    int max_terms)
{
    if (X.rows() != y.rows() || y.rows() != w.rows()) {
        throw std::runtime_error("invalid dataset lengths");
    }
    return new MarsAlgo(X.data(), y.data(), w.data(),
                        X.rows(), X.cols(), max_terms, X.outerStride());
}

///////////////////////////////////////////////////////////////////////////////

py::tuple eval(MarsAlgo &algo, const Ref<const MatrixXbC> &mask,
               int endspan, bool linear, int threads)
{
    typedef Array<double,Dynamic,Dynamic,RowMajor> matrix_t;
    typedef Array<double,1,Dynamic> vector_t;

    matrix_t dsse1 = matrix_t::Zero(mask.rows(), mask.cols());
    matrix_t dsse2 = matrix_t::Zero(mask.rows(), mask.cols());
    matrix_t h_cut = matrix_t::Zero(mask.rows(), mask.cols());

    threads = threads <= 0? omp_get_num_procs() : std::min(threads, omp_get_num_procs());
    threads = std::min<int>(threads, mask.rows());

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

            vector_t dsse1_row = vector_t::Zero(mask.cols());
            vector_t dsse2_row = vector_t::Zero(mask.cols());
            vector_t h_cut_row = vector_t::Zero(mask.cols());

            #pragma omp for schedule(static)
            for (int i = 0; i < mask.rows(); ++i) {
                if (ok) {
                    algo.eval(dsse1_row.data(), dsse2_row.data(), h_cut_row.data(),
                              i, mask.row(i).data(), endspan, linear);
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
    m.attr("__version__") = "0.2";

    py::class_<MarsAlgo>(m, "MarsAlgo")
    .def(py::init(&new_algo)
         , py::arg("X").noconvert()
         , py::arg("y").noconvert()
         , py::arg("w").noconvert()
         , py::arg("max_terms")
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
