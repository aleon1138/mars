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
    typedef Array<double,Dynamic,Dynamic,RowMajor> ArrayXXdC;
    ArrayXXdC dsse1 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC dsse2 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC h_cut = ArrayXXdC::Constant(mask.rows(), mask.cols(), NAN);

    if (threads <= 0) {
        threads = omp_get_num_procs();
    }
    else {
        threads = std::min(threads, omp_get_num_procs());
    }
    threads = std::min<int>(threads, mask.rows());

    // `mask` is a (r,c) boolean matrix where `r` is the number of columns of
    // the design matrix `X` and `c` is the number of basis already chosen by
    // the algorithm.

    bool ok = true;
    {
        py::gil_scoped_release gil_r;
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int i = 0; i < mask.rows(); ++i) {
            if (i % 8 == 0) { // hacky optimization
                py::gil_scoped_acquire gil_a;
                ok &= (PyErr_CheckSignals() == 0);
            }

            if (ok) {
                algo.eval(dsse1.row(i).data(), dsse2.row(i).data(),
                          h_cut.row(i).data(), i, mask.row(i).data(), endspan, linear);
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
    m.attr("__version__") = "dev";

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
