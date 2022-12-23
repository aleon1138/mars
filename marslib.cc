#include "marsalgo.h"
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

py::tuple all_dsse(MarsAlgo &algo,
          const Ref<const MatrixXbC> &mask,
          int endspan,
          bool linear_only)
{
    typedef Array<double,Dynamic,Dynamic,RowMajor> ArrayXXdC;
    ArrayXXdC dsse1 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC dsse2 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC h_cut = ArrayXXdC::Constant(mask.rows(), mask.cols(), NAN);

    for (int i = 0; i < mask.rows(); ++i) {
        if (mask.row(i).any()) {
            algo.eval(
                dsse1.row(i).data(), dsse2.row(i).data(), h_cut.row(i).data(),
                i, mask.row(i).data(), endspan, linear_only);
        }
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
    .def("all_dsse",&all_dsse
         , py::arg("mask").noconvert()
         , py::arg("endspan")
         , py::arg("linear_only")
        )
    .def("nbasis", &MarsAlgo::nbasis)
    .def("yvar",   &MarsAlgo::yvar)
    .def("append", &MarsAlgo::append)
    ;
}
