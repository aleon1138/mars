#include "marsalgo.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

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

void eval(MarsAlgo &algo,
            Ref<ArrayXd> &linear_dsse,
            Ref<ArrayXd> &hinge_dsse,
            Ref<ArrayXd> &hinge_cuts,
            const Ref<const ArrayXb> &mask,
            int xcol,
            int endspan,
            bool linear_only)
{
    if (mask.rows() != algo.nbasis()) {
      throw std::runtime_error("invalid basis mask length");
    }

    if (linear_dsse.rows() != algo.nbasis() ||
        hinge_dsse.rows()  != algo.nbasis() ||
        hinge_cuts.rows()  != algo.nbasis()) {
        throw std::runtime_error("invalid dataset lengths");
    }

    algo.eval(linear_dsse.data(), hinge_dsse.data(), hinge_cuts.data(),
              xcol, mask.data(), endspan, linear_only);
}

///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(marslib, m) {
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
    .def("eval",   &eval)
    .def("dsse",   &MarsAlgo::dsse)
    .def("nbasis", &MarsAlgo::nbasis)
    .def("yvar",   &MarsAlgo::yvar)
    .def("append", &MarsAlgo::append)
    ;
}
