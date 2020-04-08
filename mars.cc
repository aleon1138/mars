#include "marsalgo.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;


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


void dsse(MarsAlgo &algo,
          Ref<ArrayXd> &linear_sse,
          Ref<ArrayXd> &hinge_sse,
          Ref<ArrayXd> &hinge_cut,
          const Ref<const ArrayXb> &mask,
          int xcol,
          int endspan,
          bool linear_only)
{
    if (linear_sse.rows() != mask.rows() ||
            hinge_sse.rows()  != mask.rows() ||
            hinge_cut.rows()  != mask.rows()) {
        throw std::runtime_error("invalid dataset lengths");
    }
    algo.dsse(linear_sse.data(), hinge_sse.data(), hinge_cut.data(),
              xcol, mask.data(), endspan, linear_only);
}


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
    .def("dsse", &dsse)
    ;
}
