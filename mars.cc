#include "marsalgo.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;


MarsAlgo * new_algo(
    const Ref<const MatrixXf> &X,
    const Ref<const VectorXf> &y,
    const Ref<const VectorXf> &w, int p)
{
    return new MarsAlgo(X.data(), y.data(), w.data(),
        X.rows(), X.cols(), p, X.outerStride());
}

using namespace pybind11::literals;

PYBIND11_MODULE(mars, m) {
    m.doc() = "Multivariate Adaptive Regression Splines";

    py::class_<MarsAlgo>(m, "MarsAlgo")
        .def(py::init(&new_algo))
        , "X"_a.noconvert()
        , "y"_a.noconvert()
        , "w"_a.noconvert()
        , "p"_a
        ;
}
