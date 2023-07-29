#include "marsalgo.h"
#include <omp.h>

#if 0
py::tuple eval(MarsAlgo &algo,
               const Ref<const MatrixXbC> &mask,
               int endspan,
               bool linear_only)
{
    typedef Array<double,Dynamic,Dynamic,RowMajor> ArrayXXdC;
    ArrayXXdC dsse1 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC dsse2 = ArrayXXdC::Zero(mask.rows(), mask.cols());
    ArrayXXdC h_cut = ArrayXXdC::Constant(mask.rows(), mask.cols(), NAN);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mask.rows(); ++i) {
        algo.eval(
            dsse1.row(i).data(), dsse2.row(i).data(), h_cut.row(i).data(),
            i, mask.row(i).data(), endspan, linear_only);
    }
    return py::make_tuple(algo.dsse(), dsse1, dsse2, h_cut);
}

///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(marslib, m)
{
    py::class_<MarsAlgo>(m, "MarsAlgo")
    .def("eval",&eval
         , py::arg("mask").noconvert()
         , py::arg("endspan")
         , py::arg("linear_only")
        )
    .def("append", &MarsAlgo::append)
    ;
}
#endif



#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cstddef>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

struct array_t {
    int  ndim;
    char type;
    long dims[2];
    long strides[2];
    void *data;
};

array_t from_object(PyArrayObject *obj)
{
    int   ndim = PyArray_NDIM(obj);
    long *dims = PyArray_DIMS(obj);
    long *strides = PyArray_STRIDES(obj);
    return array_t {
        .ndim    = ndim,
        .type    = PyArray_DTYPE(obj)->type,
        .dims    = {ndim > 0? dims[0]    : 0, ndim > 1? dims[1]    : 0},
        .strides = {ndim > 0? strides[0] : 0, ndim > 1? strides[1] : 0},
        .data    = PyArray_DATA(obj),
    };
}

struct MarsAlgoObject {
    PyObject_HEAD
    PyArrayObject *X_obj;
    PyArrayObject *y_obj;
    PyArrayObject *w_obj;
    MarsAlgo *algo;
    int nbasis;
    double yvar;
};

static int mars_init(MarsAlgoObject *self, PyObject *args, PyObject *kwds)
{
    int max_terms=0;
    PyArrayObject *X_obj=NULL, *y_obj=NULL, *w_obj=NULL, *tmp=NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!i",
                          &PyArray_Type, &X_obj,
                          &PyArray_Type, &y_obj,
                          &PyArray_Type, &w_obj,
                          &max_terms)) {
        return -1;
    }
    // *INDENT-OFF*
    tmp = self->X_obj; Py_INCREF(X_obj); self->X_obj = X_obj; Py_XDECREF(tmp);
    tmp = self->y_obj; Py_INCREF(y_obj); self->y_obj = y_obj; Py_XDECREF(tmp);
    tmp = self->w_obj; Py_INCREF(w_obj); self->w_obj = w_obj; Py_XDECREF(tmp);
    // *INDENT-ON*

    array_t X = from_object(self->X_obj);
    array_t y = from_object(self->y_obj);
    array_t w = from_object(self->w_obj);
    if (X.type != 'f' || y.type != 'f' || w.type != 'f') {
        PyErr_SetString(PyExc_TypeError, "input data must be 32-bit floating-point");
        return -1;
    }
    if (X.strides[0] != 4 || y.strides[0] != 4 || w.strides[0] != 4) {
        PyErr_SetString(PyExc_TypeError, "input data must be column-contiguous");
        return -1;
    }
    if (X.dims[0] != y.dims[0] || X.dims[0] != w.dims[0]) {
        PyErr_SetString(PyExc_TypeError, "invalid dataset lengths");
        return -1;
    }

    delete self->algo;
    self->algo = new MarsAlgo((const float*)X.data,
                              (const float*)y.data,
                              (const float*)w.data,
                              X.dims[0], X.dims[1],
                              max_terms, X.strides[1]);
    self->nbasis = self->algo->nbasis();
    self->yvar   = self->algo->yvar();
    return 0;
}

static void mars_dealloc(MarsAlgoObject *self)
{
    Py_XDECREF(self->X_obj);
    Py_XDECREF(self->y_obj);
    Py_XDECREF(self->w_obj);
    delete self->algo;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *mars_eval(MarsAlgoObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyUnicode_FromFormat("hello world");
}

static PyObject *mars_append(MarsAlgoObject *self, PyObject *Py_UNUSED(ignored))
{
    self->nbasis += 1;
    return PyUnicode_FromFormat("hello world");
}

static PyMethodDef methods[] = {
    {"eval",   (PyCFunction)mars_eval,   METH_NOARGS, "Evaluate an iteration"},
    {"append", (PyCFunction)mars_append, METH_NOARGS, "Append a basis to the model"},
    {NULL}
};

static PyMemberDef members[] = {
    {"nbasis", T_INT,    offsetof(MarsAlgoObject, nbasis), READONLY, "Number of basis found"},
    {"yvar",   T_DOUBLE, offsetof(MarsAlgoObject, yvar),   READONLY, "Sample variance of target Y"},
    {NULL}
};

static PyTypeObject MarsAlgoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "marslib.MarsAlgo",
    .tp_basicsize = sizeof(MarsAlgoObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)mars_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "MARS algorithm object",
    .tp_methods   = methods,
    .tp_members   = members,
    .tp_init      = (initproc)mars_init,
    .tp_new       = PyType_GenericNew,
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "marslib",
    .m_doc  = "Multivariate Adaptive Regression Splines",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_marslib()
{
    if (PyType_Ready(&MarsAlgoType) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&MarsAlgoType);
    if (PyModule_AddObject(m, "MarsAlgo", (PyObject *) &MarsAlgoType) < 0) {
        Py_DECREF(&MarsAlgoType);
        Py_DECREF(m);
        return NULL;
    }

    import_array();
    return m;
}
