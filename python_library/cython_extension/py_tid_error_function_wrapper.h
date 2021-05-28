//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_PREDICTOR_ERROR_WRAPPER_H
#define DL85_PY_PREDICTOR_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"

class PyTidErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyTidErrorWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PyTidErrorWrapper(const PyTidErrorWrapper& rhs): PyTidErrorWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PyTidErrorWrapper(PyTidErrorWrapper&& rhs): PyTidErrorWrapper(rhs.pyFunction) {
        rhs.pyFunction = nullptr;
    }

    // need no-arg constructor to stack allocate in Cython
    PyTidErrorWrapper(): PyTidErrorWrapper(nullptr) {
    }

    ~PyTidErrorWrapper() {
        Py_XDECREF(pyFunction);
    }

    PyTidErrorWrapper& operator=(const PyTidErrorWrapper& rhs) {
        PyTidErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyTidErrorWrapper& operator=(PyTidErrorWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = nullptr;
        return *this;
    }

    float operator()(RCover* ar) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_tid_error_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_PREDICTOR_ERROR_WRAPPER_H
