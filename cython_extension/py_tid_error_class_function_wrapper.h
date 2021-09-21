//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_ERROR_WRAPPER_H
#define DL85_PY_ERROR_WRAPPER_H

#include <Python.h>
#include "rCover.h"
#include "error_function.h" // cython helper file

class PyTidErrorClassWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyTidErrorClassWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PyTidErrorClassWrapper(const PyTidErrorClassWrapper& rhs): PyTidErrorClassWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PyTidErrorClassWrapper(PyTidErrorClassWrapper&& rhs): pyFunction(rhs.pyFunction) {
        rhs.pyFunction = nullptr;
    }

    // need no-arg constructor to stack allocate in Cython
    PyTidErrorClassWrapper(): PyTidErrorClassWrapper(nullptr) {
    }

    ~PyTidErrorClassWrapper() {
        Py_XDECREF(pyFunction);
    }

    PyTidErrorClassWrapper& operator=(const PyTidErrorClassWrapper& rhs) {
        PyTidErrorClassWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyTidErrorClassWrapper& operator=(PyTidErrorClassWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = nullptr;
        return *this;
    }

    vector<float> operator()(RCover* ar) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_tid_error_class_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_ERROR_WRAPPER_H
