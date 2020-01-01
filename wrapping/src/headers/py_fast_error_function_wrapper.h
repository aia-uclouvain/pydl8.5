//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_FAST_ERROR_WRAPPER_H
#define DL85_PY_FAST_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"

class PyFastErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyFastErrorWrapper(PyObject* o): pyFastFunction(o) {
        Py_XINCREF(o);
    }

    PyFastErrorWrapper(const PyFastErrorWrapper& rhs): PyFastErrorWrapper(rhs.pyFastFunction) { // C++11 onwards only
    }

    PyFastErrorWrapper(PyFastErrorWrapper&& rhs): pyFastFunction(rhs.pyFastFunction) {
        rhs.pyFastFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyFastErrorWrapper(): PyFastErrorWrapper(nullptr) {
    }

    ~PyFastErrorWrapper() {
        Py_XDECREF(pyFastFunction);
    }

    PyFastErrorWrapper& operator=(const PyFastErrorWrapper& rhs) {
        PyFastErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyFastErrorWrapper& operator=(PyFastErrorWrapper&& rhs) {
        pyFastFunction = rhs.pyFastFunction;
        rhs.pyFastFunction = 0;
        return *this;
    }

    vector<float> operator()(RCover* ar) {
        PyInit_error_function();
        if (pyFastFunction) { // nullptr check
            return call_python_fast_error_function(pyFastFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFastFunction;
};

#endif //DL85_PY_FAST_ERROR_WRAPPER_H
