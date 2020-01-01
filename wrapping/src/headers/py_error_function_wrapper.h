//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_ERROR_WRAPPER_H
#define DL85_PY_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"

class PyErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyErrorWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PyErrorWrapper(const PyErrorWrapper& rhs): PyErrorWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PyErrorWrapper(PyErrorWrapper&& rhs): pyFunction(rhs.pyFunction) {
        rhs.pyFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyErrorWrapper(): PyErrorWrapper(nullptr) {
    }

    ~PyErrorWrapper() {
        Py_XDECREF(pyFunction);
    }

    PyErrorWrapper& operator=(const PyErrorWrapper& rhs) {
        PyErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyErrorWrapper& operator=(PyErrorWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = 0;
        return *this;
    }

    vector<float> operator()(RCover* ar) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_error_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_ERROR_WRAPPER_H
