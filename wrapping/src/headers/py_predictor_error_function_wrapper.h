//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_PREDICTOR_ERROR_WRAPPER_H
#define DL85_PY_PREDICTOR_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"

class PyPredictorErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyPredictorErrorWrapper(PyObject* o): pyPredictorFunction(o) {
        Py_XINCREF(o);
    }

    PyPredictorErrorWrapper(const PyPredictorErrorWrapper& rhs): PyPredictorErrorWrapper(rhs.pyPredictorFunction) { // C++11 onwards only
    }

    PyPredictorErrorWrapper(PyPredictorErrorWrapper&& rhs): PyPredictorErrorWrapper(rhs.pyPredictorFunction) {
        rhs.pyPredictorFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyPredictorErrorWrapper(): PyPredictorErrorWrapper(nullptr) {
    }

    ~PyPredictorErrorWrapper() {
        Py_XDECREF(pyPredictorFunction);
    }

    PyPredictorErrorWrapper& operator=(const PyPredictorErrorWrapper& rhs) {
        PyPredictorErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyPredictorErrorWrapper& operator=(PyPredictorErrorWrapper&& rhs) {
        pyPredictorFunction = rhs.pyPredictorFunction;
        rhs.pyPredictorFunction = 0;
        return *this;
    }

    float operator()(RCover* ar) {
        PyInit_error_function();
        if (pyPredictorFunction) { // nullptr check
            return call_python_predictor_error_function(pyPredictorFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyPredictorFunction;
};

#endif //DL85_PY_PREDICTOR_ERROR_WRAPPER_H
