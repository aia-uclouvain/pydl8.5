//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_FAST_ERROR_WRAPPER_H
#define DL85_PY_FAST_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"

class PySupportErrorClassWrapper {
public:
    // constructors and destructors mostly do reference counting
    PySupportErrorClassWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PySupportErrorClassWrapper(const PySupportErrorClassWrapper& rhs): PySupportErrorClassWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PySupportErrorClassWrapper(PySupportErrorClassWrapper&& rhs): pyFunction(rhs.pyFunction) {
        rhs.pyFunction = nullptr;
    }

    // need no-arg constructor to stack allocate in Cython
    PySupportErrorClassWrapper(): PySupportErrorClassWrapper(nullptr) {
    }

    ~PySupportErrorClassWrapper() {
        Py_XDECREF(pyFunction);
    }

    PySupportErrorClassWrapper& operator=(const PySupportErrorClassWrapper& rhs) {
        PySupportErrorClassWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PySupportErrorClassWrapper& operator=(PySupportErrorClassWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = nullptr;
        return *this;
    }

    vector<float> operator()(RCover* ar) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_support_error_class_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_FAST_ERROR_WRAPPER_H
