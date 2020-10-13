//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_TEST_ERROR_WRAPPER_H
#define DL85_PY_TEST_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file


class PyTestErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyTestErrorWrapper(PyObject* o): pyTestErrorFunction(o) {
        Py_XINCREF(o);
    }

    PyTestErrorWrapper(const PyTestErrorWrapper& rhs): PyTestErrorWrapper(rhs.pyTestErrorFunction) { // C++11 onwards only
    }

    PyTestErrorWrapper(PyTestErrorWrapper&& rhs): PyTestErrorWrapper(rhs.pyTestErrorFunction) {
        rhs.pyTestErrorFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyTestErrorWrapper(): PyTestErrorWrapper(nullptr) {
    }

    ~PyTestErrorWrapper() {
        Py_XDECREF(pyTestErrorFunction);
    }

    PyTestErrorWrapper& operator=(const PyTestErrorWrapper& rhs) {
        PyTestErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyTestErrorWrapper& operator=(PyTestErrorWrapper&& rhs) {
        pyTestErrorFunction = rhs.pyTestErrorFunction;
        rhs.pyTestErrorFunction = 0;
        return *this;
    }

    float operator()(float error) {
        PyInit_error_function();
        if (pyTestErrorFunction) { // nullptr check
            return call_python_test_error_function(pyTestErrorFunction, error); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyTestErrorFunction;
};

#endif //DL85_PY_TEST_ERROR_WRAPPER_H
