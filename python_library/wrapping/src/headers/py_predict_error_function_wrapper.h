//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_TEST_ERROR_WRAPPER_H
#define DL85_PY_TEST_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file


class PyPredictErrorWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyPredictErrorWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PyPredictErrorWrapper(const PyPredictErrorWrapper& rhs): PyPredictErrorWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PyPredictErrorWrapper(PyPredictErrorWrapper&& rhs): PyPredictErrorWrapper(rhs.pyFunction) {
        rhs.pyFunction = nullptr;
    }

    // need no-arg constructor to stack allocate in Cython
    PyPredictErrorWrapper(): PyPredictErrorWrapper(nullptr) {
    }

    ~PyPredictErrorWrapper() {
        Py_XDECREF(pyFunction);
    }

    PyPredictErrorWrapper& operator=(const PyPredictErrorWrapper& rhs) {
        PyPredictErrorWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyPredictErrorWrapper& operator=(PyPredictErrorWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = nullptr;
        return *this;
    }

    vector<float> operator()(string tree_json) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_predict_error_function(pyFunction, tree_json); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_TEST_ERROR_WRAPPER_H
