//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_EXAMPLE_WEIGHT_WRAPPER_H
#define DL85_PY_EXAMPLE_WEIGHT_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include <vector>
#include <string>

using namespace std;

class PyExampleWeightWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyExampleWeightWrapper(PyObject* o): pyExampleWeightFunction(o) {
        Py_XINCREF(o);
    }

    PyExampleWeightWrapper(const PyExampleWeightWrapper& rhs): PyExampleWeightWrapper(rhs.pyExampleWeightFunction) { // C++11 onwards only
    }

    PyExampleWeightWrapper(PyExampleWeightWrapper&& rhs): PyExampleWeightWrapper(rhs.pyExampleWeightFunction) {
        rhs.pyExampleWeightFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyExampleWeightWrapper(): PyExampleWeightWrapper(nullptr) {
    }

    ~PyExampleWeightWrapper() {
        Py_XDECREF(pyExampleWeightFunction);
    }

    PyExampleWeightWrapper& operator=(const PyExampleWeightWrapper& rhs) {
        PyExampleWeightWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyExampleWeightWrapper& operator=(PyExampleWeightWrapper&& rhs) {
        pyExampleWeightFunction = rhs.pyExampleWeightFunction;
        rhs.pyExampleWeightFunction = 0;
        return *this;
    }

    std::vector<float> operator()(string tree_json) {
        PyInit_error_function();
        if (pyExampleWeightFunction) { // nullptr check
            return call_python_example_weight_function(pyExampleWeightFunction, tree_json); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* pyExampleWeightFunction;
};

#endif //DL85_PY_EXAMPLE_WEIGHT_WRAPPER_H
