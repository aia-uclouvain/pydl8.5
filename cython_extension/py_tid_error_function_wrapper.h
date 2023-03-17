//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_PREDICTOR_ERROR_WRAPPER_H
#define DL85_PY_PREDICTOR_ERROR_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "rCover.h"
#include <limits>

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
        float result = std::numeric_limits<float>::max();
        if (pyFunction != nullptr) { // nullptr check
            result = call_python_tid_error_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
        return result;
    }

    /*float operator()(RCover* ar) {
        std::cout << "Calling sans check Python error function" << std::endl;
        int status = PyImport_AppendInittab("myerror", PyInit_error_function);
        if (status == -1) {
            cout << "Error in PyImport_AppendInittab" << endl;
            PyErr_Print();
            return std::numeric_limits<float>::max();
        }
        Py_Initialize();
        PyObject* module = PyImport_ImportModule("myerror");
        if (module == nullptr) {
            cout << "Error in PyImport_ImportModule" << endl;
            PyErr_Print();
            Py_Finalize();
            return std::numeric_limits<float>::max();
        }
        float result = std::numeric_limits<float>::max();
        if (pyFunction != nullptr) { // nullptr check
            std::cout << "Calling Python error function" << std::endl;
            result = call_python_tid_error_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
            std::cout << "Python error function returned " << result << std::endl;
        }
        else {
            std::cout << "Python error function is nullptr" << std::endl;
        }
        Py_Finalize();
        return result;
    }*/

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_PREDICTOR_ERROR_WRAPPER_H
