//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_OBJ_WRAPPER_H
#define DL85_PY_OBJ_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file
#include "globals.h"

class PyObjWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PyObjWrapper(const PyObjWrapper& rhs): PyObjWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs): pyFunction(rhs.pyFunction) {
        rhs.pyFunction = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper(): PyObjWrapper(nullptr) {
    }

    ~PyObjWrapper() {
        Py_XDECREF(pyFunction);
    }

    PyObjWrapper& operator=(const PyObjWrapper& rhs) {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = 0;
        return *this;
    }

    /*float operator()(Array<int>::iterator it) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_function(pyFunction, it); // note, no way of checking for errors until you return to Python
        }
    }*/

    vector<float> operator()(Array<int>* ar) {
        PyInit_error_function();
        if (pyFunction) { // nullptr check
            return call_python_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }
    }

    /*float operator()(Array<int>::iterator a) {
        //PyInit_call_obj();
        float error = -1;
        int status=PyImport_AppendInittab("call_obj", PyInit_call_obj);
        if(status==-1){
            return error;//error
        }
        Py_Initialize();
        PyObject *module = PyImport_ImportModule("call_obj");

        if(module==NULL){
            Py_Finalize();
            return error;//error
        }
        if (held) { // nullptr check
            //std::vector<int> vec(a, a+b);
            cout << "eya 2" << endl;
            cout << *a << endl;
            error = call_obj(held, a); // note, no way of checking for errors until you return to Python
        }
        Py_Finalize();
        return error;
    }*/

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_OBJ_WRAPPER_H
