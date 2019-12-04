//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_OBJ_WRAPPER_H
#define DL85_PY_OBJ_WRAPPER_H

#include <Python.h>
//#include <vector>
#include "call_obj.h" // cython helper file

class PyObjWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyObjWrapper(const PyObjWrapper& rhs): PyObjWrapper(rhs.held) { // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper(): PyObjWrapper(nullptr) {
    }

    ~PyObjWrapper() {
        Py_XDECREF(held);
    }

    PyObjWrapper& operator=(const PyObjWrapper& rhs) {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    float operator()(int* a, int b) {
        if (held) { // nullptr check
            std::vector<int> vec(a, a+b);
            return call_obj(held, vec); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* held;
};

#endif //DL85_PY_OBJ_WRAPPER_H
