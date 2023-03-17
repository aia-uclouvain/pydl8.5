from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np

cdef extern from "../core/src/dataManager.h":
    cdef cppclass DataManager:
        int getNClasses()
        int* getSupports()

cdef extern from "../core/src/rCover.h":
    cdef cppclass RCover:
        cppclass iterator:
            int wordIndex
            int operator*()
            iterator operator++()
            bool operator==(iterator)
            bool operator!=(iterator)
        iterator begin(bool trans_loop)
        iterator end(bool trans_loop)
        DataManager* dm
        stack[int] limit

cdef class ArrayIterator:
    cdef RCover* arr
    cdef RCover.iterator it
    cdef bool trans_loop

    def __init__(self, trans):
        self.trans_loop = trans

    def __iter__(self):
        return self

    def __next__(self):
        if self.trans_loop:
            if self.it.wordIndex < self.arr.limit.top():
                val = deref(self.it)
                self.it = inc(self.it)
                return val
            else:
                raise StopIteration()
        else:
            if self.it.wordIndex < deref(self.arr.dm).getNClasses():
                val = deref(self.it)
                self.it = inc(self.it)
                return val
            else:
                raise StopIteration()

    def init_iterator(self):
        self.it = self.arr.begin(self.trans_loop)

cdef public wrap_array(RCover *ar, bool trans):
    tid_python_object = ArrayIterator(trans)
    tid_python_object.arr = ar
    tid_python_object.init_iterator()
    return tid_python_object

# Windows does not allow to return a vector. So we return a pointer as a workaround
cdef public float* call_python_tid_error_class_function(py_function, RCover *ar):
    error_class_array = np.array(py_function(wrap_array(ar, True)), dtype=np.float32)
    if not error_class_array.flags['C_CONTIGUOUS']:
        error_class_array = np.ascontiguousarray(error_class_array) # Makes a contiguous copy of the numpy array.
    cdef float [::1] error_class_view = error_class_array
    return &error_class_view[0]

# Windows does not allow to return a vector. So we return a pointer to the vector data as a workaround
cdef public float* call_python_support_error_class_function(py_function, RCover *ar):
    error_class_array = np.array(py_function(wrap_array(ar, False)), dtype=np.float32)
    if not error_class_array.flags['C_CONTIGUOUS']:
        error_class_array = np.ascontiguousarray(error_class_array) # Makes a contiguous copy of the numpy array.
    cdef float [::1] error_class_view = error_class_array
    return &error_class_view[0]

cdef public float call_python_tid_error_function(py_function, RCover *ar):
    return py_function(wrap_array(ar, True))

