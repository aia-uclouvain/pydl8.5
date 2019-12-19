from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "globals.h":
    cdef cppclass Array[T]:
        cppclass iterator:
            int nIndex
            T operator*()
            iterator operator++()
            bool operator==(iterator)
            bool operator!=(iterator)
        iterator begin()
        iterator end()
        int getSize()

cdef class ArrayIterator:
    cdef Array[int]* arr
    cdef Array[int].iterator it

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.it.nIndex < self.arr.getSize():
            val = deref(self.it)
            self.it = inc(self.it)
            return val
        else:
            raise StopIteration()

    def init_iterator(self):
            self.it = self.arr.begin()

cdef public wrap_array(Array[int] *ar):
    tid_python_object = ArrayIterator()
    tid_python_object.arr = ar
    tid_python_object.init_iterator()
    return tid_python_object

cdef public vector[float] call_python_error_function(python_function, Array[int] *ar):
    return python_function(wrap_array(ar))

cdef public vector[float] call_python_fast_error_function(python_fast_function, Array[int] *ar):
    return python_fast_function(wrap_array(ar))

cdef public float call_python_predictor_error_function(python_predictor_function, Array[int] *ar):
    return python_predictor_function(wrap_array(ar))

