from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "globals.h":
    cdef cppclass Array[T]:
        cppclass iterator:
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

    def init_iterator(self):
        self.it = self.arr.begin()

    def get_value(self):
        return deref(self.it)

    def inc_iterator(self):
        self.it = inc(self.it)

    def get_size(self):
        return self.arr.getSize()

cdef public wrap_array(Array[int] *ar):
    tid_python_object = ArrayIterator()
    tid_python_object.arr = ar
    return tid_python_object

cdef public vector[float] call_python_error_function(python_function, Array[int] *ar):
    return python_function(wrap_array(ar))

cdef public vector[float] call_python_fast_error_function(python_fast_function, Array[int] *ar):
    return python_fast_function(wrap_array(ar))

