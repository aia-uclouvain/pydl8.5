from libcpp.vector cimport vector

cdef public float call_obj(obj, vector[int] a):
    return obj(a)