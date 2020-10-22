from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp cimport bool, nullptr
from libcpp.vector cimport vector
from libcpp.functional cimport function
import numpy as np

cdef extern from "src/headers/globals.h":
    cdef cppclass Array[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bool operator==(iterator)
            bool operator!=(iterator)
        iterator begin()
        iterator end()
        int getSize()

cdef extern from "src/headers/py_tid_error_class_function_wrapper.h":
    cdef cppclass PyTidErrorClassWrapper:
        PyTidErrorClassWrapper()
        PyTidErrorClassWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "src/headers/py_support_error_class_function_wrapper.h":
    cdef cppclass PySupportErrorClassWrapper:
        PySupportErrorClassWrapper()
        PySupportErrorClassWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "src/headers/py_tid_error_function_wrapper.h":
    cdef cppclass PyTidErrorWrapper:
        PyTidErrorWrapper()
        PyTidErrorWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "src/headers/py_example_weight_function_wrapper.h":
    cdef cppclass PyExampleWeightWrapper:
        PyExampleWeightWrapper()
        PyExampleWeightWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "src/headers/py_predict_error_function_wrapper.h":
    cdef cppclass PyPredictErrorWrapper:
        PyPredictErrorWrapper()
        PyPredictErrorWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!


cdef extern from "src/headers/dl85.h":
    string search ( float* supports,
                    int ntransactions,
                    int nattributes,
                    int nclasses,
                    int *data,
                    int *target,
                    float maxError,
                    bool stopAfterError,
                    bool iterative,
                    PyTidErrorClassWrapper tids_error_class_callback,
                    PySupportErrorClassWrapper supports_error_class_callback,
                    PyTidErrorWrapper tids_error_callback,
                    PyExampleWeightWrapper example_weight_callback,
                    PyPredictErrorWrapper predict_error_callback,
                    float* in_weights,
                    bool tids_error_class_is_null,
                    bool supports_error_class_is_null,
                    bool tids_error_is_null,
                    bool example_weight_is_null,
                    bool predict_error_is_null,
                    # bool in_weights_is_null,
                    int maxdepth,
                    int minsup,
                    int max_estimators,
                    bool infoGain,
                    bool infoAsc,
                    bool repeatSort,
                    int timeLimit,
                    map[int, pair[int, int]]* continuousMap,
                    bool save,
                    bool verbose_param) except +


#cdef get_pointer(object):
#    if object is None:
#        return NULL
#    else:
#        return &object


def solve(data,
          target,
          tec_func_=None,
          sec_func_=None,
          te_func_=None,
          exw_func_=None,
          pred_func_=None,
          max_depth=1,
          min_sup=1,
          max_estimators=1,
          example_weights=[],
          max_error=0,
          stop_after_better=False,
          iterative=False,
          time_limit=0,
          verb=False,
          desc=False,
          asc=False,
          repeat_sort=False,
          continuousMap=None,
          bin_save=False,
          predictor=False):

    cdef PyTidErrorClassWrapper tec_func = PyTidErrorClassWrapper(tec_func_)
    tec_null_flag = True
    if tec_func_ is not None:
        tec_null_flag = False

    cdef PySupportErrorClassWrapper sec_func = PySupportErrorClassWrapper(sec_func_)
    sec_null_flag = True
    if sec_func_ is not None:
        sec_null_flag = False

    cdef PyTidErrorWrapper te_func = PyTidErrorWrapper(te_func_)
    te_null_flag = True
    if te_func_ is not None:
        te_null_flag = False

    cdef PyExampleWeightWrapper exw_func = PyExampleWeightWrapper(exw_func_)
    exw_null_flag = True
    if exw_func_ is not None:
        exw_null_flag = False

    cdef PyPredictErrorWrapper pred_func = PyPredictErrorWrapper(pred_func_)
    pred_null_flag = True
    if pred_func_ is not None:
        pred_null_flag = False

    if max_estimators <= 1:
            # test_func = None
            # test_null_flag = True
            weight_func = None
            weight_null_flag = True

    data = data.astype('int32')
    ntransactions, nattributes = data.shape
    data = data.transpose()
    classes, supports = np.unique(target, return_counts=True)
    nclasses = len(classes)
    supports = supports.astype('float32')

    if np.array_equal(data, data.astype('bool')) is False:  # WARNING: maybe categorical (not binary) inputs will be supported in the future
        raise ValueError("Bad input type. DL8.5 actually only supports binary (0/1) inputs")

    # get pointer from data
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data) # Makes a contiguous copy of the numpy array.
    cdef int [:, ::1] data_view = data
    cdef int *data_matrix = &data_view[0][0]

    # get pointer form target
    cdef int [::1] target_view
    cdef int *target_array = NULL
    if target is not None:
        target = target.astype('int32')
        if not target.flags['C_CONTIGUOUS']:
            target = np.ascontiguousarray(target) # Makes a contiguous copy of the numpy array.
        target_view = target
        target_array = &target_view[0]
    else:
        nclasses = 0

    # get pointer from support
    if not supports.flags['C_CONTIGUOUS']:
        supports = np.ascontiguousarray(supports) # Makes a contiguous copy of the numpy array.
    cdef float [::1] supports_view = supports
    cdef float * supports_pointer = &supports_view[0]

    # get pointer from example weights
    cdef float [::1] ex_weights_view
    cdef float *ex_weights_pointer = NULL
    # print("len =", len(example_weights))
    if len(example_weights) > 0:  # not none, not empty
        ex_weights = np.asarray(example_weights, dtype=np.float32)
        if not ex_weights.flags['C_CONTIGUOUS']:
            ex_weights = np.ascontiguousarray(ex_weights) # Makes a contiguous copy of the numpy array.
        ex_weights_view = ex_weights
        ex_weights_pointer = &ex_weights_view[0]

    # max_err = max_error - 1  # because maxError but not be reached
    if max_error < 0:  # raise error when incompatibility between max_error value and stop_after_better value
        stop_after_better = False

    cont_map = NULL
    if continuousMap is not None:
        #cont_map must be defined properly
        cont_map = NULL

    info_gain = not (desc == False and asc == False)

    # pred = not predictor

    out = search(supports_pointer,
                 ntransactions,
                 nattributes,
                 nclasses,
                 data_matrix,
                 target_array,
                 max_error,
                 stop_after_better,
                 iterative,
                 tids_error_class_callback = tec_func,
                 supports_error_class_callback = sec_func,
                 tids_error_callback = te_func,
                 example_weight_callback = exw_func,
                 predict_error_callback = pred_func,
                 in_weights = ex_weights_pointer,
                 tids_error_class_is_null = tec_null_flag,
                 supports_error_class_is_null = sec_null_flag,
                 tids_error_is_null = te_null_flag,
                 example_weight_is_null = exw_null_flag,
                 predict_error_is_null = pred_null_flag,
                 maxdepth = max_depth,
                 minsup = min_sup,
                 max_estimators = max_estimators,
                 infoGain = info_gain,
                 infoAsc = asc,
                 repeatSort = repeat_sort,
                 timeLimit = time_limit,
                 continuousMap = NULL,
                 save = bin_save,
                 verbose_param = verb)

    return out.decode("utf-8")