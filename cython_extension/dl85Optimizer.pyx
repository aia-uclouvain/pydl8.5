from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp cimport bool, nullptr
from libcpp.vector cimport vector
from libcpp.functional cimport function
import numpy as np

cdef extern from "../core/src/globals.h":
    cdef cppclass Array[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bool operator==(iterator)
            bool operator!=(iterator)
        iterator begin()
        iterator end()
        int getSize()
    cdef int MISCLASSIFICATION_ERROR
    cdef int MSE_ERROR 
    cdef int QUANTILE_ERROR

cdef extern from "py_tid_error_class_function_wrapper.h":
    cdef cppclass PyTidErrorClassWrapper:
        PyTidErrorClassWrapper()
        PyTidErrorClassWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "py_support_error_class_function_wrapper.h":
    cdef cppclass PySupportErrorClassWrapper:
        PySupportErrorClassWrapper()
        PySupportErrorClassWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

cdef extern from "py_tid_error_function_wrapper.h":
    cdef cppclass PyTidErrorWrapper:
        PyTidErrorWrapper()
        PyTidErrorWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!


cdef extern from "../core/src/dl85.h":
    string search ( float* supports,
                    int ntransactions,
                    int nattributes,
                    int nclasses,
                    int *data,
                    int *target,
                    double *float_target,
                    int maxdepth,
                    int minsup,
                    float *maxError,
                    bool* stopAfterError,
                    # bool iterative,
                    PyTidErrorClassWrapper tids_error_class_callback,
                    PySupportErrorClassWrapper supports_error_class_callback,
                    PyTidErrorWrapper tids_error_callback,
                    float* in_weights,
                    bool tids_error_class_is_null,
                    bool supports_error_class_is_null,
                    bool tids_error_is_null,
                    bool infoGain,
                    bool infoAsc,
                    bool repeatSort,
                    int backup_error,
                    float* quantiles,
                    int nquantiles,
                    int timeLimit,
                    # map[int, pair[int, int]]* continuousMap,
                    # bool save,
                    bool verbose_param) except +


def solve(data,
          target,
          tec_func_=None, # tec means that it takes "t"ransaction_ids as param and return "e"rror and "c"lass
          sec_func_=None, # sec means that it takes "s"upports as param and return "e"rror and "c"lass
          te_func_=None, # tec means that it takes "t"ransaction_ids as param and return "e"rror
          max_depth=1,
          min_sup=1,
          example_weights=[],
          max_error=None,
          stop_after_better=None,
          # iterative=False,
          time_limit=0,
          verb=False,
          desc=False,
          asc=False,
          repeat_sort=False,
          backup_error="misclassification",
          quantiles=np.array([0.5]),
          # continuousMap=None,
          # bin_save=False,
          # predictor=False
          ):

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

    if backup_error == "misclassification":
        backup_error_code = MISCLASSIFICATION_ERROR
    elif backup_error == "mse":
        backup_error_code = MSE_ERROR 
    elif backup_error == "quantile":
        backup_error_code = QUANTILE_ERROR

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

    cdef double *float_target_array = NULL

    cdef float [::1] max_errors_view
    cdef float *max_errors_array = NULL 
    
    if max_error is not None:
        max_error = np.array(max_error)
        max_error = max_error.astype('float32')
        if not max_error.flags['C_CONTIGUOUS']:
            max_error = np.ascontiguousarray(max_error) # Makes a contiguous copy of the numpy array.
        max_errors_array = <float *> malloc(len(max_error)*sizeof(float))
        for i, v in enumerate(max_error):
            max_errors_array[i] = v
    

    cdef bool [::1] stop_after_better_view
    cdef bool *stop_after_better_array = NULL 

    if stop_after_better is not None:
        stop_after_better = np.array(stop_after_better)
        stop_after_better = stop_after_better.astype('bool')
        if not stop_after_better.flags['C_CONTIGUOUS']:
            stop_after_better = np.ascontiguousarray(stop_after_better) # Makes a contiguous copy of the numpy array.
        stop_after_better_array = <bool *> malloc(len(stop_after_better)*sizeof(bool))
        for i, v in enumerate(stop_after_better):
            stop_after_better_array[i] = v
    #stop_after_better_view = stop_after_better
    #stop_after_better_array = &stop_after_better_view[0]

    cdef float [::1] quantiles_view
    cdef float *quantiles_array = NULL 
    
    quantiles = np.array(quantiles)
    quantiles = quantiles.astype('float32')
    if not quantiles.flags['C_CONTIGUOUS']:
        quantiles = np.ascontiguousarray(quantiles) # Makes a contiguous copy of the numpy array.
    quantiles_array = <float *> malloc(len(quantiles)*sizeof(float))
    for i, v in enumerate(quantiles):
        quantiles_array[i] = v

    #quantiles_view = quantiles
    #quantiles_array = &quantiles_view[0]

    nquantiles = len(quantiles)


    if target is None:
        nclasses = 0 
    
    elif backup_error in ["misclassification"]:
        target = target.astype('int32')
        if not target.flags['C_CONTIGUOUS']:
            target = np.ascontiguousarray(target) # Makes a contiguous copy of the numpy array.
        target_view = target
        #target_array = &target_view[0]

    elif backup_error in ["mse", "quantile"]:
        nclasses = 0 
        target = target.astype(np.double)
        if not target.flags['C_CONTIGUOUS']:
            target = np.ascontiguousarray(target) # Makes a contiguous copy of the numpy array.

        # with simple memory views and no malloc, we get free errors
        float_target_array = <double *> malloc(len(target)*sizeof(double))
        for i, v in enumerate(target):
            float_target_array[i] = v

    # get pointer from support
    if not supports.flags['C_CONTIGUOUS']:
        supports = np.ascontiguousarray(supports) # Makes a contiguous copy of the numpy array.
    cdef float [::1] supports_view = supports
    #cdef float *supports_pointer = &supports_view[0]

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
    # if max_error < 0:  # raise error when incompatibility between max_error value and stop_after_better value
    #     stop_after_better = False

    # cont_map = NULL
    # if continuousMap is not None:
        # #cont_map must be defined properly
        # cont_map = NULL

    info_gain = not (desc == False and asc == False)

    # pred = not predictor

    out = search(supports = &supports_view[0],
                 ntransactions = ntransactions,
                 nattributes = nattributes,
                 nclasses = nclasses,
                 data = data_matrix,
                 target = target_array,
                 float_target = float_target_array,
                 maxdepth = max_depth,
                 minsup = min_sup,
                 maxError = max_errors_array,
                 stopAfterError = stop_after_better_array,
                 # iterative = iterative,
                 tids_error_class_callback = tec_func,
                 supports_error_class_callback = sec_func,
                 tids_error_callback = te_func,
                 in_weights = ex_weights_pointer,
                 tids_error_class_is_null = tec_null_flag,
                 supports_error_class_is_null = sec_null_flag,
                 tids_error_is_null = te_null_flag,
                 infoGain = info_gain,
                 infoAsc = asc,
                 repeatSort = repeat_sort,
                 backup_error = backup_error_code,
                 quantiles = quantiles_array,
                 nquantiles = nquantiles,
                 timeLimit = time_limit,
                 # continuousMap = NULL,
                 # save = bin_save,
                 verbose_param = verb)

    return out.decode("utf-8")