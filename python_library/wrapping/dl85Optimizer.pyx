from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.functional cimport function
import numpy as np

cdef extern from "src/headers/dl85.h":
    string search ( PyObjWrapper,
                    int* supports,
                    int ntransactions,
                    int nattributes,
                    int nclasses,
                    int *data,
                    int *target,
                    float maxError,
                    bool stopAfterError,
                    bool iterative,
                    bool user,
                    int maxdepth,
                    int minsup,
                    bool infoGain,
                    bool infoAsc,
                    bool repeatSort,
                    int timeLimit,
                    map[int, pair[int, int]]* continuousMap,
                    bool save,
                    bool nps_param,
                    bool verbose_param ) except +

cdef extern from "src/headers/py_obj_wrapper.h":
    cdef cppclass PyObjWrapper:
        PyObjWrapper()
        PyObjWrapper(object) # define a constructor that takes a Python object
             # note - doesn't match c++ signature - that's fine!

def readBinData(dataPath):
    dataset = np.genfromtxt(dataPath, delimiter = ' ')
    dataset = dataset.astype('int32')
    return dataset

def readBinCSVData(dataPath):
    dataset = np.genfromtxt(dataPath, delimiter = ',')
    dataset = dataset.astype('int32')
    return dataset

def readBinCSV2Data(dataPath):
    dataset = np.genfromtxt(dataPath, delimiter = ';')
    dataset = dataset.astype('int32')
    return dataset

def splitClassFirst(dataset):
    target = dataset[:,0]
    data = dataset[:,1:]
    return (data, target)

def splitClassLast(dataset):
    target = dataset[:,-1]
    data = dataset[:,0:-1]
    return (data, target)

original_targets = []
original_class_support = []

def default_error_function(tid_iterator):
    tid_iterator.init_iterator()
    size = tid_iterator.get_size()

    tid_list = []
    for i in range(size):
        tid_list.append(tid_iterator.get_value())
        if i != size - 1:
            tid_iterator.inc_iterator()

    target_subset = original_targets.take(tid_list)
    classes, supports = np.unique(target_subset, return_counts=True)
    class_support = dict(zip(classes, supports))
    maxclass = -1
    maxclassval = minclassval = -1
    conflict = 0

    for classe, sup in class_support.items():
        if sup > maxclassval:
            maxclass = classe
            maxclassval = sup
        elif sup == maxclassval:
            conflict += 1
            if original_class_support[classe] > original_class_support[maxclass]:
                maxclass = classe
        else:
            minclassval = sup

    error_score = sum(supports) - maxclassval
    return [error_score, maxclass, conflict, minclassval]

def default_error(entry):
    return 2


def solve(func,
          data,
          target,
          max_depth=1,
          min_sup=1,
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
          nps=False):

    cdef PyObjWrapper f_user = PyObjWrapper(func)
    cdef PyObjWrapper f_default = PyObjWrapper(default_error)

    target = target.astype('int32')
    data = data.astype('int32')
    if np.array_equal(data, data.astype('bool')) is False:  # WARNING: maybe categorical (not binary) inputs will be supported in the future
        raise ValueError("Bad input type. DL8.5 actually only supports binary (0/1) inputs")

    global original_targets
    original_targets = target

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data) # Makes a contiguous copy of the numpy array.
    if not target.flags['C_CONTIGUOUS']:
        target = np.ascontiguousarray(target) # Makes a contiguous copy of the numpy array.

    cdef int [:, ::1] data_view = data
    cdef int [::1] target_view = target

    ntransactions, nattributes = data.shape
    classes, supports = np.unique(target, return_counts=True)
    nclasses = len(classes)
    supports = supports.astype('int32')

    global original_class_support
    original_class_support = dict(zip(classes, supports))

    if not supports.flags['C_CONTIGUOUS']:
        supports = np.ascontiguousarray(supports) # Makes a contiguous copy of the numpy array.
    cdef int [::1] supports_view = supports

    max_err = max_error - 1  # because maxError but not be reached
    if max_err == -1:  # raise error when incompatibility between max_error value and stop_after_better value
        stop_after_better = False

    cont_map = NULL
    if continuousMap is not None:
        #cont_map must be defined properly
        cont_map = NULL

    info_gain = not (desc == False and asc == False)

    if not callable(func):

        out = search(f_default,
                     &supports_view[0],
                     ntransactions,
                     nattributes,
                     nclasses,
                     &data_view[0][0],
                     &target_view[0],
                     max_err,
                     stop_after_better,
                     iterative,
                     user = False,
                     maxdepth = max_depth,
                     minsup = min_sup,
                     infoGain = info_gain,
                     infoAsc = asc,
                     repeatSort = repeat_sort,
                     timeLimit = time_limit,
                     continuousMap = NULL,
                     save = bin_save,
                     nps_param = nps,
                     verbose_param = verb)
    else:

        out = search(f_user,
                 &supports_view[0],
                 ntransactions,
                 nattributes,
                 nclasses,
                 &data_view[0][0],
                 &target_view[0],
                 max_err,
                 stop_after_better,
                 iterative,
                 user = True,
                 maxdepth = max_depth,
                 minsup = min_sup,
                 infoGain = info_gain,
                 infoAsc = asc,
                 repeatSort = repeat_sort,
                 timeLimit = time_limit,
                 continuousMap = NULL,
                 save = bin_save,
                 nps_param = nps,
                 verbose_param = verb)

    return out.decode("utf-8")