from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp cimport bool
import numpy as np

cdef extern from "src/headers/dl85.h":
    string search ( int* supports,
                    int ntransactions,
                    int nattributes,
                    int nclasses,
                    int *data,
                    int *target,
                    float maxError,
                    bool stopAfterError,
                    bool iterative,
                    int maxdepth,
                    int minsup,
                    bool infoGain,
                    bool infoAsc,
                    bool repeatSort,
                    int timeLimit,
                    map[int, pair[int, int]]* continuousMap,
                    bool save,
                    bool nps_param,
                    bool verbose_param )

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


def solve(data,
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

    target = target.astype('int32')
    data = data.astype('int32')
    if np.array_equal(data, data.astype('bool')) is False:  # WARNING: maybe categorical (not binary) inputs will be supported in the future
        raise ValueError("Bad input type. DL8.5 actually only supports binary (0/1) inputs")

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
    out = search(&supports_view[0],
                 ntransactions,
                 nattributes,
                 nclasses,
                 &data_view[0][0],
                 &target_view[0],
                 max_err,
                 stop_after_better,
                 iterative,
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