from libc.stdlib cimport malloc, free
from libcpp.string cimport string
from libcpp cimport bool
import numpy as np

cdef extern from "src/headers/dl85.h":
    string search ( int argc, char *argv[], int* supports, int ntransactions, int nattributes, int nclasses, int* data, int* target, float maxError, bool stopAfterError, bool iterative )

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
          verbose=False,
          desc=False,
          asc=False,
          repeat_sort=False,
          continuous=False,
          bin_save=False,
          nps=False):
    target = target.astype('int32')
    #check that variable are set
    args = ["None"]
    args.append("-d")
    args.append(max_depth)
    args.append("-s")
    args.append(min_sup)
    if time_limit > 0:
        args.append("-t")
        args.append(time_limit)
    if verbose is True:
        args.append("-v")
    #plan to print something in incorrect cases (eg: asc and desc are set to True)
    if desc is True and asc is False:
        args.append("-i")
    if asc is True and desc is True:
        args.append("-I")
    if repeat_sort is True:
        args.append("-l")
    if continuous is True:
        data = data.astype('float32')
        args.append("-n")
    else:
        data = data.astype('int32')
        if np.array_equal(data, data.astype('bool')) is False:  # WARNING: maybe categorical (not binary) inputs will be supported in the future
            raise ValueError("Bad input type. DL8.5 actually only supports binary (0/1) inputs")
    if bin_save is True:
        args.append("-e")
    if nps is True:
        args.append("-T")

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data) # Makes a contiguous copy of the numpy array.

    cdef float [:, ::1] data_view_float
    cdef int [:, ::1] data_view_int
    if continuous is True:
        data_view_float = data
        #cdef float [:, ::1] data_view = data
    else:
        data_view_int = data
        #cdef int [:, ::1] data_view = data

    ntransactions, nattributes = data.shape

    if not target.flags['C_CONTIGUOUS']:
        target = np.ascontiguousarray(target) # Makes a contiguous copy of the numpy array.
    cdef int [::1] target_view = target

    classes, supports = np.unique(target, return_counts=True)
    nclasses = len(classes)
    supports = supports.astype('int32')

    if not supports.flags['C_CONTIGUOUS']:
        supports = np.ascontiguousarray(supports) # Makes a contiguous copy of the numpy array.
    cdef int [::1] supports_view = supports

    cdef char** c_argv
    # Allocate memory
    c_argv = <char**>malloc(len(args) * sizeof(char*))
    # Check if allocation went fine
    if c_argv is NULL:
        raise MemoryError()
    # Convert str to char* and store it into our char**
    b_args = []
    for i in range(len(args)):
        b_args.append(str(args[i]).encode())
        c_argv[i] = b_args[i]
    # Grabbing return value

    max_err = max_error - 1  # because maxError but not be reached
    if max_err == -1:  # raise error when incompatibility between max_error value and stop_after_better value
        stop_after_better = False

    if continuous is True:
        return "TrainingError: DL8.5 ODTClassifier is not yet implemented for continuous dataset."
        #out = search_float(len(args), c_argv, &supports_view[0], ntransactions, nattributes, nclasses, &data_view_float[0][0], &target_view[0], max_err, stop_after_better)
    else:
        out = search(len(args), c_argv, &supports_view[0], ntransactions, nattributes, nclasses, &data_view_int[0][0], &target_view[0], max_err, stop_after_better, iterative)
    #print()
    #print(out.decode("utf-8"))
    #print()

    # Let him go
    free(c_argv)

    return out.decode("utf-8")