cmake_minimum_required(VERSION 3.15)
project(dl85)

set(CMAKE_CXX_STANDARD 17)

#[[add_executable(test
        tests_run/test.cpp)]]

include_directories(src/)

add_executable(dl85
        main.cpp
#        tests_run/main_arg.cpp
        src/cache.h
        src/cache.cpp
        src/cache_hash.h
        src/cache_hash.cpp
        src/cache_priority.h
        src/cache_priority.cpp
        src/cache_trie.h
        src/cache_trie.cpp
        src/cache_ltd_trie.h
        src/cache_ltd_trie.cpp
        src/data.h
        src/data.cpp
        src/dataBinaryPython.h
        src/dataBinaryPython.cpp
        src/dataManager.h
        src/dataManager.cpp
        src/depthTwoComputer.h
        src/depthTwoComputer.cpp
        src/dl85.h
        src/dl85.cpp
        src/experror.h
        src/experror.cpp
        src/freq_nodedataManager.h
        src/freq_nodedataManager.cpp
        src/freq_Solution.h
        src/freq_Solution.cpp
        src/globals.h
        src/globals.cpp
#        src/lcm_iterative.h
#        src/lcm_iterative.cpp
        src/search.h
        src/search.cpp
        src/search_nocache.h
        src/search_nocache.cpp
        src/logger.h
        src/nodedataManager.h
        src/nodedataManager.cpp
        src/solution.h
        src/solution.cpp
        src/rCover.h
        src/rCover.cpp
        src/rCoverTotalFreq.h
        src/rCoverTotalFreq.cpp
        src/rCoverWeighted.h
        src/rCoverWeighted.cpp src/search_base.cpp src/search_base.h)
