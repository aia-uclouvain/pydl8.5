cmake_minimum_required(VERSION 3.15)
project(dl85)

set(CMAKE_CXX_STANDARD 11)


include_directories(src/)


add_executable(dl85
        main.cpp
        src/dataManager.h
        src/dataManager.cpp
        src/depthTwoComputer.h
        src/depthTwoComputer.cpp
        src/dl85.h
        src/dl85.cpp
        src/globals.h
        src/globals.cpp
        src/lcm_pruned.h
        src/lcm_pruned.cpp
        src/logger.h
        src/query.h
        src/query.cpp
        src/query_best.h
        src/query_best.cpp
        src/query_totalfreq.h
        src/query_totalfreq.cpp
        src/rCover.h
        src/rCover.cpp
        src/rCoverTotalFreq.h
        src/rCoverTotalFreq.cpp
        src/rCoverWeighted.h
        src/rCoverWeighted.cpp
        src/trie.h
        src/trie.cpp)