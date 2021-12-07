cmake_minimum_required(VERSION 3.15)
project(dl85)

set(CMAKE_CXX_STANDARD 17)

#[[add_executable(test
        tests_run/main.cpp)]]

include_directories(src/)

add_executable(dl85
        main.cpp
#        tests_run/main_arg.cpp
        src/cache.h
        src/cache.cpp
        src/cache_hash.h
        src/cache_hash.cpp
        src/cache_hash_cover.h
        src/cache_hash_cover.cpp
#        src/cache_priority.h
#        src/cache_priority.cpp
#        src/cache_trie.h
#        src/cache_trie.cpp
        src/cache_trie.h
        src/cache_trie.cpp
        src/dataManager.h
        src/dataManager.cpp
        src/depthTwoComputer.h
        src/depthTwoComputer.cpp
        src/dl85.h
        src/dl85.cpp
        src/nodeDataManagerFreq.h
        src/nodeDataManagerFreq.cpp
        src/solutionFreq.h
        src/solutionFreq.cpp
        src/globals.h
        src/globals.cpp
        src/search_base.h
        src/search_base.cpp
        src/search_cache.h
        src/search_cache.cpp
        src/search_hash_cover.h
        src/search_hash_cover.cpp
        src/search_nocache.h
        src/search_nocache.cpp
        src/logger.h
        src/nodeDataManager.h
        src/nodeDataManager.cpp
        src/solution.h
        src/solution.cpp
        src/rCover.h
        src/rCover.cpp
        src/rCoverFreq.h
        src/rCoverFreq.cpp
        src/rCoverWeight.h
        src/rCoverWeight.cpp
        )


#        src/dataHandler.h
#        src/dataHandler.cpp
#        src/lcm_iterative.h
#        src/lcm_iterative.cpp
#        src/longRCover.h
#        src/longRCover.cpp
#        src/longRCoverFreq.h
#        src/longRCoverFreq.cpp
#        src/nodedataHandler.h
#        src/nodedataHandler.cpp
#        src/nodedataHandlerFreq.h
#        src/nodedataHandlerFreq.cpp
#        src/cache_wipe.h