cmake_minimum_required(VERSION 3.15)
project(dl85)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer -g -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fno-omit-frame-pointer -g -O3")


#[[add_executable(test
        tests_run/main.cpp)]]

include_directories(src/)

add_executable(dl85
        main.cpp
        src/cache.h
        src/cache.cpp
        src/cache_hash_cover.h
        src/cache_hash_cover.cpp
        src/cache_hash_itemset.h
        src/cache_hash_itemset.cpp
        src/cache_trie.h
        src/cache_trie.cpp
        src/dataManager.h
        src/dataManager.cpp
        src/depthTwoComputer.h
        src/depthTwoComputer.cpp
        src/depthTwoNodeData.h
        src/dl85.h
        src/dl85.cpp
        src/globals.h
        src/globals.cpp
        src/logger.h
        src/nodeDataManager.h
        src/nodeDataManager.cpp
        src/nodeDataManager_Cover.h
        src/nodeDataManager_Cover.cpp
        src/nodeDataManager_Trie.h
        src/nodeDataManager_Trie.cpp
        src/rCover.h
        src/rCover.cpp
        src/rCoverFreq.h
        src/rCoverFreq.cpp
        src/rCoverWeight.h
        src/rCoverWeight.cpp
        src/search_base.h
        src/search_base.cpp
        src/search_cover_cache.h
        src/search_cover_cache.cpp
        src/search_nocache.h
        src/search_nocache.cpp
        src/search_trie_cache.h
        src/search_trie_cache.cpp
        src/solution.h
        src/solution.cpp
        src/solution_Cover.h
        src/solution_Cover.cpp
        src/solution_Trie.h
        src/solution_Trie.cpp
        argparse.cpp)