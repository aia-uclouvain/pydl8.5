cmake_minimum_required(VERSION 3.15)
project(dl85)

set(CMAKE_CXX_STANDARD 11)


include_directories(headers/)


add_executable(dl85
        main.cpp

        headers/data.h
        codes/data.cpp

        headers/dataContinuous.h
        codes/dataContinuous.cpp

        headers/dataBinary.h
        codes/dataBinary.cpp

        headers/dl85.h
        codes/dl85.cpp

        headers/experror.h
        codes/experror.cpp

        headers/globals.h
        codes/globals.cpp

        headers/lcm_pruned.h
        codes/lcm_pruned.cpp

        # dl85_dist_source/wrapping/src/headers/nodedataManager.h
        # dl85_dist_source/wrapping/src/codes/nodedataManager.cpp

        headers/query.h
        codes/query.cpp

        headers/query_best.h
        codes/query_best.cpp

        headers/query_totalfreq.h
        codes/query_totalfreq.cpp

        # dl85_dist_source/wrapping/src/headers/solution.h
        # dl85_dist_source/wrapping/src/codes/solution.cpp

        # dl85_dist_source/wrapping/src/headers/freq_nodedataManager.h
        # dl85_dist_source/wrapping/src/codes/freq_nodedataManager.cpp

        #        dl85_dist_source/wrapping/src/headers/query_bioptimization.h
        #        dl85_dist_source/wrapping/src/codes/query_bioptimization.cpp

        #        dl85_dist_source/wrapping/src/headers/query_weighted.h
        #        dl85_dist_source/wrapping/src/codes/query_weighted.cpp

        headers/depthTwoComputer.h
        codes/depthTwoComputer.cpp

        #        dl85_dist_source/wrapping/src/headers/cache.h
        #        dl85_dist_source/wrapping/src/codes/cache.cpp
        #
        #        dl85_dist_source/wrapping/src/headers/cache_hash.h
        #        dl85_dist_source/wrapping/src/codes/cache_hash.cpp
        #
        #        dl85_dist_source/wrapping/src/headers/cache_priority.h
        #        dl85_dist_source/wrapping/src/codes/cache_priority.cpp
        #
        #        dl85_dist_source/wrapping/src/headers/cache_trie.h
        #        dl85_dist_source/wrapping/src/codes/cache_trie.cpp

        headers/trie.h
        codes/trie.cpp

        headers/dataBinaryPython.h
        codes/dataBinaryPython.cpp

        headers/lcm_iterative.h
        codes/lcm_iterative.cpp

        headers/dataManager.h
        codes/dataManager.cpp

        headers/rCover.h
        codes/rCover.cpp

        headers/rCoverTotalFreq.h
        codes/rCoverTotalFreq.cpp

        headers/rCoverWeighted.h
        codes/rCoverWeighted.cpp

        headers/logger.h

        #        dl85_dist_source/wrapping/src/headers/model.h
        #        main_for_stashed_code.cpp dl85_dist_source/wrapping/src/codes/model.cpp
        # dl85_dist_source/wrapping/src/headers/freq_Solution.h
        # dl85_dist_source/wrapping/src/codes/freq_Solution.cpp
        )

#target_link_libraries(dl85
#        libgurobi_g++4.2.a
#        libgurobi81.dylib
#        nlohmann_json::nlohmann_json)

#target_link_libraries(dl85_predict
#        libgurobi_g++4.2.a
#        libgurobi81.dylib
#        nlohmann_json::nlohmann_json)