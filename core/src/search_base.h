//
// Created by Gael Aglin on 19/10/2021.
//

#ifndef DL85_SEARCH_BASE_H
#define DL85_SEARCH_BASE_H

#include <utility>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <climits>
#include <cassert>
#include <cmath>
#include <chrono>
#include "globals.h"
#include "nodedataManager.h"
#include "dataManager.h"
#include "rCover.h"
#include "depthTwoComputer.h"
#include "logger.h"

class Search_base {

public:
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
    Support minsup;
    Depth maxdepth;
    int timeLimit;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    bool specialAlgo = true;
    time_point<high_resolution_clock> startTime;
    bool timeLimitReached = false;
    NodeDataManager *nodeDataManager;

    Search_base(NodeDataManager *nodeDataManager,
                bool infoGain,
                bool infoAsc,
                bool repeatSort,
                Support minsup,
                Depth maxdepth,
                int timeLimit,
                float maxError = NO_ERR,
                bool specialAlgo = true,
                bool stopAfterError = false);

    virtual void run() = 0;
};

// a variable to express whether the error computation is not performed in python or not
#define no_python_error !nodeDataManager->tids_error_callback && !nodeDataManager->tids_error_class_callback && !nodeDataManager->supports_error_class_callback


#endif //DL85_SEARCH_BASE_H
