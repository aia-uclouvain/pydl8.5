//
// Created by Gael Aglin on 19/10/2021.
//

#ifndef SEARCH_BASE_H
#define SEARCH_BASE_H

#include <map>
#include "globals.h"
#include "logger.h"
#include "nodeDataManager.h"
#include "dataManager.h"
#include "rCover.h"
#include "cache.h"
#include "depthTwoComputer.h"

class Search_base {

public:
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
    Support minsup;
    Depth maxdepth;
    int timeLimit;
    Cache* cache = nullptr;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    bool specialAlgo = true;
    bool timeLimitReached = false;
    NodeDataManager *nodeDataManager;
    bool from_cpp = true;

    Search_base(NodeDataManager *nodeDataManager,
                bool infoGain,
                bool infoAsc,
                bool repeatSort,
                Support minsup,
                Depth maxdepth,
                int timeLimit,
                Cache *cache = nullptr,
                float maxError = NO_ERR,
                bool specialAlgo = true,
                bool stopAfterError = false,
                bool from_cpp = true);

    virtual ~Search_base(){}

    virtual void run() = 0;
};

// a variable to express whether the error computation is performed in python or not
#define is_python_error nodeDataManager->tids_error_callback or nodeDataManager->tids_error_class_callback or nodeDataManager->supports_error_class_callback
// a variable to express whether the error computation is not performed in python or not
#define no_python_error not nodeDataManager->tids_error_callback and not nodeDataManager->tids_error_class_callback and not nodeDataManager->supports_error_class_callback
#define get_node first
#define has_intersected second
#define is_new second

#endif //SEARCH_BASE_H
