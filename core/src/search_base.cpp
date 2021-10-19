//
// Created by Gael Aglin on 19/10/2021.
//

#include "search_base.h"

Search_base::Search_base(
        NodeDataManager *nodeDataManager,
        bool infoGain,
        bool infoAsc,
        bool repeatSort,
        Support minsup,
        Depth maxdepth,
        int timeLimit,
        float maxError,
        bool specialAlgo,
        bool stopAfterError
) :
        nodeDataManager(nodeDataManager),
        infoGain(infoGain),
        infoAsc(infoAsc),
        repeatSort(repeatSort),
        minsup(minsup),
        maxdepth(maxdepth),
        timeLimit(timeLimit),
        maxError(maxError),
        specialAlgo(specialAlgo),
        stopAfterError(stopAfterError) {}
