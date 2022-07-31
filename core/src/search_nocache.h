//
// Created by Gael Aglin on 19/10/2021.
//

#ifndef SEARCH_NOCACHE_H
#define SEARCH_NOCACHE_H

#include "search_base.h"

class Search_nocache : public Search_base{
public:
    bool use_ub;

    Search_nocache(NodeDataManager *nodeDataManager,
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
                   bool use_ub = true);

    void run();
    Error recurse(Attribute last_added, Attributes &attributes_to_visit, Depth depth, Error ub);
    Attributes getSuccessors(Attributes &last_attributes, Attribute last_added);
    float informationGain(ErrorVals notTaken, ErrorVals taken);
    ~Search_nocache();

};

#endif //SEARCH_NOCACHE_H
