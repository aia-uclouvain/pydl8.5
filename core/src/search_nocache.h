//
// Created by Gael Aglin on 19/10/2021.
//

#ifndef DL85_SEARCH_NOCACHE_H
#define DL85_SEARCH_NOCACHE_H

#include "search_base.h"


class Search_nocache : public Search_base{
public:
    Search_nocache(NodeDataManager *nodeDataManager,
                   bool infoGain,
                   bool infoAsc,
                   bool repeatSort,
                   Support minsup,
                   Depth maxdepth,
                   int timeLimit,
                   float maxError = NO_ERR,
                   bool specialAlgo = true,
                   bool stopAfterError = false);

    ~Search_nocache();

    void run();


    Error recurse(Attribute last_added, Array <Attribute> attributes_to_visit, Depth depth, Error ub);

    Array <Attribute> getSuccessors(Array <Attribute> last_freq_attributes, Attribute last_added);

    float informationGain(Supports notTaken, Supports taken);




private:
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth);

};

// a variable to express whether the error computation is not performed in python or not
#define no_python_error !nodeDataManager->tids_error_callback && !nodeDataManager->tids_error_class_callback && !nodeDataManager->supports_error_class_callback



#endif //DL85_SEARCH_NOCACHE_H
