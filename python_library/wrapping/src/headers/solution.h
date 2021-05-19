#ifndef QUERY_BEST_H
#define QUERY_BEST_H

#include "cache.h"
//#include "experror.h"
#include <nodedataManager.h>
#include <vector>
#include "dataContinuous.h"

typedef void *Tree;


class Solution {
public:
    Solution(void*, NodeDataManager*);

    virtual ~Solution();

    virtual Tree * getTree() = 0;

    void* searcher;
    NodeDataManager* nodeDataManager;
};

#endif
