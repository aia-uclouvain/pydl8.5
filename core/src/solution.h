#ifndef SOLUTION_H
#define SOLUTION_H

#include "cache.h"
#include "nodeDataManager.h"
#include <vector>

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
