#ifndef SOLUTION_H
#define SOLUTION_H

#include "cache.h"
#include "nodeDataManager.h"
#include <vector>

struct Tree {
    string expression;
    int size;
    Depth depth;
    Error trainingError;
    int cacheSize;
    float runtime;
    float accuracy;
    bool timeout;

    virtual string to_str() const = 0;
};


class Solution {
public:
    Solution(void*, NodeDataManager*);

    virtual ~Solution();

    virtual Tree * getTree() = 0;

    void* searcher;
    NodeDataManager* nodeDataManager;
};

#endif
