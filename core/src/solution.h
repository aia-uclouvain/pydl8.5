#ifndef SOLUTION_H
#define SOLUTION_H

#include "search_base.h"

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
    virtual ~Tree() {}
};


class Solution {
public:
    Solution(Search_base*);
    virtual ~Solution();

    virtual Tree * getTree() = 0;

    Tree* tree;
    Search_base* searcher;
};

#endif
