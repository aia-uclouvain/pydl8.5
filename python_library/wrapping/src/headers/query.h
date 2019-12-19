#ifndef QUERY_H
#define QUERY_H
#include <utility>
#include "globals.h"
#include "data.h"
#include <iostream>
#include <cfloat>
#include <functional>
#include <vector>

class TrieNode;
class Trie;

using namespace std;

typedef void *QueryData; // using void pointers is much lighter than class derivation



class Query {
public:
    Query( Trie * trie, Data *data, int timeLimit, bool continuous, function<vector<float>(Array<int>*)>* error_callback = nullptr, function<vector<float>(Array<int>*)>* fast_error_callback = nullptr, function<float(Array<int>*)>*  predictor_error_callback = nullptr, float maxError = NO_ERR, bool stopAfterError = false );

    virtual ~Query();
    virtual bool is_freq ( pair<Supports,Support> supports ) = 0;
    virtual bool is_pure ( pair<Supports,Support> supports ) = 0;
    virtual bool canimprove ( QueryData *left, Error ub ) = 0;
    virtual bool canSkip ( QueryData *actualBest ) = 0;
    virtual QueryData *initData ( Array<Transaction> tid, Error parent_ub, Support minsup, Depth currentMaxDepth = -1) = 0;
    virtual bool updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right ) = 0;
    virtual string printResult ( Data *data ) = 0;
    void setStartTime( clock_t sTime ){startTime = sTime;}

    Data *data; // we need to have information about the data for default predictions
    Trie *trie;
    TrieNode *realroot; // as the empty itemset may not have an empty closure
    Support minsup;
    Depth maxdepth;
    clock_t startTime;
    int timeLimit;
    bool timeLimitReached = false;
    bool continuous = false;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    function<vector<float>(Array<int>*)>* error_callback;
    function<vector<float>(Array<int>*)>* fast_error_callback;
    function<float(Array<int>*)>*  predictor_error_callback;
};

#endif