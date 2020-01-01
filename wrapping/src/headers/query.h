#ifndef QUERY_H
#define QUERY_H
#include <utility>
#include "globals.h"
#include "rCover.h"
#include "dataManager.h"
#include <iostream>
#include <cfloat>
#include <functional>
#include <vector>

struct TrieNode;
class Trie;

using namespace std;

typedef void *QueryData; // using void pointers is much lighter than class derivation



class Query {
public:
    Query( Trie * trie, DataManager *data, int timeLimit, bool continuous, function<vector<float>(RCover*)>* error_callback = nullptr, function<vector<float>(RCover*)>* fast_error_callback = nullptr, function<float(RCover*)>*  predictor_error_callback = nullptr, float maxError = NO_ERR, bool stopAfterError = false );

    virtual ~Query();
    virtual bool is_freq ( pair<Supports,Support> supports ) = 0;
    virtual bool is_pure ( pair<Supports,Support> supports ) = 0;
    virtual bool canimprove ( QueryData *left, Error ub ) = 0;
    virtual bool canSkip ( QueryData *actualBest ) = 0;
    //virtual QueryData *initData ( Array<Transaction> tid, Error parent_ub, Support minsup, Depth currentMaxDepth = -1) = 0;
    virtual QueryData *initData ( RCover* tid, Error parent_ub, Support minsup, Depth currentMaxDepth = -1) = 0;
    virtual bool updateData ( QueryData *best, Error upperBound, Attribute attribute, QueryData *left, QueryData *right ) = 0;
    virtual string printResult ( DataManager *data ) = 0;
    void setStartTime( clock_t sTime ){startTime = sTime;}

    DataManager *data; // we need to have information about the data for default predictions
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
    function<vector<float>(RCover*)>* error_callback;
    function<vector<float>(RCover*)>* fast_error_callback;
    function<float(RCover*)>*  predictor_error_callback;
};

#endif