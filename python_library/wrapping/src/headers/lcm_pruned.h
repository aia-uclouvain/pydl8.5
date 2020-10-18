#ifndef LCMB_H
#define LCMB_H

#include <utility>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <climits>
#include <cassert>
#include <cmath>
#include <chrono>
#include "globals.h"
#include "trie.h"
#include "query.h"
#include "dataManager.h"
#include "rCover.h"
#include "depthTwoComputer.h"
#include "query_best.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"
#include "dataContinuous.h"



class LcmPruned {
public:
    LcmPruned ( RCover *cover, Query *query, bool infoGain, bool infoAsc, bool repeatSort );

    ~LcmPruned();

    void run ();

    int latticesize = 0;

    Query *query;

    RCover *cover;


protected:
    TrieNode* recurse ( Array<Item> itemset, Attribute last_added, TrieNode* node, Array<Attribute> attributes_to_visit, Depth depth, Error ub, Error lb = 0 );

    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes, Attribute last_added);

    Array<Attribute> getExistingSuccessors(TrieNode* node);

    Error computeLowerBound(bitset<M> *b1_cover, bitset<M> *b2_cover);

    void addInfoForLowerBound(QueryData *node_data, bitset<M> *&b1_cover,
                              bitset<M> *&b2_cover, Error &highest_error, Support& highest_coversize);

    float informationGain ( Supports notTaken, Supports taken);


    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
    //bool timeLimitReached = false;
};

#endif