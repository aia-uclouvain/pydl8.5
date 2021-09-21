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
//#include <algorithm>
#include "globals.h"
#include "trie.h"
#include "query.h"
#include "dataManager.h"
#include "rCover.h"
#include "depthTwoComputer.h"
#include "query_best.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"



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

    Error computeSimilarityLowerBound(bitset<M> *b1_cover, bitset<M> *b2_cover, Error b1_error, Error b2_error);

    void addInfoForLowerBound(QueryData *node_data, bitset<M> *&b1_cover, bitset<M> *&b2_cover,
                              Error &b1_error, Error &b2_error, Support &highest_coversize);

    float informationGain ( Supports notTaken, Supports taken);


    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
    //bool timeLimitReached = false;
};

// a variable to express whether the error computation is not performed in python or not
#define no_python_error !query->tids_error_callback && !query->tids_error_class_callback && !query->supports_error_class_callback

// a variable to express whether the error computation is performed in python or not
#define is_python_error query->tids_error_callback || query->tids_error_class_callback || query->supports_error_class_callback

#endif