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
#include "cache.h"
//#include "cache_hash.h"
#include "cache_trie.h"
#include "nodedataManager.h"
#include "dataManager.h"
#include "rCover.h"
#include "depthTwoComputer.h"
#include "solution.h" // if cannot link is specified, we need a clustering problem!!!
#include "logger.h"
#include "dataContinuous.h"


class LcmPruned {
public:
    LcmPruned ( NodeDataManager *nodeDataManager,
                bool infoGain,
                bool infoAsc,
                bool repeatSort,
                Support minsup,
                Depth maxdepth,
                Cache *cache,
                int timeLimit,
                bool continuous,
                float maxError = NO_ERR,
                bool stopAfterError = false);

    ~LcmPruned();

    void run ();

    int latticesize = 0;

    NodeDataManager *nodeDataManager;



//protected:
    Node* recurse ( Array<Item> itemset, Attribute last_added, Node* node, Array<Attribute> attributes_to_visit, Depth depth, Error ub, bool newnode);

    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes, Attribute last_added, Node* node);

//    Array<Attribute> getExistingSuccessors(TrieNode* node);

    Error computeSimilarityLowerBound(bitset<M> *b1_cover, bitset<M> *b2_cover, Error b1_error, Error b2_error);

    void addInfoForLowerBound(NodeData *node_data, bitset<M> *&b1_cover, bitset<M> *&b2_cover,
                              Error &b1_error, Error &b2_error, Support &highest_coversize);

    float informationGain ( Supports notTaken, Supports taken);


    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
//    DataManager *dm; // we need to have information about the data for default predictions
    Cache *cache;
    Support minsup;
    Depth maxdepth;
    int timeLimit;
    bool continuous = false;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    time_point<high_resolution_clock> startTime;
//    TrieNode *realroot; // as the empty itemset may not have an empty closure
    bool timeLimitReached = false;

private:
    Node *getSolutionIfExists(Node *node, Error ub, Depth depth);

};

// a variable to express whether the error computation is not performed in python or not
#define no_python_error !nodeDataManager->tids_error_callback && !nodeDataManager->tids_error_class_callback && !nodeDataManager->supports_error_class_callback

// a variable to express whether the error computation is performed in python or not
#define is_python_error nodeDataManager->tids_error_callback || nodeDataManager->tids_error_class_callback || nodeDataManager->supports_error_class_callback

#endif