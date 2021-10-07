//
// Created by Gael Aglin on 2019-10-25.
//

#ifndef DL85_LCM_ITERATIVE_H
#define DL85_LCM_ITERATIVE_H
#include <utility>
#include "globals.h"
#include "cache_trie.h"
#include "nodedataManager.h"
#include "dataManager.h"
#include "rCoverTotalFreq.h"


class LcmIterative {
public:
    LcmIterative ( DataManager *data, NodeDataManager *query, Cache *cache, bool infoGain, bool infoAsc, bool allDepths,
                   Support minsup,
                   Depth maxdepth,
                   int timeLimit,
                   bool continuous,
                   float maxError = NO_ERR,
                   bool stopAfterError = false);

    ~LcmIterative();

    void run ();

    int latticesize = 0;


//protected:
    Node* recurse ( Array<Item> itemset,
                        Item added,
                        Array<pair<bool,Attribute> > a_attributes,
                        RCover* a_transactions,
                        Depth depth,
                        float priorUbFromParent,
                        int currentMaxDepth);

    Array<pair<bool,Attribute>> getSuccessors(Array<pair<bool,Attribute > > a_attributes,
                                              RCover* a_transactions,
                                              Item added);

    void printItemset(Array<Item> itemset);

    float informationGain ( pair<Supports,Support> notTaken, pair<Supports,Support> taken);

    DataManager *dataReader;
    Cache *cache;
    NodeDataManager *nodeDataManager;
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool allDepths = false;
    //bool timeLimitReached = false;
    Support minsup;
    Depth maxdepth;
    int timeLimit;
    bool continuous = false;
    float maxError = NO_ERR;
    bool stopAfterError = false;
    time_point<high_resolution_clock> startTime;
    TrieNode *realroot; // as the empty itemset may not have an empty closure
    bool timeLimitReached = false;
};


#endif //DL85_LCM_ITERATIVE_H
