//
// Created by Gael Aglin on 2019-10-25.
//

#ifndef DL85_LCM_ITERATIVE_H
#define DL85_LCM_ITERATIVE_H
#include <utility>
#include "globals.h"
#include "trie.h"
#include "query.h"
#include "dataManager.h"
#include "rCover.h"


class LcmIterative {
public:
    LcmIterative ( DataManager *data, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool allDepths );

    ~LcmIterative();

    void run ();

    int latticesize = 0;


protected:
    TrieNode* recurse ( Array<Item> itemset,
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
    Trie *trie;
    Query *query;
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool allDepths = false;
    //bool timeLimitReached = false;
};


#endif //DL85_LCM_ITERATIVE_H
