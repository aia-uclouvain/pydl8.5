#ifndef LCMB_H
#define LCMB_H
#include <utility>
#include "globals.h"
#include "trie.h"
#include "query.h"
#include "dataManager.h"
#include "rCover.h"

class LcmPruned {
public:
    LcmPruned ( DataManager *dataReader, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool allDepths );

    ~LcmPruned();

    void run ();

    int latticesize = 0;


protected:
    TrieNode* recurse ( Array<Item> itemset, Item added, Array<pair<bool,Attribute>> a_attributes, RCover* a_transactions, Depth depth, float priorUbFromParent );

    Array<pair<bool,Attribute>> getSuccessors(Array<pair<bool,Attribute > > a_attributes,RCover* a_transactions, Item added);

    Array<pair<bool, Attribute> > getExistingSuccessors(TrieNode* node);

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

#endif