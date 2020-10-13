#ifndef LCMB_H
#define LCMB_H
#include <utility>
#include "globals.h"
#include "trie.h"
#include "query.h"
#include "dataManager.h"
#include "rCover.h"
#include <map>
#include <unordered_set>
#include <unordered_map>
#include "depthTwoComputer.h"


class LcmPruned {
public:
    LcmPruned ( DataManager *dataReader, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool repeatSort );

    ~LcmPruned();

    void run ();

    int latticesize = 0;

    Query *query;


protected:
    TrieNode* recurse ( Array<Item> itemset, Attribute last_added, TrieNode* node, Array<Attribute> attributes_to_visit, RCover* a_transactions, Depth depth, Error ub, Error lb = 0 );

//    TrieNode* getdepthtwotree(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb = 0);

    TrieNode* getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Attribute last_added, Array<Item> itemset, TrieNode* node, Error lb = 0);

    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes,RCover* a_transactions, Attribute last_added);

    Array<Attribute> getExistingSuccessors(TrieNode* node);

    Error computeLowerBound(RCover *cover, bitset<M> *b1_cover, bitset<M> *b2_cover);

    void addInfoForLowerBound(RCover *cover, QueryData *node_data, bitset<M> *&b1_cover,
                              bitset<M> *&b2_cover, Error &highest_error, Support& highest_coversize);

    float informationGain ( Supports notTaken, Supports taken);

    DataManager *dataReader;
    Trie *trie;
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool repeatSort = false;
    //bool timeLimitReached = false;
};

#endif