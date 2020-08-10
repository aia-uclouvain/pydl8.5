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

class LcmPruned {
public:
    LcmPruned ( DataManager *dataReader, Query *query, Trie *trie, bool infoGain, bool infoAsc, bool allDepths );

    ~LcmPruned();

    void run ();

    int latticesize = 0;


protected:
    TrieNode* recurse ( Array<Item> itemset, Item added, TrieNode* node, Array<Attribute> attributes_to_visit, RCover* a_transactions, Depth depth, Error ub, Error lb = 0 );

//    TrieNode* getdepthtwotree(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb = 0);

    TrieNode* getdepthtwotrees(RCover* cover, Error ub, Array<Attribute> attributes_to_visit, Item added, Array<Item> itemset, TrieNode* node, Error lb = 0);

    Array<Attribute> getSuccessors(Array<Attribute> last_freq_attributes,RCover* a_transactions, Item added, unordered_set<int> frequent_attr = {});

    unordered_set<int> getExistingSuccessors(TrieNode* node);

    Error computeLowerBound(RCover* cover, Array<bitset<M>*>& covers, Array<float>& errors);

    void addInfoForLowerBound(RCover* cover, QueryData * node_data, Array<bitset<M>*>& covers, Array<float>& errors);

    float informationGain ( Supports notTaken, Supports taken);

    DataManager *dataReader;
    Trie *trie;
    Query *query;
    bool infoGain = false;
    bool infoAsc = false; //if true ==> items with low IG are explored first
    bool allDepths = false;
    //bool timeLimitReached = false;
};

#endif