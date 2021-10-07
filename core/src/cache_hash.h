#ifndef CACHE_HASH_H
#define CACHE_HASH_H
#include "cache.h"
#include <cmath>
#include <unordered_map>
#include "freq_nodedataManager.h"

using namespace std;

struct HashNode : Node {
    /*Array<Item> itemset;
    HashNode(Array<Item> itemset1): Node() { itemset = itemset1; }
    HashNode(): Node() {itemset.size = 0; itemset.elts = nullptr;}
    ~HashNode() { if (itemset.elts) itemset.free(); }*/
};

class Cache_Hash: public Cache {
public:
    Cache_Hash(Depth maxdepth, WipeType wipe_type, int maxcachesize);
    ~Cache_Hash() {}

    Array<unordered_map<Array<Item>, HashNode*>> store;

    pair<Node *, bool>insert ( Array<Item> itemset, NodeDataManager* );
    Node *get ( Array<Item> itemset);
    void updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc=false);
    void updateItemsetLoad ( Array<Item> itemset, bool inc=false );
    int getCacheSize();
    void wipe(Node* node);

private:


};

#endif
