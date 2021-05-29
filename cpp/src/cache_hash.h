#ifndef CACHE_HASH_H
#define CACHE_HASH_H
#include "cache.h"
#include <cmath>

using namespace std;

struct HashNode : Node {
    Array<Item> itemset;
    HashNode(Array<Item> itemset1): Node() { itemset = itemset1; }
    HashNode(): Node() {itemset.size = 0; itemset.elts = nullptr;}
    ~HashNode() { if (itemset.elts) itemset.free(); }
};

class Cache_Hash: public Cache {
public:
    Cache_Hash(int maxcachesize, int maxlength);
    ~Cache_Hash() { delete[] bucket; }

    int maxcachesize;
    int maxlength;
    HashNode** bucket;
    pair<Node *, bool>insert ( Array<Item> itemset, NodeDataManager* );

private:
    int gethashcode ( Array<Item> itemset );
    void remove( int hashcode );

};

#endif
