#ifndef CACHE_HASH_H
#define CACHE_HASH_H
#include "cache.h"
#include <cmath>
#include <unordered_map>
#include "nodeDataManagerFreq.h"

using namespace std;

struct HashNode : Node {
    /*Array<Item> itemset;
    HashNode(Array<Item> itemset1): Node() { itemset = itemset1; }
    HashNode(): Node() {itemset.size = 0; itemset.elts = nullptr;}
    ~HashNode() { if (itemset.elts) itemset.free(); }*/
};

template<>
struct hash<Itemset> {
    std::size_t operator()(const Itemset& array) const noexcept {
        std::size_t h = array.size();
        for (auto elt: array) h ^= elt + 0x9e3779b9 + 64 * h + h / 4;
        return h;
    }
};

class Cache_Hash: public Cache {
public:
    Cache_Hash(Depth maxdepth, WipeType wipe_type, int maxcachesize);
    ~Cache_Hash() {delete root; for(auto &depth : store) for(auto &elt: depth) delete elt.second;}

    vector<unordered_map<Itemset, HashNode*, hash<Itemset>>> store;

    pair<Node*, bool> insert ( Itemset& itemset );
    Node *get ( Itemset& itemset);
    void updateSubTreeLoad(Itemset &itemset, Item firstItem, Item secondItem, bool inc=false);
    void updateItemsetLoad ( Itemset &itemset, bool inc=false );
    int getCacheSize();
    void wipe(Node* node);

private:


};

#endif
