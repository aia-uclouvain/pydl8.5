#include "cache_hash_itemset.h"

using namespace std;

// the implementation of the memory wiping for the current cache is similar to the one described in cache_hash_cover
Cache_Hash_Itemset::Cache_Hash_Itemset(Depth maxdepth, WipeType wipe_type, int maxcachesize, float wipe_factor) : Cache(maxdepth, wipe_type, maxcachesize) {
    root = new HashItemsetNode();
}

pair<Node *, bool> Cache_Hash_Itemset::insert(Itemset &itemset) {
    if (itemset.empty()) {
        cachesize++;
        return {root, true};
    }
    auto *node = new HashItemsetNode();
    auto info = store.insert({itemset, node});
    if (not info.second) delete node; // if node already exists
    else cachesize++;
    return {info.first->second, info.second};
}

Node *Cache_Hash_Itemset::get(const Itemset &itemset) {
    auto it = store.find(itemset);
    if (it != store.end()) return it->second;
    else return nullptr;
}

int Cache_Hash_Itemset::getCacheSize() {
    return store.size() + 1;
}
