#include "cache_hash.h"

using namespace std;

Cache_Hash::Cache_Hash(Depth maxdepth, WipeType wipe_type, int maxcachesize): Cache(maxdepth, wipe_type, maxcachesize) {
    root = new HashNode();
    store.reserve(maxdepth);
    for (int i = 0; i < maxdepth; ++i) {
        store.emplace_back(unordered_map<Itemset, HashNode*>());
    }
}

pair<Node*, bool> Cache_Hash::insert(Itemset &itemset) {
    if (itemset.empty()) {
        cachesize++;
        return {root, true};
    }
    else {
        if (cachesize >= maxcachesize && maxcachesize > 0) wipe();
        auto* node = new HashNode();
        store[itemset.size() - 1].insert({itemset, node});
        cachesize++;
        return {node, true};
    }
}

Node *Cache_Hash::get ( Itemset &itemset){
    return store[itemset.size() - 1][itemset];
}


int Cache_Hash::getCacheSize() {
    int size = 1;
    for (auto & depth : store) size += depth.size();
    return size;
}

void Cache_Hash::wipe() {
    for (auto & depth : store) {
        for (auto itr = depth.begin(); itr != depth.end(); ++itr) {
            if (itr->second->data && not itr->second->is_used) depth.erase(itr->first);
            //else if (itr->second->count_opti_path < 0) for(;;) cout << "g";
        }
    }
}
