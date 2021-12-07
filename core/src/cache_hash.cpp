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

void Cache_Hash::updateSubTreeLoad(Itemset &itemset, Item firstItem, Item secondItem, bool inc) {
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) continue;

        Itemset child_itemset = addItem(itemset, item);

        // when we build both branches itemsets, we don't need the parent anymore
        if (item == secondItem) Itemset().swap(itemset);

        updateItemsetLoad(child_itemset, inc);
        auto *child_node = (HashNode *) get(child_itemset);
        Itemset().swap(child_itemset);

        if (child_node and child_node->data and ((FND) child_node->data)->left and ((FND) child_node->data)->right) {
            Item nextFirstItem = item(((FND) child_node->data)->test, 0);
            Item nextSecondItem = item(((FND) child_node->data)->test, 1);
            updateSubTreeLoad(child_itemset, nextFirstItem, nextSecondItem, inc);
        }
    }
}

void Cache_Hash::updateItemsetLoad ( Itemset &itemset, bool inc ){
    if (store[itemset.size() - 1].find(itemset) != store[itemset.size() - 1].end() && store[itemset.size() - 1][itemset] != nullptr){
        if (inc) store[itemset.size() - 1][itemset]->count_opti_path++;
        else store[itemset.size() - 1][itemset]->count_opti_path--;
    }
}

int Cache_Hash::getCacheSize() {
    int size = 1;
    for (auto & depth : store) size += depth.size();
    return size;
}

void Cache_Hash::wipe() {
    for (auto & depth : store) {
        for (auto itr = depth.begin(); itr != depth.end(); ++itr) {
            if (itr->second->data && itr->second->count_opti_path == 0) depth.erase(itr->first);
            //else if (itr->second->count_opti_path < 0) for(;;) cout << "g";
        }
    }
}
