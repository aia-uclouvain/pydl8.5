#ifndef CACHE_HASH_ITEMSET_H
#define CACHE_HASH_ITEMSET_H
#include "cache.h"
#include <unordered_map>

using namespace std;

struct HashItemsetNode;

struct HashItemsetNode : public Node {

    HashItemsetNode() : Node() {}

    ~HashItemsetNode() {}
};

template<>
struct std::hash<Itemset> {
    std::size_t operator()(const Itemset& array) const noexcept {
        std::size_t h = array.size();
        for (int i = 0; i < array.size(); ++i) {
            h ^= array[i] + 0x9e3779b9 + 64 * h + h / 4;
        }
        return h;
    }
};

template<>
struct std::equal_to<Itemset> {
    bool operator()(const Itemset& lhs, const Itemset& rhs) const noexcept {
        if (lhs.size() != rhs.size()) return false;
        return std::equal(lhs.begin(), lhs.begin() + lhs.size(), rhs.begin());
    }
};

// the implementation of the memory wiping for the current cache is similar to the one described in cache_hash_cover
class Cache_Hash_Itemset : public Cache {
public:
    Cache_Hash_Itemset(Depth maxdepth, WipeType wipe_type, int maxcachesize=0, float wipe_factor=.5f);
    ~Cache_Hash_Itemset() {
        delete root;
        for(auto &elt: store) delete elt.second;
    }

    unordered_map<Itemset, HashItemsetNode*> store;

    pair<Node*, bool> insert (Itemset &itemset);
    Node *get ( const Itemset &itemset);
    int getCacheSize();
    void wipe() override {}

};

#endif
