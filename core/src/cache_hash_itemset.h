#ifndef CACHE_HASH_ITEMSET_H
#define CACHE_HASH_ITEMSET_H
#include "cache.h"
//#include <cmath>
// #include <iostream>
// #include <fstream>
#include <unordered_map>
// #include <unordered_set>
// #include "globals.h"

using namespace std;

struct HashItemsetNode;

/*template<>
struct std::hash<pair<HashCoverNode*,Itemset>> {
    std::size_t operator()(const pair<HashCoverNode*,Itemset>& array) const noexcept {
        return std::hash<HashCoverNode*>{}(array.first);
    }
};

template<>
struct std::equal_to<pair<HashCoverNode*,Itemset>> {
    bool operator()(const pair<HashCoverNode*,Itemset>& lhs, const pair<HashCoverNode*,Itemset>& rhs) const noexcept {
        return lhs.first == rhs.first;
    }
};*/

struct HashItemsetNode : public Node {

//    NodeData *data; // data is the information kept by a node during the tree search
//    unordered_set<HashCoverNode*> search_parents;
//    unordered_set<pair<HashCoverNode*,Itemset>> search_parents;
//    int n_reuse = 0;

    HashItemsetNode() : Node() {
//        data = nullptr;
    }

    ~HashItemsetNode() {
//        delete data;
    }
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

class Cache_Hash_Itemset : public Cache {
public:
    Cache_Hash_Itemset(Depth maxdepth, WipeType wipe_type, int maxcachesize=0, float wipe_factor=.5f);
    ~Cache_Hash_Itemset() {
        delete root;
        for(auto &elt: store) delete elt.second;
    }

    unordered_map<Itemset, HashItemsetNode*> store;
    float wipe_factor;
//    vector<pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Itemset>>* heap;
//    vector<const unordered_map<MyCover, HashCoverNode*>::iterator*>* heap;
//    vector<pair<HashCoverNode*, Itemset>>* heap;
//    vector<HashCoverNode*>* heap;

    pair<Node*, bool> insert (Itemset &itemset);
    Node *get ( const Itemset &itemset);
    int getCacheSize();
//    void wipe() override;
    void wipe() override {}
//    void updateParents(Node* best, Node* left, Node* right, Itemset = Itemset()) override;
//    void wipe(Node* node);

private:
//    void setOptimalNodes(HashCoverNode* node, int& n_used);
//    void setUsingNodes(HashCoverNode* node, int& n_used);


};

#endif
