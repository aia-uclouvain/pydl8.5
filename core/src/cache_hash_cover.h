#ifndef CACHE_HASH_COVER_H
#define CACHE_HASH_COVER_H
#include "cache.h"
//#include <cmath>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct HashCoverNode;

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

struct HashCoverNode : public Node {

//    NodeData *data; // data is the information kept by a node during the tree search
//    unordered_set<HashCoverNode*> search_parents;
//    unordered_set<pair<HashCoverNode*,Itemset>> search_parents;
//    int n_reuse = 0;

    HashCoverNode() : Node() {
//        data = nullptr;
    }

    ~HashCoverNode() {
//        delete data;
    }
};

struct MyCover{
    unsigned long* cover;
    int nwords;

    explicit MyCover(RCover* c){
        cover = new unsigned long[c->limit.top()];
        int pos = 0;
        for (int i = 0; i < c->nWords; ++i) {
            if (c->coverWords[i].top().any()) {
                cover[pos] = c->coverWords[i].top().to_ulong();
                pos++;
            }
        }
        nwords = pos;
    }
};

template<>
struct std::hash<MyCover> {
    std::size_t operator()(const MyCover& array) const noexcept {
        std::size_t h = array.nwords;
        for (int i = 0; i < array.nwords; ++i) {
            h ^= array.cover[i] + 0x9e3779b9 + 64 * h + h / 4;
        }
        return h;
    }
};

template<>
struct std::equal_to<MyCover> {
    bool operator()(const MyCover& lhs, const MyCover& rhs) const noexcept {
        if (lhs.nwords != rhs.nwords) return false;
        for (int i = 0; i < lhs.nwords; ++i) {
            if (lhs.cover[i] != rhs.cover[i]) return false;
        }
        return true;
    }
};

class Cache_Hash_Cover : public Cache {
public:
    Cache_Hash_Cover(Depth maxdepth, WipeType wipe_type, int maxcachesize=0, float wipe_factor=.5f);
    ~Cache_Hash_Cover() {
        delete root;
        for (int i = 0; i < maxdepth; ++i) {
            for(auto &elt: store[i]) delete elt.second;
        }
    }

    unordered_map<MyCover, HashCoverNode*>* store;
    float wipe_factor;
//    vector<pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Itemset>>* heap;
//    vector<const unordered_map<MyCover, HashCoverNode*>::iterator*>* heap;
//    vector<pair<HashCoverNode*, Itemset>>* heap;
//    vector<HashCoverNode*>* heap;

    pair<Node*, bool> insert ( NodeDataManager*, int depth = 0, Itemset itemset = Itemset());
    Node *get ( NodeDataManager*, int);
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
