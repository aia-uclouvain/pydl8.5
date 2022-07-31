#ifndef CACHE_HASH_COVER_H
#define CACHE_HASH_COVER_H
#include "cache.h"
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct HashCoverNode;

struct HashCoverNode : public Node {
    // add variable necessary to compute metrics used to decide deletion order of nodes
    // keep in mind that too much variable will impact your memory consumption
    int n_reuse = 0;
    int n_subnodes = 0;

    HashCoverNode() : Node() {}

    ~HashCoverNode() {}
};

// structure used to represent the key of the cache. Here it is the cover, so a duplicate of the current state of the
// cover is performed as the RSBS data structure keeps only one instance of the cover through the whose search
struct MyCover{
    unsigned long* cover;
    int nwords;

    explicit MyCover(RCover* c){
        nwords = c->nWords;
        cover = new unsigned long[c->nWords];
        for (int i = 0; i < c->nWords; ++i) {
            cover[i] = c->coverWords[i].top().to_ulong();
        }
    }
};

// hash code function to convert the cover state into an integer
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

// equals function to check integrity in case of collision
template<>
struct std::equal_to<MyCover> {
    bool operator()(const MyCover& lhs, const MyCover& rhs) const noexcept {
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

    // the cache itself
    unordered_map<MyCover, HashCoverNode*>* store;

    // the deletion queue is a set of iterators pointing to each node of the hashtable. It also maintains the depth
    // of each node to ease the deletion. For this, pay attention to the iterator invalidation while removing nodes
    vector<pair<const unordered_map<MyCover, HashCoverNode*>::iterator*, Depth>> deletion_queue;
    float wipe_factor; // the percentage of the cache to wipe

    pair<Node*, bool> insert ( NodeDataManager*, int depth = 0);
    Node *get ( NodeDataManager*, int);
    int getCacheSize();
    //void wipe(); // override {}

//private:
//    void setOptimalNodes(HashCoverNode* node);
//    void setUsingNodes(HashCoverNode* node);

};

#endif
