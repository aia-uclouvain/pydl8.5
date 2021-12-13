#ifndef CACHE_HASH_COVER_H
#define CACHE_HASH_COVER_H
#include "cache.h"
#include <cmath>
#include <unordered_map>
#include "nodeDataManagerFreq.h"

using namespace std;

struct HashCoverNode : public Node {
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
    Cache_Hash_Cover(Depth maxdepth, WipeType wipe_type, int maxcachesize);
    ~Cache_Hash_Cover() {
        delete root;
        for (int i = 0; i < maxdepth; ++i) {
            for(auto &elt: store[i]) delete elt.second;
//            store[i].clear();
        }
    }

    unordered_map<MyCover, HashCoverNode*>* store;

    pair<Node*, bool> insert ( NodeDataManager*, int depth = 0, bool rootnode = false );
    Node *get ( NodeDataManager*, int);
    int getCacheSize();
//    void updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc=false);
//    void updateItemsetLoad ( Array<Item> itemset, bool inc=false );
//    void wipe(Node* node);

private:


};

#endif
