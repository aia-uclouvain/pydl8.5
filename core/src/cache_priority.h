#ifndef CACHE_PRIORITY_H
#define CACHE_PRIORITY_H

#include "cache.h"
#include <queue>

using namespace std;
typedef pair<int, int> NodePriority;

struct PriorityNode : Node {
    Array<Item> itemset;
    PriorityNode(Array<Item> itemset1): Node() { itemset = itemset1; }
    PriorityNode(): Node() {itemset.size = 0; itemset.elts = nullptr;}
    ~PriorityNode() { if (itemset.elts) itemset.free(); }
};

class Cache_Priority: public Cache {
public:
    Cache_Priority(Depth maxdepth, WipeType wipe_type, int maxcachesize);
    ~Cache_Priority() { delete[] bucket; }

    struct cmp {
        bool operator()(const NodePriority &a, const NodePriority &b) {
            return a.first > b.first; // < for Max heap (highest first) and > for min heap (lowest first)
        };
    };
    PriorityNode** bucket;
    priority_queue<NodePriority, vector<NodePriority>, cmp> nodemapper;
    pair<Node*, bool> insert ( Array<Item> itemset );

private:
    void remove( int index );
    void addpriority( int priority, int index );
    int removelessimportantnode();

};

#endif
