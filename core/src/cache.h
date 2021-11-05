#ifndef CACHE_H
#define CACHE_H
#include "globals.h"
#include "nodeDataManager.h"

using namespace std;

enum CacheType {
    CacheTrie, CacheLtdTrie, CacheHash, CachePriority
};

enum WipeType {
    All, Subnodes, Recall
};

/*This struct is used to represent a node in the tree search algorithm*/
struct Node {
    NodeData *data; // data is the information kept by a node during the tree search
    int count_opti_path;
    Node() { data = nullptr; }
    virtual ~Node() { delete data; }
};


/*This class represents the tree structure built during the tree search algorithm*/
class Cache {
public:
    Cache(Depth maxdepth, WipeType wipe_type, Size maxcachesize);
    virtual ~Cache() {}

    Node *root; // the root node of the tree
    Size cachesize; // the size (number of nodes) of the cache
    Size maxcachesize; // the maximum size allowed by the cache system
    Depth maxdepth;
    WipeType wipe_type;
    virtual Node * insert ( Array<Item> itemset, NodeDataManager* ) = 0; // add a node to the tree
    virtual Node* get ( Array<Item> itemset){return nullptr;} // get a node in the tree based on its corresponding itemset
    virtual void updateSubTreeLoad(Array<Item> itemset, Item firstI, Item secondI, bool inc=false){}
    virtual void updateItemsetLoad ( Array<Item> itemset, bool inc=false ){}
    virtual int getCacheSize(){return cachesize;}
    virtual void updateRootPath(Array<Item> itemset, int value){}
};

#endif
