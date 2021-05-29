#ifndef CACHE_H
#define CACHE_H
#include "globals.h"
#include "nodedataManager.h"

using namespace std;

struct Node {
    NodeData *data; // data used to answer a query, if null this itemset is not closed
    Node() { data = nullptr; }
    virtual ~Node() { if (data) delete data; }
    virtual void update(vector<Item>&, vector<Node*>&) {}
    virtual void updateNode(Attribute attr, Attribute old, bool hasupdated) {}
    virtual void updateImportance(Node* old_first, Node* old_second, Node* new_first, Node* new_second, bool hasUpdated, Array<Item> itemset, Item old_firstI, Item old_secondI, Item new_firstI, Item new_secondI, Cache* cache ){}
};

class Cache {
public:
    Cache();
    virtual ~Cache() { delete root; }

    Node *root;
    int cachesize;
    virtual pair<Node *, bool>insert ( Array<Item> itemset, NodeDataManager* ) = 0;
    virtual Node *get ( Array<Item> itemset, Item item=-1 ){return nullptr;}
    virtual void uncountItemset ( Array<Item> itemset, bool inc=false ){return;}
};

#endif
