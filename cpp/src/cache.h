#ifndef CACHE_H
#define CACHE_H
#include "globals.h"
#include "nodedataManager.h"

using namespace std;

/*This struct is used to represent a node in the tree search algorithm*/
struct Node {
    NodeData *data; // data is the information kept by a node during the tree search
    int count_opti_path;
    Node() { data = nullptr; }
    virtual ~Node() { if (data) delete data; }
    /*virtual void update(vector<Item>&, vector<Node*>&) {}
    virtual void updateNode(Attribute attr, Attribute old, bool hasupdated) {}
    virtual void updateSubTreeLoad(Node* old_first, Node* old_second, Node* new_first, Node* new_second, bool hasUpdated, Array<Item> itemset, Item old_firstI, Item old_secondI, Item new_firstI, Item new_secondI, Cache* cache ){}*/
};


/*This class represents the tree structure built during the tree search algorithm*/
class Cache {
public:
    Cache();
    virtual ~Cache() { delete root; }

    Node *root; // the root node of the tree
    int cachesize; // the size (number of nodes) of the tree
    virtual pair<Node *, bool>insert ( Array<Item> itemset, NodeDataManager* ) = 0; // add a node to the tree
    virtual Node *get ( Array<Item> itemset){return nullptr;} // get a node in the tree based on its corresponding itemset
    virtual void updateSubTreeLoad(Array<Item> itemset, Item firstI, Item secondI, bool inc=false){}
    virtual void updateItemsetLoad ( Array<Item> itemset, bool inc=false ){return;}
    virtual int getCacheSize(){return cachesize;}
};

#endif
