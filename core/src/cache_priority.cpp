#include "cache_priority.h"

using namespace std;

Cache_Priority::Cache_Priority(Depth maxdepth, WipeType wipe_type, int maxcachesize): Cache(maxdepth, wipe_type, maxcachesize) {
    root = new PriorityNode();
    bucket = new PriorityNode*[maxcachesize];
    for (int i=0; i<maxcachesize; i++) bucket[i] = nullptr;
}

void Cache_Priority::addpriority( int priority, int index ){
    nodemapper.push({priority, index});
}

int Cache_Priority::removelessimportantnode(){
    int index = nodemapper.top().second;
    remove(index);
    nodemapper.pop();
}

void Cache_Priority::remove(int index){
    delete bucket[index];
    bucket[index] = nullptr;
}

Node * Cache_Priority::insert(Array<Item> itemset, NodeDataManager* nodeDataManager) {
    if (itemset.size == 0) {
        root->data = nodeDataManager->initData();
        return root;
    }
    return root;
}
