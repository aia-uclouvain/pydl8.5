#include "cache_hash.h"

using namespace std;

Cache_Hash::Cache_Hash(int maxcachesize, int maxdepth): Cache(), maxcachesize(maxcachesize) {
    root = new HashNode();
    store.alloc(maxdepth);
    for (int i = 0; i < maxdepth; ++i) {
        store[i] = unordered_map<Array<Item>, HashNode*>();
    }
    /*bucket = new HashNode*[maxcachesize];
    for (int i=0; i<maxcachesize; i++) bucket[i] = nullptr;*/
}

pair<Node *, bool> Cache_Hash::insert(Array<Item> itemset, NodeDataManager* nodeDataManager) {
    if (itemset.size == 0) {
        //cout << "zero" << endl;
        root->data = nodeDataManager->initData();
        return {root, true};
    }
    else {
        //cout << "plus que zero" << endl;
        HashNode* node = new HashNode();
        node->data = nodeDataManager->initData();
        store[itemset.size - 1].insert({itemset, node});
        return {node, true};
    }
}

Node *Cache_Hash::get ( Array<Item> itemset){
    return store[itemset.size - 1][itemset];
}

void Cache_Hash::updateSubTreeLoad(Array<Item> itemset, Item firstItem, Item secondItem, bool inc){
    for (auto item: {firstItem, secondItem}) {
        if (item == -1) continue;
        Array<Item> itemset1 = addItem(itemset, item);
        if (item == secondItem) itemset.free();
        updateItemsetLoad(itemset1, inc);

        auto* node = (HashNode*)get(itemset1);

        if ( ((FND)node->data)->left && ((FND)node->data)->right ){
            Item nextFirstItem = item(((FND)node->data)->test, 0);
            Item nextSecondItem = item(((FND)node->data)->test, 1);
            updateSubTreeLoad( itemset1, nextFirstItem, nextSecondItem, inc );
        }
        else if (item == secondItem) itemset1.free();
    }
}

void Cache_Hash::updateItemsetLoad ( Array<Item> itemset, bool inc ){
    if (inc) store[itemset.size - 1][itemset]->count_opti_path++;
    else store[itemset.size - 1][itemset]->count_opti_path--;
}

int Cache_Hash::getCacheSize() {
    int size = 0;
    for (int i = 0; i < store.size; ++i) {
        size += store[i].size();
    }
    return size + 1;
}

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = itemset[0] + 2 + 0x9e3779b9;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= itemset[i] + 2 + 0x9e3779b9 + (val<<6) + (val>>2);
        else val ^= 1 + 0x9e3779b9 + (val<<6) + (val>>2);
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = ((itemset[0] + 2) * pow(37, maxlength-1)) + 0x9e3779b9;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= ((int)((itemset[i] + 2) * pow(37, maxlength-i-1))) + 0x9e3779b9 + (val<<6) + (val>>2);
        else val ^= ((int)((1) * pow(37, maxlength-i-1))) + 0x9e3779b9 + (val<<6) + (val>>2);
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = (itemset[0] + 2) * pow(37, maxlength-1);
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= (int)((itemset[i] + 2) * pow(37, maxlength-i-1));
        else val ^= (int)((1) * pow(37, maxlength-i-1));
    }
    return val % maxcachesize;
}*/

/*int Cache_Hash::gethashcode(Array<Item> itemset){
    int val = itemset[0] + 2;
    for (int i = 1; i < maxlength; ++i) {
        if (i < itemset.size) val ^= itemset[i] + 2;
        else val ^= 1;
    }
    return val % maxcachesize;
}

void Cache_Hash::remove(int hashcode){
    delete bucket[hashcode];
    bucket[hashcode] = nullptr;
}*/

/*pair<Node *, bool> Cache_Hash::insert(Array<Item> itemset, NodeDataManager* nodeDataManager) {
//    cout << "insert" << endl;
    if (itemset.size == 0) {
        root->data = nodeDataManager->initData();
        return {root, true};
    }
    int in_hashcode = gethashcode(itemset);
    if (bucket[in_hashcode]) {
        if (bucket[in_hashcode]->itemset == itemset) return {bucket[in_hashcode], false};
//        if (itemset.size == 1 && itemset[0] == 0) cout << "\t\t\t\t\t\tREM 00000000" << endl;
//        if (in_hashcode == 2) { cout << "\t\t\t\t\t\tREM 00000000" << endl;
//            printItemset(itemset); }
        remove(in_hashcode);
    } else cachesize++;
    Array<Item> copy_itemset;
    copy_itemset.duplicate(itemset); // like copy constructor
    HashNode* node = new HashNode(copy_itemset);
    bucket[in_hashcode] = node;
    node->data = nodeDataManager->initData();
    return {node, true};
}*/
