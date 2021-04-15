#include "trie.h"
#include "query_best.h"
#include <algorithm>

using namespace std;


Trie::Trie(int maxsize) {
    cachesize = maxsize;
    root = new TrieNode;
    root->itemset.size = 0;
    root->itemset.elts = nullptr;
    root->data = nullptr;
    bucket = new TrieNode*[maxsize];
    for (int i=0; i<maxsize; i++) bucket[i] = nullptr;
    cout << "init" << endl;
}

TrieNode::~TrieNode() {
//    cout << "nodes deleted" << endl;
    if (data) delete data; //free (data); assumed allocated with malloc
}

Trie::~Trie() {
//    cout << "root veut delete" << endl;
    delete[] bucket;
//    cout << "root est delete" << endl;
}

hashcode Trie::gethashcode(Array<Item> itemset){
    if (itemset.size == 0) return 0;
    cout << "0" << endl;
    int val = itemset[0];
    cout << "1" << endl;
    for (int i = 1; i < itemset.size; ++i) val ^= itemset[i];
    cout << "2" << endl;
    return val % cachesize;
}

hashcode Trie::getlasthashcode(Array<Item> itemset){
    int val = itemset[0];
    for (int i = 1; i < itemset.size-1; ++i) val ^= itemset[i];
    return val % cachesize;
}

void Trie::remove(hashcode code){
    TrieNode* node = bucket[code];
//    for (auto p: node->parent){
//        TrieNode* par = bucket[p];
//        par->children.erase(find(par->children.begin(), par->children.end(), code));
//    }
    node->itemset.free();
    delete node;
}

/// insert itemset. Check from root and insert items only they do not exist using createTree
TrieNode *Trie::insert(Array<Item> itemset) {
// check size before insert. remove an element to potentially liberate space first
// implement the remove and the pop. impement the remove from the trie
    cout << "deb insert" << endl;
    hashcode toinsert = gethashcode(itemset);
    cout << "hash" << endl;
    if (bucket[toinsert]) {
        cout << "tor" << endl;
        if (bucket[toinsert]->itemset == itemset) return bucket[toinsert];
        remove(toinsert);
    }
    cout << "hg" << endl;
    TrieNode* node = new TrieNode;
    if (itemset.size == 0) return root;
    else{
        node->itemset = itemset; //copy constructor
        node->data = nullptr;
        bucket[toinsert] = node;
    }
    cout << "ty" << endl;
    /*if (itemset.size == 0){
        cout << "tt" << endl;
        return root;
        cout << "tttt" << endl;
    }
    else if (itemset.size == 1){
        cout << "tt" << endl;
        node->itemset = itemset; //copy constructor
        node->data = nullptr;
        bucket[toinsert] = node;
        cout << "tttt" << endl;
    }
    else {
        node->item = itemset[itemset.size-1];
        node->data = nullptr;
        bucket[gethashcode(itemset)] = node;
        bucket[getlasthashcode(itemset)]->children.push_back(toinsert);
        bucket[toinsert]->parent.push_back(getlasthashcode(itemset));
    }*/
    cout << "insert" << endl;
    return node;
}
