#ifndef CACHE_TRIE_H
#define CACHE_TRIE_H

#include "cache.h"

struct TrieNode;

struct TrieEdge {
    Item item;
    TrieNode *subtrie;
};

struct TrieNode : Node {
    int n_subnodes = 0;
    int n_reuse = 0;
    Depth depth = 0;
    std::vector<TrieEdge> edges;
    TrieNode* trie_parent;

    TrieNode(): Node() {}
    ~TrieNode() {}
};

class Cache_Trie : public Cache {

public:
    Cache_Trie(Depth maxdepth, WipeType wipe_type=Subnodes, int maxcachesize=0, float wipe_factor=.5f);

    ~Cache_Trie(){ delete root; for (auto node: deletion_queue) { delete node; } };

    float wipe_factor;
    vector<TrieNode*> deletion_queue;
    pair<Node*, bool> insert ( Itemset &itemset );
    Node *get ( const Itemset &itemset ) override;
    int getCacheSize() override;
    void wipe() override;

private:
    TrieNode *addNonExistingItemsetPart (Itemset &itemset, int pos, vector<TrieEdge>::iterator& geqEdge_it, TrieNode *parent);
    int computeSubNodes(TrieNode* node);
    void setOptimalNodes(TrieNode* node, const Itemset& itemset);
    void setUsingNodes(TrieNode* node, const Itemset& itemset);
    TrieNode *getandSet (const Itemset &itemset);

};

#endif