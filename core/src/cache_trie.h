#ifndef CACHE_TRIE_H
#define CACHE_TRIE_H

#include "cache.h"
#include <vector>
#include "nodeDataManagerFreq.h"
#include <tuple>

using namespace std;

struct TrieNode;

struct TrieEdge {
  Item item;
  TrieNode *subtrie;
};

struct TrieNode : Node {
    int n_subnodes = 0;
    int n_reuse = 0;
    Depth depth = 0;
    vector<TrieEdge> edges;
    vector<TrieNode*> search_parents;
    TrieNode* trie_parent;

    TrieNode(): Node() { count_opti_path = 1; }
    ~TrieNode() {}

    void invalidateChildren() {
        for (const auto & edge: edges) {
            edge.subtrie->count_opti_path = INT32_MIN;
            edge.subtrie->invalidateChildren();
        }
    }
};

class Cache_Trie : public Cache {

public:
    Cache_Trie(Depth maxdepth, WipeType wipe_type=Subnodes, int maxcachesize=0, float wipe_factor=.5f);
    ~Cache_Trie(){ delete root; for (auto node: heap) { delete node; } };

    pair<Node*, bool> insert ( Itemset &itemset );
    Node *get ( const Itemset &itemset );
    void updateItemsetLoad(Itemset &itemset, bool inc=false);
    void updateSubTreeLoad(Itemset &itemset, Item firstI, Item secondI, bool inc=false);
    int getCacheSize();
    void wipe();
    void updateRootPath(Itemset &itemset, int value);
    vector<TrieNode*> heap;
    float wipe_factor;
    void updateParents(Node* best, Node* left, Node* right);

    void printItemsetLoad(Itemset &itemset, bool inc=false);
    void printSubTreeLoad(Itemset &itemset, Item firstI, Item secondI, bool inc=false);
    bool isLoadConsistent(TrieNode* node, Itemset itemset=Itemset());
    bool isNonNegConsistent(TrieNode* node);

    Node* newNode(){return new TrieNode();}

private:
    TrieNode *addNonExistingItemsetPart (Itemset &itemset, int pos, vector<TrieEdge>::iterator& geqEdge_it, TrieNode *parent);
    int computeSubNodes(TrieNode* node);
    bool isConsistent(TrieNode* node);

    bool isConsistent(TrieNode* node, vector<Item>);

};

#endif