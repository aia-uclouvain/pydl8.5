#ifndef CACHE_TRIE_H
#define CACHE_TRIE_H

#include "cache.h"
#include <vector>
#include "nodeDataManagerFreq.h"
#include <tuple>
#include <unordered_set>

using namespace std;

struct TrieNode;

struct TrieEdge {
  Item item;
  TrieNode *subtrie;
};

template<>
struct std::hash<pair<TrieNode*,Itemset>> {
    std::size_t operator()(const pair<TrieNode*,Itemset>& array) const noexcept {
        return std::hash<TrieNode*>{}(array.first);
    }
};

template<>
struct std::equal_to<pair<TrieNode*,Itemset>> {
    bool operator()(const pair<TrieNode*,Itemset>& lhs, const pair<TrieNode*,Itemset>& rhs) const noexcept {
        return lhs.first == rhs.first;
    }
};

struct TrieNode : Node {
    int n_subnodes = 0;
    int n_reuse = 0;
    Depth depth = 0;
    bool is_used = false;
    vector<TrieEdge> edges;
//    vector<TrieNode*> search_parents;
//    vector<pair<TrieNode*,Itemset>> search_parents;
    unordered_set<pair<TrieNode*,Itemset>> search_parents;
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
//    ~Cache_Trie(){ delete root; for (auto node: heap) { delete node; } };
    ~Cache_Trie(){ delete root; for (auto node: heap) { delete node.first; } };

    pair<Node*, bool> insert ( Itemset &itemset );
    Node *get ( const Itemset &itemset );
    void updateItemsetLoad(Itemset &itemset, bool inc=false);
    void updateSubTreeLoad(Itemset &itemset, Item firstI, Item secondI, bool inc=false);
    int getCacheSize();
    void wipe();
    void updateRootPath(Itemset &itemset, int value);
//    vector<TrieNode*> heap;
    vector<pair<TrieNode*,Itemset>> heap;
    float wipe_factor;
    void updateParents(Node* best, Node* left, Node* right, Itemset = Itemset());
//    void updateParents(Node* best, Node* left, Node* right);

    void printItemsetLoad(Itemset &itemset, bool inc=false);
    void printSubTreeLoad(Itemset &itemset, Item firstI, Item secondI, bool inc=false);
    bool isLoadConsistent(TrieNode* node, Itemset itemset=Itemset());
    bool isNonNegConsistent(TrieNode* node);

    Node* newNode(){return new TrieNode();}

private:
    TrieNode *addNonExistingItemsetPart (Itemset &itemset, int pos, vector<TrieEdge>::iterator& geqEdge_it, TrieNode *parent);
    int computeSubNodes(TrieNode* node);
    bool isConsistent(TrieNode* node);
    void setOptimalNodes(TrieNode* node, int& n_used);
    void setUsingNodes(TrieNode* node, Itemset& itemset, int& n_used);
    Node *getandSet ( const Itemset &itemset, int& n_used );
    void retroPropagate(TrieNode* node);

    bool isConsistent(TrieNode* node, vector<Item>);

};

#endif