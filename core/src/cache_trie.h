#ifndef CACHE_TRIE_H
#define CACHE_TRIE_H

#include "cache.h"
//#include "node_trie.h"
//#include <vector>
//#include "nodeDataManager_TrieFreq.h"
//#include <tuple>
//#include <unordered_set>

//using namespace std;

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
//    unordered_set<TrieNode*> search_parents;
//    unordered_set<pair<TrieNode*,Itemset>> search_parents;
    TrieNode* trie_parent;
//    TrieFreq_NodeData *data; // data is the information kept by a node during the tree search

    TrieNode(): Node() {
//        data = nullptr;
    }
    ~TrieNode() {
//        delete data;
    }

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



class Cache_Trie : public Cache {

public:
    Cache_Trie(Depth maxdepth, WipeType wipe_type=Subnodes, int maxcachesize=0, float wipe_factor=.5f);

    //~Cache_Trie(){ delete root; for (auto node: heap) { delete node; } };
    ~Cache_Trie(){ delete root; for (const auto& node: heap) { delete node.first; } };

    float wipe_factor;
    vector<pair<TrieNode*,Itemset>> heap;
//    vector<pair<TrieNode*,TrieNode*>> heap;
    pair<Node*, bool> insert ( Itemset &itemset );
    Node *get ( const Itemset &itemset ) override;
    int getCacheSize() override;
    void wipe() override;
//    void updateParents(Node* best, Node* left, Node* right, Itemset = Itemset()) override;
    //vector<TrieNode*> heap;
    //void updateParents(Node* best, Node* left, Node* right);

private:
    TrieNode *addNonExistingItemsetPart (Itemset &itemset, int pos, vector<TrieEdge>::iterator& geqEdge_it, TrieNode *parent);
    int computeSubNodes(TrieNode* node);
    bool isConsistent(TrieNode* node);
    void setOptimalNodes(TrieNode* node, const Itemset& itemset, int& n_used);
    void setUsingNodes(TrieNode* node, const Itemset& itemset, int& n_used);
    TrieNode *getandSet ( const Itemset &itemset, int& n_used );
//    void retroPropagate(TrieNode* node);
    //bool isConsistent(TrieNode* node, vector<Item>);

};

#endif